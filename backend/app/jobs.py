"""DB-backed job queue with one asyncio worker. No Redis.

Stages chain: ingest -> extract -> match -> notify. Each is a row in `jobs`, claimed with
SELECT ... FOR UPDATE SKIP LOCKED (ignored by SQLite), retried 3 times with exponential backoff,
then marked failed with the error kept.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import db as dbmod
from .models import Job

log = logging.getLogger("reunite.jobs")
MAX_ATTEMPTS = 3
POLL_SECONDS = 0.5

Handler = Callable[[Session, dict], None]
_handlers: dict[str, Handler] = {}


def handler(kind: str):
    def deco(fn: Handler) -> Handler:
        _handlers[kind] = fn
        return fn

    return deco


def enqueue(db: Session, kind: str, payload: dict, delay_seconds: float = 0) -> Job:
    job = Job(kind=kind, payload=payload, status="queued", run_after=datetime.now(timezone.utc) + timedelta(seconds=delay_seconds))
    db.add(job)
    db.flush()
    return job


@handler("ingest")
def _ingest(db: Session, p: dict) -> None:
    from .services import items

    items.run_ingest(db, uuid.UUID(p["item_id"]))
    enqueue(db, "extract", p)


@handler("extract")
def _extract(db: Session, p: dict) -> None:
    from .services import items

    items.run_extract(db, uuid.UUID(p["item_id"]))
    enqueue(db, "match", p)


@handler("match")
def _match(db: Session, p: dict) -> None:
    from .services import matching

    matching.run_match(db, uuid.UUID(p["item_id"]))
    enqueue(db, "notify", p)


@handler("notify")
def _notify(db: Session, p: dict) -> None:
    from .services import matching

    matching.notify_matches(db, uuid.UUID(p["item_id"]))


def process_next(db: Session) -> bool:
    """Run one due job. Returns False when the queue is empty."""
    now = datetime.now(timezone.utc)
    job = db.scalar(
        select(Job).where(Job.status == "queued", Job.run_after <= now).order_by(Job.created_at).limit(1).with_for_update(skip_locked=True)
    )
    if job is None:
        return False
    job.status, job.attempts = "running", job.attempts + 1
    db.commit()
    try:
        _handlers[job.kind](db, job.payload)
        job.status, job.finished_at, job.last_error = "done", datetime.now(timezone.utc), None
    except Exception as e:  # noqa: BLE001
        db.rollback()
        log.exception("job %s (%s) failed on attempt %s", job.id, job.kind, job.attempts)
        job = db.get(Job, job.id)
        assert job is not None
        job.last_error = f"{type(e).__name__}: {e}"[:2000]
        if job.attempts >= MAX_ATTEMPTS:
            job.status, job.finished_at = "failed", datetime.now(timezone.utc)
        else:
            job.status = "queued"
            job.run_after = datetime.now(timezone.utc) + timedelta(seconds=2 ** job.attempts)
    db.commit()
    return True


def drain(db: Session, limit: int = 100) -> int:
    """Run queued jobs until none are due. Used by tests, the seed script and the eval harness."""
    n = 0
    while n < limit and process_next(db):
        n += 1
    return n


def recover_stale(db: Session) -> int:
    """Requeue jobs left 'running' by a crash or restart. Call once at startup, before the worker starts.

    There is one worker, so a job still marked running when the server boots was cut off (an out of memory kill
    does this) and nothing will ever pick it up again: its report would stay at "Reading it" forever. A job that
    has already used all its attempts is failed instead, so a job that keeps crashing the server cannot loop."""
    now = datetime.now(timezone.utc)
    stale = db.scalars(select(Job).where(Job.status == "running")).all()
    for job in stale:
        if job.attempts >= MAX_ATTEMPTS:
            job.status, job.finished_at, job.last_error = "failed", now, "interrupted by a restart or crash"
        else:
            job.status, job.run_after = "queued", now
    db.commit()
    return len(stale)


def _tick() -> bool:
    with dbmod.session_scope() as db:
        return process_next(db)


async def worker(stop: asyncio.Event) -> None:
    """The single asyncio worker. DB and model calls run in a thread so the event loop stays free."""
    log.info("job worker started")
    while not stop.is_set():
        try:
            did = await asyncio.to_thread(_tick)
        except Exception:  # noqa: BLE001
            log.exception("worker tick failed")
            did = False
        if not did:
            try:
                await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
            except asyncio.TimeoutError:
                pass
    log.info("job worker stopped")
