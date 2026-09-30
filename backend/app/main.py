from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from . import db as dbmod
from .api import admin, auth, claims, items, matches, media, notifications, reference, tags
from .config import get_settings
from .errors import install_error_handlers
from .jobs import recover_stale, worker
from .ml import registry
from .models import Item
from .services import media as media_svc, routing
from .services.zones import load_zones

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("reunite")


def purge_old(db) -> int:  # noqa: ANN001
    """Delete closed and returned items (and their photos) past RETENTION_DAYS."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=get_settings().retention_days)
    n = 0
    for it in db.scalars(select(Item).where(Item.status.in_(("closed", "returned")), Item.updated_at < cutoff)):
        for img in it.images:
            media_svc.delete_files(img)
        db.delete(it)
        n += 1
    db.commit()
    return n


@asynccontextmanager
async def lifespan(app: FastAPI):
    if dbmod.SessionLocal is None:
        dbmod.configure()
    registry.setup_cache_dirs()
    media_svc.media_root()
    with dbmod.session_scope() as db:
        load_zones(db)
        removed = purge_old(db)
        if removed:
            log.info("purged %s closed items past retention", removed)
        routed = routing.route_unclaimed(db)
        if routed:
            log.info("routed %s unclaimed items to the desk", routed)
        recovered = recover_stale(db)
        if recovered:
            log.info("recovered %s jobs interrupted by a restart", recovered)
    stop = asyncio.Event()
    task = None
    if os.environ.get("DISABLE_WORKER") != "1":
        task = asyncio.create_task(worker(stop))
    yield
    stop.set()
    if task:
        await task


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="Reunite", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=s.origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"], expose_headers=["Retry-After"])
    install_error_handlers(app)
    for r in (auth.router, reference.router, items.router, matches.router, claims.router, notifications.router, admin.router, tags.router):
        app.include_router(r, prefix="/api/v1")
    app.include_router(media.router)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "ml": registry.status(), "campus": s.campus_name}

    return app


app = create_app()
