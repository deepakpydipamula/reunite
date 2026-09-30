from __future__ import annotations

import io
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import DeskOrAdmin, current_user
from ..errors import ApiError
from ..jobs import enqueue
from ..ml import pipeline
from ..models import STAFF_ROLES, Item, Match, User
from ..services import events, items as items_svc, media
from ..services.serializers import item_out, match_out, public_item

router = APIRouter(tags=["items"])


class ItemPatch(BaseModel):
    text: str | None = None
    attributes: dict | None = None
    hidden: dict | None = None
    zone_id: str | None = None
    custody_zone_id: str | None = None
    occurred_from: datetime | None = None
    occurred_to: datetime | None = None


class BulkRow(BaseModel):
    text: str
    zone_id: str
    custody_zone_id: str | None = None
    occurred_at: datetime | None = None


class BulkBody(BaseModel):
    items: list[BulkRow]


class CloseBody(BaseModel):
    reason: str | None = None


def _dt(raw: str | None, field: str) -> datetime:
    try:
        return datetime.fromisoformat((raw or "").replace("Z", "+00:00"))
    except ValueError as e:
        raise ApiError(422, "invalid_request", "Use an ISO 8601 date and time.", {field: "invalid date"}) from e


async def _files(form, *names: str) -> list[bytes]:  # noqa: ANN001
    out: list[bytes] = []
    for n in names:
        for f in form.getlist(n):
            if hasattr(f, "read"):
                out.append(await f.read())
    return out


def _visible(db: Session, item_id: str, user: User) -> Item:
    try:
        item = db.get(Item, uuid.UUID(item_id))
    except ValueError:
        item = None
    if item is None or (item.owner_id != user.id and user.role not in STAFF_ROLES):
        raise ApiError(404, "not_found", "That item isn't here. It may have been returned or closed.")
    return item


@router.post("/items/extract-preview")
async def extract_preview(request: Request, _: User = Depends(current_user)) -> dict:
    form = await request.form()
    text = str(form.get("text") or "")
    kind = str(form.get("kind") or "")
    photos = await _files(form, "photo", "photos", "photos[]")
    for p in photos:
        media.validate_upload(p)
    if not text.strip() and not photos:
        raise ApiError(422, "invalid_request", "Describe it or take a photo.", {"text": "add a description or a photo"})

    def run() -> dict:
        images = [media.open_upload(p) for p in photos[:3]]
        ex = pipeline.extract(text, images, embed=False)  # the preview shows attributes only, so skip the embedding models
        attrs = ex.attributes
        if kind == "found":  # suggest privacy for what verifies an owner
            for f in items_svc.DEFAULT_HIDDEN_FIELDS:
                v = attrs.get(f)
                for a in (v if isinstance(v, list) else [v] if v else []):
                    a["hidden"] = True
        return {
            "attributes": attrs,
            "detections": [d for r in ex.images for d in r.detections],
            "ocr_text": ex.ocr_text,
            "zone_hint": ex.zone_hint,
            "suspected_id_card": ex.suspected_id_card,
        }

    return await run_in_threadpool(run)


def _num(v, field: str) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(str(v))
    except ValueError:
        raise ApiError(422, "invalid_request", "That map position isn't valid.", {field: "not a number"})


@router.post("/items", status_code=201)
async def create_item(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    form = await request.form()
    photos = await _files(form, "photos", "photos[]", "photo")
    now = datetime.now().astimezone()
    occurred_from = _dt(str(form.get("occurred_from") or now.isoformat()), "occurred_from")
    occurred_to = _dt(str(form.get("occurred_to") or form.get("occurred_from") or now.isoformat()), "occurred_to")

    def run() -> dict:
        item = items_svc.create_item(
            db, user, kind=str(form.get("kind") or ""), text=str(form.get("text") or ""), zone_id=str(form.get("zone_id") or ""),
            occurred_from=occurred_from, occurred_to=occurred_to, custody_zone_id=(str(form.get("custody_zone_id")) if form.get("custody_zone_id") else None),
            attributes=(str(form.get("attributes")) if form.get("attributes") else None), photos=photos,
            lat=_num(form.get("lat"), "lat"), lon=_num(form.get("lon"), "lon"),
        )
        enqueue(db, "ingest", {"item_id": str(item.id)})
        db.commit()
        return item_out(db, item)

    return await run_in_threadpool(run)


@router.post("/items/bulk", status_code=201)
def bulk_found(body: BulkBody, user: User = Depends(DeskOrAdmin), db: Session = Depends(get_db)) -> dict:
    """Desk staff log a batch of found items from a text list; photos can be added one by one afterwards."""
    if not 1 <= len(body.items) <= 50:
        raise ApiError(422, "invalid_request", "Send between 1 and 50 items at a time.", {"items": "1 to 50"})
    now = datetime.now().astimezone()
    made = []
    for row in body.items:
        when = row.occurred_at or now
        item = items_svc.create_item(
            db, user, kind="found", text=row.text, zone_id=row.zone_id, occurred_from=when, occurred_to=when,
            custody_zone_id=row.custody_zone_id or row.zone_id, attributes=None, photos=[],
        )
        enqueue(db, "ingest", {"item_id": str(item.id)})
        made.append(item)
    db.commit()
    return {"created": len(made), "items": [item_out(db, i) for i in made]}


@router.get("/items/public")
def public_items(zone_id: str | None = None, category: str | None = None, q: str | None = None, _: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    stmt = select(Item).where(Item.kind == "found", Item.status.in_(("open", "matched")))
    if zone_id:
        stmt = stmt.where(Item.zone_id == zone_id)
    if category:
        stmt = stmt.where(Item.category == category)
    rows = list(db.scalars(stmt.order_by(Item.occurred_from.desc()).limit(200)))
    out = [public_item(i) for i in rows]
    if q:
        needle = q.strip().lower()
        out = [o for o in out if needle in " ".join([o.get("category") or "", o.get("brand") or "", *o.get("colors", [])]).replace("_", " ").lower()]
    return out


@router.get("/items")
def list_items(mine: int = 0, kind: str | None = None, status: str | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    stmt = select(Item)
    if mine or user.role not in STAFF_ROLES:
        stmt = stmt.where(Item.owner_id == user.id)
    if kind:
        stmt = stmt.where(Item.kind == kind)
    if status:
        stmt = stmt.where(Item.status == status)
    return [item_out(db, i) for i in db.scalars(stmt.order_by(Item.created_at.desc()).limit(200))]


@router.get("/items/{item_id}")
def get_item(item_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return item_out(db, _visible(db, item_id, user))


@router.patch("/items/{item_id}")
def patch_item(item_id: str, body: ItemPatch, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    item = _visible(db, item_id, user)
    if item.owner_id != user.id:
        raise ApiError(403, "forbidden", "Only the person who reported this can change it.")
    text_changed, changed = items_svc.apply_edit(db, item, user, body.model_dump(exclude_unset=True))
    if changed:
        enqueue(db, "extract" if text_changed else "match", {"item_id": str(item.id)})
    db.commit()
    return item_out(db, item)


@router.post("/items/{item_id}/close")
def close_item(item_id: str, body: CloseBody, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    item = _visible(db, item_id, user)
    return item_out(db, items_svc.close_item(db, item, user, body.reason))


@router.get("/items/{item_id}/events")
def item_events(item_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    item = _visible(db, item_id, user)
    return events.timeline(db, item.id)


@router.get("/items/{item_id}/matches")
def item_matches(item_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    item = _visible(db, item_id, user)
    if item.kind != "lost":
        raise ApiError(404, "not_found", "Matches are listed on lost reports.")
    rows = db.scalars(select(Match).where(Match.lost_item_id == item.id, Match.score >= get_settings().min_show).order_by(Match.rank))
    return [match_out(m) for m in rows if m.found_item.status not in ("returned", "closed")]
