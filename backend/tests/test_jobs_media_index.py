import io
import json
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import jobs
from app.ml.matching.candidates import passes_gates, top_candidates
from app.ml.matching.index import NumpyIndex
from app.models import Item, ItemImage, Job
from app.services import media
from tests import factories as fx
from tests.helpers import T0, png_bytes, unit, view


# ---- job queue --------------------------------------------------------------------

def test_jobs_run_in_order_and_chain(db):
    u = fx.user(db, "a@x.edu")
    it = fx.item(db, u, "lost", "bottle", status="processing")
    jobs.enqueue(db, "ingest", {"item_id": str(it.id)})
    db.commit()
    n = jobs.drain(db)
    kinds = [j.kind for j in db.query(Job).order_by(Job.created_at)]
    assert kinds == ["ingest", "extract", "match", "notify"] and n == 4
    assert all(j.status == "done" for j in db.query(Job))
    db.refresh(it)
    assert it.status in ("open", "matched")


def test_failed_job_retries_with_backoff_then_fails(db, monkeypatch):
    calls = []

    def boom(_db, _p):
        calls.append(1)
        raise RuntimeError("model exploded")

    monkeypatch.setitem(jobs._handlers, "notify", boom)
    j = jobs.enqueue(db, "notify", {"item_id": str(uuid.uuid4())})
    db.commit()
    assert jobs.process_next(db) is True
    db.refresh(j)
    assert j.status == "queued" and j.attempts == 1 and "model exploded" in j.last_error
    ra = j.run_after if j.run_after.tzinfo else j.run_after.replace(tzinfo=timezone.utc)
    assert ra > datetime.now(timezone.utc)          # backed off into the future
    assert jobs.process_next(db) is False        # backoff: not due yet
    for _ in range(2):
        j.run_after = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
        assert jobs.process_next(db) is True
    db.refresh(j)
    assert j.status == "failed" and j.attempts == 3 and j.finished_at is not None and len(calls) == 3


def test_queue_skips_jobs_that_are_not_due(db):
    jobs.enqueue(db, "notify", {"item_id": str(uuid.uuid4())}, delay_seconds=3600)
    db.commit()
    assert jobs.process_next(db) is False


def test_recover_stale_requeues_interrupted_jobs_and_fails_exhausted_ones(db):
    u = fx.user(db, "a@x.edu")
    it = fx.item(db, u, "lost", "bottle", status="processing")
    cut_off = jobs.enqueue(db, "ingest", {"item_id": str(it.id)})
    cut_off.status, cut_off.attempts = "running", 1            # the server died while this was running
    looping = jobs.enqueue(db, "notify", {"item_id": str(uuid.uuid4())})
    looping.status, looping.attempts = "running", jobs.MAX_ATTEMPTS   # it has crashed the server every time
    done = jobs.enqueue(db, "notify", {"item_id": str(uuid.uuid4())})
    done.status = "done"
    db.commit()

    assert jobs.recover_stale(db) == 2
    db.refresh(cut_off), db.refresh(looping), db.refresh(done)
    assert cut_off.status == "queued" and looping.status == "failed" and "interrupted" in looping.last_error
    assert done.status == "done"                                # finished jobs are left alone

    jobs.drain(db)                                              # the requeued job now runs and chains on
    db.refresh(it)
    assert it.status in ("open", "matched")


# ---- media -------------------------------------------------------------------------

def test_exif_is_stripped_and_public_copy_is_blurred(db):
    im = Image.new("RGB", (400, 300), (10, 10, 10))
    for x in range(100, 300):
        im.putpixel((x, 150), (255, 255, 255))  # a thin bright line that a blur must smear
    exif = Image.Exif()
    exif[0x010F] = "ACME Phone"      # make
    exif[0x8825] = {1: "N", 2: (17.0, 23.0, 6.0)}   # a GPS block
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    row = ItemImage(item_id=uuid.uuid4(), path="x")
    path, sha = media.stash_incoming(buf.getvalue())
    row.path = path
    row.id = uuid.uuid4()
    media.process_image(row)
    original = Image.open(media.media_root() / row.path)
    assert not original.getexif() and "exif" not in original.info                    # no location, no device
    public = Image.open(media.media_root() / row.public_path)
    assert public.width == media.PUBLIC_WIDTH
    arr = np.asarray(public.convert("L"), dtype=float)
    assert arr.max() < 200                                              # the bright line was smeared away


def test_redacted_plate_is_used_for_id_cards():
    row = ItemImage(item_id=uuid.uuid4(), path="x", id=uuid.uuid4())
    path, _ = media.stash_incoming(png_bytes((200, 30, 30)))
    row.path = path
    media.process_image(row, redact=True)
    arr = np.asarray(Image.open(media.media_root() / row.public_path).convert("RGB")).astype(int)
    assert arr[:, :, 0].mean() > 200 and abs(arr[:, :, 0].mean() - arr[:, :, 2].mean()) < 20   # plate, not the red photo


def test_upload_validation():
    from app.errors import ApiError

    for bad in (b"", b"not an image"):
        with pytest.raises(ApiError) as e:
            media.validate_upload(bad)
        assert e.value.status == 422
    with pytest.raises(ApiError) as e:
        media.validate_upload(b"0" * (media.MAX_BYTES + 1))
    assert e.value.status == 413


# ---- index and candidates ----------------------------------------------------------

def test_numpy_index_search_upsert_remove():
    idx = NumpyIndex(8)
    vecs = {f"k{i}": np.random.default_rng(i).standard_normal(8).astype(np.float32) for i in range(20)}
    for k, v in vecs.items():
        idx.upsert(k, v)
    assert len(idx) == 20
    top = idx.search(vecs["k3"], 3)
    assert top[0][0] == "k3" and top[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [s for _, s in top] == sorted((s for _, s in top), reverse=True)
    restricted = idx.search(vecs["k3"], 5, allowed=["k1", "k2", "k9"])
    assert {k for k, _ in restricted} == {"k1", "k2", "k9"}
    idx.upsert("k3", vecs["k4"])
    assert idx.search(vecs["k4"], 2)[0][1] == pytest.approx(1.0, abs=1e-5)
    idx.remove("k3")
    assert len(idx) == 19 and all(k != "k3" for k, _ in idx.search(vecs["k3"], 19))
    idx.remove("missing")
    assert idx.search(vecs["k1"], 3, allowed=[]) == []


def test_candidate_gates():
    lost = view("l", "lost", category="laptop", category_conf=0.9)
    assert passes_gates(lost, view("f", "found", category="laptop", category_conf=0.9))
    assert not passes_gates(lost, view("f", "found", category="umbrella", category_conf=0.9))     # other category group
    assert passes_gates(lost, view("f", "found", category="umbrella", category_conf=0.3))         # unsure: relaxed
    assert passes_gates(lost, view("f", "found", category="tablet", category_conf=0.9))           # same group
    early = view("f", "found", category="laptop", category_conf=0.9, occurred_from=T0 - timedelta(hours=2), occurred_to=T0 - timedelta(hours=2))
    assert not passes_gates(lost, early)                                                         # found before it was lost


def test_top_candidates_ranks_by_embedding_and_caps():
    q = view("q", "lost", category="laptop", category_conf=0.9, image_embs=[unit(1)])
    pool = [view(f"f{i}", "found", category="laptop", category_conf=0.9, image_embs=[unit(1) if i == 7 else unit(100 + i)]) for i in range(30)]
    pool.append(view("noemb", "found", category="laptop", category_conf=0.9))
    top = top_candidates(q, pool, k=5)
    assert len(top) == 5 and top[0].id == "f7"
    everything = top_candidates(q, pool, k=100)
    assert everything[-1].id == "noemb"          # no embedding: ranked last, still present (recency fallback)


def test_cross_modal_candidate_similarity():
    e = unit(5)
    q = view("q", "lost", category="laptop", category_conf=0.9, clip_text_emb=e)
    match = view("m", "found", category="laptop", category_conf=0.9, image_embs=[e])
    other = view("o", "found", category="laptop", category_conf=0.9, image_embs=[unit(6)])
    assert top_candidates(q, [other, match], 2)[0].id == "m"


# ---- zone script ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]


def test_zone_builder_from_saved_overpass_elements(tmp_path):
    els = {"elements": [
        {"type": "way", "center": {"lat": 17.385, "lon": 78.486}, "tags": {"name": "Central Library", "amenity": "library", "building": "yes"}},
        {"type": "way", "center": {"lat": 17.386, "lon": 78.487}, "tags": {"name": "Food Court", "amenity": "food_court"}},
        {"type": "node", "lat": 17.383, "lon": 78.485, "tags": {"barrier": "gate", "name": "Main Gate"}},
        {"type": "way", "center": {"lat": 17.387, "lon": 78.489}, "tags": {"name": "Boys Hostel 2", "building": "dormitory"}},
        {"type": "way", "center": {"lat": 17.3915, "lon": 78.4915}, "tags": {"name": "Cricket Ground", "leisure": "pitch"}},
    ]}
    src = tmp_path / "overpass.json"
    src.write_text(json.dumps(els))
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_campus_zones as b

    geo, graph, aliases = b.build_from_elements(els["elements"])
    kinds = {f["properties"]["kind"] for f in geo["features"]}
    assert kinds == {"library", "canteen", "gate", "hostel", "sports"}
    ids = {f["properties"]["id"] for f in geo["features"]}
    assert len(ids) == 5 and set(graph["nodes"]) == ids
    connected = {n for e in graph["edges"] for n in e}
    assert connected == ids                                   # nothing is left unreachable
    assert "library" in aliases[next(i for i in ids if "library" in i)]
    g, gr, _ = b.placeholder()
    assert len(g["features"]) == 8 and len(gr["edges"]) == 10


def test_purge_removes_old_closed_items(db):
    from app.main import purge_old

    u = fx.user(db, "a@x.edu")
    old = fx.item(db, u, "lost", "bottle", status="closed")
    old.updated_at = datetime.now(timezone.utc) - timedelta(days=200)
    fresh = fx.item(db, u, "lost", "bottle", status="closed")
    live = fx.item(db, u, "lost", "bottle", status="open")
    live.updated_at = datetime.now(timezone.utc) - timedelta(days=200)
    db.commit()
    assert purge_old(db) == 1
    remaining = {i.id for i in db.query(Item)}
    assert old.id not in remaining and fresh.id in remaining and live.id in remaining


def test_an_unsure_photo_is_gated_by_what_it_might_be():
    """Regression: a lost laptop was matching a found bottle because one low-confidence guess relaxed the whole category gate."""
    laptop = view("l", "lost", category="laptop", category_conf=0.95)
    bottle_photo = view("f", "found", category="bottle", category_conf=0.40, category_dist={"bottle": 0.40, "watch": 0.20, "spectacles": 0.15})
    assert not passes_gates(laptop, bottle_photo)                       # 75% of its mass is classified, and none of it is computers
    maybe_laptop = view("f2", "found", category="bottle", category_conf=0.40, category_dist={"bottle": 0.40, "laptop": 0.30})
    assert passes_gates(laptop, maybe_laptop)                           # it could be a laptop: keep it as a candidate
    fog = view("f3", "found", category="bottle", category_conf=0.20, category_dist={"bottle": 0.20})
    assert passes_gates(laptop, fog)                                    # almost nothing is known about it: do not gate on it
    assert not passes_gates(laptop, view("f4", "found", category="umbrella", category_conf=0.95))
