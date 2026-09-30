"""Extraction pipeline: photos and text in, one fused attribute record plus embeddings out.

    photo -> detector -> primary crop -> { category / material (CLIP), colors (k-means), OCR }
    text  -> parser (spaCy + gazetteers)
    both  -> fuse: agreeing sources raise confidence, disagreeing sources keep the stronger one
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .. import taxonomy
from ..config import get_settings
from .attrs import attr, noisy_or
from .text import embedder as text_embedder
from .text.parser import ParsedText, parse_text
from .vision import attributes as vattrs
from .vision import color, detector, embedder as clip, ocr

MIN_CATEGORY_CONF = 0.15


@dataclass
class ImageResult:
    detections: list[dict] = field(default_factory=list)
    primary_crop: dict | None = None
    ocr_text: str = ""
    embedding: np.ndarray | None = None
    category: list[tuple[str, float]] = field(default_factory=list)
    material: tuple[str, float] | None = None
    colors: list[dict] = field(default_factory=list)
    brand: dict | None = None
    serial: dict | None = None
    roll_no: str | None = None


@dataclass
class Extraction:
    attributes: dict
    images: list[ImageResult]
    ocr_text: str
    text_embedding: np.ndarray | None
    clip_text_embedding: np.ndarray | None
    suspected_id_card: bool
    zone_hint: dict | None = None


# ---- one image -----------------------------------------------------------------


def _crop(img: Image.Image, box: list[float] | None) -> Image.Image:
    if not box:
        return img
    w, h = img.size
    x, y, bw, bh = box
    x0, y0, x1, y1 = int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)
    return img.crop((x0, y0, x1, y1)) if x1 - x0 >= 24 and y1 - y0 >= 24 else img


def analyze_image(img: Image.Image) -> ImageResult:
    img = img.convert("RGB")
    res = ImageResult()
    res.detections = detector.detect(img)
    pc = detector.primary_crop(res.detections)
    res.primary_crop = pc
    crop = _crop(img, pc["box"] if pc else None)

    res.embedding = clip.embed_images([crop])[0]
    ranked = vattrs.predict_category(res.embedding)
    # A confident COCO / trained detection overrides zero-shot when both are present.
    if pc and pc["confidence"] >= 0.5:
        prob = dict(ranked).get(pc["label"], 0.0)
        ranked = [(pc["label"], max(pc["confidence"] * 0.95, prob))] + [r for r in ranked if r[0] != pc["label"]]
    res.category = ranked[:3]

    if not get_settings().light_ml:
        mats = vattrs.predict_material(res.embedding)
        res.material = mats[0]

    rgb = np.asarray(crop)
    res.colors = color.dominant_colors(rgb)

    # OCR on the crop; for a suspected ID card also on the full image at higher resolution, where the printed text is.
    top = res.category[0][0] if res.category else None
    lines: list[tuple[str, float]] = []
    if top not in ocr.NO_TEXT_CATEGORIES:
        lines = ocr.read_text(rgb)
    if top == "id_card":
        lines += ocr.read_text(np.asarray(img), max_side=960)
    parsed = ocr.parse_ocr(lines)
    res.ocr_text = parsed["text"]
    res.brand = parsed.get("brand")
    res.serial = parsed.get("serial")
    res.roll_no = parsed.get("roll_no")
    return res


# ---- fusion --------------------------------------------------------------------


def _photo_category(images: list[ImageResult]) -> dict | None:
    votes: dict[str, float] = {}
    best_im: ImageResult | None = None
    for im in images:
        for cat, p in im.category[:1]:
            if p >= votes.get(cat, 0.0):
                votes[cat] = p
    if not votes:
        return None
    cat, p = max(votes.items(), key=lambda kv: kv[1])
    for im in images:
        if im.category and im.category[0][0] == cat:
            best_im = im
            break
    a = attr(cat, min(0.95, p), "photo")
    if best_im:  # the runners-up, so matching can score a photo it was unsure about honestly
        a["alts"] = [[c, round(float(q), 3)] for c, q in best_im.category[1:3] if q >= 0.03]
    return a


def fuse(text: ParsedText, images: list[ImageResult]) -> dict:
    out: dict = {"colors": [], "marks": list(text.marks)}

    # category: text and photo vote; agreement raises confidence, disagreement keeps the stronger
    photo_cat = _photo_category(images)
    text_cat = text.category
    if text_cat and photo_cat and text_cat["value"] == photo_cat["value"]:
        out["category"] = attr(text_cat["value"], noisy_or(text_cat["confidence"], photo_cat["confidence"]), "text")
    else:
        best = max((c for c in (text_cat, photo_cat) if c), key=lambda c: c["confidence"], default=None)
        if best and best["confidence"] >= MIN_CATEGORY_CONF:
            out["category"] = best
    # ID cards are recognised by their printed roll number even when the classifier is unsure.
    if any(im.roll_no for im in images) and (not out.get("category") or out["category"]["value"] != "id_card"):
        if not text_cat or text_cat["value"] == "id_card" or text_cat["confidence"] < 0.9:
            out["category"] = attr("id_card", 0.9, "ocr")

    # brand: text, else OCR; agreement raises confidence
    ocr_brand = next((im.brand for im in images if im.brand), None)
    if text.brand and ocr_brand and text.brand["value"] == ocr_brand["value"]:
        out["brand"] = attr(text.brand["value"], noisy_or(text.brand["confidence"], ocr_brand["confidence"]), "text")
    elif text.brand or ocr_brand:
        out["brand"] = max((b for b in (text.brand, ocr_brand) if b), key=lambda b: b["confidence"])

    # colors: named text colors first (people say what matters), then photo clusters not already named
    colors = list(text.colors)
    seen = {c["value"] for c in colors}
    for im in images[:1]:
        for c in im.colors:
            conf = min(0.9, 0.5 + 0.5 * c["share"])
            if c["name"] in seen:
                for existing in colors:
                    if existing["value"] == c["name"]:
                        existing["confidence"] = round(noisy_or(existing["confidence"], conf), 3)
            elif len(colors) < 2:
                colors.append(attr(c["name"], conf, "photo"))
                seen.add(c["name"])
    out["colors"] = colors[:2]

    material = next((im.material for im in images if im.material), None)
    if material and material[1] >= 0.35:
        out["material"] = attr(material[0], min(0.9, material[1]), "photo")

    serial = text.serial or next((im.serial for im in images if im.serial), None)
    if serial:
        out["serial"] = serial
    return out


def merge_user_attributes(extracted: dict, provided: dict | None) -> dict:
    """User edits (source 'user') always win; the `hidden` flag from the client is honoured."""
    if not provided:
        return extracted
    merged = {k: (list(v) if isinstance(v, list) else v) for k, v in extracted.items()}
    for field_name in ("category", "brand", "material", "serial"):
        if field_name not in provided:
            continue
        p = provided[field_name]
        if p is None:  # the user removed it
            merged.pop(field_name, None)
        elif p.get("source") == "user":
            merged[field_name] = attr(p["value"], 1.0, "user", p.get("hidden"))
        elif p.get("hidden") and merged.get(field_name, {}).get("value") == p["value"]:
            merged[field_name]["hidden"] = True  # same machine value the finder marked private
    for list_field in ("colors", "marks"):
        if list_field in provided:
            prov = provided[list_field] or []
            if any(p.get("source") == "user" for p in prov) or len(prov) < len(merged.get(list_field, [])):
                merged[list_field] = [attr(p["value"], 1.0 if p.get("source") == "user" else p.get("confidence", 0.9), p.get("source", "user"), p.get("hidden")) for p in prov]
            else:  # only flags changed
                hidden = {p["value"] for p in prov if p.get("hidden")}
                for a in merged.get(list_field, []):
                    if a["value"] in hidden:
                        a["hidden"] = True
    return merged


# ---- entry points -----------------------------------------------------------------


def extract(text: str | None, images: list[Image.Image], embed: bool = True) -> Extraction:
    """Read attributes from text and photos. `embed=False` skips the text embeddings: the report form's live preview
    never uses them, and computing them loads two neural models (MiniLM and CLIP), which is slow and memory hungry."""
    parsed = parse_text(text)
    results = [analyze_image(im) for im in images]
    attributes = fuse(parsed, results)

    text_emb = clip_text_emb = None
    if embed and text and text.strip():
        text_emb = text_embedder.embed_text(text.strip())
        clip_text_emb = clip.embed_texts([text.strip()])[0]
    return Extraction(
        attributes=attributes,
        images=results,
        ocr_text=" ".join(r.ocr_text for r in results if r.ocr_text),
        text_embedding=text_emb,
        clip_text_embedding=clip_text_emb,
        suspected_id_card=(attributes.get("category") or {}).get("value") == "id_card",
        zone_hint=parsed.zone_hint,
    )


def split_hidden(attributes: dict) -> dict:
    """The subset of attributes flagged hidden (kept for claim questions, never shown publicly)."""
    hidden: dict = {}
    for k, v in attributes.items():
        if isinstance(v, list):
            keep = [a for a in v if a.get("hidden")]
            if keep:
                hidden[k] = keep
        elif isinstance(v, dict) and v.get("hidden"):
            hidden[k] = v
    return hidden


def public_attributes(attributes: dict) -> dict:
    """Attributes with every hidden entry removed. This is what PublicItem is built from."""
    out: dict = {}
    for k, v in attributes.items():
        if isinstance(v, list):
            out[k] = [a for a in v if not a.get("hidden")]
        elif isinstance(v, dict) and not v.get("hidden"):
            out[k] = v
    return out


def category_group_of(attributes: dict) -> str | None:
    cat = (attributes.get("category") or {}).get("value")
    return taxonomy.category_group(cat)
