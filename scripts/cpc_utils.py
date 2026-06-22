from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gaic_utils import IMAGE_EXTS, box_features, clip_xyxy, dump_jsonl, read_image_size, valid_box, write_json


@dataclass(frozen=True)
class CpcView:
    view_id: str
    original_index: int
    box: List[int]
    raw_box: List[float]
    raw_score: Optional[float]
    score_unit: float
    final_score: float


@dataclass(frozen=True)
class CpcPreference:
    winner: str
    loser: str
    weight: float = 1.0
    source: str = "cpc_pairwise"


@dataclass(frozen=True)
class CpcImageRecord:
    sample_id: str
    image_name: str
    image_path: Path
    views: List[CpcView]
    preferences: List[CpcPreference]
    annotation_payload: Dict[str, Any]


def find_cpc_annotation_file(cpc_root: str | Path, annotation_file: str | Path = "") -> Path:
    if annotation_file:
        path = Path(annotation_file)
        if not path.is_absolute():
            path = Path(cpc_root) / path
        if not path.exists():
            raise FileNotFoundError(f"CPC annotation file not found: {path}")
        return path

    root = Path(cpc_root)
    preferred = [
        root / "image_crop.json",
        root / "annotations" / "image_crop.json",
        root / "metadata" / "image_crop.json",
        root / "CPCDataset" / "image_crop.json",
    ]
    for path in preferred:
        if path.exists():
            return path

    matches = sorted(root.rglob("image_crop.json"), key=lambda p: (len(p.parts), str(p)))
    if matches:
        return matches[0]
    jsons = sorted(root.rglob("*.json"), key=lambda p: (len(p.parts), str(p)))
    if not jsons:
        raise FileNotFoundError(f"No JSON annotation file found under {root}")
    raise FileNotFoundError(
        "Could not find image_crop.json automatically. "
        f"Found JSON files such as {jsons[0]}; pass --annotation-file explicitly."
    )


def build_image_index(cpc_root: str | Path, image_dir: str | Path = "") -> Dict[str, Path]:
    root = Path(cpc_root)
    search_root = Path(image_dir) if image_dir else root
    if image_dir and not search_root.is_absolute():
        search_root = root / search_root
    if not search_root.exists():
        raise FileNotFoundError(f"CPC image directory not found: {search_root}")

    paths = sorted(p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    index: Dict[str, Path] = {}
    for path in paths:
        keys = {
            path.name,
            path.stem,
            path.relative_to(root).as_posix() if _is_relative_to(path, root) else path.as_posix(),
        }
        for key in keys:
            index.setdefault(str(key), path)
    return index


def load_cpc_records(
    cpc_root: str | Path,
    annotation_file: str | Path = "",
    image_dir: str | Path = "",
    coord_mode: str = "auto",
    min_pair_score_gap: float = 0.0,
    max_pairs_per_image: int = 0,
    seed: int = 42,
    clip_boxes: bool = True,
) -> Tuple[List[CpcImageRecord], Dict[str, Any]]:
    ann_path = find_cpc_annotation_file(cpc_root, annotation_file)
    image_index = build_image_index(cpc_root, image_dir)
    with open(ann_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = _iter_annotation_entries(raw)
    rng = random.Random(seed)
    records: List[CpcImageRecord] = []
    skipped: List[Dict[str, Any]] = []
    total_views = 0
    total_pairs = 0
    for image_name, payload in entries:
        image_path = resolve_image_path(image_name, image_index)
        if image_path is None:
            skipped.append({"image_name": image_name, "reason": "missing_image"})
            continue
        image_w, image_h = read_image_size(image_path)
        boxes = _extract_boxes(payload)
        if not boxes:
            skipped.append({"image_name": image_name, "reason": "missing_boxes"})
            continue
        raw_scores = _extract_scores(payload, len(boxes))
        views = build_cpc_views(
            boxes,
            raw_scores,
            image_w,
            image_h,
            coord_mode=coord_mode,
            clip_boxes=clip_boxes,
        )
        if len(views) < 2:
            skipped.append({"image_name": image_name, "reason": "too_few_valid_views"})
            continue
        preferences = _extract_payload_preferences(payload, views)
        if not preferences:
            preferences = build_preferences_from_scores(
                views,
                min_score_gap=min_pair_score_gap,
                max_pairs=max_pairs_per_image,
                rng=rng,
            )
        sample_id = Path(image_name).stem
        records.append(
            CpcImageRecord(
                sample_id=sample_id,
                image_name=image_name,
                image_path=image_path,
                views=views,
                preferences=preferences,
                annotation_payload=payload if isinstance(payload, dict) else {},
            )
        )
        total_views += len(views)
        total_pairs += len(preferences)

    summary = {
        "annotation_file": str(ann_path.resolve()),
        "image_index_size": len(image_index),
        "records": len(records),
        "skipped": skipped,
        "views": total_views,
        "pairwise_preferences": total_pairs,
        "coord_mode": coord_mode,
        "min_pair_score_gap": min_pair_score_gap,
        "max_pairs_per_image": max_pairs_per_image,
    }
    return records, summary


def resolve_image_path(image_name: str, image_index: Dict[str, Path]) -> Optional[Path]:
    candidates = [image_name, Path(image_name).name, Path(image_name).stem]
    if Path(image_name).suffix:
        candidates.append(Path(image_name).with_suffix("").as_posix())
    for key in candidates:
        if key in image_index:
            return image_index[key]
    return None


def build_cpc_views(
    boxes: Sequence[Sequence[Any]],
    raw_scores: Sequence[Optional[float]],
    image_w: int,
    image_h: int,
    coord_mode: str = "auto",
    clip_boxes: bool = True,
) -> List[CpcView]:
    score_units = normalize_scores(raw_scores)
    views: List[CpcView] = []
    for idx, raw_box in enumerate(boxes):
        try:
            box_f = [float(v) for v in raw_box[:4]]
        except (TypeError, ValueError):
            continue
        box = convert_cpc_box(box_f, image_w, image_h, coord_mode=coord_mode, clip=clip_boxes)
        if not valid_box(box):
            continue
        unit = float(score_units[idx]) if idx < len(score_units) else 0.5
        final_score = 1.0 + 4.0 * max(0.0, min(1.0, unit))
        views.append(
            CpcView(
                view_id=f"cpc_{idx:03d}",
                original_index=idx,
                box=box,
                raw_box=box_f,
                raw_score=raw_scores[idx] if idx < len(raw_scores) else None,
                score_unit=max(0.0, min(1.0, unit)),
                final_score=final_score,
            )
        )
    return views


def convert_cpc_box(
    raw_box: Sequence[float],
    image_w: int,
    image_h: int,
    coord_mode: str = "auto",
    clip: bool = True,
) -> List[int]:
    mode = coord_mode.lower()
    x1, y1, x2, y2 = [float(v) for v in raw_box[:4]]
    max_val = max(x1, y1, x2, y2)
    min_val = min(x1, y1, x2, y2)

    if mode == "auto":
        if 0.0 <= min_val and max_val <= 1.5:
            mode = "normalized_xyxy"
        elif x2 <= x1 or y2 <= y1:
            mode = "image_xywh"
        else:
            mode = "image_xyxy"

    if mode in {"xyxy", "image_xyxy"}:
        vals = [x1, y1, x2, y2]
    elif mode in {"xywh", "image_xywh"}:
        vals = [x1, y1, x1 + x2, y1 + y2]
    elif mode in {"normalized_xyxy", "norm_xyxy"}:
        vals = [x1 * image_w, y1 * image_h, x2 * image_w, y2 * image_h]
    elif mode in {"normalized_xywh", "norm_xywh"}:
        vals = [x1 * image_w, y1 * image_h, (x1 + x2) * image_w, (y1 + y2) * image_h]
    else:
        raise ValueError(f"Unknown CPC coord mode: {coord_mode}")

    out = [int(round(v)) for v in vals]
    return clip_xyxy(out, image_w, image_h) if clip else out


def normalize_scores(raw_scores: Sequence[Optional[float]]) -> List[float]:
    values = [float(s) for s in raw_scores if s is not None]
    if not raw_scores:
        return []
    if not values:
        return [0.5 for _ in raw_scores]
    lo = min(values)
    hi = max(values)
    out: List[float] = []
    for score in raw_scores:
        if score is None:
            out.append(0.5)
            continue
        value = float(score)
        if 0.0 <= lo and hi <= 1.0:
            unit = value
        elif 1.0 <= lo and hi <= 5.0:
            unit = (value - 1.0) / 4.0
        elif hi > lo:
            unit = (value - lo) / (hi - lo)
        else:
            unit = 0.5
        out.append(max(0.0, min(1.0, unit)))
    return out


def build_preferences_from_scores(
    views: Sequence[CpcView],
    min_score_gap: float = 0.0,
    max_pairs: int = 0,
    rng: Optional[random.Random] = None,
) -> List[CpcPreference]:
    prefs: List[CpcPreference] = []
    for i, a in enumerate(views):
        for b in views[i + 1 :]:
            diff = a.score_unit - b.score_unit
            if abs(diff) < min_score_gap:
                continue
            if diff > 0:
                winner, loser = a, b
            elif diff < 0:
                winner, loser = b, a
            else:
                continue
            prefs.append(
                CpcPreference(
                    winner=winner.view_id,
                    loser=loser.view_id,
                    weight=max(1e-3, abs(diff)),
                    source="cpc_score_ordered_pair",
                )
            )
    if max_pairs > 0 and len(prefs) > max_pairs:
        rng = rng or random.Random(42)
        prefs = rng.sample(prefs, max_pairs)
    return prefs


def cpc_record_to_dacc_json(record: CpcImageRecord, cpc_root: str | Path) -> Dict[str, Any]:
    image_w, image_h = read_image_size(record.image_path)
    candidates = [cpc_view_to_candidate(v, image_w, image_h) for v in sorted(record.views, key=lambda item: item.final_score, reverse=True)]
    for rank, cand in enumerate(candidates, start=1):
        cand["rank"] = rank
    best = candidates[0]
    rel_path = _rel_path_or_name(record.image_path, Path(cpc_root))
    return {
        "sample_id": record.sample_id,
        "image_path": str(record.image_path.resolve()),
        "rel_path": rel_path,
        "image_width": image_w,
        "image_height": image_h,
        "target_aspect_ratio": "free",
        "caption": "",
        "source_caption": "",
        "semantic_type": "unknown",
        "semantic_info": {},
        "vlm_understanding": {},
        "composition_middle_state": {},
        "detections": [],
        "crop_state_graph": {},
        "masks": {},
        "candidates": candidates,
        "pairwise_preferences": [pref_to_json(p) for p in record.preferences],
        "best_crop": best["box"],
        "best_score": best["scores"]["final_score"],
        "best_action": "unknown",
        "main_issue": "cpc_pairwise_supervision",
        "cpc_supervision": {
            "source": "CPC",
            "image_name": record.image_name,
            "candidate_scores": "pairwise_preference",
            "candidate_pseudo_scores": "derived_from_cpc_view_scores",
            "num_candidates": len(candidates),
            "num_pairwise_preferences": len(record.preferences),
        },
        "quality_flags": {
            "has_valid_subject_mask": False,
            "has_valid_key_object_mask": False,
            "has_enough_candidates": len(candidates) >= 16,
            "has_pairwise_preferences": len(record.preferences) > 0,
            "has_vlm_middle_state": False,
            "needs_manual_review": False,
        },
    }


def cpc_view_to_candidate(view: CpcView, image_w: int, image_h: int) -> Dict[str, Any]:
    return {
        "candidate_id": view.view_id,
        "box": view.box,
        "box_format": "xyxy",
        "source": "cpc_view",
        "action": "unknown",
        "issue": "cpc_pairwise_supervision",
        "reason": "CPC candidate view with human comparative preference supervision.",
        "features": box_features(view.box, image_w, image_h),
        "scores": {
            "final_score": round(view.final_score, 6),
            "pseudo_score_unit": round(view.score_unit, 6),
            "cpc_raw_score": None if view.raw_score is None else round(float(view.raw_score), 6),
        },
        "rank": None,
        "quality_label": quality_label_from_unit(view.score_unit),
        "cpc_original_index": view.original_index,
        "cpc_original_box": [round(float(v), 6) for v in view.raw_box],
    }


def pref_to_json(pref: CpcPreference) -> Dict[str, Any]:
    return {
        "winner": pref.winner,
        "loser": pref.loser,
        "weight": round(float(pref.weight), 6),
        "source": pref.source,
    }


def quality_label_from_unit(score_unit: float) -> str:
    if score_unit >= 0.8:
        return "excellent"
    if score_unit >= 0.6:
        return "good"
    if score_unit >= 0.4:
        return "fair"
    if score_unit >= 0.2:
        return "poor"
    return "bad"


def split_records(
    records: Sequence[CpcImageRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_records: int = 0,
) -> Dict[str, List[CpcImageRecord]]:
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    if max_records > 0:
        items = items[:max_records]
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def _iter_annotation_entries(raw: Any) -> List[Tuple[str, Dict[str, Any]]]:
    if isinstance(raw, dict):
        if "images" in raw and isinstance(raw["images"], list):
            return _iter_coco_like_entries(raw)
        entries: List[Tuple[str, Dict[str, Any]]] = []
        for key, value in raw.items():
            if key in {"metadata", "info", "categories"}:
                continue
            if isinstance(value, dict):
                image_name = str(value.get("image") or value.get("image_name") or value.get("file_name") or value.get("filename") or key)
                entries.append((image_name, value))
            elif isinstance(value, list):
                entries.append((str(key), {"bboxes": value}))
        return entries
    if isinstance(raw, list):
        entries = []
        for idx, value in enumerate(raw):
            if not isinstance(value, dict):
                continue
            image_name = str(
                value.get("image")
                or value.get("image_name")
                or value.get("file_name")
                or value.get("filename")
                or value.get("name")
                or f"{idx}"
            )
            entries.append((image_name, value))
        return entries
    raise ValueError("Unsupported CPC annotation JSON structure")


def _iter_coco_like_entries(raw: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    images = {img.get("id", idx): img for idx, img in enumerate(raw.get("images", [])) if isinstance(img, dict)}
    ann_by_image: Dict[Any, List[Dict[str, Any]]] = {}
    for ann in raw.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        ann_by_image.setdefault(ann.get("image_id"), []).append(ann)
    entries: List[Tuple[str, Dict[str, Any]]] = []
    for image_id, image in images.items():
        anns = ann_by_image.get(image_id, [])
        boxes = [ann.get("bbox") or ann.get("box") for ann in anns]
        scores = [ann.get("score") for ann in anns]
        image_name = str(image.get("file_name") or image.get("filename") or image.get("name") or image_id)
        entries.append((image_name, {"bboxes": boxes, "scores": scores, "image_id": image_id}))
    return entries


def _extract_boxes(payload: Dict[str, Any]) -> List[List[Any]]:
    for key in ["bboxes", "boxes", "views", "view_boxes", "crop_boxes", "crops"]:
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            boxes: List[List[Any]] = []
            for item in value:
                if isinstance(item, dict):
                    box = item.get("bbox") or item.get("box") or item.get("crop") or item.get("rect")
                else:
                    box = item
                if isinstance(box, list) and len(box) >= 4:
                    boxes.append(box[:4])
            if boxes:
                return boxes
    return []


def _extract_scores(payload: Dict[str, Any], expected_len: int) -> List[Optional[float]]:
    raw_scores = None
    for key in ["scores", "score", "mos", "view_scores", "quality", "qualities"]:
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            raw_scores = value
            break
    if raw_scores is None:
        views = payload.get("views") or payload.get("crops") if isinstance(payload, dict) else None
        if isinstance(views, list):
            raw_scores = [
                item.get("score") if isinstance(item, dict) else None
                for item in views
            ]
    scores: List[Optional[float]] = []
    for idx in range(expected_len):
        try:
            value = raw_scores[idx] if raw_scores is not None and idx < len(raw_scores) else None
            scores.append(None if value is None else float(value))
        except (TypeError, ValueError):
            scores.append(None)
    return scores


def _extract_payload_preferences(payload: Dict[str, Any], views: Sequence[CpcView]) -> List[CpcPreference]:
    if not isinstance(payload, dict):
        return []
    value = payload.get("pairwise_preferences") or payload.get("pairs") or payload.get("preferences")
    if not isinstance(value, list):
        return []
    valid_ids = {v.view_id for v in views}
    by_index = {v.original_index: v.view_id for v in views}
    prefs: List[CpcPreference] = []
    for item in value:
        winner: Optional[str] = None
        loser: Optional[str] = None
        weight = 1.0
        if isinstance(item, dict):
            winner = _resolve_view_ref(item.get("winner") or item.get("better") or item.get("positive") or item.get("pos"), valid_ids, by_index)
            loser = _resolve_view_ref(item.get("loser") or item.get("worse") or item.get("negative") or item.get("neg"), valid_ids, by_index)
            weight = _float_or_default(item.get("weight"), 1.0)
        elif isinstance(item, list) and len(item) >= 2:
            winner = _resolve_view_ref(item[0], valid_ids, by_index)
            loser = _resolve_view_ref(item[1], valid_ids, by_index)
            if len(item) >= 3:
                weight = _float_or_default(item[2], 1.0)
        if winner and loser and winner != loser:
            prefs.append(CpcPreference(winner=winner, loser=loser, weight=weight, source="cpc_human_pairwise"))
    return prefs


def load_pairwise_file(path: str | Path, sample_to_candidates: Dict[str, Sequence[str]]) -> Dict[str, List[CpcPreference]]:
    pair_path = Path(path)
    if not pair_path.exists():
        raise FileNotFoundError(f"Pairwise file not found: {pair_path}")
    if pair_path.suffix.lower() == ".json":
        with open(pair_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return _load_pairwise_json(raw, sample_to_candidates)
    return _load_pairwise_table(pair_path, sample_to_candidates)


def _load_pairwise_json(raw: Any, sample_to_candidates: Dict[str, Sequence[str]]) -> Dict[str, List[CpcPreference]]:
    out: Dict[str, List[CpcPreference]] = {}
    if isinstance(raw, dict):
        iterator = raw.items()
    elif isinstance(raw, list):
        iterator = [(None, item) for item in raw]
    else:
        return out
    for sample_id, value in iterator:
        if isinstance(value, dict):
            sid = str(value.get("sample_id") or value.get("image") or sample_id or "")
            pair_items = value.get("pairs") or value.get("pairwise_preferences") or value.get("preferences") or []
        else:
            sid = str(sample_id or "")
            pair_items = value if isinstance(value, list) else []
        candidate_ids = set(sample_to_candidates.get(sid, []))
        by_index = {idx: cid for idx, cid in enumerate(sample_to_candidates.get(sid, []))}
        prefs = []
        for item in pair_items:
            if isinstance(item, dict):
                winner = _resolve_view_ref(item.get("winner") or item.get("better"), candidate_ids, by_index)
                loser = _resolve_view_ref(item.get("loser") or item.get("worse"), candidate_ids, by_index)
                weight = _float_or_default(item.get("weight"), 1.0)
            elif isinstance(item, list) and len(item) >= 2:
                winner = _resolve_view_ref(item[0], candidate_ids, by_index)
                loser = _resolve_view_ref(item[1], candidate_ids, by_index)
                weight = _float_or_default(item[2], 1.0) if len(item) > 2 else 1.0
            else:
                continue
            if winner and loser and winner != loser:
                prefs.append(CpcPreference(winner=winner, loser=loser, weight=weight, source="cpc_pairwise_file"))
        if sid and prefs:
            out[sid] = prefs
    return out


def _load_pairwise_table(path: Path, sample_to_candidates: Dict[str, Sequence[str]]) -> Dict[str, List[CpcPreference]]:
    out: Dict[str, List[CpcPreference]] = {}
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        delimiter = "," if "," in first else None
        reader = csv.DictReader(f, delimiter=delimiter) if any(k in first.lower() for k in ["winner", "loser", "image"]) else None
        if reader is not None:
            for row in reader:
                sid = row.get("sample_id") or row.get("image") or row.get("image_name") or row.get("filename")
                if not sid:
                    continue
                candidate_ids = set(sample_to_candidates.get(str(sid), []))
                by_index = {idx: cid for idx, cid in enumerate(sample_to_candidates.get(str(sid), []))}
                winner = _resolve_view_ref(row.get("winner") or row.get("better"), candidate_ids, by_index)
                loser = _resolve_view_ref(row.get("loser") or row.get("worse"), candidate_ids, by_index)
                if winner and loser:
                    out.setdefault(str(sid), []).append(CpcPreference(winner=winner, loser=loser, weight=_float_or_default(row.get("weight"), 1.0), source="cpc_pairwise_file"))
        else:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 3:
                    continue
                sid = parts[0]
                candidate_ids = set(sample_to_candidates.get(sid, []))
                by_index = {idx: cid for idx, cid in enumerate(sample_to_candidates.get(sid, []))}
                winner = _resolve_view_ref(parts[1], candidate_ids, by_index)
                loser = _resolve_view_ref(parts[2], candidate_ids, by_index)
                if winner and loser:
                    out.setdefault(sid, []).append(CpcPreference(winner=winner, loser=loser, weight=_float_or_default(parts[3] if len(parts) > 3 else None, 1.0), source="cpc_pairwise_file"))
    return out


def write_cpc_splits(out_dir: str | Path, splits: Dict[str, Sequence[CpcImageRecord]], cpc_root: str | Path) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    for split, records in splits.items():
        if not records:
            continue
        json_records = [cpc_record_to_dacc_json(r, cpc_root) for r in records]
        path = out / f"{split}.jsonl"
        dump_jsonl(path, json_records)
        summary[split] = {
            "records": len(json_records),
            "candidates": sum(len(r.get("candidates", [])) for r in json_records),
            "pairwise_preferences": sum(len(r.get("pairwise_preferences", [])) for r in json_records),
            "jsonl": str(path.resolve()),
        }
    return summary


def _resolve_view_ref(value: Any, valid_ids: Iterable[str], by_index: Dict[int, str]) -> Optional[str]:
    if value is None:
        return None
    valid = set(valid_ids)
    text = str(value)
    if text in valid:
        return text
    try:
        idx = int(float(text))
    except (TypeError, ValueError):
        return None
    return by_index.get(idx)


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rel_path_or_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
