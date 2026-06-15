from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_stem(path_or_key: str) -> str:
    value = str(path_or_key).replace("\\", "/")
    value = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff/]+", "_", value)
    value = value.replace("/", "__")
    return Path(value).with_suffix("").name


def iter_images(image_root: str | Path) -> Iterator[Path]:
    root = Path(image_root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_json(path: str | Path, data: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_captions(path: Optional[str | Path]) -> Dict[str, str]:
    if not path:
        return {}
    data = load_json(path)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    raise ValueError("caption file must be a JSON object: {relative_image_path: caption}")


def caption_for_image(captions: Dict[str, str], image_path: Path, image_root: Path) -> str:
    rel = image_path.relative_to(image_root).as_posix()
    candidates = [rel, image_path.name, str(image_path)]
    for key in candidates:
        if key in captions:
            return captions[key]
    return ""


def read_image_bgr(path: str | Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imread failed: {path}")
    return img


def crop_image(img_bgr, box) -> Any:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box.to_xyxy_int()
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return img_bgr.copy()
    return img_bgr[y1:y2, x1:x2].copy()

