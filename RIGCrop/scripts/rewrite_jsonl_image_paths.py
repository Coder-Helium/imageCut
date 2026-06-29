#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite image_path fields in RIG/DACC JSONL after moving servers.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--old-prefix", default="", help="Old absolute path prefix to replace.")
    parser.add_argument("--new-prefix", default="", help="New absolute path prefix.")
    parser.add_argument("--image-root", default="", help="Build image_path as IMAGE_ROOT / rel_path.")
    parser.add_argument("--path-field", default="image_path")
    parser.add_argument("--rel-field", default="rel_path")
    parser.add_argument("--check-exists", action="store_true")
    parser.add_argument("--max-missing", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.old_prefix and not args.image_root:
        raise SystemExit("Use either --old-prefix/--new-prefix or --image-root.")
    if args.old_prefix and not args.new_prefix:
        raise SystemExit("--new-prefix is required when --old-prefix is used.")

    out_path = Path(args.out_jsonl)
    if out_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{out_path} exists. Use --overwrite.")
    if args.overwrite and out_path.exists() and not args.dry_run:
        out_path.unlink()

    processed = 0
    changed = 0
    missing = 0
    missing_examples = []
    examples = []
    image_root = Path(args.image_root).expanduser() if args.image_root else None
    records = []

    for rec in _iter_jsonl(args.input_jsonl):
        processed += 1
        out = dict(rec)
        old_path = str(out.get(args.path_field, ""))
        new_path = _rewrite_path(out, old_path, args.old_prefix, args.new_prefix, image_root, args.rel_field)
        if new_path != old_path:
            changed += 1
            if len(examples) < 5:
                examples.append({"old": old_path, "new": new_path})
        out[args.path_field] = new_path
        if args.check_exists and not Path(new_path).exists():
            missing += 1
            if len(missing_examples) < args.max_missing:
                missing_examples.append({"sample_id": out.get("sample_id", ""), "image_path": new_path})
        records.append(out)

    report: Dict[str, Any] = {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "out_jsonl": str(out_path.resolve()),
        "processed": processed,
        "changed": changed,
        "missing": missing,
        "dry_run": args.dry_run,
        "examples": examples,
        "missing_examples": missing_examples,
    }

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        _write_json(out_path.with_suffix(out_path.suffix + ".rewrite_summary.json"), report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


def _rewrite_path(
    rec: Dict[str, Any],
    old_path: str,
    old_prefix: str,
    new_prefix: str,
    image_root: Path | None,
    rel_field: str,
) -> str:
    if image_root is not None:
        rel = str(rec.get(rel_field, "")).strip()
        if rel:
            return str((image_root / rel).resolve())
        return str((image_root / Path(old_path).name).resolve())
    old = str(Path(old_prefix).expanduser())
    new = str(Path(new_prefix).expanduser())
    if old_path.startswith(old):
        suffix = old_path[len(old) :].lstrip("/")
        return str(Path(new, suffix))
    return old_path


def _iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
