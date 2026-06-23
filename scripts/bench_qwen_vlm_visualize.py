#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import os
import re
import ssl
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from composition_dataset_builder.semantic import route_caption  # noqa: E402
from composition_dataset_builder.vlm import _build_qwen_prompt  # noqa: E402
from enrich_gaic_with_vlm_semantics import draw_enrichment_visualization  # noqa: E402


TIMING_KEYS = [
    "compress_image",
    "connect_tls",
    "send_headers",
    "send_body_upload",
    "wait_response_headers",
    "download_response_body",
    "parse_http_json",
    "parse_qwen_content_json",
    "total_api_observed",
]


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser()
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input jsonl not found: {input_jsonl}")

    run_dir = resolve_run_dir(args)
    out_jsonl = Path(args.out_jsonl).expanduser() if args.out_jsonl else run_dir / "records.jsonl"
    summary_path = Path(args.summary_json).expanduser() if args.summary_json else run_dir / "summary.json"
    vis_dir = Path(args.vis_dir).expanduser() if args.vis_dir else run_dir / "visualizations"

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        unlink_if_exists(out_jsonl)
        unlink_if_exists(summary_path)
    elif out_jsonl.exists():
        raise FileExistsError(f"{out_jsonl} exists. Use --overwrite or set OUT_JSONL to a new path.")

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("Need DASHSCOPE_API_KEY or QWEN_API_KEY")

    records = read_records(input_jsonl, args.n)
    print(f"input={input_jsonl}")
    print(f"model={args.model}")
    print(f"n={len(records)} max_side={args.max_side} jpeg_quality={args.jpeg_quality} max_image_kb={args.max_image_kb}")
    print(f"out_jsonl={out_jsonl}")
    if args.visualize:
        print(f"vis_dir={vis_dir} (visualization time excluded from averages)")

    client = QwenHttpClient(
        api_key=api_key,
        base_url=args.qwen_base_url,
        model=args.model,
    )

    all_times: List[Dict[str, float]] = []
    vis_times: List[float] = []

    for i, rec in enumerate(records, 1):
        image_path = resolve_image_path(str(rec["image_path"]), input_jsonl)
        sid = str(rec.get("sample_id") or image_path.stem or i)
        caption = str(rec.get("caption") or rec.get("source_caption") or "").strip()
        semantic_info = rec.get("semantic_info") if isinstance(rec.get("semantic_info"), dict) and rec.get("semantic_info") else {}
        if not semantic_info:
            semantic_info = route_caption(caption, args.caption_rule_root or None)

        image_url, img_bytes, img_size, q_used, compress_dt = compress_image_to_data_url(
            image_path=image_path,
            max_side=args.max_side,
            jpeg_quality=args.jpeg_quality,
            max_image_kb=args.max_image_kb,
        )
        prompt = _build_qwen_prompt(caption, semantic_info)
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        parsed, timings = client.post_timed(payload)
        parsed = finalize_for_visualization(parsed, semantic_info)
        timings["compress_image"] = compress_dt
        all_times.append(timings)

        enriched = build_visualization_record(
            rec=rec,
            parsed=parsed,
            semantic_info=semantic_info,
            image_path=image_path,
            sample_id=sid,
            model=args.model,
            img_bytes=img_bytes,
            img_size=img_size,
            q_used=q_used,
            timings=timings,
        )

        vis_msg = ""
        if args.visualize:
            vis_path = vis_dir / f"{safe_file_name(sid)}.jpg"
            vis_t0 = time.perf_counter()
            try:
                draw_enrichment_visualization(enriched, vis_path, topk=args.vis_topk)
                vis_dt = time.perf_counter() - vis_t0
                vis_times.append(vis_dt)
                enriched["visualization_path"] = str(vis_path.resolve())
                enriched["qwen_benchmark"]["visualization_sec_excluded"] = vis_dt
                vis_msg = f" vis={vis_path}"
            except Exception as exc:  # noqa: BLE001
                vis_dt = time.perf_counter() - vis_t0
                enriched.setdefault("quality_flags", {})["visualization_error"] = repr(exc)
                enriched["qwen_benchmark"]["visualization_sec_excluded"] = vis_dt
                vis_msg = f" vis_warning={repr(exc)}"

        append_jsonl(out_jsonl, enriched)

        print(
            f"[{i}/{len(records)}] {sid} "
            f"img={img_bytes / 1024:.1f}KB size={img_size[0]}x{img_size[1]} q={q_used} "
            f"compress={compress_dt:.3f}s "
            f"connect={timings['connect_tls']:.3f}s "
            f"headers={timings['send_headers']:.3f}s "
            f"upload={timings['send_body_upload']:.3f}s "
            f"server_wait={timings['wait_response_headers']:.3f}s "
            f"download={timings['download_response_body']:.3f}s "
            f"parse_http={timings['parse_http_json']:.4f}s "
            f"parse_content={timings['parse_qwen_content_json']:.4f}s "
            f"total={timings['total_api_observed']:.3f}s"
            f"{vis_msg}"
        )

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print_average_timing(all_times)
    print_payload_size(all_times)
    if vis_times:
        print("\n==== visualization timing (excluded) ====")
        print(f"visualization avg={statistics.mean(vis_times):.4f}s min={min(vis_times):.4f}s max={max(vis_times):.4f}s")

    summary = build_summary(args, input_jsonl, out_jsonl, summary_path, vis_dir if args.visualize else None, all_times, vis_times)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsummary={summary_path}")


class QwenHttpClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        url = urlparse(self.base_url + "/chat/completions")
        self.host = url.hostname
        self.port = url.port or (443 if url.scheme == "https" else 80)
        self.path = url.path
        self.use_https = url.scheme == "https"
        if not self.host:
            raise ValueError(f"invalid Qwen base URL: {base_url}")

    def post_timed(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Connection": "close",
        }

        timings: Dict[str, float] = {"request_body_bytes": float(len(body))}
        conn_cls = http.client.HTTPSConnection if self.use_https else http.client.HTTPConnection
        conn_kwargs: Dict[str, Any] = {"host": self.host, "port": self.port, "timeout": 180}
        if self.use_https:
            conn_kwargs["context"] = ssl.create_default_context()
        conn = conn_cls(**conn_kwargs)

        t0 = time.perf_counter()
        conn.connect()
        t1 = time.perf_counter()

        conn.putrequest("POST", self.path)
        for key, value in headers.items():
            conn.putheader(key, value)

        t2 = time.perf_counter()
        conn.endheaders()
        t3 = time.perf_counter()

        conn.send(body)
        t4 = time.perf_counter()

        resp = conn.getresponse()
        t5 = time.perf_counter()

        raw = resp.read()
        t6 = time.perf_counter()

        if resp.status >= 400:
            conn.close()
            raise RuntimeError(f"HTTP {resp.status}: {raw[:500]!r}")

        t7 = time.perf_counter()
        data = json.loads(raw.decode("utf-8"))
        t8 = time.perf_counter()

        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

        parsed = json.loads(content)
        t9 = time.perf_counter()
        conn.close()

        timings.update(
            {
                "connect_tls": t1 - t0,
                "prepare_headers": t2 - t1,
                "send_headers": t3 - t2,
                "send_body_upload": t4 - t3,
                "wait_response_headers": t5 - t4,
                "download_response_body": t6 - t5,
                "parse_http_json": t8 - t7,
                "parse_qwen_content_json": t9 - t8,
                "total_api_observed": t9 - t0,
                "response_body_bytes": float(len(raw)),
            }
        )
        return parsed, timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen-VL calls and save visualization outputs.")
    parser.add_argument(
        "--input-jsonl",
        default=os.environ.get("INPUT_JSONL", "data/gaic_semantic_qwen/metadata/inputFile/test.jsonl"),
        help="Input JSONL. Defaults to INPUT_JSONL or data/gaic_semantic_qwen/metadata/inputFile/test.jsonl.",
    )
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen3-vl-flash"))
    parser.add_argument("--qwen-base-url", default=os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--n", type=int, default=int(os.environ.get("N", "10")))
    parser.add_argument("--max-side", type=int, default=int(os.environ.get("MAX_SIDE", "480")))
    parser.add_argument("--jpeg-quality", type=int, default=int(os.environ.get("JPEG_QUALITY", "65")))
    parser.add_argument("--max-image-kb", type=int, default=int(os.environ.get("MAX_IMAGE_KB", "180")))
    parser.add_argument("--caption-rule-root", default=os.environ.get("CAPTION_RULE_ROOT", ""))
    parser.add_argument("--run-dir", default=os.environ.get("RUN_DIR", ""))
    parser.add_argument("--out-jsonl", default=os.environ.get("OUT_JSONL", ""))
    parser.add_argument("--summary-json", default=os.environ.get("SUMMARY_JSON", ""))
    parser.add_argument("--vis-dir", default=os.environ.get("VIS_DIR", ""))
    parser.add_argument("--vis-topk", type=int, default=int(os.environ.get("VIS_TOPK", "5")))
    parser.add_argument("--sleep-sec", type=float, default=float(os.environ.get("SLEEP_SEC", "0")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    args = parser.parse_args()
    args.visualize = not args.no_visualize and env_bool("VISUALIZE", default=True)
    return args


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "qwen_timing_visualize" / stamp


def compress_image_to_data_url(
    image_path: Path,
    max_side: int,
    jpeg_quality: int,
    max_image_kb: int,
) -> Tuple[str, int, Tuple[int, int], int, float]:
    t0 = time.perf_counter()
    with Image.open(image_path) as src:
        img = src.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    quality = jpeg_quality
    min_quality = 35

    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        out_bytes = buf.getvalue()
        if len(out_bytes) <= max_image_kb * 1024 or quality <= min_quality:
            break
        quality -= 5

    while len(out_bytes) > max_image_kb * 1024 and max(img.size) > 448:
        w, h = img.size
        img = img.resize((int(w * 0.85), int(h * 0.85)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=max(quality, min_quality), optimize=True, progressive=True)
        out_bytes = buf.getvalue()

    b64 = base64.b64encode(out_bytes).decode("ascii")
    dt = time.perf_counter() - t0
    return "data:image/jpeg;base64," + b64, len(out_bytes), img.size, quality, dt


def read_records(input_jsonl: Path, limit: int) -> List[Dict[str, Any]]:
    records = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records


def resolve_image_path(raw_path: str, input_jsonl: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        input_jsonl.parent / path,
        input_jsonl.parent.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def finalize_for_visualization(parsed: Dict[str, Any], semantic_info: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(parsed or {})
    out.setdefault("semantic_type", semantic_info.get("semantic_type", "unknown"))
    out.setdefault("source", "qwen_dashscope")
    return out


def build_visualization_record(
    rec: Dict[str, Any],
    parsed: Dict[str, Any],
    semantic_info: Dict[str, Any],
    image_path: Path,
    sample_id: str,
    model: str,
    img_bytes: int,
    img_size: Tuple[int, int],
    q_used: int,
    timings: Dict[str, float],
) -> Dict[str, Any]:
    final_caption = str(parsed.get("caption") or rec.get("caption") or rec.get("source_caption") or "").strip()
    semantic_type = str(parsed.get("semantic_type") or semantic_info.get("semantic_type") or "unknown")
    enriched = dict(rec)
    enriched["sample_id"] = sample_id
    enriched["image_path"] = str(image_path)
    enriched["source_caption"] = rec.get("source_caption", rec.get("caption", ""))
    enriched["caption"] = final_caption
    enriched["semantic_type"] = semantic_type
    enriched["semantic_info"] = {**semantic_info, "semantic_type": semantic_type}
    enriched["vlm_understanding"] = parsed
    enriched["best_action"] = first_action(parsed) or rec.get("best_action", "unknown")
    enriched["main_issue"] = initial_issue(parsed) or rec.get("main_issue", "unknown")
    enriched["quality_flags"] = dict(rec.get("quality_flags", {}) or {})
    enriched["quality_flags"]["has_vlm_middle_state"] = True
    enriched["quality_flags"]["semantic_teacher_source"] = parsed.get("source", "qwen_dashscope")
    enriched["qwen_benchmark"] = {
        "model": model,
        "image_bytes": img_bytes,
        "image_size": list(img_size),
        "jpeg_quality": q_used,
        "timings_excluding_visualization": timings,
    }
    return enriched


def first_action(understanding: Dict[str, Any]) -> str:
    intent = understanding.get("composition_intent", {}) if isinstance(understanding.get("composition_intent"), dict) else {}
    actions = intent.get("suggested_actions", [])
    if isinstance(actions, list) and actions:
        return str(actions[0])
    if isinstance(actions, str):
        return actions
    return ""


def initial_issue(understanding: Dict[str, Any]) -> str:
    intent = understanding.get("composition_intent", {}) if isinstance(understanding.get("composition_intent"), dict) else {}
    return str(intent.get("initial_issue") or "")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_average_timing(all_times: List[Dict[str, float]]) -> None:
    if not all_times:
        print("\n==== average timing ====")
        print("no records")
        return

    print("\n==== average timing ====")
    for key in TIMING_KEYS:
        vals = [float(item[key]) for item in all_times]
        print(f"{key:24s} avg={statistics.mean(vals):.4f}s min={min(vals):.4f}s max={max(vals):.4f}s")


def print_payload_size(all_times: List[Dict[str, float]]) -> None:
    if not all_times:
        return
    print("\n==== payload size ====")
    print(f"request_body_avg_kb={statistics.mean([x['request_body_bytes'] for x in all_times]) / 1024:.1f}")
    print(f"response_body_avg_kb={statistics.mean([x['response_body_bytes'] for x in all_times]) / 1024:.1f}")


def build_summary(
    args: argparse.Namespace,
    input_jsonl: Path,
    out_jsonl: Path,
    summary_path: Path,
    vis_dir: Optional[Path],
    all_times: List[Dict[str, float]],
    vis_times: List[float],
) -> Dict[str, Any]:
    average_timing = {}
    if all_times:
        for key in TIMING_KEYS:
            vals = [float(item[key]) for item in all_times]
            average_timing[key] = {
                "avg": statistics.mean(vals),
                "min": min(vals),
                "max": max(vals),
            }

    summary: Dict[str, Any] = {
        "input_jsonl": str(input_jsonl.resolve()),
        "out_jsonl": str(out_jsonl.resolve()),
        "summary_json": str(summary_path.resolve()),
        "vis_dir": str(vis_dir.resolve()) if vis_dir is not None else "",
        "model": args.model,
        "n": len(all_times),
        "max_side": args.max_side,
        "jpeg_quality": args.jpeg_quality,
        "max_image_kb": args.max_image_kb,
        "timing_excludes_visualization": True,
        "average_timing": average_timing,
    }
    if all_times:
        summary["request_body_avg_kb"] = statistics.mean([x["request_body_bytes"] for x in all_times]) / 1024
        summary["response_body_avg_kb"] = statistics.mean([x["response_body_bytes"] for x in all_times]) / 1024
    if vis_times:
        summary["visualization_timing_excluded"] = {
            "avg": statistics.mean(vis_times),
            "min": min(vis_times),
            "max": max(vis_times),
        }
    return summary


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def safe_file_name(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    return text or "sample"


def unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
