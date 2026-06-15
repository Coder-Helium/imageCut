#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines = [line for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(lines)
    n_train = int(len(lines) * args.train_ratio)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text("\n".join(lines[:n_train]) + "\n", encoding="utf-8")
    (out_dir / "val.jsonl").write_text("\n".join(lines[n_train:]) + "\n", encoding="utf-8")
    print({"train": n_train, "val": len(lines) - n_train, "out_dir": str(out_dir)})


if __name__ == "__main__":
    main()

