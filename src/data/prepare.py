#!/usr/bin/env python3
"""Download raw datasets and write the processed 2D cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.download_acdc import download_acdc
from src.data.download_kipa import download_kipa
from src.data.download_kvasir import download_kvasir
from src.data.paths import ACDC_RAW, KIPA_RAW, KVASIR_RAW, PROCESSED_ROOT
from src.data.preprocess import preprocess_acdc, preprocess_kipa, preprocess_kvasir

ALL_DATASETS = ("polyp", "kidney_tumor", "cardiac")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=list(ALL_DATASETS),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing raw folders; do not fetch archives",
    )
    parser.add_argument("--kvasir-raw", type=Path, default=KVASIR_RAW)
    parser.add_argument("--kipa-raw", type=Path, default=KIPA_RAW)
    parser.add_argument("--acdc-raw", type=Path, default=ACDC_RAW)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    return parser.parse_args(argv)


def run_prepare(args: argparse.Namespace) -> dict[str, list[str]]:
    processed = args.processed_root
    processed.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}

    if "polyp" in args.datasets:
        raw = args.kvasir_raw if args.skip_download else download_kvasir(args.kvasir_raw)
        written["polyp"] = preprocess_kvasir(raw, processed)

    if "kidney_tumor" in args.datasets:
        raw = args.kipa_raw if args.skip_download else download_kipa(args.kipa_raw)
        ids, _stats = preprocess_kipa(raw, processed)
        written["kidney_tumor"] = ids

    if "cardiac" in args.datasets:
        raw = args.acdc_raw if args.skip_download else download_acdc(args.acdc_raw)
        written["cardiac"] = preprocess_acdc(raw, processed)

    return written


def main(argv: list[str] | None = None) -> int:
    run_prepare(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
