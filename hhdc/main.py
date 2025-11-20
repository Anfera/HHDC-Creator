#!/usr/bin/env python3
"""
End-to-end pipeline for downloading NEON LiDAR data and building Hyperheight cubes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from .cube_generator import generate_cubes_from_config
from .download import DEFAULT_PATTERN, download_site_lidar


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NEON LiDAR data for a site and generate Hyperheight cubes."
    )
    parser.add_argument("--site", required=True, help="NEON site code (e.g. ABBY, OSBS).")
    parser.add_argument(
        "--product",
        default="DP1.30003.001",
        help="NEON product code to download (default: DP1.30003.001 discrete return LiDAR).",
    )
    parser.add_argument(
        "--month",
        help="Optional month in YYYY-MM format. Omitting downloads all available months.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("neon_lidar"),
        help="Base directory for downloads (files saved under <download-dir>/<site>/<month>).",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Substring to filter files when downloading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("cube_config_sample.json"),
        help="Cube configuration JSON file.",
    )
    parser.add_argument(
        "--cube-output-dir",
        type=Path,
        default=Path("cubes"),
        help="Base directory for generated cubes.",
    )
    parser.add_argument(
        "--cube-prefix",
        help="Prefix for generated cube filenames (defaults to the site code).",
    )
    parser.add_argument(
        "--jobs",
        "--cube-jobs",
        dest="cube_jobs",
        type=int,
        default=1,
        help="Number of workers for both reading tiles and cube generation (alias: --cube-jobs).",
    )
    parser.add_argument(
        "--tiles-per-batch",
        type=int,
        default=0,
        help="Process at most NxN tiles per KD-tree build to cap memory (0 disables batching).",
    )
    return parser.parse_args(argv)


def _discover_month_dirs(site_dir: Path, month_filter: Optional[str]) -> List[Path]:
    if month_filter:
        candidate = site_dir / month_filter
        if not candidate.exists():
            raise ValueError(f"No downloaded data found for month {month_filter} in {site_dir}.")
        if not candidate.is_dir():
            raise ValueError(f"{candidate} is not a directory.")
        return [candidate]

    month_dirs = sorted(p for p in site_dir.iterdir() if p.is_dir())
    if not month_dirs:
        raise ValueError(f"No monthly subdirectories were found in {site_dir}.")
    return month_dirs


def _has_laz_files(path: Path) -> bool:
    return next(path.rglob("*.laz"), None) is not None


def _compose_month_prefix(base: str, month_label: str) -> str:
    if base.endswith("_"):
        return f"{base}{month_label}"
    return f"{base}_{month_label}"


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    site_code = args.site.strip().upper()

    try:
        site_dir, downloaded = download_site_lidar(
            product=args.product,
            site=site_code,
            outdir=args.download_dir,
            pattern=args.pattern,
            month=args.month,
            overwrite=args.overwrite,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    try:
        month_dirs = _discover_month_dirs(site_dir, args.month)
    except ValueError as exc:
        print(f"Cube generation skipped: {exc}", file=sys.stderr)
        return 1

    site_cube_root = args.cube_output_dir / site_code
    cube_prefix_base = args.cube_prefix or site_code

    total_cubes = 0
    processed: List[tuple[str, int]] = []
    skipped: List[tuple[str, str]] = []

    for month_dir in month_dirs:
        month_label = month_dir.name
        if not _has_laz_files(month_dir):
            message = "No LiDAR files found"
            print(f"{month_label}: {message}, skipping.")
            skipped.append((month_label, message))
            continue

        month_output = site_cube_root / month_label
        month_prefix = _compose_month_prefix(cube_prefix_base, month_label)
        print(f"{month_label}: generating cubes from {month_dir}...")
        try:
            cubes_created = generate_cubes_from_config(
                config=args.config,
                pointcloud_dir=month_dir,
                output_dir=month_output,
                prefix=month_prefix,
                jobs=args.cube_jobs,
                tiles_per_batch=args.tiles_per_batch,
            )
        except ValueError as exc:
            msg = str(exc)
            print(f"{month_label}: skipped ({msg}).")
            skipped.append((month_label, msg))
            continue

        processed.append((month_label, cubes_created))
        total_cubes += cubes_created
        print(f"{month_label}: {cubes_created} cubes written to {month_output}.")
        import gc
        gc.collect()  # Force garbage collection

    if total_cubes == 0:
        print("No cubes were generated for any month.", file=sys.stderr)
        for label, reason in skipped:
            print(f"  {label}: {reason}", file=sys.stderr)
        return 1

    print("Monthly cube counts:")
    for label, count in processed:
        print(f"  {label}: {count} cube(s) in {site_cube_root / label}")

    if skipped:
        print("Skipped months:")
        for label, reason in skipped:
            print(f"  {label}: {reason}")

    print(
        f"Pipeline complete. Downloaded {downloaded} files into {site_dir} "
        f"and generated {total_cubes} cubes across {len(processed)} month(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
