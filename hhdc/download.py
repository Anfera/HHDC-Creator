#!/usr/bin/env python3
"""
Utility for downloading NEON LiDAR data packages.

Example:
    python neon_lidar_download.py --site ABBY --month 2019-06 --outdir ./downloads
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import requests

API_BASE = "https://data.neonscience.org/api/v0"
DEFAULT_PATTERN = "_classified_point_cloud_colorized.laz"


def fetch_json(url: str) -> Dict:
    """Download JSON data from the NEON API."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.json()


def get_product_metadata(product: str) -> Dict:
    """Return metadata for a NEON data product."""
    url = f"{API_BASE}/products/{product}"
    payload = fetch_json(url)
    return payload.get("data", {})


def get_site_entry(metadata: Dict, site: str) -> Dict:
    """Return the metadata block for the requested site."""
    for entry in metadata.get("siteCodes", []):
        if entry.get("siteCode") == site:
            return entry
    raise ValueError(f"Site {site} has no data for product {metadata.get('productCode')}.")


def get_site_months(metadata: Dict, site: str) -> Iterable[str]:
    """Return available months for the selected site."""
    entry = get_site_entry(metadata, site)
    return entry.get("availableMonths", [])


def get_site_codes(metadata: Dict) -> Sequence[str]:
    """Return a sorted list of available site codes."""
    site_codes = [entry.get("siteCode") for entry in metadata.get("siteCodes", []) if entry.get("siteCode")]
    return sorted(site_codes)


def display_sites(site_codes: Sequence[str]) -> None:
    """Print the available site codes without additional data."""
    if not site_codes:
        print("No sites found for this product.")
        return

    print("Available NEON sites:")
    for code in site_codes:
        print(f"- {code}")


def select_site(metadata: Dict, provided_site: Optional[str] = None) -> str:
    """Display sites and return the site code selected by the user."""
    site_codes = get_site_codes(metadata)
    display_sites(site_codes)

    if not site_codes:
        raise ValueError("Cannot proceed without available sites.")

    if provided_site:
        site = provided_site.strip().upper()
        if site not in site_codes:
            raise ValueError(f"Site {site} is not in the available NEON site list.")
        print(f"Selected site via CLI argument: {site}")
        return site

    while True:
        choice = input("Enter a site code from the list above: ").strip().upper()
        if not choice:
            print("Please enter a site code.")
            continue
        if choice in site_codes:
            print(f"Selected site: {choice}")
            return choice
        print("Invalid site code. Please choose from the displayed list.")


def normalize_site_code(site: str) -> str:
    """Return an uppercase site code and validate it is non-empty."""
    site_code = (site or "").strip().upper()
    if not site_code:
        raise ValueError("A NEON site code is required.")
    return site_code


def validate_site_code(metadata: Dict, site: str) -> str:
    """Ensure the requested site exists for the product."""
    site_code = normalize_site_code(site)
    site_codes = set(get_site_codes(metadata))
    if site_code not in site_codes:
        raise ValueError(
            f"Site {site_code} is not available for product {metadata.get('productCode')}."
        )
    return site_code


def resolve_months(metadata: Dict, site: str, month: Optional[str] = None) -> Sequence[str]:
    """Return the months that should be downloaded, validating an optional filter."""
    available_months = sorted(get_site_months(metadata, site))
    if not available_months:
        raise ValueError(f"No months are available for site {site}.")

    if month:
        if month not in available_months:
            raise ValueError(
                f"Month {month} not available for site {site}. "
                f"Available months: {', '.join(available_months)}"
            )
        return [month]
    return available_months


def iter_files(product: str, site: str, month: str) -> Iterator[Dict]:
    """Yield file metadata dictionaries for the requested site/month."""
    url: Optional[str] = f"{API_BASE}/data/{product}/{site}/{month}"
    while url:
        payload = fetch_json(url)
        data_entries = payload.get("data", [])
        if isinstance(data_entries, dict):
            data_entries = [data_entries]
        elif not isinstance(data_entries, list):
            data_entries = []
        for entry in data_entries:
            if not isinstance(entry, dict):
                continue
            files = entry.get("files", [])
            if not isinstance(files, list):
                continue
            for file in files:
                if isinstance(file, dict):
                    yield file
        url = payload.get("next")


def download_file(file_meta: Dict, output_dir: Path, overwrite: bool = False) -> Path:
    """Download a single NEON file to the output directory."""
    dest = output_dir / file_meta["name"]
    if dest.exists() and not overwrite:
        return dest

    with requests.get(file_meta["url"], stream=True, timeout=300) as resp:
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return dest


def filter_files(files: Iterable[Dict], pattern: Optional[str]) -> Iterator[Dict]:
    """Yield files matching the optional substring pattern."""
    for file_meta in files:
        name = file_meta.get("name", "")
        if not pattern or pattern.lower() in name.lower():
            yield file_meta


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LiDAR point cloud data from the NEON API."
    )
    parser.add_argument(
        "--product",
        default="DP1.30003.001",
        help="NEON product code (default: DP1.30003.001 discrete return LiDAR)",
    )
    parser.add_argument("--site", help="NEON site code (e.g. ABBY, OSBS)")
    parser.add_argument(
        "--month",
        help="Specific month in YYYY-MM format. Omit to download all months for the site.",
    )
    parser.add_argument(
        "--outdir",
        default=Path("neon_lidar"),
        type=Path,
        help="Base directory to place downloaded files (data stored under <outdir>/<site>).",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Substring to filter files (default downloads *_classified_point_cloud_colorized.laz)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist locally",
    )
    return parser.parse_args(argv)


def download_site_lidar(
    *,
    product: str,
    site: str,
    outdir: Path,
    pattern: Optional[str] = DEFAULT_PATTERN,
    month: Optional[str] = None,
    overwrite: bool = False,
    metadata: Optional[Dict] = None,
    quiet: bool = False,
) -> Tuple[Path, int]:
    """
    Download LiDAR files for a single NEON site. Returns the site directory and count of files.
    """
    metadata = metadata or get_product_metadata(product)
    site_code = validate_site_code(metadata, site)
    months_to_download = resolve_months(metadata, site_code, month)

    site_dir = outdir / site_code
    total_downloaded = 0
    for month_id in months_to_download:
        month_files = list(filter_files(iter_files(product, site_code, month_id), pattern))
        if not month_files:
            if not quiet:
                print(f"{month_id}: No files matched {pattern!r}.")
            continue

        month_dir = site_dir / month_id
        if not quiet:
            print(f"{month_id}: downloading {len(month_files)} files to {month_dir}...")
        for file_meta in month_files:
            dest = download_file(file_meta, month_dir, overwrite=overwrite)
            total_downloaded += 1
            if not quiet:
                print(f"  {dest}")

    if total_downloaded == 0:
        raise RuntimeError("No files matching the requested pattern were found.")

    if not quiet:
        print(f"Download complete. {total_downloaded} files saved in {site_dir}.")

    return site_dir, total_downloaded


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    metadata = get_product_metadata(args.product)

    try:
        site = select_site(metadata, args.site)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        _, total_downloaded = download_site_lidar(
            product=args.product,
            site=site,
            outdir=args.outdir,
            pattern=args.pattern,
            month=args.month,
            overwrite=args.overwrite,
            metadata=metadata,
            quiet=False,
        )
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if total_downloaded == 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
