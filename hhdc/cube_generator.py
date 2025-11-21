#!/usr/bin/env python3
"""
Hyperheight data cube generator (Optimized).

Loads NEON LiDAR point clouds (.laz) and aggregates them into voxelized histograms
according to a JSON configuration. The script tiles the full spatial extent into
as many cubes as will fit the requested dimensions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import laspy
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

try:
    # pykdtree keeps coordinates in float32, significantly lowering peak RAM.
    from pykdtree.kdtree import KDTree as PykdTree

    PYKDTREE_AVAILABLE = True
    PYKDTREE_SUPPORTS_BALL_QUERY = hasattr(PykdTree, "query_ball_point")
except Exception:
    PykdTree = None
    PYKDTREE_AVAILABLE = False
    PYKDTREE_SUPPORTS_BALL_QUERY = False


@dataclass
class CubeConfig:
    cube_length: int
    vertical_height: float
    vertical_resolution: float
    footprint_separation: float
    footprint_radius: float
    # New setting: default to 1% (0.01)
    outlier_quantile: float = 0.01 

    @classmethod
    def from_json(cls, path: Path) -> "CubeConfig":
        payload = json.loads(path.read_text())
        return cls(
            cube_length=int(payload["cube_length"]),
            vertical_height=float(payload["vertical_height"]),
            vertical_resolution=float(payload["vertical_resolution"]),
            footprint_separation=float(payload["footprint_separation"]),
            footprint_radius=float(payload["footprint_radius"]),
            # Load optional quantile, default to 0.01 if missing
            outlier_quantile=payload.get("outlier_quantile", 0.01),
        )


@dataclass
class LazTileInfo:
    path: Path
    point_count: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def _scan_laz_files(files: Sequence[Path]) -> List[LazTileInfo]:
    """
    Read LAS/LAZ headers once to collect point counts and spatial bounds
    without pulling point data into memory.
    """
    infos: List[LazTileInfo] = []
    for path in tqdm(files, desc="Scanning headers"):
        with laspy.open(path) as reader:
            hdr = reader.header
            infos.append(
                LazTileInfo(
                    path=path,
                    point_count=int(hdr.point_count),
                    x_min=float(hdr.mins[0]),
                    x_max=float(hdr.maxs[0]),
                    y_min=float(hdr.mins[1]),
                    y_max=float(hdr.maxs[1]),
                )
            )
    return infos


def _extent_from_infos(infos: Sequence[LazTileInfo]) -> Tuple[float, float, float, float]:
    x_min = min(info.x_min for info in infos)
    x_max = max(info.x_max for info in infos)
    y_min = min(info.y_min for info in infos)
    y_max = max(info.y_max for info in infos)
    return x_min, x_max, y_min, y_max


def _get_point_count(path: Path) -> int:
    """Reads only the header to get the point count fast."""
    with laspy.open(path) as reader:
        return reader.header.point_count


def _read_file_into_buffer(
    path: Path,
    start_idx: int,
    xs_shared: np.ndarray,
    ys_shared: np.ndarray,
    zs_shared: np.ndarray,
    classes_shared: Optional[np.ndarray] = None,
) -> None:
    """Reads a single file and writes directly into the pre-allocated arrays."""
    with laspy.open(path) as reader:
        # laspy.read() is optimized to read all dimensions, but we simply
        # extract what we need and let the object die immediately to free RAM.
        las = reader.read()

        # Verify we don't overflow if the header count was wrong (rare but possible)
        count = len(las)
        end_idx = start_idx + count

        # Direct assignment avoids creating intermediate large copies
        xs_shared[start_idx:end_idx] = las.x
        ys_shared[start_idx:end_idx] = las.y
        zs_shared[start_idx:end_idx] = las.z
        if classes_shared is not None:
            if "classification" in las.point_format.dimension_names:
                classes_shared[start_idx:end_idx] = np.asarray(las.classification, dtype=np.uint8)
            else:
                classes_shared[start_idx:end_idx] = 0


def read_points(
    files: Sequence[Union[Path, LazTileInfo]], jobs: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized reader:
    1. Scans headers for total count (unless pre-scanned LazTileInfo objects are provided).
    2. Pre-allocates one continuous memory block.
    3. Fills in parallel.
    """
    if not files:
        raise ValueError("No point cloud files provided.")

    first = files[0]
    if isinstance(first, LazTileInfo):
        file_infos: List[LazTileInfo] = list(files)  # type: ignore[arg-type]
        file_counts = [(info.path, info.point_count) for info in file_infos]
        total_points = sum(info.point_count for info in file_infos)
    else:
        print("Scanning file headers...")
        file_counts = []
        total_points = 0
        for p in tqdm(files, desc="Scanning headers"):
            count = _get_point_count(p)  # type: ignore[arg-type]
            file_counts.append((p, count))
            total_points += count

    print(f"Total points to load: {total_points:,}")

    if total_points == 0:
        raise ValueError("Point clouds contain no points.")

    # 2. Pre-allocate memory (Float32 is sufficient and saves 50% RAM vs Float64)
    xs = np.empty(total_points, dtype=np.float32)
    ys = np.empty(total_points, dtype=np.float32)
    zs = np.empty(total_points, dtype=np.float32)
    classifications = np.empty(total_points, dtype=np.uint8)

    # 3. Parallel Load
    current_start = 0
    tasks = []

    for path, count in file_counts:
        tasks.append((path, current_start))
        current_start += count

    if jobs and jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(_read_file_into_buffer, path, start, xs, ys, zs, classifications)
                for path, start in tasks
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
                pass  # Just updating the progress bar
    else:
        for path, start in tqdm(tasks, desc="Loading data"):
            _read_file_into_buffer(path, start, xs, ys, zs, classifications)

    return filter_invalid_points(xs, ys, zs, classifications)


def filter_invalid_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    classification: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Removes NaN/Inf records and optional noise points (classification == 7) that
    would contaminate downstream histogram bin edges.
    """
    if xs.size == 0:
        return xs, ys, zs

    mask = np.ones(xs.shape[0], dtype=bool)
    removal_details: List[str] = []

    if classification is not None:
        if classification.shape[0] != xs.shape[0]:
            raise ValueError("Classification array length does not match coordinate arrays.")
        noise_mask = classification != 7
        noise_removed = int(mask.sum() - np.count_nonzero(mask & noise_mask))
        mask &= noise_mask
        if noise_removed:
            removal_details.append(f"{noise_removed} with classification 7")

    finite_mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    invalid_removed = int(mask.sum() - np.count_nonzero(mask & finite_mask))
    mask &= finite_mask
    if invalid_removed:
        removal_details.append(f"{invalid_removed} invalid (NaN/Inf)")

    valid_points = int(mask.sum())
    removed_points = int(xs.size - valid_points)

    if valid_points == 0:
        raise ValueError("All point records were filtered out (noise or invalid values).")

    if removed_points:
        detail = "; ".join(removal_details) if removal_details else "filtered points"
        print(f"Discarding {removed_points} points ({detail}) before cube generation.")

    return xs[mask], ys[mask], zs[mask]


def _bounds_intersect(
    a_min_x: float, a_max_x: float, a_min_y: float, a_max_y: float, b_bounds: Tuple[float, float, float, float]
) -> bool:
    b_min_x, b_max_x, b_min_y, b_max_y = b_bounds
    return not (a_max_x < b_min_x or a_min_x > b_max_x or a_max_y < b_min_y or a_min_y > b_max_y)


def _select_infos_for_bounds(
    infos: Sequence[LazTileInfo], bounds: Tuple[float, float, float, float]
) -> List[LazTileInfo]:
    return [
        info
        for info in infos
        if _bounds_intersect(info.x_min, info.x_max, info.y_min, info.y_max, bounds)
    ]


def _expand_bounds(bounds: Tuple[float, float, float, float], padding: float) -> Tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = bounds
    return x_min - padding, x_max + padding, y_min - padding, y_max + padding


def read_points_clipped(
    infos: Sequence[LazTileInfo],
    bounds: Tuple[float, float, float, float],
    jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load only points that fall inside the provided XY bounds. This is used for
    batched cube generation to keep the KD-tree small.
    """
    if not infos:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    x_min, x_max, y_min, y_max = bounds

    def _read_clip(info: LazTileInfo) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with laspy.open(info.path) as reader:
            las = reader.read()
        classification = (
            np.asarray(las.classification, dtype=np.uint8)
            if "classification" in las.point_format.dimension_names
            else np.zeros(len(las), dtype=np.uint8)
        )
        mask = (las.x >= x_min) & (las.x <= x_max) & (las.y >= y_min) & (las.y <= y_max)
        if not mask.any():
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.uint8),
            )
        return (
            np.asarray(las.x[mask], dtype=np.float32),
            np.asarray(las.y[mask], dtype=np.float32),
            np.asarray(las.z[mask], dtype=np.float32),
            classification[mask],
        )

    results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    if jobs and jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_read_clip, info): info for info in infos}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading batch"):
                results.append(future.result())
    else:
        for info in tqdm(infos, desc="Loading batch"):
            results.append(_read_clip(info))

    xs_parts = [r[0] for r in results if r[0].size]
    ys_parts = [r[1] for r in results if r[1].size]
    zs_parts = [r[2] for r in results if r[2].size]
    cls_parts = [r[3] for r in results if r[3].size]

    if not xs_parts:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    xs = np.concatenate(xs_parts)
    ys = np.concatenate(ys_parts)
    zs = np.concatenate(zs_parts)
    classification = np.concatenate(cls_parts) if cls_parts else None
    return filter_invalid_points(xs, ys, zs, classification)


def build_grid(
    tile_x_min: float,
    tile_x_max: float,
    tile_y_min: float,
    tile_y_max: float,
    cfg: CubeConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates grid coordinates based on fixed cube dimensions."""
    num_x = int(np.round((tile_x_max - tile_x_min) / cfg.footprint_separation))
    num_y = int(np.round((tile_y_max - tile_y_min) / cfg.footprint_separation))

    offset = cfg.footprint_separation / 2.0

    x_coords = tile_x_min + offset + (np.arange(num_x) * cfg.footprint_separation)
    y_coords = tile_y_min + offset + (np.arange(num_y) * cfg.footprint_separation)

    return x_coords, y_coords


def iter_tile_bounds(
    x_min: float, x_max: float, y_min: float, y_max: float, cfg: CubeConfig
) -> Iterator[Tuple[int, int, float, float, float, float]]:
    if (x_max - x_min) < cfg.cube_length or (y_max - y_min) < cfg.cube_length:
        return

    x_stop = x_max - cfg.cube_length
    y_stop = y_max - cfg.cube_length

    x_starts = np.arange(x_min, x_stop + 1e-6, cfg.cube_length)
    y_starts = np.arange(y_min, y_stop + 1e-6, cfg.cube_length)

    for y_idx, y0 in enumerate(y_starts):
        y1 = y0 + cfg.cube_length
        for x_idx, x0 in enumerate(x_starts):
            x1 = x0 + cfg.cube_length
            yield x_idx, y_idx, float(x0), float(x1), float(y0), float(y1)


def _tile_batches(
    tasks: Sequence[Tuple[int, int, float, float, float, float]], tiles_per_batch: int
) -> List[Tuple[Tuple[int, int, float, float, float, float], List[Tuple[int, int, float, float, float, float]]]]:
    """
    Group tile tasks into spatial batches of size tiles_per_batch x tiles_per_batch.
    Returns a list of (batch_bounds, batch_tasks).
    """
    buckets: Dict[Tuple[int, int], List[Tuple[int, int, float, float, float, float]]] = {}
    for task in tasks:
        x_idx, y_idx = task[0], task[1]
        key = (x_idx // tiles_per_batch, y_idx // tiles_per_batch)
        buckets.setdefault(key, []).append(task)

    batches: List[
        Tuple[Tuple[int, int, float, float, float, float], List[Tuple[int, int, float, float, float, float]]]
    ] = []
    for key in sorted(buckets):
        batch_tasks = buckets[key]
        x_min = min(t[2] for t in batch_tasks)
        x_max = max(t[3] for t in batch_tasks)
        y_min = min(t[4] for t in batch_tasks)
        y_max = max(t[5] for t in batch_tasks)
        batches.append(((key[0], key[1], x_min, x_max, y_min, y_max), batch_tasks))
    return batches


def compute_histogram(heights: np.ndarray, z_min: float, z_max: float, bins: int) -> np.ndarray:
    if heights.size == 0:
        return np.zeros(bins, dtype=np.int32)
    hist, _ = np.histogram(heights, bins=bins, range=(z_min, z_max))
    return hist.astype(np.int32)


def build_kdtree(coords: np.ndarray):
    """
    Build the spatial index. Prefer pykdtree to keep float32 internally and save memory;
    fall back to scipy's cKDTree if pykdtree is not installed.
    """
    if PYKDTREE_AVAILABLE and PYKDTREE_SUPPORTS_BALL_QUERY:
        return PykdTree(coords.astype(np.float32, copy=False), leafsize=32)

    # cKDTree upcasts to float64 internally, so memory is larger; keep leafsize modest.
    return cKDTree(coords.astype(np.float32, copy=False), leafsize=32)


def generate_cube(
    tree: cKDTree,
    zs: np.ndarray,
    cfg: CubeConfig,
    bounds: Tuple[float, float, float, float],
    z_base: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_top = z_base + cfg.vertical_height
    bin_count = int(math.ceil(cfg.vertical_height / cfg.vertical_resolution))
    bin_edges = np.linspace(z_base, z_top, bin_count + 1)

    x_min, x_max, y_min, y_max = bounds
    x_centers, y_centers = build_grid(x_min, x_max, y_min, y_max, cfg)
    
    cube = np.zeros((y_centers.size, x_centers.size, bin_count), dtype=np.int32)
    footprint_counts = np.zeros((y_centers.size, x_centers.size), dtype=np.int32)

    for yi, cy in enumerate(y_centers):
        for xi, cx in enumerate(x_centers):
            idx = tree.query_ball_point([cx, cy], cfg.footprint_radius)
            if not idx:
                continue
            heights = zs[np.asarray(idx, dtype=int)]
            heights = heights[(heights >= z_base) & (heights <= z_top)]
            count = heights.size
            footprint_counts[yi, xi] = count
            if count == 0:
                continue
            cube[yi, xi, :] = compute_histogram(heights, z_base, z_top, bin_count)

    return cube, x_centers, y_centers, bin_edges, footprint_counts


def tile_has_points(tree: cKDTree, bounds: Tuple[float, float, float, float], radius: float) -> bool:
    x_min, x_max, y_min, y_max = bounds
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    diagonal = math.hypot(x_max - x_min, y_max - y_min) / 2.0 + radius
    indices = tree.query_ball_point([cx, cy], diagonal)
    return bool(indices)


def _tile_point_indices(tree: cKDTree, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Returns the indices of points strictly inside the rectangular tile bounds.
    We query with a circumscribed circle for speed, then mask to the box.
    """
    x_min, x_max, y_min, y_max = bounds
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    radius = math.hypot(x_max - x_min, y_max - y_min) / 2.0

    candidate_idx = tree.query_ball_point([cx, cy], radius)
    if not candidate_idx:
        return np.empty(0, dtype=int)

    candidate_idx = np.asarray(candidate_idx, dtype=int)
    pts = tree.data[candidate_idx]
    mask = (
        (pts[:, 0] >= x_min)
        & (pts[:, 0] <= x_max)
        & (pts[:, 1] >= y_min)
        & (pts[:, 1] <= y_max)
    )
    return candidate_idx[mask]


GLOBAL_STATE: dict = {}


def _set_global_state(
    zs: np.ndarray,
    tree: cKDTree,
    cfg: CubeConfig,
    output_dir: Path,
    prefix: str,
) -> None:
    """
    Note: We do not store xs/ys in global state anymore as they are consumed
    by the KDTree and not needed for the histogram step, saving RAM.
    """
    GLOBAL_STATE.clear()
    GLOBAL_STATE.update(
        {
            "zs": zs,
            "tree": tree,
            "cfg": cfg,
            "output_dir": output_dir,
            "prefix": prefix,
        }
    )


def _run_tile_tasks(tasks: Sequence[Tuple[int, int, float, float, float, float]], jobs: int, desc: str):
    results = []
    if jobs and jobs > 1:
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
        with ctx.Pool(processes=jobs) as pool:
            for result in tqdm(pool.imap_unordered(_process_tile, tasks), total=len(tasks), desc=desc):
                if result:
                    results.append(result)
    else:
        for task in tqdm(tasks, desc=desc):
            result = _process_tile(task)
            if result:
                results.append(result)
    return results


def _process_tile(task: Tuple[int, int, float, float, float, float]):
    if not GLOBAL_STATE:
        raise RuntimeError("Global state not initialized.")

    x_idx, y_idx, x0, x1, y0, y1 = task
    bounds = (x0, x1, y0, y1)
    tree: cKDTree = GLOBAL_STATE["tree"]
    cfg: CubeConfig = GLOBAL_STATE["cfg"]

    # 1. Geometric checks
    if (x1 - x0) <= 2 * cfg.footprint_radius or (y1 - y0) <= 2 * cfg.footprint_radius:
        return None

    if not tile_has_points(tree, bounds, cfg.footprint_radius):
        return None

    tile_indices = _tile_point_indices(tree, bounds)
    if tile_indices.size == 0:
        return None

    zs = GLOBAL_STATE["zs"]
    local_zs = zs[tile_indices]

    # 2. Automatic Outlier Filtering (Percentile Clipping)
    # We use getattr to be safe in case 'outlier_quantile' isn't in your Config class yet.
    # Default is 0.0 (no filtering).
    quantile = getattr(cfg, "outlier_quantile", 0.0)
    
    if quantile > 0.0:
        # Calculate the safe range (e.g., 1st to 99th percentile)
        q_lower = quantile * 100
        q_upper = (1.0 - quantile) * 100
        
        # np.percentile handles the sorting and interpolation automatically
        z_min_cutoff, z_max_cutoff = np.percentile(local_zs, [q_lower, q_upper])
        
        # We set the base of the cube to the robust percentile, not the outlier min
        tile_z_base = float(z_min_cutoff)
        
        # Optional: If you want to ensure the cube doesn't include points above the
        # percentile max, you could rely on cfg.vertical_height, or strictly mask
        # inside generate_cube. Usually, fixing z_base is enough to fix the grid alignment.
    else:
        # Standard behavior: use the absolute minimum point found
        tile_z_base = float(np.min(local_zs))

    try:
        # 3. Generate Cube
        # Note: generate_cube naturally filters out points below tile_z_base,
        # so the low outliers are automatically discarded here.
        cube, x_centers, y_centers, bin_edges, footprint_counts = generate_cube(
            tree,
            zs,
            cfg,
            bounds,
            tile_z_base,
        )
    except ValueError:
        return None

    if not footprint_counts.any():
        return None

    metadata = {
        "cube_length": float(cfg.cube_length),
        "vertical_height": float(cfg.vertical_height),
        "vertical_resolution": float(cfg.vertical_resolution),
        "footprint_separation": float(cfg.footprint_separation),
        "footprint_radius": float(cfg.footprint_radius),
        "tile_min_z": float(tile_z_base),
        "outlier_quantile_used": float(quantile),
        "footprints": {
            "total": int(footprint_counts.size),
            "empty": int(np.count_nonzero(footprint_counts == 0)),
        },
        "tile_bounds": {
            "x_min": float(x0),
            "x_max": float(x1),
            "y_min": float(y0),
            "y_max": float(y1),
        },
        "tile_indices": {"x": int(x_idx), "y": int(y_idx)},
    }

    out_dir: Path = GLOBAL_STATE["output_dir"]
    prefix: str = GLOBAL_STATE["prefix"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_x{x_idx:03d}_y{y_idx:03d}.npz"
    np.savez_compressed(
        out_path,
        cube=cube,
        x_centers=x_centers,
        y_centers=y_centers,
        bin_edges=bin_edges,
        footprint_counts=footprint_counts,
        metadata=json.dumps(metadata),
    )
    return str(out_path), cube.shape


def find_laz_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.laz") if p.is_file())


def _generate_cubes_batched(
    *,
    tasks: Sequence[Tuple[int, int, float, float, float, float]],
    file_infos: Sequence[LazTileInfo],
    cfg: CubeConfig,
    output_dir: Path,
    prefix: str,
    jobs: int,
    tiles_per_batch: int,
) -> int:
    batches = _tile_batches(tasks, tiles_per_batch)
    if not batches:
        return 0

    print(f"Processing {len(batches)} batch(es) with at most {tiles_per_batch}x{tiles_per_batch} tiles each.")
    total_results = []
    padding = cfg.footprint_radius + (cfg.footprint_separation / 2.0)
    read_jobs = max(1, jobs)

    for batch_idx, (batch_bounds, batch_tasks) in enumerate(batches, start=1):
        bx, by, x_min, x_max, y_min, y_max = batch_bounds
        expanded_bounds = _expand_bounds((x_min, x_max, y_min, y_max), padding)
        relevant_infos = _select_infos_for_bounds(file_infos, expanded_bounds)
        if not relevant_infos:
            continue

        approx_pts = sum(info.point_count for info in relevant_infos)
        print(
            f"Batch ({bx}, {by}) [{batch_idx}/{len(batches)}]: "
            f"{len(relevant_infos)} tile(s), header total {approx_pts:,} pts, "
            f"bounds {expanded_bounds}."
        )

        xs, ys, zs = read_points_clipped(relevant_infos, expanded_bounds, jobs=read_jobs)
        if zs.size == 0:
            print("  No points survived filtering; skipping batch.")
            continue

        coords = np.empty((xs.size, 2), dtype=np.float32)
        coords[:, 0] = xs
        coords[:, 1] = ys
        tree = build_kdtree(coords)
        del coords, xs, ys
        gc.collect()

        _set_global_state(zs, tree, cfg, output_dir, prefix)
        batch_results = _run_tile_tasks(batch_tasks, jobs, desc=f"Processing tiles (batch {batch_idx})")
        total_results.extend(batch_results)

        del tree, zs
        gc.collect()

    return len(total_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hyperheight data cubes from NEON point clouds.")
    parser.add_argument("--config", required=True, type=Path, help="Path to JSON config file.")
    parser.add_argument("--pointcloud-dir", required=True, type=Path, help="Directory containing .laz tiles.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write per-tile cube .npz files.",
    )
    parser.add_argument(
        "--prefix",
        default="cube",
        help="Filename prefix for generated cubes (default: cube).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of workers to use (applied to reading threads and cube generation; default: 1).",
    )
    parser.add_argument(
        "--tiles-per-batch",
        type=int,
        default=0,
        help="Limit memory by processing at most NxN tiles per KD-tree build (0 disables batching).",
    )
    return parser.parse_args()


def generate_cubes_from_config(
    *,
    config: Path,
    pointcloud_dir: Path,
    output_dir: Path,
    prefix: str = "cube",
    jobs: int = 1,
    tiles_per_batch: Optional[int] = None,
) -> int:
    """Convenience wrapper that loads a config file before generating cubes."""
    cfg = CubeConfig.from_json(config)
    return generate_cubes(
        cfg=cfg,
        pointcloud_dir=pointcloud_dir,
        output_dir=output_dir,
        prefix=prefix,
        jobs=jobs,
        tiles_per_batch=tiles_per_batch,
    )


def generate_cubes(
    *,
    cfg: CubeConfig,
    pointcloud_dir: Path,
    output_dir: Path,
    prefix: str,
    jobs: int,
    tiles_per_batch: Optional[int] = None,
) -> int:
    """Generate cubes for the given configuration and directory."""
    laz_files = find_laz_files(pointcloud_dir)
    if not laz_files:
        raise ValueError("No .laz files found in the specified directory.")

    print(f"Found {len(laz_files)} files.")
    print("Scanning file headers...")
    file_infos = _scan_laz_files(laz_files)
    total_points = sum(info.point_count for info in file_infos)
    x_min, x_max, y_min, y_max = _extent_from_infos(file_infos)
    print(f"Header summary: ~{total_points:,} points, extent x[{x_min:.2f},{x_max:.2f}] y[{y_min:.2f},{y_max:.2f}]")

    print("Calculating tile bounds...")
    tasks = list(iter_tile_bounds(x_min, x_max, y_min, y_max, cfg))

    if not tasks:
        raise ValueError("Point cloud extent is smaller than the requested cube length.")

    read_jobs = max(1, jobs)
    tiles_per_batch = tiles_per_batch or 0

    if tiles_per_batch > 0:
        total_cubes = _generate_cubes_batched(
            tasks=tasks,
            file_infos=file_infos,
            cfg=cfg,
            output_dir=output_dir,
            prefix=prefix,
            jobs=jobs,
            tiles_per_batch=tiles_per_batch,
        )
        if total_cubes == 0:
            raise ValueError(
                "No cubes were generated with the current configuration and spatial extent (batched mode)."
            )
        print(f"Finished generating {total_cubes} cubes in {output_dir}.")
        return total_cubes

    xs, ys, zs = read_points(file_infos, jobs=read_jobs)

    print("Building spatial index (this may take a moment)...")
    coords = np.empty((xs.size, 2), dtype=np.float32)
    coords[:, 0] = xs
    coords[:, 1] = ys

    if PYKDTREE_AVAILABLE and PYKDTREE_SUPPORTS_BALL_QUERY:
        print("Using pykdtree for lower-memory float32 KD-tree.")
    elif PYKDTREE_AVAILABLE and not PYKDTREE_SUPPORTS_BALL_QUERY:
        print("pykdtree lacks query_ball_point; using scipy.spatial.cKDTree instead.")
    else:
        print("pykdtree not available; falling back to scipy.spatial.cKDTree (higher memory).")

    tree = build_kdtree(coords)
    del coords  # drop temporary XY array now that the KD-tree is built
    del xs, ys  # raw coordinate arrays no longer needed after KD-tree construction
    gc.collect()
    print("Spatial index built.")

    _set_global_state(zs, tree, cfg, output_dir, prefix)

    print(f"Generating {len(tasks)} cubes...")
    results = _run_tile_tasks(tasks, jobs, desc="Processing tiles")

    total_cubes = len(results)
    if total_cubes == 0:
        raise ValueError("No cubes were generated with the current configuration and spatial extent.")

    del tree, zs
    gc.collect()

    print(f"Finished generating {total_cubes} cubes in {output_dir}.")
    return total_cubes


def main() -> int:
    args = parse_args()
    try:
        generate_cubes_from_config(
            config=args.config,
            pointcloud_dir=args.pointcloud_dir,
            output_dir=args.output_dir,
            prefix=args.prefix,
            jobs=args.jobs,
            tiles_per_batch=args.tiles_per_batch,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
