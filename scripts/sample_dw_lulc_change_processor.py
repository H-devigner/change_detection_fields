#!/usr/bin/env python3
"""Sample change detection processor for Dynamic World LULC raster snapshots.

The default file pattern targets names like:
  kursh_2021_february_april__dw_lulc.tif
  kursh_2021_june_august__dw_lulc.tif
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires rasterio. Install with: pip install -e ."
    ) from exc


DW_CLASSES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}

DW_COLORS = {
    0: "#419bdf",
    1: "#397d49",
    2: "#88b053",
    3: "#7a87c6",
    4: "#e49635",
    5: "#dfc35a",
    6: "#c4281b",
    7: "#a59b8f",
    8: "#b39fe1",
}

CROP_CLASS = 4
LOGGER = logging.getLogger("dw_lulc_change")

CHANGED_MASK_COLORMAP = {
    0: (245, 245, 238, 255),
    1: (214, 39, 40, 255),
    255: (0, 0, 0, 0),
}

CROP_STATE_COLORMAP = {
    0: (245, 245, 238, 255),
    1: (44, 160, 44, 255),
    2: (214, 39, 40, 255),
    3: (31, 119, 180, 255),
    255: (0, 0, 0, 0),
}


@dataclass(frozen=True)
class Snapshot:
    year: int
    season: str
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process sample 2021-2023 Dynamic World LULC seasonal change."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing LULC raster snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/change_detection/kursh_2021_2023_dw_lulc_sample"),
        help="Output directory for rasters, tables, and figures.",
    )
    parser.add_argument(
        "--years",
        default="2021,2022,2023",
        help="Comma-separated years to process.",
    )
    parser.add_argument(
        "--seasons",
        default="february_april,june_august",
        help="Comma-separated ordered season tokens.",
    )
    parser.add_argument(
        "--filename-template",
        default="kursh_{year}_{season}__dw_lulc",
        help="Filename stem template. Extension is discovered automatically.",
    )
    parser.add_argument(
        "--extensions",
        default=".tif,.tiff,.vrt",
        help="Comma-separated raster extensions to try inside files or matching snapshot directories.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively inside --input-dir for snapshot files/directories.",
    )
    parser.add_argument(
        "--nodata-values",
        default="",
        help="Optional comma-separated pixel values to treat as nodata.",
    )
    parser.add_argument(
        "--no-reproject",
        action="store_true",
        help="Fail instead of aligning rasters to the first snapshot grid.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip PNG figure exports. Useful for fast diagnostic runs.",
    )
    parser.add_argument(
        "--skip-rasters",
        action="store_true",
        help="Skip GeoTIFF change raster exports. CSV tables are still written.",
    )
    parser.add_argument(
        "--preview-max-size",
        type=int,
        default=2048,
        help="Maximum width/height for PNG previews. Use 0 for full-resolution figures.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print warnings and errors.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debug logs.",
    )
    return parser.parse_args()


def setup_logging(args: argparse.Namespace) -> None:
    if args.debug:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def candidate_listing(input_dir: Path, stem: str) -> str:
    if not input_dir.exists():
        return "input directory does not exist"
    candidates = sorted(p.name for p in input_dir.glob(f"*{stem}*"))[:20]
    if candidates:
        return "nearby candidates: " + ", ".join(candidates)
    return "no nearby candidates matched the expected stem"


def pick_raster_from_directory(directory: Path, extensions: list[str]) -> Path | None:
    matches: list[Path] = []
    for ext in extensions:
        matches.extend(directory.rglob(f"*{ext}"))
    matches = sorted(
        p for p in matches
        if p.is_file() and not p.name.startswith(".") and ".aux.xml" not in p.name
    )
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    preferred = [
        p for p in matches
        if any(token in p.name.lower() for token in ("dw", "lulc", "classification", "class"))
    ]
    return sorted(preferred or matches, key=lambda p: (len(p.parts), p.name))[0]


def find_snapshot_file(input_dir: Path, stem: str, extensions: list[str], recursive: bool) -> Path:
    directory_candidate = input_dir / stem
    if directory_candidate.is_dir():
        raster = pick_raster_from_directory(directory_candidate, extensions)
        if raster is not None:
            return raster

    for ext in extensions:
        candidate = input_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(input_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0]

    if recursive:
        directory_matches = sorted(p for p in input_dir.rglob(stem) if p.is_dir())
        for directory_match in directory_matches:
            raster = pick_raster_from_directory(directory_match, extensions)
            if raster is not None:
                return raster
        for ext in extensions:
            recursive_matches = sorted(input_dir.rglob(f"{stem}{ext}"))
            if recursive_matches:
                return recursive_matches[0]

    raise FileNotFoundError(
        f"No raster found for stem '{stem}' in {input_dir}. "
        f"Expected either '{stem}.tif' or a directory named '{stem}' containing a raster. "
        f"{candidate_listing(input_dir, stem)}"
    )


def discover_snapshots(args: argparse.Namespace) -> list[Snapshot]:
    years = parse_csv_ints(args.years)
    seasons = parse_csv_strings(args.seasons)
    extensions = parse_csv_strings(args.extensions)
    snapshots = []
    for year in years:
        for season in seasons:
            stem = args.filename_template.format(year=year, season=season)
            path = find_snapshot_file(args.input_dir, stem, extensions, recursive=args.recursive)
            LOGGER.info("Discovered %s -> %s", f"{year}_{season}", path)
            snapshots.append(Snapshot(year=year, season=season, label=f"{year}_{season}", path=path))
    return snapshots


def read_reference(snapshot: Snapshot) -> tuple[np.ndarray, dict, np.ndarray]:
    with rasterio.open(snapshot.path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        valid = np.ones(array.shape, dtype=bool)
        if src.nodata is not None:
            valid &= array != src.nodata
    return array, profile, valid


def read_aligned(snapshot: Snapshot, ref_profile: dict, no_reproject: bool) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(snapshot.path) as src:
        same_grid = (
            src.crs == ref_profile["crs"]
            and src.transform == ref_profile["transform"]
            and src.width == ref_profile["width"]
            and src.height == ref_profile["height"]
        )
        valid = np.ones((ref_profile["height"], ref_profile["width"]), dtype=bool)
        if same_grid:
            LOGGER.debug("%s already matches reference grid", snapshot.label)
            array = src.read(1)
            if src.nodata is not None:
                valid &= array != src.nodata
            return array, valid

        if no_reproject:
            raise ValueError(f"Raster grid differs from reference: {snapshot.path}")

        LOGGER.info("Aligning %s to reference grid with nearest-neighbor resampling", snapshot.label)
        array = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
        )
        if src.nodata is not None:
            valid &= array != src.nodata
        return array, valid


def pixel_area_ha(profile: dict) -> float:
    transform = profile["transform"]
    crs = profile.get("crs")
    if crs is not None and getattr(crs, "is_geographic", False):
        center_lat = transform.f + transform.e * (profile["height"] / 2.0)
        meters_per_degree = 111_320.0
        area_m2 = (
            abs(transform.a)
            * meters_per_degree
            * abs(transform.e)
            * meters_per_degree
            * np.cos(np.deg2rad(center_lat))
        )
    else:
        area_m2 = abs(transform.a * transform.e)
    return float(area_m2 / 10_000.0)


def write_raster(
    path: Path,
    array: np.ndarray,
    profile: dict,
    nodata: int,
    colormap: dict[int, tuple[int, int, int, int]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype=str(array.dtype),
        compress="lzw",
        nodata=nodata,
        BIGTIFF="YES",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    write_start = time.perf_counter()
    LOGGER.info("Writing raster %s: shape=%s, dtype=%s, BigTIFF=YES", path.name, array.shape, array.dtype)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array, 1)
        if colormap is not None:
            dst.write_colormap(1, colormap)
    LOGGER.info("Wrote raster %s in %.1fs", path.name, time.perf_counter() - write_start)


def class_counts(array: np.ndarray, valid: np.ndarray, snapshot: Snapshot, px_ha: float) -> list[dict]:
    rows = []
    for value, count in zip(*np.unique(array[valid], return_counts=True)):
        value = int(value)
        rows.append(
            {
                "snapshot": snapshot.label,
                "year": snapshot.year,
                "season": snapshot.season,
                "class_id": value,
                "class_name": DW_CLASSES.get(value, f"class_{value}"),
                "pixels": int(count),
                "area_ha": float(count * px_ha),
            }
        )
    return rows


def transition_rows(transition_code: np.ndarray, valid: np.ndarray, from_label: str, to_label: str, px_ha: float) -> list[dict]:
    rows = []
    counts = np.bincount(transition_code[valid].ravel())
    values = np.flatnonzero(counts)
    for encoded_value in values:
        count = int(counts[encoded_value])
        from_class = int(encoded_value // 1000)
        to_class = int(encoded_value % 1000)
        rows.append(
            {
                "from_snapshot": from_label,
                "to_snapshot": to_label,
                "from_class_id": from_class,
                "from_class_name": DW_CLASSES.get(from_class, f"class_{from_class}"),
                "to_class_id": to_class,
                "to_class_name": DW_CLASSES.get(to_class, f"class_{to_class}"),
                "pixels": count,
                "area_ha": float(count * px_ha),
            }
        )
    return rows


def pair_summary(from_arr: np.ndarray, to_arr: np.ndarray, valid: np.ndarray, from_label: str, to_label: str, px_ha: float) -> dict:
    changed = valid & (from_arr != to_arr)
    from_crop = valid & (from_arr == CROP_CLASS)
    to_crop = valid & (to_arr == CROP_CLASS)
    crop_gain = valid & (~from_crop) & to_crop
    crop_loss = valid & from_crop & (~to_crop)
    crop_stable = valid & from_crop & to_crop
    valid_pixels = int(valid.sum())
    changed_pixels = int(changed.sum())
    return {
        "from_snapshot": from_label,
        "to_snapshot": to_label,
        "valid_pixels": valid_pixels,
        "valid_area_ha": float(valid_pixels * px_ha),
        "changed_pixels": changed_pixels,
        "changed_area_ha": float(changed_pixels * px_ha),
        "changed_pct": float(changed_pixels / valid_pixels) if valid_pixels else 0.0,
        "crop_from_area_ha": float(from_crop.sum() * px_ha),
        "crop_to_area_ha": float(to_crop.sum() * px_ha),
        "crop_gain_area_ha": float(crop_gain.sum() * px_ha),
        "crop_loss_area_ha": float(crop_loss.sum() * px_ha),
        "crop_stable_area_ha": float(crop_stable.sum() * px_ha),
    }


def preview_step(shape: tuple[int, ...], max_size: int) -> int:
    if max_size <= 0:
        return 1
    return max(1, int(np.ceil(max(shape[:2]) / max_size)))


def rgba8(color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    return tuple(int(round(channel * 255)) for channel in mcolors.to_rgba(color, alpha=alpha))


def class_rgba(class_id: int) -> tuple[int, int, int, int]:
    if class_id in DW_COLORS:
        return rgba8(DW_COLORS[class_id])
    hue = (class_id * 0.61803398875) % 1.0
    red, green, blue = mcolors.hsv_to_rgb((hue, 0.65, 0.85))
    return int(red * 255), int(green * 255), int(blue * 255), 255


def downsample_binary_any(mask: np.ndarray, max_size: int) -> tuple[np.ndarray, int]:
    step = preview_step(mask.shape, max_size)
    if step == 1:
        return mask, step
    height = (mask.shape[0] // step) * step
    width = (mask.shape[1] // step) * step
    if height == 0 or width == 0:
        return mask, 1
    blocks = mask[:height, :width].reshape(height // step, step, width // step, step)
    return blocks.any(axis=(1, 3)), step


def save_class_map(path: Path, array: np.ndarray, valid: np.ndarray, title: str, max_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    step = preview_step(array.shape, max_size)
    preview_array = array[::step, ::step]
    preview_valid = valid[::step, ::step]
    rgb = np.full((*preview_array.shape, 4), rgba8("#f5f5ee"), dtype=np.uint8)
    preview_values = np.unique(preview_array[preview_valid])
    for value in preview_values:
        class_id = int(value)
        rgb[preview_valid & (preview_array == value)] = class_rgba(class_id)
    rgb[~preview_valid] = rgba8("#202020")
    unknown_values = [int(value) for value in preview_values if int(value) not in DW_COLORS]
    if unknown_values:
        LOGGER.warning(
            "%s has values outside Dynamic World 0-8 classes in the preview: %s",
            title,
            unknown_values[:20],
        )
    plt.figure(figsize=(9, 9))
    plt.imshow(rgb, interpolation="nearest")
    plt.title(f"{title} (preview 1:{step})" if step > 1 else title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_binary_map(path: Path, mask: np.ndarray, title: str, max_size: int, color: str = "#d62728") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview_mask, step = downsample_binary_any(mask, max_size)
    rgba = np.full((*preview_mask.shape, 4), rgba8("#f5f5ee"), dtype=np.uint8)
    rgba[preview_mask] = rgba8(color)
    plt.figure(figsize=(9, 9))
    plt.imshow(rgba, interpolation="nearest")
    plt.title(f"{title} (preview 1:{step})" if step > 1 else title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_pair_trend(path: Path, summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = summary["from_snapshot"] + " -> " + summary["to_snapshot"]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(labels, summary["changed_pct"] * 100, marker="o", color="#d62728", label="changed %")
    ax1.set_ylabel("Changed area (%)")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(labels, summary["crop_to_area_ha"], marker="s", color="#2ca02c", label="crop area")
    ax2.set_ylabel("Crop area (ha)")
    ax1.set_title("Adjacent Snapshot Change Trend")
    fig.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    setup_logging(args)
    LOGGER.info("Starting Dynamic World LULC change processor")
    LOGGER.info("Input directory: %s", args.input_dir)
    LOGGER.info("Output directory: %s", args.output_dir)
    if args.skip_figures:
        LOGGER.info("PNG figure export: disabled")
    elif args.preview_max_size <= 0:
        LOGGER.info("PNG figure export: full resolution")
    else:
        LOGGER.info("PNG figure export: preview max dimension=%d px", args.preview_max_size)

    discovery_start = time.perf_counter()
    snapshots = discover_snapshots(args)
    LOGGER.info(
        "Discovered %d snapshots in %.1fs",
        len(snapshots),
        time.perf_counter() - discovery_start,
    )

    output_dir = args.output_dir
    rasters_dir = output_dir / "rasters"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    rasters_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading reference grid from %s", snapshots[0].path)
    _, ref_profile, _ = read_reference(snapshots[0])
    nodata_values = set(parse_csv_ints(args.nodata_values))
    px_ha = pixel_area_ha(ref_profile)
    LOGGER.info(
        "Reference grid: %sx%s, CRS=%s, approx pixel area=%.6f ha",
        ref_profile["width"],
        ref_profile["height"],
        ref_profile.get("crs"),
        px_ha,
    )

    arrays: dict[str, np.ndarray] = {}
    valids: dict[str, np.ndarray] = {}
    class_count_rows = []
    for index, snapshot in enumerate(snapshots, start=1):
        snapshot_start = time.perf_counter()
        LOGGER.info("Reading snapshot %d/%d: %s", index, len(snapshots), snapshot.label)
        array, valid = read_aligned(snapshot, ref_profile, no_reproject=args.no_reproject)
        for nodata_value in nodata_values:
            valid &= array != nodata_value
        arrays[snapshot.label] = array
        valids[snapshot.label] = valid
        class_count_rows.extend(class_counts(array, valid, snapshot, px_ha))
        valid_pixels = int(valid.sum())
        LOGGER.info(
            "Loaded %s: shape=%s, dtype=%s, valid_pixels=%d, elapsed=%.1fs",
            snapshot.label,
            array.shape,
            array.dtype,
            valid_pixels,
            time.perf_counter() - snapshot_start,
        )
        if not args.skip_figures:
            fig_start = time.perf_counter()
            save_class_map(
                figures_dir / "snapshots" / f"{snapshot.label}.png",
                array,
                valid,
                snapshot.label,
                max_size=args.preview_max_size,
            )
            LOGGER.info(
                "Saved snapshot figure for %s in %.1fs",
                snapshot.label,
                time.perf_counter() - fig_start,
            )

    pair_summaries = []
    transition_all = []
    pair_total = max(0, len(snapshots) - 1)
    for pair_index, (previous, current) in enumerate(zip(snapshots[:-1], snapshots[1:]), start=1):
        pair_start = time.perf_counter()
        pair_label = f"{previous.label}_to_{current.label}"
        LOGGER.info("Processing adjacent pair %d/%d: %s", pair_index, pair_total, pair_label)
        from_arr = arrays[previous.label]
        to_arr = arrays[current.label]
        valid = valids[previous.label] & valids[current.label]

        changed = valid & (from_arr != to_arr)
        crop_state = np.zeros(from_arr.shape, dtype=np.uint8)
        crop_state[valid & (from_arr == CROP_CLASS) & (to_arr == CROP_CLASS)] = 1
        crop_state[valid & (from_arr == CROP_CLASS) & (to_arr != CROP_CLASS)] = 2
        crop_state[valid & (from_arr != CROP_CLASS) & (to_arr == CROP_CLASS)] = 3
        crop_state[~valid] = 255

        transition_code = np.full(from_arr.shape, 65535, dtype=np.uint16)
        transition_code[valid] = (from_arr[valid].astype(np.uint16) * 1000) + to_arr[valid].astype(np.uint16)
        changed_mask = np.zeros(from_arr.shape, dtype=np.uint8)
        changed_mask[changed] = 1
        changed_mask[~valid] = 255

        if not args.skip_rasters:
            raster_start = time.perf_counter()
            write_raster(
                rasters_dir / f"{pair_label}_changed_mask.tif",
                changed_mask,
                ref_profile,
                nodata=255,
                colormap=CHANGED_MASK_COLORMAP,
            )
            write_raster(
                rasters_dir / f"{pair_label}_crop_change_state.tif",
                crop_state,
                ref_profile,
                nodata=255,
                colormap=CROP_STATE_COLORMAP,
            )
            write_raster(rasters_dir / f"{pair_label}_transition_code.tif", transition_code, ref_profile, nodata=65535)
            LOGGER.info("Saved rasters for %s in %.1fs", pair_label, time.perf_counter() - raster_start)
        if not args.skip_figures:
            fig_start = time.perf_counter()
            save_binary_map(
                figures_dir / "pairs" / f"{pair_label}_changed.png",
                changed,
                f"Changed pixels: {pair_label}",
                max_size=args.preview_max_size,
            )
            save_binary_map(
                figures_dir / "pairs" / f"{pair_label}_crop_gain.png",
                valid & (from_arr != CROP_CLASS) & (to_arr == CROP_CLASS),
                f"Crop gain: {pair_label}",
                max_size=args.preview_max_size,
                color="#2ca02c",
            )
            save_binary_map(
                figures_dir / "pairs" / f"{pair_label}_crop_loss.png",
                valid & (from_arr == CROP_CLASS) & (to_arr != CROP_CLASS),
                f"Crop loss: {pair_label}",
                max_size=args.preview_max_size,
                color="#d62728",
            )
            LOGGER.info("Saved pair figures for %s in %.1fs", pair_label, time.perf_counter() - fig_start)
        summary = pair_summary(from_arr, to_arr, valid, previous.label, current.label, px_ha)
        pair_summaries.append(summary)
        LOGGER.info(
            "Pair stats for %s: changed=%.4f%%, crop_gain_ha=%.2f, crop_loss_ha=%.2f",
            pair_label,
            summary["changed_pct"] * 100,
            summary["crop_gain_area_ha"],
            summary["crop_loss_area_ha"],
        )
        transition_start = time.perf_counter()
        transition_all.extend(transition_rows(transition_code, valid, previous.label, current.label, px_ha))
        LOGGER.info(
            "Computed transition table for %s in %.1fs",
            pair_label,
            time.perf_counter() - transition_start,
        )
        LOGGER.info("Finished %s in %.1fs", pair_label, time.perf_counter() - pair_start)

    LOGGER.info("Writing CSV summary tables")
    class_counts_df = pd.DataFrame(class_count_rows)
    pair_summary_df = pd.DataFrame(pair_summaries)
    transitions_df = pd.DataFrame(transition_all)
    class_counts_df.to_csv(tables_dir / "snapshot_class_counts.csv", index=False)
    pair_summary_df.to_csv(tables_dir / "adjacent_pair_summary.csv", index=False)
    transitions_df.to_csv(tables_dir / "adjacent_transition_matrix_long.csv", index=False)
    if not args.skip_figures:
        save_pair_trend(figures_dir / "adjacent_change_trend.png", pair_summary_df)

    manifest = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "snapshots": [
            {"label": s.label, "year": s.year, "season": s.season, "path": str(s.path.resolve())}
            for s in snapshots
        ],
        "outputs": {
            "snapshot_class_counts": str((tables_dir / "snapshot_class_counts.csv").resolve()),
            "adjacent_pair_summary": str((tables_dir / "adjacent_pair_summary.csv").resolve()),
            "adjacent_transition_matrix_long": str((tables_dir / "adjacent_transition_matrix_long.csv").resolve()),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    LOGGER.info("Processed %d snapshots", len(snapshots))
    LOGGER.info("Adjacent comparisons: %d", len(pair_summaries))
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Total elapsed time: %.1fs", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
