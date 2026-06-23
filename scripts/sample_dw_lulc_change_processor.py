#!/usr/bin/env python3
"""Sample agricultural field raster change processor.

This compares agricultural field detections through time. A pixel is treated as
field when it is a non-zero valid value by default, which supports binary masks
and rasters where each delineated field has a separate positive object ID.

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
from PIL import Image, ImageDraw

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires rasterio. Install with: pip install -e ."
    ) from exc


LOGGER = logging.getLogger("field_change")

CHANGED_MASK_COLORMAP = {
    0: (245, 245, 238, 255),
    1: (214, 39, 40, 255),
    255: (0, 0, 0, 0),
}

FIELD_STATE_COLORMAP = {
    0: (245, 245, 238, 255),
    1: (44, 160, 44, 255),
    2: (214, 39, 40, 255),
    3: (31, 119, 180, 255),
    255: (0, 0, 0, 0),
}

FIELD_TRANSITION_COLORMAP = {
    0: (245, 245, 238, 255),
    1: (31, 119, 180, 255),
    10: (214, 39, 40, 255),
    11: (44, 160, 44, 255),
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
        description="Process sample 2021-2023 agricultural field raster change."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing agricultural field raster snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/change_detection/kursh_2021_2023_field_sample"),
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
        help="Filename stem template. Extension is discovered automatically. The default keeps the existing folder naming.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=("same-season-yearly", "same-season-all-years", "adjacent", "all"),
        default="adjacent",
        help=(
            "Comparison pairs to run. The default follows the chronological "
            "snapshot order, including season-to-season field comparisons."
        ),
    )
    parser.add_argument(
        "--field-values",
        default="",
        help=(
            "Optional comma-separated raster values to treat as field. "
            "Default: every non-zero valid value is field."
        ),
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
        "--skip-gifs",
        action="store_true",
        help="Skip animated GIF timeline exports.",
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
        "--gif-duration-ms",
        type=int,
        default=900,
        help="Duration of each GIF frame in milliseconds.",
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
        if any(token in p.name.lower() for token in ("field", "mask", "delineation", "dw", "lulc"))
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


def build_pairs(snapshots: list[Snapshot], pair_mode: str) -> list[tuple[Snapshot, Snapshot]]:
    if pair_mode == "adjacent":
        return list(zip(snapshots[:-1], snapshots[1:]))
    if pair_mode == "all":
        return [
            (snapshots[i], snapshots[j])
            for i in range(len(snapshots))
            for j in range(i + 1, len(snapshots))
        ]

    by_season: dict[str, list[Snapshot]] = {}
    for snapshot in snapshots:
        by_season.setdefault(snapshot.season, []).append(snapshot)

    pairs: list[tuple[Snapshot, Snapshot]] = []
    for season_snapshots in by_season.values():
        ordered = sorted(season_snapshots, key=lambda snapshot: snapshot.year)
        if pair_mode == "same-season-yearly":
            pairs.extend(zip(ordered[:-1], ordered[1:]))
        elif pair_mode == "same-season-all-years":
            pairs.extend(
                (ordered[i], ordered[j])
                for i in range(len(ordered))
                for j in range(i + 1, len(ordered))
            )
    return pairs


def read_reference_profile(snapshot: Snapshot) -> dict:
    with rasterio.open(snapshot.path) as src:
        return src.profile.copy()


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


def make_field_mask(array: np.ndarray, valid: np.ndarray, field_values: set[int]) -> np.ndarray:
    if field_values:
        return valid & np.isin(array, list(field_values))
    return valid & (array != 0)


def snapshot_field_summary(field: np.ndarray, valid: np.ndarray, snapshot: Snapshot, px_ha: float) -> dict:
    valid_pixels = int(valid.sum())
    field_pixels = int((valid & field).sum())
    non_field_pixels = valid_pixels - field_pixels
    return {
        "snapshot": snapshot.label,
        "year": snapshot.year,
        "season": snapshot.season,
        "valid_pixels": valid_pixels,
        "valid_area_ha": float(valid_pixels * px_ha),
        "field_pixels": field_pixels,
        "field_area_ha": float(field_pixels * px_ha),
        "field_pct": float(field_pixels / valid_pixels) if valid_pixels else 0.0,
        "non_field_pixels": non_field_pixels,
        "non_field_area_ha": float(non_field_pixels * px_ha),
    }


def field_transition_rows(from_field: np.ndarray, to_field: np.ndarray, valid: np.ndarray, from_label: str, to_label: str, px_ha: float) -> list[dict]:
    transitions = [
        (0, "non_field", 0, "non_field", valid & (~from_field) & (~to_field)),
        (0, "non_field", 1, "field", valid & (~from_field) & to_field),
        (1, "field", 0, "non_field", valid & from_field & (~to_field)),
        (1, "field", 1, "field", valid & from_field & to_field),
    ]
    rows = []
    for from_state, from_name, to_state, to_name, mask in transitions:
        pixels = int(mask.sum())
        rows.append(
            {
                "from_snapshot": from_label,
                "to_snapshot": to_label,
                "from_state": from_state,
                "from_state_name": from_name,
                "to_state": to_state,
                "to_state_name": to_name,
                "change_type": f"{from_name}_to_{to_name}",
                "pixels": pixels,
                "area_ha": float(pixels * px_ha),
            }
        )
    return rows


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def pair_summary(from_field: np.ndarray, to_field: np.ndarray, valid: np.ndarray, from_label: str, to_label: str, px_ha: float) -> dict:
    changed = valid & (from_field != to_field)
    field_gain = valid & (~from_field) & to_field
    field_loss = valid & from_field & (~to_field)
    stable_field = valid & from_field & to_field
    stable_non_field = valid & (~from_field) & (~to_field)
    valid_pixels = int(valid.sum())
    changed_pixels = int(changed.sum())
    from_field_pixels = int((valid & from_field).sum())
    to_field_pixels = int((valid & to_field).sum())
    stable_field_pixels = int(stable_field.sum())
    stable_non_field_pixels = int(stable_non_field.sum())
    field_gain_pixels = int(field_gain.sum())
    field_loss_pixels = int(field_loss.sum())
    union_pixels = int((valid & (from_field | to_field)).sum())
    gross_change_pixels = field_gain_pixels + field_loss_pixels
    net_change_pixels = to_field_pixels - from_field_pixels
    true_positive = stable_field_pixels
    false_positive = field_gain_pixels
    false_negative = field_loss_pixels
    true_negative = stable_non_field_pixels
    precision = safe_div(true_positive, true_positive + false_positive)
    recall = safe_div(true_positive, true_positive + false_negative)
    specificity = safe_div(true_negative, true_negative + false_positive)
    negative_predictive_value = safe_div(true_negative, true_negative + false_negative)
    accuracy = safe_div(true_positive + true_negative, valid_pixels)
    balanced_accuracy = (recall + specificity) / 2.0
    denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = safe_div(
        (true_positive * true_negative) - (false_positive * false_negative),
        denominator,
    )
    expected_accuracy = safe_div(
        ((true_positive + false_positive) * (true_positive + false_negative))
        + ((false_negative + true_negative) * (false_positive + true_negative)),
        valid_pixels * valid_pixels,
    )
    kappa = safe_div(accuracy - expected_accuracy, 1.0 - expected_accuracy)
    return {
        "from_snapshot": from_label,
        "to_snapshot": to_label,
        "valid_pixels": valid_pixels,
        "valid_area_ha": float(valid_pixels * px_ha),
        "changed_pixels": changed_pixels,
        "changed_area_ha": float(changed_pixels * px_ha),
        "changed_pct": safe_div(changed_pixels, valid_pixels),
        "gross_change_pixels": gross_change_pixels,
        "gross_change_area_ha": float(gross_change_pixels * px_ha),
        "gross_change_pct": safe_div(gross_change_pixels, valid_pixels),
        "from_field_pixels": from_field_pixels,
        "from_field_area_ha": float(from_field_pixels * px_ha),
        "to_field_pixels": to_field_pixels,
        "to_field_area_ha": float(to_field_pixels * px_ha),
        "field_gain_pixels": field_gain_pixels,
        "field_gain_area_ha": float(field_gain_pixels * px_ha),
        "field_gain_rate_vs_from": safe_div(field_gain_pixels, from_field_pixels),
        "field_gain_rate_vs_to": safe_div(field_gain_pixels, to_field_pixels),
        "field_loss_pixels": field_loss_pixels,
        "field_loss_area_ha": float(field_loss_pixels * px_ha),
        "field_loss_rate_vs_from": safe_div(field_loss_pixels, from_field_pixels),
        "stable_field_pixels": stable_field_pixels,
        "stable_field_area_ha": float(stable_field_pixels * px_ha),
        "stable_non_field_pixels": stable_non_field_pixels,
        "stable_non_field_area_ha": float(stable_non_field_pixels * px_ha),
        "net_field_pixels": net_change_pixels,
        "net_field_area_change_ha": float(net_change_pixels * px_ha),
        "field_area_ratio_to_from": safe_div(to_field_pixels, from_field_pixels),
        "field_iou": safe_div(stable_field_pixels, union_pixels),
        "field_dice_f1": safe_div(2 * stable_field_pixels, from_field_pixels + to_field_pixels),
        "field_precision_current_vs_previous": precision,
        "field_recall_persistence": recall,
        "field_specificity": specificity,
        "field_negative_predictive_value": negative_predictive_value,
        "field_accuracy": accuracy,
        "field_balanced_accuracy": balanced_accuracy,
        "field_mcc": mcc,
        "field_cohen_kappa": kappa,
        "field_jaccard_distance": 1.0 - safe_div(stable_field_pixels, union_pixels),
        "field_churn_rate": safe_div(gross_change_pixels, union_pixels),
        "field_retention_rate": safe_div(stable_field_pixels, from_field_pixels),
        "field_expansion_rate": safe_div(field_gain_pixels, from_field_pixels),
    }


def preview_step(shape: tuple[int, ...], max_size: int) -> int:
    if max_size <= 0:
        return 1
    return max(1, int(np.ceil(max(shape[:2]) / max_size)))


def rgba8(color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    return tuple(int(round(channel * 255)) for channel in mcolors.to_rgba(color, alpha=alpha))


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


def field_preview_rgba(field: np.ndarray, valid: np.ndarray, max_size: int) -> tuple[np.ndarray, int]:
    preview_field, step = downsample_binary_any(field & valid, max_size)
    preview_valid, _ = downsample_binary_any(valid, max_size)
    rgb = np.full((*preview_field.shape, 4), rgba8("#f5f5ee"), dtype=np.uint8)
    rgb[preview_field] = rgba8("#2ca02c")
    rgb[~preview_valid] = rgba8("#202020")
    return rgb, step


def field_state_preview_rgba(from_field: np.ndarray, to_field: np.ndarray, valid: np.ndarray, max_size: int) -> tuple[np.ndarray, int]:
    preview_valid, step = downsample_binary_any(valid, max_size)
    stable_preview, _ = downsample_binary_any(valid & from_field & to_field, max_size)
    gain_preview, _ = downsample_binary_any(valid & (~from_field) & to_field, max_size)
    loss_preview, _ = downsample_binary_any(valid & from_field & (~to_field), max_size)
    rgb = np.full((*preview_valid.shape, 4), rgba8("#f5f5ee"), dtype=np.uint8)
    rgb[stable_preview] = rgba8("#2ca02c")
    rgb[gain_preview] = rgba8("#1f77b4")
    rgb[loss_preview] = rgba8("#d62728")
    rgb[gain_preview & loss_preview] = rgba8("#7f3c8d")
    rgb[~preview_valid] = rgba8("#202020")
    return rgb, step


def save_field_map(path: Path, field: np.ndarray, valid: np.ndarray, title: str, max_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb, step = field_preview_rgba(field, valid, max_size)
    plt.figure(figsize=(9, 9))
    plt.imshow(rgb, interpolation="nearest")
    plt.title(f"{title} (preview 1:{step})" if step > 1 else title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_field_state_map(path: Path, from_field: np.ndarray, to_field: np.ndarray, valid: np.ndarray, title: str, max_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb, step = field_state_preview_rgba(from_field, to_field, valid, max_size)
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


def compose_frame(
    rgba: np.ndarray,
    title: str,
    legend: list[tuple[str, tuple[int, int, int, int]]],
) -> Image.Image:
    image = Image.fromarray(rgba, mode="RGBA")
    width, height = image.size
    title_height = 38
    legend_height = 28 if legend else 0
    canvas = Image.new("RGBA", (width, height + title_height + legend_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(20, 20, 20, 255))
    canvas.paste(image, (0, title_height))
    x = 10
    y = title_height + height + 7
    for label, color in legend:
        draw.rectangle((x, y, x + 14, y + 14), fill=color)
        draw.text((x + 20, y - 1), label, fill=(20, 20, 20, 255))
        x += 150
    return canvas.convert("P", palette=Image.Palette.ADAPTIVE)


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_snapshot_timeline_gif(
    path: Path,
    frame_dir: Path,
    snapshots: list[Snapshot],
    fields: dict[str, np.ndarray],
    valids: dict[str, np.ndarray],
    max_size: int,
    duration_ms: int,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    legend = [
        ("field", rgba8("#2ca02c")),
        ("non-field", rgba8("#f5f5ee")),
        ("nodata", rgba8("#202020")),
    ]
    frames = []
    for index, snapshot in enumerate(snapshots, start=1):
        rgba, step = field_preview_rgba(fields[snapshot.label], valids[snapshot.label], max_size)
        frame = compose_frame(rgba, f"{snapshot.label} field extent (preview 1:{step})", legend)
        frame.save(frame_dir / f"{index:03d}_{snapshot.label}.png")
        frames.append(frame)
    save_gif(frames, path, duration_ms)


def save_change_timeline_gif(
    path: Path,
    frame_dir: Path,
    pairs: list[tuple[Snapshot, Snapshot]],
    fields: dict[str, np.ndarray],
    valids: dict[str, np.ndarray],
    max_size: int,
    duration_ms: int,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    legend = [
        ("stable", rgba8("#2ca02c")),
        ("gain", rgba8("#1f77b4")),
        ("loss", rgba8("#d62728")),
        ("mixed", rgba8("#7f3c8d")),
        ("non-field", rgba8("#f5f5ee")),
    ]
    frames = []
    for index, (previous, current) in enumerate(pairs, start=1):
        valid = valids[previous.label] & valids[current.label]
        rgba, step = field_state_preview_rgba(
            fields[previous.label],
            fields[current.label],
            valid,
            max_size,
        )
        pair_label = f"{previous.label}_to_{current.label}"
        frame = compose_frame(rgba, f"{previous.label} -> {current.label} change state (preview 1:{step})", legend)
        frame.save(frame_dir / f"{index:03d}_{pair_label}.png")
        frames.append(frame)
    save_gif(frames, path, duration_ms)


def save_pair_trend(path: Path, summary: pd.DataFrame, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = summary["from_snapshot"] + " -> " + summary["to_snapshot"]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(labels, summary["changed_pct"] * 100, marker="o", color="#d62728", label="changed %")
    ax1.set_ylabel("Changed area (%)")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(labels, summary["to_field_area_ha"], marker="s", color="#2ca02c", label="field area")
    ax2.set_ylabel("Field area (ha)")
    ax1.set_title(title)
    fig.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_metrics_dashboard(path: Path, snapshot_summary: pd.DataFrame, pair_summary_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pair_labels = pair_summary_df["from_snapshot"] + " -> " + pair_summary_df["to_snapshot"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(snapshot_summary["snapshot"], snapshot_summary["field_area_ha"], marker="o", color="#2ca02c")
    axes[0, 0].set_title("Field Area Timeline")
    axes[0, 0].set_ylabel("Area (ha)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].plot(pair_labels, pair_summary_df["field_iou"], marker="o", label="IoU", color="#1f77b4")
    axes[0, 1].plot(pair_labels, pair_summary_df["field_dice_f1"], marker="s", label="Dice/F1", color="#ff7f0e")
    axes[0, 1].plot(pair_labels, pair_summary_df["field_mcc"], marker="^", label="MCC", color="#9467bd")
    axes[0, 1].set_title("Agreement Metrics")
    axes[0, 1].set_ylim(-1.05, 1.05)
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis="x", rotation=45)

    x_positions = np.arange(len(pair_labels))
    width = 0.42
    axes[1, 0].bar(
        x_positions - width / 2,
        pair_summary_df["field_gain_area_ha"],
        width,
        label="gain",
        color="#1f77b4",
    )
    axes[1, 0].bar(
        x_positions + width / 2,
        pair_summary_df["field_loss_area_ha"],
        width,
        label="loss",
        color="#d62728",
    )
    axes[1, 0].set_title("Field Gain / Loss")
    axes[1, 0].set_ylabel("Area (ha)")
    axes[1, 0].set_xticks(x_positions, pair_labels, rotation=45, ha="right")
    axes[1, 0].legend()

    axes[1, 1].bar(pair_labels, pair_summary_df["net_field_area_change_ha"], color="#2ca02c")
    axes[1, 1].axhline(0, color="#202020", linewidth=0.8)
    axes[1, 1].set_title("Net Field Area Change")
    axes[1, 1].set_ylabel("Area (ha)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    setup_logging(args)
    LOGGER.info("Starting agricultural field raster change processor")
    LOGGER.info("Input directory: %s", args.input_dir)
    LOGGER.info("Output directory: %s", args.output_dir)
    LOGGER.info("Pair mode: %s", args.pair_mode)
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
    ref_profile = read_reference_profile(snapshots[0])
    nodata_values = set(parse_csv_ints(args.nodata_values))
    field_values = set(parse_csv_ints(args.field_values))
    px_ha = pixel_area_ha(ref_profile)
    if field_values:
        LOGGER.info("Field definition: values in %s", sorted(field_values))
    else:
        LOGGER.info("Field definition: every non-zero valid pixel")
    LOGGER.info(
        "Reference grid: %sx%s, CRS=%s, approx pixel area=%.6f ha",
        ref_profile["width"],
        ref_profile["height"],
        ref_profile.get("crs"),
        px_ha,
    )

    fields: dict[str, np.ndarray] = {}
    valids: dict[str, np.ndarray] = {}
    snapshot_rows = []
    for index, snapshot in enumerate(snapshots, start=1):
        snapshot_start = time.perf_counter()
        LOGGER.info("Reading snapshot %d/%d: %s", index, len(snapshots), snapshot.label)
        array, valid = read_aligned(snapshot, ref_profile, no_reproject=args.no_reproject)
        for nodata_value in nodata_values:
            valid &= array != nodata_value
        field = make_field_mask(array, valid, field_values)
        fields[snapshot.label] = field
        valids[snapshot.label] = valid
        snapshot_summary = snapshot_field_summary(field, valid, snapshot, px_ha)
        snapshot_rows.append(snapshot_summary)
        valid_pixels = int(valid.sum())
        LOGGER.info(
            "Loaded %s: shape=%s, dtype=%s, valid_pixels=%d, field_area_ha=%.2f, elapsed=%.1fs",
            snapshot.label,
            array.shape,
            array.dtype,
            valid_pixels,
            snapshot_summary["field_area_ha"],
            time.perf_counter() - snapshot_start,
        )
        if not args.skip_figures:
            fig_start = time.perf_counter()
            save_field_map(
                figures_dir / "snapshots" / f"{snapshot.label}.png",
                field,
                valid,
                snapshot.label,
                max_size=args.preview_max_size,
            )
            LOGGER.info(
                "Saved snapshot figure for %s in %.1fs",
                snapshot.label,
                time.perf_counter() - fig_start,
            )

    pairs = build_pairs(snapshots, args.pair_mode)
    if not pairs:
        raise ValueError(f"No comparison pairs were built for pair mode '{args.pair_mode}'")

    pair_summaries = []
    transition_all = []
    pair_total = len(pairs)
    for pair_index, (previous, current) in enumerate(pairs, start=1):
        pair_start = time.perf_counter()
        pair_label = f"{previous.label}_to_{current.label}"
        LOGGER.info("Processing pair %d/%d: %s", pair_index, pair_total, pair_label)
        from_field = fields[previous.label]
        to_field = fields[current.label]
        valid = valids[previous.label] & valids[current.label]

        changed = valid & (from_field != to_field)
        field_state = np.zeros(from_field.shape, dtype=np.uint8)
        field_state[valid & from_field & to_field] = 1
        field_state[valid & from_field & (~to_field)] = 2
        field_state[valid & (~from_field) & to_field] = 3
        field_state[~valid] = 255

        transition_code = np.full(from_field.shape, 255, dtype=np.uint8)
        transition_code[valid] = (from_field[valid].astype(np.uint8) * 10) + to_field[valid].astype(np.uint8)
        changed_mask = np.zeros(from_field.shape, dtype=np.uint8)
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
                rasters_dir / f"{pair_label}_field_change_state.tif",
                field_state,
                ref_profile,
                nodata=255,
                colormap=FIELD_STATE_COLORMAP,
            )
            write_raster(
                rasters_dir / f"{pair_label}_field_transition_code.tif",
                transition_code,
                ref_profile,
                nodata=255,
                colormap=FIELD_TRANSITION_COLORMAP,
            )
            LOGGER.info("Saved rasters for %s in %.1fs", pair_label, time.perf_counter() - raster_start)
        if not args.skip_figures:
            fig_start = time.perf_counter()
            save_field_state_map(
                figures_dir / "pairs" / f"{pair_label}_field_change_state.png",
                from_field,
                to_field,
                valid,
                f"Field change state: {pair_label}",
                max_size=args.preview_max_size,
            )
            save_binary_map(
                figures_dir / "pairs" / f"{pair_label}_changed.png",
                changed,
                f"Changed pixels: {pair_label}",
                max_size=args.preview_max_size,
            )
            save_binary_map(
                figures_dir / "pairs" / f"{pair_label}_field_gain.png",
                valid & (~from_field) & to_field,
                f"Field gain: {pair_label}",
                max_size=args.preview_max_size,
                color="#2ca02c",
            )
            save_binary_map(
                figures_dir / "pairs" / f"{pair_label}_field_loss.png",
                valid & from_field & (~to_field),
                f"Field loss: {pair_label}",
                max_size=args.preview_max_size,
                color="#d62728",
            )
            LOGGER.info("Saved pair figures for %s in %.1fs", pair_label, time.perf_counter() - fig_start)
        summary = pair_summary(from_field, to_field, valid, previous.label, current.label, px_ha)
        pair_summaries.append(summary)
        LOGGER.info(
            "Pair stats for %s: changed=%.4f%%, field_gain_ha=%.2f, field_loss_ha=%.2f",
            pair_label,
            summary["changed_pct"] * 100,
            summary["field_gain_area_ha"],
            summary["field_loss_area_ha"],
        )
        transition_start = time.perf_counter()
        transition_all.extend(field_transition_rows(from_field, to_field, valid, previous.label, current.label, px_ha))
        LOGGER.info(
            "Computed field transition table for %s in %.1fs",
            pair_label,
            time.perf_counter() - transition_start,
        )
        LOGGER.info("Finished %s in %.1fs", pair_label, time.perf_counter() - pair_start)

    LOGGER.info("Writing CSV summary tables")
    snapshot_df = pd.DataFrame(snapshot_rows)
    pair_summary_df = pd.DataFrame(pair_summaries)
    transitions_df = pd.DataFrame(transition_all)
    snapshot_df.to_csv(tables_dir / "snapshot_field_summary.csv", index=False)
    pair_summary_df.to_csv(tables_dir / "pair_summary.csv", index=False)
    transitions_df.to_csv(tables_dir / "field_transition_matrix_long.csv", index=False)
    transitions_df.to_csv(tables_dir / "transition_matrix_long.csv", index=False)
    pair_summary_df.to_csv(tables_dir / "adjacent_pair_summary.csv", index=False)
    transitions_df.to_csv(tables_dir / "adjacent_transition_matrix_long.csv", index=False)
    if not args.skip_figures:
        save_pair_trend(
            figures_dir / "change_trend.png",
            pair_summary_df,
            title=f"Field Change Trend ({args.pair_mode})",
        )
        save_pair_trend(
            figures_dir / "adjacent_change_trend.png",
            pair_summary_df,
            title=f"Field Change Trend ({args.pair_mode})",
        )
        save_metrics_dashboard(figures_dir / "field_change_metrics_dashboard.png", snapshot_df, pair_summary_df)
        if not args.skip_gifs:
            timeline_dir = figures_dir / "timelines"
            save_snapshot_timeline_gif(
                timeline_dir / "field_extent_timeline.gif",
                timeline_dir / "frames" / "field_extent",
                snapshots,
                fields,
                valids,
                max_size=args.preview_max_size,
                duration_ms=args.gif_duration_ms,
            )
            save_change_timeline_gif(
                timeline_dir / "field_change_timeline.gif",
                timeline_dir / "frames" / "field_change",
                pairs,
                fields,
                valids,
                max_size=args.preview_max_size,
                duration_ms=args.gif_duration_ms,
            )

    manifest = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "pair_mode": args.pair_mode,
        "snapshots": [
            {"label": s.label, "year": s.year, "season": s.season, "path": str(s.path.resolve())}
            for s in snapshots
        ],
        "pairs": [
            {"from_snapshot": previous.label, "to_snapshot": current.label}
            for previous, current in pairs
        ],
        "outputs": {
            "snapshot_field_summary": str((tables_dir / "snapshot_field_summary.csv").resolve()),
            "pair_summary": str((tables_dir / "pair_summary.csv").resolve()),
            "field_transition_matrix_long": str((tables_dir / "field_transition_matrix_long.csv").resolve()),
            "metrics_dashboard": str((figures_dir / "field_change_metrics_dashboard.png").resolve()),
            "field_extent_timeline_gif": str((figures_dir / "timelines" / "field_extent_timeline.gif").resolve()),
            "field_change_timeline_gif": str((figures_dir / "timelines" / "field_change_timeline.gif").resolve()),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    LOGGER.info("Processed %d snapshots", len(snapshots))
    LOGGER.info("Comparisons: %d", len(pair_summaries))
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Total elapsed time: %.1fs", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
