#!/usr/bin/env python3
"""Track agricultural field polygon changes across snapshot GeoJSON/GPKG files."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


LOGGER = logging.getLogger("vector_field_change")


@dataclass(frozen=True)
class VectorSnapshot:
    year: int
    season: str
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track field polygon changes using overlap and IoU metrics.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing vector field snapshots.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/vector_field_change"))
    parser.add_argument("--years", default="2021,2022,2023")
    parser.add_argument("--seasons", default="february_april,june_august")
    parser.add_argument("--filename-template", default="kursh_{year}_{season}__fields")
    parser.add_argument("--extensions", default=".geojson,.gpkg,.shp,.json")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--pair-mode",
        choices=("adjacent", "same-season-yearly", "same-season-all-years", "all"),
        default="adjacent",
    )
    parser.add_argument("--input-crs", default="", help="CRS to assign when a file has no CRS, for example EPSG:4326.")
    parser.add_argument("--metric-crs", default="EPSG:32636", help="Projected CRS used for area and overlap metrics.")
    parser.add_argument("--min-iou", type=float, default=0.30, help="Minimum IoU for a stable one-to-one match.")
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.10,
        help="Minimum source/target overlap ratio for split, merge, new, and disappeared diagnostics.",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-gifs", action="store_true")
    parser.add_argument("--gif-duration-ms", type=int, default=900)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
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
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def find_snapshot_file(input_dir: Path, stem: str, extensions: list[str], recursive: bool) -> Path:
    for ext in extensions:
        direct = input_dir / f"{stem}{ext}"
        if direct.exists():
            return direct
    directory = input_dir / stem
    if directory.is_dir():
        for ext in extensions:
            matches = sorted(directory.rglob(f"*{ext}"))
            if matches:
                return matches[0]
    if recursive:
        for ext in extensions:
            matches = sorted(input_dir.rglob(f"{stem}{ext}"))
            if matches:
                return matches[0]
        directories = sorted(path for path in input_dir.rglob(stem) if path.is_dir())
        for directory in directories:
            for ext in extensions:
                matches = sorted(directory.rglob(f"*{ext}"))
                if matches:
                    return matches[0]
    raise FileNotFoundError(f"No vector snapshot found for stem '{stem}' in {input_dir}")


def discover_snapshots(args: argparse.Namespace) -> list[VectorSnapshot]:
    years = parse_csv_ints(args.years)
    seasons = parse_csv_strings(args.seasons)
    extensions = parse_csv_strings(args.extensions)
    snapshots = []
    for year in years:
        for season in seasons:
            stem = args.filename_template.format(year=year, season=season)
            path = find_snapshot_file(args.input_dir, stem, extensions, args.recursive)
            LOGGER.info("Discovered %s -> %s", f"{year}_{season}", path)
            snapshots.append(VectorSnapshot(year=year, season=season, label=f"{year}_{season}", path=path))
    return snapshots


def build_pairs(snapshots: list[VectorSnapshot], pair_mode: str) -> list[tuple[VectorSnapshot, VectorSnapshot]]:
    if pair_mode == "adjacent":
        return list(zip(snapshots[:-1], snapshots[1:]))
    if pair_mode == "all":
        return [
            (snapshots[i], snapshots[j])
            for i in range(len(snapshots))
            for j in range(i + 1, len(snapshots))
        ]
    by_season: dict[str, list[VectorSnapshot]] = {}
    for snapshot in snapshots:
        by_season.setdefault(snapshot.season, []).append(snapshot)
    pairs = []
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


def load_snapshot(snapshot: VectorSnapshot, input_crs: str, metric_crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(snapshot.path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        if not input_crs:
            raise ValueError(f"{snapshot.path} has no CRS. Pass --input-crs.")
        gdf = gdf.set_crs(input_crs)
    if metric_crs:
        gdf = gdf.to_crs(metric_crs)
    gdf["field_uid"] = [f"{snapshot.label}_{index}" for index in range(len(gdf))]
    gdf["snapshot"] = snapshot.label
    gdf["area_ha"] = gdf.geometry.area / 10_000.0
    return gdf[["field_uid", "snapshot", "area_ha", "geometry"]]


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compare_pair(
    previous: VectorSnapshot,
    current: VectorSnapshot,
    from_gdf: gpd.GeoDataFrame,
    to_gdf: gpd.GeoDataFrame,
    min_iou: float,
    min_overlap_ratio: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_label = f"{previous.label}_to_{current.label}"
    from_fields = from_gdf.rename(columns={"field_uid": "from_field_id", "area_ha": "from_area_ha"})
    to_fields = to_gdf.rename(columns={"field_uid": "to_field_id", "area_ha": "to_area_ha"})
    from_fields = from_fields[["from_field_id", "from_area_ha", "geometry"]]
    to_fields = to_fields[["to_field_id", "to_area_ha", "geometry"]]

    if from_fields.empty or to_fields.empty:
        overlaps = pd.DataFrame()
    else:
        overlaps_gdf = gpd.overlay(from_fields, to_fields, how="intersection", keep_geom_type=False)
        overlaps_gdf = overlaps_gdf[overlaps_gdf.geometry.notna() & ~overlaps_gdf.geometry.is_empty].copy()
        overlaps_gdf["intersection_area_ha"] = overlaps_gdf.geometry.area / 10_000.0
        overlaps_gdf["union_area_ha"] = (
            overlaps_gdf["from_area_ha"] + overlaps_gdf["to_area_ha"] - overlaps_gdf["intersection_area_ha"]
        )
        overlaps_gdf["iou"] = overlaps_gdf["intersection_area_ha"] / overlaps_gdf["union_area_ha"]
        overlaps_gdf["from_overlap_ratio"] = overlaps_gdf["intersection_area_ha"] / overlaps_gdf["from_area_ha"]
        overlaps_gdf["to_overlap_ratio"] = overlaps_gdf["intersection_area_ha"] / overlaps_gdf["to_area_ha"]
        overlaps = pd.DataFrame(overlaps_gdf.drop(columns="geometry"))

    if overlaps.empty:
        matched = pd.DataFrame()
        split_from_ids: set[str] = set()
        merge_to_ids: set[str] = set()
        covered_from_ids: set[str] = set()
        covered_to_ids: set[str] = set()
    else:
        best_to_by_from = overlaps.loc[overlaps.groupby("from_field_id")["iou"].idxmax()]
        best_from_by_to = overlaps.loc[overlaps.groupby("to_field_id")["iou"].idxmax()]
        mutual_pairs = set(zip(best_to_by_from["from_field_id"], best_to_by_from["to_field_id"])) & set(
            zip(best_from_by_to["from_field_id"], best_from_by_to["to_field_id"])
        )
        mutual_mask = overlaps.apply(lambda row: (row["from_field_id"], row["to_field_id"]) in mutual_pairs, axis=1)
        matched = overlaps[mutual_mask & (overlaps["iou"] >= min_iou)].copy()

        split_counts = overlaps[overlaps["from_overlap_ratio"] >= min_overlap_ratio].groupby("from_field_id")[
            "to_field_id"
        ].nunique()
        merge_counts = overlaps[overlaps["to_overlap_ratio"] >= min_overlap_ratio].groupby("to_field_id")[
            "from_field_id"
        ].nunique()
        split_from_ids = set(split_counts[split_counts >= 2].index)
        merge_to_ids = set(merge_counts[merge_counts >= 2].index)
        covered_from_ids = set(overlaps[overlaps["from_overlap_ratio"] >= min_overlap_ratio]["from_field_id"])
        covered_to_ids = set(overlaps[overlaps["to_overlap_ratio"] >= min_overlap_ratio]["to_field_id"])

    from_ids = set(from_gdf["field_uid"])
    to_ids = set(to_gdf["field_uid"])
    disappeared_ids = from_ids - covered_from_ids
    new_ids = to_ids - covered_to_ids

    events = []
    events.extend({"pair": pair_label, "event_type": "split", "field_id": field_id} for field_id in split_from_ids)
    events.extend({"pair": pair_label, "event_type": "merge", "field_id": field_id} for field_id in merge_to_ids)
    events.extend({"pair": pair_label, "event_type": "disappeared", "field_id": field_id} for field_id in disappeared_ids)
    events.extend({"pair": pair_label, "event_type": "new", "field_id": field_id} for field_id in new_ids)
    events_df = pd.DataFrame(events)

    matched_area = float(matched["intersection_area_ha"].sum()) if not matched.empty else 0.0
    matched_iou_mean = float(matched["iou"].mean()) if not matched.empty else 0.0
    matched_iou_median = float(matched["iou"].median()) if not matched.empty else 0.0
    matched_iou_area_weighted = safe_div(
        float((matched["iou"] * matched["intersection_area_ha"]).sum()) if not matched.empty else 0.0,
        matched_area,
    )
    from_area = float(from_gdf["area_ha"].sum())
    to_area = float(to_gdf["area_ha"].sum())
    summary = {
        "pair": pair_label,
        "from_snapshot": previous.label,
        "to_snapshot": current.label,
        "from_count": int(len(from_gdf)),
        "to_count": int(len(to_gdf)),
        "matched_count": int(len(matched)),
        "new_count": int(len(new_ids)),
        "disappeared_count": int(len(disappeared_ids)),
        "split_candidate_count": int(len(split_from_ids)),
        "merge_candidate_count": int(len(merge_to_ids)),
        "from_area_ha": from_area,
        "to_area_ha": to_area,
        "net_area_change_ha": to_area - from_area,
        "matched_intersection_area_ha": matched_area,
        "matched_iou_mean": matched_iou_mean,
        "matched_iou_median": matched_iou_median,
        "matched_iou_area_weighted": matched_iou_area_weighted,
    }
    if not overlaps.empty:
        overlaps.insert(0, "pair", pair_label)
    if not matched.empty:
        matched.insert(0, "pair", pair_label)
    return summary, overlaps, matched, events_df


def save_snapshot_plot(path: Path, gdf: gpd.GeoDataFrame, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    if not gdf.empty:
        gdf.plot(ax=ax, facecolor="#2ca02c", edgecolor="#174d1a", linewidth=0.15)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pair_overlay(path: Path, from_gdf: gpd.GeoDataFrame, to_gdf: gpd.GeoDataFrame, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    if not from_gdf.empty:
        from_gdf.boundary.plot(ax=ax, color="#d62728", linewidth=0.25, label="from")
    if not to_gdf.empty:
        to_gdf.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.25, label="to")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_gif_from_pngs(frame_paths: list[Path], gif_path: Path, duration_ms: int) -> None:
    if not frame_paths:
        return
    frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_summary_dashboard(path: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = summary["from_snapshot"] + " -> " + summary["to_snapshot"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(labels, summary["matched_iou_area_weighted"], marker="o", color="#1f77b4")
    axes[0, 0].set_title("Area-Weighted Matched IoU")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(labels, summary["net_area_change_ha"], color="#2ca02c")
    axes[0, 1].axhline(0, color="#202020", linewidth=0.8)
    axes[0, 1].set_title("Net Field Area Change")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].plot(labels, summary["split_candidate_count"], marker="o", label="split", color="#ff7f0e")
    axes[1, 0].plot(labels, summary["merge_candidate_count"], marker="s", label="merge", color="#9467bd")
    axes[1, 0].set_title("Split / Merge Candidates")
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].plot(labels, summary["new_count"], marker="o", label="new", color="#1f77b4")
    axes[1, 1].plot(labels, summary["disappeared_count"], marker="s", label="disappeared", color="#d62728")
    axes[1, 1].set_title("New / Disappeared Fields")
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    setup_logging(args)
    LOGGER.info("Starting vector field change tracker")
    LOGGER.info("Input directory: %s", args.input_dir)
    LOGGER.info("Output directory: %s", args.output_dir)
    snapshots = discover_snapshots(args)
    pairs = build_pairs(snapshots, args.pair_mode)
    if not pairs:
        raise ValueError(f"No comparison pairs were built for pair mode '{args.pair_mode}'")

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    snapshots_dir = figures_dir / "snapshots"
    pairs_dir = figures_dir / "pairs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    gdfs = {}
    snapshot_rows = []
    snapshot_frame_paths = []
    for snapshot in snapshots:
        gdf = load_snapshot(snapshot, args.input_crs, args.metric_crs)
        gdfs[snapshot.label] = gdf
        snapshot_rows.append(
            {
                "snapshot": snapshot.label,
                "year": snapshot.year,
                "season": snapshot.season,
                "field_count": int(len(gdf)),
                "area_ha": float(gdf["area_ha"].sum()),
                "mean_field_area_ha": float(gdf["area_ha"].mean()) if len(gdf) else 0.0,
                "median_field_area_ha": float(gdf["area_ha"].median()) if len(gdf) else 0.0,
            }
        )
        LOGGER.info("Loaded %s: fields=%d, area_ha=%.2f", snapshot.label, len(gdf), gdf["area_ha"].sum())
        if not args.skip_figures:
            frame_path = snapshots_dir / f"{snapshot.label}.png"
            save_snapshot_plot(frame_path, gdf, f"{snapshot.label} field polygons")
            snapshot_frame_paths.append(frame_path)

    summaries = []
    overlaps_all = []
    matches_all = []
    events_all = []
    pair_frame_paths = []
    for index, (previous, current) in enumerate(pairs, start=1):
        pair_label = f"{previous.label}_to_{current.label}"
        LOGGER.info("Processing vector pair %d/%d: %s", index, len(pairs), pair_label)
        summary, overlaps, matched, events = compare_pair(
            previous,
            current,
            gdfs[previous.label],
            gdfs[current.label],
            min_iou=args.min_iou,
            min_overlap_ratio=args.min_overlap_ratio,
        )
        summaries.append(summary)
        if not overlaps.empty:
            overlaps_all.append(overlaps)
        if not matched.empty:
            matches_all.append(matched)
        if not events.empty:
            events_all.append(events)
        LOGGER.info(
            "%s: matched=%d, new=%d, disappeared=%d, split=%d, merge=%d, aw_iou=%.3f",
            pair_label,
            summary["matched_count"],
            summary["new_count"],
            summary["disappeared_count"],
            summary["split_candidate_count"],
            summary["merge_candidate_count"],
            summary["matched_iou_area_weighted"],
        )
        if not args.skip_figures:
            frame_path = pairs_dir / f"{pair_label}.png"
            save_pair_overlay(frame_path, gdfs[previous.label], gdfs[current.label], f"{previous.label} -> {current.label}")
            pair_frame_paths.append(frame_path)

    snapshot_summary = pd.DataFrame(snapshot_rows)
    pair_summary = pd.DataFrame(summaries)
    overlaps_df = pd.concat(overlaps_all, ignore_index=True) if overlaps_all else pd.DataFrame()
    matches_df = pd.concat(matches_all, ignore_index=True) if matches_all else pd.DataFrame()
    events_df = pd.concat(events_all, ignore_index=True) if events_all else pd.DataFrame()

    snapshot_summary.to_csv(tables_dir / "vector_snapshot_summary.csv", index=False)
    pair_summary.to_csv(tables_dir / "vector_pair_summary.csv", index=False)
    overlaps_df.to_csv(tables_dir / "vector_overlap_matrix.csv", index=False)
    matches_df.to_csv(tables_dir / "vector_field_matches.csv", index=False)
    events_df.to_csv(tables_dir / "vector_split_merge_events.csv", index=False)

    if not args.skip_figures:
        save_summary_dashboard(figures_dir / "vector_change_metrics_dashboard.png", pair_summary)
        if not args.skip_gifs:
            save_gif_from_pngs(snapshot_frame_paths, figures_dir / "timelines" / "vector_fields_timeline.gif", args.gif_duration_ms)
            save_gif_from_pngs(pair_frame_paths, figures_dir / "timelines" / "vector_pair_overlay_timeline.gif", args.gif_duration_ms)

    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Total elapsed time: %.1fs", time.perf_counter() - start)


if __name__ == "__main__":
    main()
