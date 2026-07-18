#!/usr/bin/env python3
"""Track agricultural field polygon changes across snapshot GeoJSON/GPKG files."""

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

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping
from shapely.validation import make_valid


LOGGER = logging.getLogger("vector_field_change")


@dataclass(frozen=True)
class VectorSnapshot:
    year: int
    season: str
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track field polygon changes using overlap and IoU metrics.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing vector field snapshots. Optional when --snapshot-paths is provided.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/vector_field_change"))
    parser.add_argument(
        "--years",
        default="2020-2023",
        help="Years to process. Accepts comma lists like '2020,2021,2022' or inclusive ranges like '2020-2023'.",
    )
    parser.add_argument("--seasons", default="february_april,june_august")
    parser.add_argument("--filename-template", default="kursh_{year}_{season}__fields")
    parser.add_argument(
        "--filename-templates",
        default="",
        help=(
            "Optional semicolon-separated fallback stem templates tried in order, "
            "for example 'kursh_{year}_{season}__fields;kursh_{year}_{season}'. "
            "Overrides --filename-template when set."
        ),
    )
    parser.add_argument("--extensions", default=".geojson,.gpkg,.shp,.json")
    parser.add_argument(
        "--tile-id",
        default="",
        help=(
            "Optional tile ID such as 36RXT. When set without --snapshot-vector-glob, "
            "the tracker selects '07_exports/{tile}/{tile}/{tile}.geojson' inside "
            "each snapshot directory. If --snapshot-vector-glob contains '{tile}', "
            "the placeholder is replaced with this value."
        ),
    )
    parser.add_argument(
        "--snapshot-vector-glob",
        default="",
        help=(
            "Optional relative glob used inside each snapshot directory, for example "
            "'*.geojson', 'outputs/*.geojson', or "
            "'07_exports/{tile}/{tile}/{tile}.geojson'. Use this when run folders "
            "contain multiple vector products."
        ),
    )
    parser.add_argument(
        "--snapshot-paths",
        default="",
        help=(
            "Optional semicolon-separated exact vector files or snapshot directories. "
            "Bare entries are assigned in --years/--seasons order. Entries may also "
            "be explicit as 'year:season:/path/to/file_or_directory'."
        ),
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--pair-mode",
        choices=("adjacent", "same-season-yearly", "same-season-all-years", "all"),
        default="adjacent",
    )
    parser.add_argument("--input-crs", default="", help="CRS to assign when a file has no CRS, for example EPSG:4326.")
    parser.add_argument("--metric-crs", default="EPSG:32636", help="Projected CRS used for area and overlap metrics.")
    parser.add_argument("--output-crs", default="EPSG:4326", help="CRS used for GeoJSON outputs.")
    parser.add_argument("--min-iou", type=float, default=0.30, help="Minimum IoU for a stable one-to-one match.")
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.10,
        help="Minimum source/target overlap ratio for split, merge, new, and disappeared diagnostics.",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-gifs", action="store_true")
    parser.add_argument("--skip-geojson", action="store_true", help="Skip GeoJSON vector output exports.")
    parser.add_argument("--gif-duration-ms", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true", help="Only discover snapshots; do not read vectors.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Continue when an expected snapshot is missing. Missing snapshots are "
            "logged and skipped. Without this flag, missing snapshots fail fast."
        ),
    )
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


def parse_years(raw: str) -> list[int]:
    years: list[int] = []
    for token in parse_csv_strings(raw):
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            step = 1 if end >= start else -1
            years.extend(range(start, end + step, step))
        else:
            years.append(int(token))
    return years


def parse_csv_strings(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def parse_semicolon_strings(raw: str) -> list[str]:
    return [value.strip() for value in raw.replace("\n", ";").split(";") if value.strip()]


def resolve_vector_glob(args: argparse.Namespace) -> str:
    tile_id = args.tile_id.strip()
    vector_glob = args.snapshot_vector_glob.strip()
    if vector_glob and tile_id:
        return vector_glob.format(tile=tile_id)
    if vector_glob:
        return vector_glob
    if tile_id:
        return f"07_exports/{tile_id}/{tile_id}/{tile_id}.geojson"
    return ""


def pick_vector_from_directory(directory: Path, extensions: list[str], vector_glob: str = "") -> Path | None:
    if vector_glob:
        matches = sorted(
            path
            for path in directory.glob(vector_glob)
            if path.is_file() and not path.name.startswith(".")
        )
        if not matches:
            return None
        if len(matches) > 1:
            candidates = ", ".join(str(path.relative_to(directory)) for path in matches[:20])
            raise ValueError(
                f"Snapshot directory {directory} has multiple vectors matching "
                f"--snapshot-vector-glob '{vector_glob}': {candidates}. "
                "Use a more specific glob."
            )
        return matches[0]

    matches: list[Path] = []
    for ext in extensions:
        matches.extend(directory.rglob(f"*{ext}"))
    matches = sorted(path for path in matches if path.is_file() and not path.name.startswith("."))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    preferred = [
        path
        for path in matches
        if any(token in path.name.lower() for token in ("field", "delineat", "polygon", "geojson"))
    ]
    ranked = sorted(preferred or matches, key=lambda path: (len(path.parts), path.name))
    candidates = ", ".join(str(path.relative_to(directory)) for path in ranked[:20])
    raise ValueError(
        f"Snapshot directory {directory} contains multiple candidate vectors: {candidates}. "
        "Pass --snapshot-vector-glob to select the intended file."
    )


def normalize_snapshot_path(raw_path: str, extensions: list[str], vector_glob: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        vector = pick_vector_from_directory(path, extensions, vector_glob)
        if vector is not None:
            return vector
        raise FileNotFoundError(f"No vector file found inside snapshot directory: {path}")
    raise FileNotFoundError(f"Snapshot path not found: {path}")


def discover_snapshots_from_paths(args: argparse.Namespace) -> list[VectorSnapshot]:
    entries = parse_semicolon_strings(args.snapshot_paths)
    if not entries:
        raise ValueError("--snapshot-paths was provided but did not contain any paths")

    extensions = parse_csv_strings(args.extensions)
    vector_glob = resolve_vector_glob(args)
    explicit_entries = [entry for entry in entries if len(entry.split(":", 2)) == 3]
    if explicit_entries:
        if len(explicit_entries) != len(entries):
            raise ValueError(
                "--snapshot-paths must use either all bare paths or all explicit "
                "'year:season:path' entries; do not mix both forms."
            )
        snapshots = []
        for entry in entries:
            year_raw, season, path_raw = entry.split(":", 2)
            year = int(year_raw.strip())
            season = season.strip()
            path = normalize_snapshot_path(path_raw.strip(), extensions, vector_glob)
            label = f"{year}_{season}"
            LOGGER.info("Discovered %s from explicit path -> %s", label, path)
            snapshots.append(VectorSnapshot(year=year, season=season, label=label, path=path))
        return snapshots

    years = parse_years(args.years)
    seasons = parse_csv_strings(args.seasons)
    expected = len(years) * len(seasons)
    if len(entries) != expected:
        raise ValueError(
            f"--snapshot-paths provided {len(entries)} bare paths, but --years/--seasons "
            f"define {expected} snapshots ({len(years)} years x {len(seasons)} seasons). "
            "Either pass the exact number of ordered paths or use explicit "
            "'year:season:path' entries."
        )

    snapshots = []
    path_index = 0
    for year in years:
        for season in seasons:
            path = normalize_snapshot_path(entries[path_index], extensions, vector_glob)
            label = f"{year}_{season}"
            LOGGER.info("Discovered %s from ordered path -> %s", label, path)
            snapshots.append(VectorSnapshot(year=year, season=season, label=label, path=path))
            path_index += 1
    return snapshots


def parse_filename_templates(args: argparse.Namespace) -> list[str]:
    if args.filename_templates:
        return [template.strip() for template in args.filename_templates.split(";") if template.strip()]
    return [args.filename_template]


def find_snapshot_file(
    input_dir: Path,
    stem: str,
    extensions: list[str],
    recursive: bool,
    vector_glob: str,
) -> Path:
    for ext in extensions:
        direct = input_dir / f"{stem}{ext}"
        if direct.exists():
            return direct
    directory = input_dir / stem
    if directory.is_dir():
        vector = pick_vector_from_directory(directory, extensions, vector_glob)
        if vector is not None:
            return vector
    if recursive:
        for ext in extensions:
            matches = sorted(input_dir.rglob(f"{stem}{ext}"))
            if matches:
                return matches[0]
        directories = sorted(path for path in input_dir.rglob(stem) if path.is_dir())
        for directory in directories:
            vector = pick_vector_from_directory(directory, extensions, vector_glob)
            if vector is not None:
                return vector
    raise FileNotFoundError(f"No vector snapshot found for stem '{stem}' in {input_dir}")


def discover_snapshots(args: argparse.Namespace) -> list[VectorSnapshot]:
    if args.snapshot_paths:
        return discover_snapshots_from_paths(args)
    if args.input_dir is None:
        raise ValueError("Either --input-dir or --snapshot-paths is required.")

    years = parse_years(args.years)
    seasons = parse_csv_strings(args.seasons)
    extensions = parse_csv_strings(args.extensions)
    filename_templates = parse_filename_templates(args)
    vector_glob = resolve_vector_glob(args)
    snapshots = []
    missing = []
    for year in years:
        for season in seasons:
            errors = []
            label = f"{year}_{season}"
            for filename_template in filename_templates:
                stem = filename_template.format(year=year, season=season)
                try:
                    path = find_snapshot_file(
                        args.input_dir,
                        stem,
                        extensions,
                        args.recursive,
                        vector_glob,
                    )
                except FileNotFoundError as exc:
                    errors.append(str(exc))
                    continue
                LOGGER.info("Discovered %s using template '%s' -> %s", label, filename_template, path)
                snapshots.append(VectorSnapshot(year=year, season=season, label=label, path=path))
                break
            else:
                message = (
                    f"No vector snapshot found for {label} using templates: {filename_templates}. "
                    f"Last errors: {' | '.join(errors[-3:])}"
                )
                if args.allow_missing:
                    LOGGER.warning("Skipping missing snapshot %s. %s", label, message)
                    missing.append(label)
                    continue
                raise FileNotFoundError(message)
    if missing:
        LOGGER.warning("Missing snapshots skipped (%d): %s", len(missing), ", ".join(missing))
    if not snapshots:
        raise FileNotFoundError("No vector snapshots were discovered.")
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


def log_pair_gap_warnings(pairs: list[tuple[VectorSnapshot, VectorSnapshot]]) -> None:
    for previous, current in pairs:
        if previous.season != current.season:
            continue
        year_gap = current.year - previous.year
        if year_gap > 1:
            LOGGER.warning(
                "Comparison %s -> %s skips %d missing year(s); interpret it as a multi-year gap.",
                previous.label,
                current.label,
                year_gap - 1,
            )


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


def compute_common_bounds(gdfs: list[gpd.GeoDataFrame]) -> tuple[float, float, float, float] | None:
    non_empty = [gdf for gdf in gdfs if not gdf.empty]
    if not non_empty:
        return None
    bounds = np.array([gdf.total_bounds for gdf in non_empty], dtype=float)
    minx = float(np.nanmin(bounds[:, 0]))
    miny = float(np.nanmin(bounds[:, 1]))
    maxx = float(np.nanmax(bounds[:, 2]))
    maxy = float(np.nanmax(bounds[:, 3]))
    width = maxx - minx
    height = maxy - miny
    pad = max(width, height) * 0.02
    if pad == 0:
        pad = 1.0
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def apply_map_view(ax, bounds: tuple[float, float, float, float] | None) -> None:
    if bounds is not None:
        minx, miny, maxx, maxy = bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def figure_size_for_bounds(
    bounds: tuple[float, float, float, float] | None,
    long_side: float = 11.0,
    min_side: float = 6.0,
    max_side: float = 14.0,
) -> tuple[float, float]:
    if bounds is None:
        return 9.0, 9.0
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    aspect = width / height
    if aspect >= 1:
        return min(max_side, long_side), max(min_side, min(max_side, long_side / aspect))
    return max(min_side, min(max_side, long_side * aspect)), min(max_side, long_side)


def polygonal_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return None
    try:
        geometry = make_valid(geometry)
    except Exception:
        geometry = geometry.buffer(0)
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, Polygon):
        return geometry if geometry.area > 0 else None
    if isinstance(geometry, MultiPolygon):
        polygons = [part for part in geometry.geoms if not part.is_empty and part.area > 0]
        if not polygons:
            return None
        return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for part in geometry.geoms:
            polygonal = polygonal_geometry(part)
            if isinstance(polygonal, Polygon):
                polygons.append(polygonal)
            elif isinstance(polygonal, MultiPolygon):
                polygons.extend(polygonal.geoms)
        if not polygons:
            return None
        return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    return None


def json_property(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_geojson_python(gdf: gpd.GeoDataFrame, path: Path) -> None:
    geometry_column = gdf.geometry.name
    with path.open("w", encoding="utf-8") as handle:
        handle.write('{"type":"FeatureCollection","features":[\n')
        first = True
        for _, row in gdf.iterrows():
            geometry = row.geometry
            if geometry is None or geometry.is_empty:
                continue
            properties = {
                column: json_property(row[column])
                for column in gdf.columns
                if column != geometry_column
            }
            feature = {
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(geometry),
            }
            if not first:
                handle.write(",\n")
            json.dump(feature, handle, ensure_ascii=False, allow_nan=False)
            first = False
        handle.write("\n]}\n")


def sanitize_geojson_layer(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    output = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if output.empty:
        return output
    before = len(output)
    output["geometry"] = output.geometry.map(polygonal_geometry)
    output = output[output.geometry.notna() & ~output.geometry.is_empty].copy()
    if len(output) < before:
        LOGGER.warning("Dropped %d non-polygonal or invalid geometries before GeoJSON export", before - len(output))
    return gpd.GeoDataFrame(output, geometry="geometry", crs=gdf.crs)


def write_geojson(gdf: gpd.GeoDataFrame, path: Path, output_crs: str) -> None:
    if gdf.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    output = sanitize_geojson_layer(gdf)
    if output.empty:
        LOGGER.warning("Skipping empty GeoJSON after geometry sanitization: %s", path)
        return
    if output_crs and output.crs is not None:
        output = output.to_crs(output_crs)
    try:
        output.to_file(path, driver="GeoJSON")
    except Exception as exc:
        LOGGER.warning("Pyogrio failed to write %s (%s). Retrying with pure-Python GeoJSON writer.", path, exc)
        write_geojson_python(output, path)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_event_geometries(
    pair_label: str,
    previous: VectorSnapshot,
    current: VectorSnapshot,
    from_gdf: gpd.GeoDataFrame,
    to_gdf: gpd.GeoDataFrame,
    split_from_ids: set[str],
    merge_to_ids: set[str],
    disappeared_ids: set[str],
    new_ids: set[str],
) -> gpd.GeoDataFrame:
    rows = []
    from_lookup = from_gdf.set_index("field_uid", drop=False)
    to_lookup = to_gdf.set_index("field_uid", drop=False)

    def add_events(field_ids: set[str], event_type: str, source: gpd.GeoDataFrame, source_snapshot: str) -> None:
        for field_id in sorted(field_ids):
            if field_id not in source.index:
                continue
            row = source.loc[field_id]
            rows.append(
                {
                    "pair": pair_label,
                    "event_type": event_type,
                    "field_id": field_id,
                    "source_snapshot": source_snapshot,
                    "area_ha": float(row["area_ha"]),
                    "geometry": row.geometry,
                }
            )

    add_events(split_from_ids, "split", from_lookup, previous.label)
    add_events(disappeared_ids, "disappeared", from_lookup, previous.label)
    add_events(merge_to_ids, "merge", to_lookup, current.label)
    add_events(new_ids, "new", to_lookup, current.label)
    event_crs = from_gdf.crs if from_gdf.crs is not None else to_gdf.crs
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=event_crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=event_crs)


def compare_pair(
    previous: VectorSnapshot,
    current: VectorSnapshot,
    from_gdf: gpd.GeoDataFrame,
    to_gdf: gpd.GeoDataFrame,
    min_iou: float,
    min_overlap_ratio: float,
) -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    pair_label = f"{previous.label}_to_{current.label}"
    from_fields = from_gdf.rename(columns={"field_uid": "from_field_id", "area_ha": "from_area_ha"})
    to_fields = to_gdf.rename(columns={"field_uid": "to_field_id", "area_ha": "to_area_ha"})
    from_fields = from_fields[["from_field_id", "from_area_ha", "geometry"]]
    to_fields = to_fields[["to_field_id", "to_area_ha", "geometry"]]
    pair_crs = from_gdf.crs if from_gdf.crs is not None else to_gdf.crs

    if from_fields.empty or to_fields.empty:
        overlaps = pd.DataFrame()
        overlaps_gdf = gpd.GeoDataFrame(geometry=[], crs=pair_crs)
    else:
        overlaps_gdf = gpd.overlay(from_fields, to_fields, how="intersection", keep_geom_type=False)
        overlaps_gdf = overlaps_gdf[overlaps_gdf.geometry.notna() & ~overlaps_gdf.geometry.is_empty].copy()
        overlaps_gdf = overlaps_gdf.reset_index(drop=True)
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
        matched_gdf = gpd.GeoDataFrame(geometry=[], crs=pair_crs)
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
        matched_gdf = overlaps_gdf.loc[matched.index].copy()

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
    events_gdf = build_event_geometries(
        pair_label,
        previous,
        current,
        from_gdf,
        to_gdf,
        split_from_ids,
        merge_to_ids,
        disappeared_ids,
        new_ids,
    )

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
        overlaps_gdf.insert(0, "pair", pair_label)
    if not matched.empty:
        matched.insert(0, "pair", pair_label)
        matched_gdf.insert(0, "pair", pair_label)
    return summary, overlaps, matched, events_df, overlaps_gdf, matched_gdf, events_gdf


def save_snapshot_plot(
    path: Path,
    gdf: gpd.GeoDataFrame,
    title: str,
    bounds: tuple[float, float, float, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figure_size_for_bounds(bounds))
    if not gdf.empty:
        gdf.plot(ax=ax, facecolor="#2ca02c", edgecolor="#174d1a", linewidth=0.15)
    ax.set_title(title)
    apply_map_view(ax, bounds)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_pair_overlay(
    path: Path,
    from_gdf: gpd.GeoDataFrame,
    to_gdf: gpd.GeoDataFrame,
    title: str,
    bounds: tuple[float, float, float, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figure_size_for_bounds(bounds))
    if not from_gdf.empty:
        from_gdf.boundary.plot(ax=ax, color="#d62728", linewidth=0.25, label="from")
    if not to_gdf.empty:
        to_gdf.boundary.plot(ax=ax, color="#1f77b4", linewidth=0.25, label="to")
    ax.set_title(title)
    apply_map_view(ax, bounds)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.08)
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


def save_grouped_gifs(
    grouped_frame_paths: dict[str, list[Path]],
    output_dir: Path,
    filename_template: str,
    duration_ms: int,
) -> list[Path]:
    saved_paths = []
    for group, frame_paths in sorted(grouped_frame_paths.items()):
        gif_path = output_dir / filename_template.format(group=group)
        save_gif_from_pngs(frame_paths, gif_path, duration_ms)
        if frame_paths:
            saved_paths.append(gif_path)
            LOGGER.info("Saved %s with %d frames", gif_path, len(frame_paths))
    return saved_paths


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
    if args.snapshot_paths:
        LOGGER.info("Input mode: explicit snapshot paths")
    else:
        LOGGER.info("Input directory: %s", args.input_dir)
    LOGGER.info("Output directory: %s", args.output_dir)
    vector_glob = resolve_vector_glob(args)
    if vector_glob:
        LOGGER.info("Vector selector: %s", vector_glob)
    snapshots = discover_snapshots(args)
    if args.dry_run:
        pairs = build_pairs(snapshots, args.pair_mode)
        log_pair_gap_warnings(pairs)
        LOGGER.info("Dry run complete. Selected snapshots: %d", len(snapshots))
        for snapshot in snapshots:
            LOGGER.info("%s -> %s", snapshot.label, snapshot.path)
        LOGGER.info("Dry run comparison pairs for pair mode '%s': %d", args.pair_mode, len(pairs))
        for previous, current in pairs:
            LOGGER.info("%s -> %s", previous.label, current.label)
        return

    pairs = build_pairs(snapshots, args.pair_mode)
    log_pair_gap_warnings(pairs)
    if not pairs:
        raise ValueError(f"No comparison pairs were built for pair mode '{args.pair_mode}'")

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    geojson_dir = output_dir / "geojson"
    snapshots_dir = figures_dir / "snapshots"
    pairs_dir = figures_dir / "pairs"
    geojson_snapshots_dir = geojson_dir / "snapshots"
    geojson_pairs_dir = geojson_dir / "pairs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_geojson:
        geojson_snapshots_dir.mkdir(parents=True, exist_ok=True)
        geojson_pairs_dir.mkdir(parents=True, exist_ok=True)

    gdfs = {}
    snapshot_rows = []
    snapshot_frame_paths = []
    snapshot_frame_paths_by_season: dict[str, list[Path]] = {}
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

    plot_bounds = compute_common_bounds(list(gdfs.values()))
    if plot_bounds is not None:
        LOGGER.info(
            "Using fixed map bounds for all figures: %.2f, %.2f, %.2f, %.2f",
            plot_bounds[0],
            plot_bounds[1],
            plot_bounds[2],
            plot_bounds[3],
        )

    if not args.skip_geojson:
        for snapshot in snapshots:
            write_geojson(
                gdfs[snapshot.label],
                geojson_snapshots_dir / f"{snapshot.label}_fields.geojson",
                args.output_crs,
            )

    if not args.skip_figures:
        for snapshot in snapshots:
            frame_path = snapshots_dir / f"{snapshot.label}.png"
            save_snapshot_plot(frame_path, gdfs[snapshot.label], f"{snapshot.label} field polygons", plot_bounds)
            snapshot_frame_paths.append(frame_path)
            snapshot_frame_paths_by_season.setdefault(snapshot.season, []).append(frame_path)

    summaries = []
    overlaps_all = []
    matches_all = []
    events_all = []
    overlaps_geo_all = []
    matches_geo_all = []
    events_geo_all = []
    pair_frame_paths = []
    pair_frame_paths_by_season: dict[str, list[Path]] = {}
    for index, (previous, current) in enumerate(pairs, start=1):
        pair_label = f"{previous.label}_to_{current.label}"
        LOGGER.info("Processing vector pair %d/%d: %s", index, len(pairs), pair_label)
        summary, overlaps, matched, events, overlaps_geo, matched_geo, events_geo = compare_pair(
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
        if not overlaps_geo.empty:
            overlaps_geo_all.append(overlaps_geo)
        if not matched_geo.empty:
            matches_geo_all.append(matched_geo)
        if not events_geo.empty:
            events_geo_all.append(events_geo)
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
        if not args.skip_geojson:
            write_geojson(
                overlaps_geo,
                geojson_pairs_dir / f"{pair_label}_overlap_intersections.geojson",
                args.output_crs,
            )
            write_geojson(
                matched_geo,
                geojson_pairs_dir / f"{pair_label}_matched_intersections.geojson",
                args.output_crs,
            )
            write_geojson(
                events_geo,
                geojson_pairs_dir / f"{pair_label}_change_events.geojson",
                args.output_crs,
            )
        if not args.skip_figures:
            frame_path = pairs_dir / f"{pair_label}.png"
            save_pair_overlay(
                frame_path,
                gdfs[previous.label],
                gdfs[current.label],
                f"{previous.label} -> {current.label}",
                plot_bounds,
            )
            pair_frame_paths.append(frame_path)
            pair_group = previous.season if previous.season == current.season else "cross_season"
            pair_frame_paths_by_season.setdefault(pair_group, []).append(frame_path)

    snapshot_summary = pd.DataFrame(snapshot_rows)
    pair_summary = pd.DataFrame(summaries)
    overlaps_df = pd.concat(overlaps_all, ignore_index=True) if overlaps_all else pd.DataFrame()
    matches_df = pd.concat(matches_all, ignore_index=True) if matches_all else pd.DataFrame()
    events_df = pd.concat(events_all, ignore_index=True) if events_all else pd.DataFrame()
    overlaps_geo_df = (
        gpd.GeoDataFrame(
            pd.concat(overlaps_geo_all, ignore_index=True),
            geometry="geometry",
            crs=overlaps_geo_all[0].crs,
        )
        if overlaps_geo_all
        else gpd.GeoDataFrame(geometry=[])
    )
    matches_geo_df = (
        gpd.GeoDataFrame(
            pd.concat(matches_geo_all, ignore_index=True),
            geometry="geometry",
            crs=matches_geo_all[0].crs,
        )
        if matches_geo_all
        else gpd.GeoDataFrame(geometry=[])
    )
    events_geo_df = (
        gpd.GeoDataFrame(
            pd.concat(events_geo_all, ignore_index=True),
            geometry="geometry",
            crs=events_geo_all[0].crs,
        )
        if events_geo_all
        else gpd.GeoDataFrame(geometry=[])
    )

    snapshot_summary.to_csv(tables_dir / "vector_snapshot_summary.csv", index=False)
    pair_summary.to_csv(tables_dir / "vector_pair_summary.csv", index=False)
    overlaps_df.to_csv(tables_dir / "vector_overlap_matrix.csv", index=False)
    matches_df.to_csv(tables_dir / "vector_field_matches.csv", index=False)
    events_df.to_csv(tables_dir / "vector_split_merge_events.csv", index=False)

    if not args.skip_geojson:
        write_geojson(overlaps_geo_df, geojson_dir / "vector_overlap_intersections.geojson", args.output_crs)
        write_geojson(matches_geo_df, geojson_dir / "vector_field_matches.geojson", args.output_crs)
        write_geojson(events_geo_df, geojson_dir / "vector_change_events.geojson", args.output_crs)

    if not args.skip_figures:
        save_summary_dashboard(figures_dir / "vector_change_metrics_dashboard.png", pair_summary)
        if not args.skip_gifs:
            timelines_dir = figures_dir / "timelines"
            save_gif_from_pngs(snapshot_frame_paths, timelines_dir / "vector_fields_timeline.gif", args.gif_duration_ms)
            save_gif_from_pngs(pair_frame_paths, timelines_dir / "vector_pair_overlay_timeline.gif", args.gif_duration_ms)
            save_grouped_gifs(
                snapshot_frame_paths_by_season,
                timelines_dir,
                "vector_fields_{group}_timeline.gif",
                args.gif_duration_ms,
            )
            save_grouped_gifs(
                pair_frame_paths_by_season,
                timelines_dir,
                "vector_pair_overlay_{group}_timeline.gif",
                args.gif_duration_ms,
            )

    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Total elapsed time: %.1fs", time.perf_counter() - start)


if __name__ == "__main__":
    main()
