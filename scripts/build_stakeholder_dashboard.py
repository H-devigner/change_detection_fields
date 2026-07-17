#!/usr/bin/env python3
"""Build a static stakeholder dashboard for vector field-change outputs.

The dashboard reads the CSV/PNG/GIF/GeoJSON artifacts produced by
vector_field_change_tracker.py and writes a self-contained HTML report. It uses
only the Python standard library so it can run in the same environment without
adding Streamlit, Plotly, or a web server.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


SEASON_LABELS = {
    "february_april": "February-April",
    "june_august": "June-August",
}

EVENT_COLORS = {
    "new_count": "#1f77b4",
    "disappeared_count": "#d62728",
    "split_candidate_count": "#ff9f1c",
    "merge_candidate_count": "#7b2cbf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static stakeholder dashboard from field-change outputs.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Processed vector output directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dashboard directory. Default: <input-dir>/dashboard.",
    )
    parser.add_argument("--title", default="Agricultural Field Change Dashboard")
    parser.add_argument("--subtitle", default="Stakeholder summary of field delineation change monitoring")
    parser.add_argument("--max-pair-cards", type=int, default=18, help="Maximum pair preview cards to show.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        parsed = float(value)  # type: ignore[arg-type]
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    return int(round(to_float(value, float(default))))


def fmt_num(value: float, digits: int = 0) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.{digits}f}"
    return f"{value:.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def snapshot_sort_key(row: dict[str, str]) -> tuple[int, int, str]:
    season = row.get("season", "")
    season_order = {"february_april": 0, "june_august": 1}.get(season, 99)
    return to_int(row.get("year")), season_order, season


def pair_label(row: dict[str, str]) -> str:
    return f"{row.get('from_snapshot', '')} -> {row.get('to_snapshot', '')}"


def season_display(season: str) -> str:
    return SEASON_LABELS.get(season, season.replace("_", " ").title())


def snapshot_display(snapshot: str) -> str:
    parts = snapshot.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return f"{parts[0]} {season_display(parts[1])}"
    return snapshot.replace("_", " ").title()


def season_from_snapshot(snapshot: str) -> str:
    parts = snapshot.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else snapshot


def rel_link(path: Path, output_dir: Path) -> str:
    return html.escape(path.resolve().relative_to(output_dir.resolve()).as_posix()) if path.resolve().is_relative_to(output_dir.resolve()) else html.escape(path.as_posix())


def rel_path(path: Path, output_dir: Path) -> str:
    try:
        return html.escape(path.resolve().relative_to(output_dir.resolve()).as_posix())
    except ValueError:
        return html.escape(path.resolve().as_posix())


def min_max(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        pad = abs(low) * 0.1 or 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def svg_line_chart(labels: list[str], values: list[float], color: str, y_suffix: str = "") -> str:
    width = 920
    height = 300
    pad_left = 58
    pad_right = 24
    pad_top = 24
    pad_bottom = 70
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    low, high = min_max(values)
    if not labels or not values:
        return empty_chart_svg("No data available")

    def x_at(index: int) -> float:
        if len(values) == 1:
            return pad_left + plot_w / 2
        return pad_left + (plot_w * index / (len(values) - 1))

    def y_at(value: float) -> float:
        return pad_top + plot_h - ((value - low) / (high - low) * plot_h)

    points = [(x_at(i), y_at(v)) for i, v in enumerate(values)]
    path_d = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))
    circles = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5"><title>{html.escape(labels[i])}: {fmt_num(values[i], 2)}{y_suffix}</title></circle>'
        for i, (x, y) in enumerate(points)
    )
    y_ticks = []
    for step in range(5):
        value = low + ((high - low) * step / 4)
        y = y_at(value)
        y_ticks.append(
            f'<line x1="{pad_left}" y1="{y:.2f}" x2="{width - pad_right}" y2="{y:.2f}" class="grid" />'
            f'<text x="{pad_left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis-label">{fmt_num(value, 1)}{html.escape(y_suffix)}</text>'
        )
    x_labels = []
    for i, label in enumerate(labels):
        if len(labels) > 8 and i % 2 == 1:
            continue
        x = x_at(i)
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - 32}" text-anchor="end" transform="rotate(-35 {x:.2f} {height - 32})" class="axis-label">{html.escape(label)}</text>'
        )
    return f"""
    <svg viewBox="0 0 {width} {height}" class="chart-svg" role="img">
      <rect width="{width}" height="{height}" rx="18" class="chart-bg" />
      {''.join(y_ticks)}
      <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" class="axis" />
      <path d="{path_d}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" />
      <g fill="{color}">{circles}</g>
      {''.join(x_labels)}
    </svg>
    """


def svg_bar_chart(labels: list[str], values: list[float], color: str = "#0f766e") -> str:
    width = 920
    height = 310
    pad_left = 58
    pad_right = 24
    pad_top = 24
    pad_bottom = 82
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    if not labels or not values:
        return empty_chart_svg("No data available")
    low = min(0.0, min(values))
    high = max(0.0, max(values))
    if low == high:
        high = 1.0
    bar_gap = 10
    bar_w = max(12, (plot_w - bar_gap * (len(values) + 1)) / len(values))

    def y_at(value: float) -> float:
        return pad_top + plot_h - ((value - low) / (high - low) * plot_h)

    zero_y = y_at(0)
    bars = []
    x_labels = []
    for i, value in enumerate(values):
        x = pad_left + bar_gap + i * (bar_w + bar_gap)
        y = min(y_at(value), zero_y)
        h = abs(zero_y - y_at(value))
        fill = color if value >= 0 else "#d62728"
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{max(h, 1):.2f}" rx="5" fill="{fill}">'
            f'<title>{html.escape(labels[i])}: {fmt_num(value, 2)}</title></rect>'
        )
        if len(labels) <= 10 or i % 2 == 0:
            tx = x + bar_w / 2
            x_labels.append(
                f'<text x="{tx:.2f}" y="{height - 34}" text-anchor="end" transform="rotate(-35 {tx:.2f} {height - 34})" class="axis-label">{html.escape(labels[i])}</text>'
            )
    ticks = []
    for step in range(5):
        value = low + ((high - low) * step / 4)
        y = y_at(value)
        ticks.append(
            f'<line x1="{pad_left}" y1="{y:.2f}" x2="{width - pad_right}" y2="{y:.2f}" class="grid" />'
            f'<text x="{pad_left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis-label">{fmt_num(value, 1)}</text>'
        )
    return f"""
    <svg viewBox="0 0 {width} {height}" class="chart-svg" role="img">
      <rect width="{width}" height="{height}" rx="18" class="chart-bg" />
      {''.join(ticks)}
      <line x1="{pad_left}" y1="{zero_y:.2f}" x2="{width - pad_right}" y2="{zero_y:.2f}" class="axis" />
      {''.join(bars)}
      {''.join(x_labels)}
    </svg>
    """


def svg_stacked_event_chart(labels: list[str], rows: list[dict[str, str]]) -> str:
    width = 920
    height = 320
    pad_left = 58
    pad_right = 24
    pad_top = 24
    pad_bottom = 88
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    if not rows:
        return empty_chart_svg("No event data available")
    totals = [sum(to_int(row.get(col)) for col in EVENT_COLORS) for row in rows]
    high = max(totals) if totals else 1
    if high <= 0:
        high = 1
    bar_gap = 10
    bar_w = max(12, (plot_w - bar_gap * (len(rows) + 1)) / len(rows))
    bars = []
    x_labels = []
    for i, row in enumerate(rows):
        x = pad_left + bar_gap + i * (bar_w + bar_gap)
        y_cursor = pad_top + plot_h
        for col, fill in EVENT_COLORS.items():
            value = to_int(row.get(col))
            h = value / high * plot_h
            y_cursor -= h
            if h > 0:
                bars.append(
                    f'<rect x="{x:.2f}" y="{y_cursor:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{fill}">'
                    f'<title>{html.escape(labels[i])} {html.escape(col.replace("_", " "))}: {value}</title></rect>'
                )
        if len(labels) <= 10 or i % 2 == 0:
            tx = x + bar_w / 2
            x_labels.append(
                f'<text x="{tx:.2f}" y="{height - 36}" text-anchor="end" transform="rotate(-35 {tx:.2f} {height - 36})" class="axis-label">{html.escape(labels[i])}</text>'
            )
    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(col.replace("_count", "").replace("_candidate", "").replace("_", " ").title())}</span>'
        for col, color in EVENT_COLORS.items()
    )
    ticks = []
    for step in range(5):
        value = high * step / 4
        y = pad_top + plot_h - (value / high * plot_h)
        ticks.append(
            f'<line x1="{pad_left}" y1="{y:.2f}" x2="{width - pad_right}" y2="{y:.2f}" class="grid" />'
            f'<text x="{pad_left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis-label">{fmt_num(value, 0)}</text>'
        )
    return f"""
    <div class="chart-legend">{legend}</div>
    <svg viewBox="0 0 {width} {height}" class="chart-svg" role="img">
      <rect width="{width}" height="{height}" rx="18" class="chart-bg" />
      {''.join(ticks)}
      <line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" y2="{pad_top + plot_h}" class="axis" />
      {''.join(bars)}
      {''.join(x_labels)}
    </svg>
    """


def empty_chart_svg(message: str) -> str:
    return f"""
    <svg viewBox="0 0 920 220" class="chart-svg" role="img">
      <rect width="920" height="220" rx="18" class="chart-bg" />
      <text x="460" y="114" text-anchor="middle" class="empty-chart">{html.escape(message)}</text>
    </svg>
    """


def table_html(headers: list[str], rows: list[list[object]], class_name: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap {class_name}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def card(title: str, value: str, detail: str = "", tone: str = "") -> str:
    return f"""
    <article class="kpi {html.escape(tone)}">
      <span>{html.escape(title)}</span>
      <strong>{html.escape(value)}</strong>
      <small>{html.escape(detail)}</small>
    </article>
    """


def find_assets(input_dir: Path) -> dict[str, list[Path]]:
    figures = input_dir / "figures"
    geojson = input_dir / "geojson"
    return {
        "dashboard_pngs": sorted(figures.glob("*.png")),
        "pair_pngs": sorted((figures / "pairs").glob("*.png")),
        "snapshot_pngs": sorted((figures / "snapshots").glob("*.png")),
        "gifs": sorted((figures / "timelines").glob("*.gif")),
        "geojson_root": sorted(geojson.glob("*.geojson")),
        "geojson_pairs": sorted((geojson / "pairs").glob("*.geojson")),
    }


def compute_summary(snapshot_rows: list[dict[str, str]], pair_rows: list[dict[str, str]]) -> dict[str, object]:
    snapshots = sorted(snapshot_rows, key=snapshot_sort_key)
    pairs = pair_rows
    first = snapshots[0] if snapshots else {}
    latest = snapshots[-1] if snapshots else {}
    first_area = to_float(first.get("area_ha"))
    latest_area = to_float(latest.get("area_ha"))
    net_area = latest_area - first_area if snapshots else 0.0
    total_new = sum(to_int(row.get("new_count")) for row in pairs)
    total_disappeared = sum(to_int(row.get("disappeared_count")) for row in pairs)
    total_split = sum(to_int(row.get("split_candidate_count")) for row in pairs)
    total_merge = sum(to_int(row.get("merge_candidate_count")) for row in pairs)
    ious = [to_float(row.get("matched_iou_area_weighted")) for row in pairs if row.get("matched_iou_area_weighted") not in (None, "")]
    net_changes = [abs(to_float(row.get("net_area_change_ha"))) for row in pairs]
    volatile_pair = None
    if pairs:
        volatile_pair = max(
            pairs,
            key=lambda row: abs(to_float(row.get("net_area_change_ha")))
            + to_int(row.get("new_count"))
            + to_int(row.get("disappeared_count")),
        )
    return {
        "snapshot_count": len(snapshots),
        "pair_count": len(pairs),
        "first_snapshot": first.get("snapshot", ""),
        "latest_snapshot": latest.get("snapshot", ""),
        "first_area": first_area,
        "latest_area": latest_area,
        "net_area": net_area,
        "net_area_pct": safe_div(net_area, first_area),
        "total_new": total_new,
        "total_disappeared": total_disappeared,
        "total_split": total_split,
        "total_merge": total_merge,
        "mean_iou": mean(ious) if ious else 0.0,
        "median_iou": median(ious) if ious else 0.0,
        "area_volatility": sum(net_changes),
        "volatile_pair": pair_label(volatile_pair) if volatile_pair else "-",
    }


def season_summary(snapshot_rows: list[dict[str, str]], pair_rows: list[dict[str, str]]) -> list[list[str]]:
    grouped_snapshots: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in snapshot_rows:
        grouped_snapshots[row.get("season", "")].append(row)
    grouped_pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        grouped_pairs[season_from_snapshot(row.get("from_snapshot", ""))].append(row)
    result = []
    for season, rows in sorted(grouped_snapshots.items()):
        ordered = sorted(rows, key=snapshot_sort_key)
        first_area = to_float(ordered[0].get("area_ha")) if ordered else 0.0
        latest_area = to_float(ordered[-1].get("area_ha")) if ordered else 0.0
        pairs = grouped_pairs.get(season, [])
        ious = [to_float(row.get("matched_iou_area_weighted")) for row in pairs]
        result.append(
            [
                season_display(season),
                len(ordered),
                fmt_num(first_area, 1),
                fmt_num(latest_area, 1),
                fmt_num(latest_area - first_area, 1),
                fmt_pct(safe_div(latest_area - first_area, first_area)),
                fmt_num(mean(ious), 3) if ious else "-",
                sum(to_int(row.get("new_count")) for row in pairs),
                sum(to_int(row.get("disappeared_count")) for row in pairs),
            ]
        )
    return result


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def distribution_rows(matches: list[dict[str, str]]) -> list[list[str]]:
    ious = [to_float(row.get("iou")) for row in matches if row.get("iou") not in (None, "")]
    from_changes = []
    for row in matches:
        from_area = to_float(row.get("from_area_ha"))
        to_area = to_float(row.get("to_area_ha"))
        if from_area:
            from_changes.append((to_area - from_area) / from_area)
    return [
        ["Matched field IoU p10", fmt_num(percentile(ious, 0.10), 3)],
        ["Matched field IoU p50", fmt_num(percentile(ious, 0.50), 3)],
        ["Matched field IoU p90", fmt_num(percentile(ious, 0.90), 3)],
        ["Matched field area change p10", fmt_pct(percentile(from_changes, 0.10))],
        ["Matched field area change p50", fmt_pct(percentile(from_changes, 0.50))],
        ["Matched field area change p90", fmt_pct(percentile(from_changes, 0.90))],
    ]


def build_narrative(summary: dict[str, object]) -> list[str]:
    net = float(summary["net_area"])
    direction = "increased" if net > 0 else "decreased" if net < 0 else "remained stable"
    return [
        f"Field area {direction} by {fmt_num(abs(net), 1)} ha from {snapshot_display(str(summary['first_snapshot']))} to {snapshot_display(str(summary['latest_snapshot']))}.",
        f"The average area-weighted IoU across comparison periods is {fmt_num(float(summary['mean_iou']), 3)}, indicating the spatial persistence of matched fields.",
        f"The most volatile comparison period is {summary['volatile_pair']}, based on area movement and field turnover counts.",
    ]


def render_pair_cards(pair_rows: list[dict[str, str]], pair_pngs: list[Path], input_dir: Path, output_dir: Path, max_cards: int) -> str:
    png_by_stem = {path.stem: path for path in pair_pngs}
    cards = []
    for row in pair_rows[:max_cards]:
        pair = row.get("pair") or f"{row.get('from_snapshot')}_to_{row.get('to_snapshot')}"
        image = png_by_stem.get(pair)
        image_html = ""
        if image is not None:
            image_html = f'<img src="{rel_path(image, output_dir)}" alt="{html.escape(pair)} overlay" loading="lazy" />'
        cards.append(
            f"""
            <article class="pair-card">
              <div>
                <h4>{html.escape(snapshot_display(row.get('from_snapshot', '')))} -> {html.escape(snapshot_display(row.get('to_snapshot', '')))}</h4>
                <p>IoU <strong>{fmt_num(to_float(row.get('matched_iou_area_weighted')), 3)}</strong> | Net area <strong>{fmt_num(to_float(row.get('net_area_change_ha')), 1)} ha</strong></p>
                <p>New {to_int(row.get('new_count'))} | Disappeared {to_int(row.get('disappeared_count'))} | Split {to_int(row.get('split_candidate_count'))} | Merge {to_int(row.get('merge_candidate_count'))}</p>
              </div>
              {image_html}
            </article>
            """
        )
    if not cards:
        return '<p class="muted">No pair previews available.</p>'
    return "\n".join(cards)


def asset_links(title: str, paths: Iterable[Path], output_dir: Path, limit: int = 30) -> str:
    paths = list(paths)
    if not paths:
        return ""
    links = []
    for path in paths[:limit]:
        links.append(f'<a href="{rel_path(path, output_dir)}">{html.escape(path.name)}</a>')
    more = f'<small>+ {len(paths) - limit} more files in the output folder</small>' if len(paths) > limit else ""
    return f"<section class=\"panel\"><h2>{html.escape(title)}</h2><div class=\"download-grid\">{''.join(links)}</div>{more}</section>"


def render_dashboard(input_dir: Path, output_dir: Path, title: str, subtitle: str, max_pair_cards: int) -> str:
    tables_dir = input_dir / "tables"
    snapshot_rows = sorted(read_csv_rows(tables_dir / "vector_snapshot_summary.csv"), key=snapshot_sort_key)
    pair_rows = read_csv_rows(tables_dir / "vector_pair_summary.csv")
    match_rows = read_csv_rows(tables_dir / "vector_field_matches.csv")
    event_rows = read_csv_rows(tables_dir / "vector_split_merge_events.csv")
    assets = find_assets(input_dir)
    summary = compute_summary(snapshot_rows, pair_rows)

    snapshot_labels = [snapshot_display(row.get("snapshot", "")) for row in snapshot_rows]
    snapshot_areas = [to_float(row.get("area_ha")) for row in snapshot_rows]
    pair_labels = [pair_label(row) for row in pair_rows]
    pair_short_labels = [label.replace("_", " ").replace(" -> ", " -> ") for label in pair_labels]
    net_area_values = [to_float(row.get("net_area_change_ha")) for row in pair_rows]
    iou_values = [to_float(row.get("matched_iou_area_weighted")) for row in pair_rows]
    event_counter = Counter(row.get("event_type", "unknown") for row in event_rows)

    kpis = "".join(
        [
            card("Snapshots", str(summary["snapshot_count"]), f"{summary['pair_count']} comparison periods", "teal"),
            card("Latest Field Area", f"{fmt_num(float(summary['latest_area']), 1)} ha", snapshot_display(str(summary["latest_snapshot"])), "green"),
            card("Net Area Change", f"{fmt_num(float(summary['net_area']), 1)} ha", fmt_pct(float(summary["net_area_pct"])), "orange"),
            card("Mean Matched IoU", fmt_num(float(summary["mean_iou"]), 3), "area-weighted pair average", "blue"),
            card("New Fields", fmt_num(float(summary["total_new"]), 0), "sum across pairs", "blue"),
            card("Disappeared Fields", fmt_num(float(summary["total_disappeared"]), 0), "sum across pairs", "red"),
            card("Split Candidates", fmt_num(float(summary["total_split"]), 0), "boundary fragmentation signal", "orange"),
            card("Merge Candidates", fmt_num(float(summary["total_merge"]), 0), "field consolidation signal", "purple"),
        ]
    )

    narrative = "".join(f"<li>{html.escape(item)}</li>" for item in build_narrative(summary))
    season_table = table_html(
        ["Season", "Snapshots", "Start Area Ha", "Latest Area Ha", "Net Ha", "Net %", "Mean IoU", "New", "Disappeared"],
        season_summary(snapshot_rows, pair_rows),
    )
    distribution_table = table_html(["Statistic", "Value"], distribution_rows(match_rows))
    event_table = table_html(
        ["Event Type", "Count"],
        [[event.replace("_", " ").title(), count] for event, count in sorted(event_counter.items())],
    )

    timeline_html = "".join(
        f'<figure><img src="{rel_path(path, output_dir)}" alt="{html.escape(path.name)}" loading="lazy" /><figcaption>{html.escape(path.name)}</figcaption></figure>'
        for path in assets["gifs"]
    ) or '<p class="muted">No timeline GIFs found.</p>'

    dashboard_png_html = "".join(
        f'<figure><img src="{rel_path(path, output_dir)}" alt="{html.escape(path.name)}" loading="lazy" /><figcaption>{html.escape(path.name)}</figcaption></figure>'
        for path in assets["dashboard_pngs"]
    )

    pair_cards = render_pair_cards(pair_rows, assets["pair_pngs"], input_dir, output_dir, max_pair_cards)
    data_links = asset_links("GeoJSON Downloads", assets["geojson_root"] + assets["geojson_pairs"], output_dir)

    generated_meta = {
        "input_dir": str(input_dir),
        "snapshot_rows": len(snapshot_rows),
        "pair_rows": len(pair_rows),
        "match_rows": len(match_rows),
        "event_rows": len(event_rows),
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #17211b;
      --muted: #637067;
      --paper: #fbfaf4;
      --panel: rgba(255, 255, 255, 0.86);
      --line: rgba(23, 33, 27, 0.12);
      --green: #2f7d32;
      --teal: #0f766e;
      --blue: #22577a;
      --red: #b42318;
      --orange: #b45309;
      --purple: #6d28d9;
      --shadow: 0 24px 70px rgba(20, 37, 28, 0.15);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(61, 132, 79, 0.22), transparent 34rem),
        radial-gradient(circle at 80% 12%, rgba(233, 179, 76, 0.2), transparent 28rem),
        linear-gradient(135deg, #f8f2df 0%, #eef5ed 45%, #f6fbf6 100%);
      min-height: 100vh;
    }}
    header {{ padding: 56px clamp(18px, 5vw, 72px) 28px; }}
    .hero {{
      max-width: 1280px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 28px;
      align-items: stretch;
    }}
    .hero-main, .panel, .kpi, .pair-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      border-radius: 28px;
    }}
    .hero-main {{ padding: 34px; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: 0.14em; color: var(--green); font-weight: 800; font-size: 0.78rem; }}
    h1 {{ font-size: clamp(2.4rem, 6vw, 5.6rem); line-height: 0.92; margin: 16px 0; letter-spacing: -0.07em; }}
    h2 {{ margin: 0 0 16px; font-size: clamp(1.35rem, 2.6vw, 2rem); letter-spacing: -0.03em; }}
    h3 {{ margin: 0 0 12px; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .hero-side {{ padding: 28px; }}
    .hero-side ul {{ margin: 0; padding-left: 20px; color: var(--muted); line-height: 1.7; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 0 clamp(18px, 5vw, 72px) 72px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 22px 0; }}
    .kpi {{ padding: 20px; min-height: 142px; }}
    .kpi span {{ display: block; color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 800; }}
    .kpi strong {{ display: block; font-size: clamp(1.7rem, 3vw, 2.7rem); margin: 14px 0 6px; letter-spacing: -0.04em; }}
    .kpi small {{ color: var(--muted); }}
    .kpi.green strong {{ color: var(--green); }}
    .kpi.teal strong {{ color: var(--teal); }}
    .kpi.blue strong {{ color: var(--blue); }}
    .kpi.red strong {{ color: var(--red); }}
    .kpi.orange strong {{ color: var(--orange); }}
    .kpi.purple strong {{ color: var(--purple); }}
    .panel {{ padding: 26px; margin: 22px 0; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }}
    .chart-card {{ background: rgba(255,255,255,0.62); border: 1px solid var(--line); border-radius: 22px; padding: 18px; }}
    .chart-svg {{ width: 100%; display: block; }}
    .chart-bg {{ fill: rgba(255,255,255,0.62); }}
    .grid {{ stroke: rgba(23,33,27,0.1); stroke-width: 1; }}
    .axis {{ stroke: rgba(23,33,27,0.35); stroke-width: 1.2; }}
    .axis-label {{ fill: #68756c; font-size: 12px; }}
    .empty-chart {{ fill: #68756c; font-size: 16px; }}
    .chart-legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 8px; color: var(--muted); font-size: 0.9rem; }}
    .chart-legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 4px; margin-right: 6px; vertical-align: -1px; }}
    .table-wrap {{ overflow-x: auto; border-radius: 18px; border: 1px solid var(--line); background: rgba(255,255,255,0.62); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 720px; }}
    th, td {{ text-align: left; padding: 13px 14px; border-bottom: 1px solid var(--line); }}
    th {{ color: #344238; background: rgba(47,125,50,0.08); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    td {{ color: var(--muted); }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; border-radius: 20px; border: 1px solid var(--line); background: white; }}
    figcaption {{ margin-top: 8px; color: var(--muted); font-size: 0.9rem; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .pair-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .pair-card {{ padding: 18px; box-shadow: none; }}
    .pair-card h4 {{ margin: 0 0 8px; }}
    .pair-card img {{ margin-top: 12px; width: 100%; border-radius: 16px; border: 1px solid var(--line); }}
    .download-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .download-grid a {{ display: block; padding: 12px; border-radius: 14px; background: rgba(47,125,50,0.08); color: #255b2b; text-decoration: none; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .muted {{ color: var(--muted); }}
    footer {{ max-width: 1280px; margin: 0 auto; padding: 12px clamp(18px, 5vw, 72px) 48px; color: var(--muted); }}
    @media (max-width: 960px) {{
      .hero, .grid-2, .media-grid, .pair-grid {{ grid-template-columns: 1fr; }}
      .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .download-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <section class="hero">
      <div class="hero-main">
        <div class="eyebrow">Field delineation change intelligence</div>
        <h1>{html.escape(title)}</h1>
        <p>{html.escape(subtitle)}</p>
      </div>
      <aside class="hero-side hero-main">
        <h2>Executive Brief</h2>
        <ul>{narrative}</ul>
      </aside>
    </section>
  </header>
  <main>
    <section class="kpi-grid">{kpis}</section>
    <section class="panel">
      <h2>Change Trends</h2>
      <div class="grid-2">
        <div class="chart-card"><h3>Field Area Timeline</h3>{svg_line_chart(snapshot_labels, snapshot_areas, '#2f7d32', ' ha')}</div>
        <div class="chart-card"><h3>Net Area Change by Period</h3>{svg_bar_chart(pair_short_labels, net_area_values)}</div>
        <div class="chart-card"><h3>Spatial Persistence: Area-Weighted IoU</h3>{svg_line_chart(pair_short_labels, iou_values, '#22577a')}</div>
        <div class="chart-card"><h3>Field Turnover Events</h3>{svg_stacked_event_chart(pair_short_labels, pair_rows)}</div>
      </div>
    </section>
    <section class="panel">
      <h2>Season-Level Summary</h2>
      {season_table}
    </section>
    <section class="panel grid-2">
      <div>
        <h2>Matched Field Distribution</h2>
        {distribution_table}
      </div>
      <div>
        <h2>Event Inventory</h2>
        {event_table}
      </div>
    </section>
    <section class="panel">
      <h2>Visual Timelines</h2>
      <div class="media-grid">{timeline_html}</div>
    </section>
    <section class="panel">
      <h2>Generated Summary Figures</h2>
      <div class="media-grid">{dashboard_png_html or '<p class="muted">No summary PNGs found.</p>'}</div>
    </section>
    <section class="panel">
      <h2>Pair Diagnostics</h2>
      <div class="pair-grid">{pair_cards}</div>
    </section>
    {data_links}
  </main>
  <footer>
    Generated from <code>{html.escape(str(input_dir))}</code>. Metadata: <script type="application/json" id="dashboard-meta">{html.escape(json.dumps(generated_meta))}</script>
  </footer>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "dashboard")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_text = render_dashboard(input_dir, output_dir, args.title, args.subtitle, args.max_pair_cards)
    out_path = output_dir / "index.html"
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Dashboard written to: {out_path}")


if __name__ == "__main__":
    main()
