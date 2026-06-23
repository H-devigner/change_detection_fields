# Kursh Dynamic World LULC Context Change

Standalone sample processor for Dynamic World LULC context around field-change
work. This is not a replacement for field-delineation geometry change detection:
use field polygons for split/merge/appearance/disappearance, and use this LULC
processor as an auxiliary layer to summarize class context across years.

If both seasonal folders in a year contain the same yearly LULC raster, same-year
season-to-season LULC comparisons are expected to be identical. The default
comparison mode therefore compares the same season across years.

## Expected Inputs

Place rasters in:

```text
data/raw/kursh_dw_lulc/
```

Default naming pattern:

```text
kursh_2021_february_april__dw_lulc.tif
kursh_2021_june_august__dw_lulc.tif
kursh_2022_february_april__dw_lulc.tif
kursh_2022_june_august__dw_lulc.tif
kursh_2023_february_april__dw_lulc.tif
kursh_2023_june_august__dw_lulc.tif
```

The processor also supports the same names as directories, for example:

```text
data/raw/kursh_dw_lulc/
  kursh_2021_february_april__dw_lulc/
    <any .tif/.tiff/.vrt inside>
```

## Setup

```bash
cd /Users/houcine/Desktop/from_oci/Delineate-Anything_just_folders_keeper/kursh_dw_lulc_change_detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run

```bash
python scripts/sample_dw_lulc_change_processor.py \
  --input-dir data/raw/kursh_dw_lulc \
  --years 2021,2022,2023 \
  --seasons february_april,june_august \
  --filename-template "kursh_{year}_{season}__dw_lulc" \
  --pair-mode same-season-yearly \
  --output-dir data/processed/kursh_2021_2023_dw_lulc_sample
```

If your snapshots live as run directories under another folder:

```bash
python scripts/sample_dw_lulc_change_processor.py \
  --input-dir /mnt/KSA-Oasis/houcine/field_delineation_data/field_delineation_runs \
  --years 2021,2022,2023 \
  --seasons february_april,june_august \
  --filename-template "kursh_{year}_{season}__dw_lulc" \
  --recursive \
  --pair-mode same-season-yearly \
  --preview-max-size 2048 \
  --output-dir data/processed/kursh_2021_2023_dw_lulc_sample
```

Default comparisons with `--pair-mode same-season-yearly`:

```text
2021_february_april -> 2022_february_april
2022_february_april -> 2023_february_april
2021_june_august -> 2022_june_august
2022_june_august -> 2023_june_august
```

Other pair modes:

```text
same-season-yearly    consecutive years for each season; default
same-season-all-years all cross-year combinations within each season
adjacent              original snapshot order, including same-year season pairs
all                   every from-to pair in snapshot order
```

GeoTIFF outputs are written as tiled BigTIFF files, so large rasters above the classic
4 GB TIFF limit are supported. PNG figures are downsampled previews by default with
`--preview-max-size 2048`; use `--preview-max-size 0` only if you explicitly need
full-resolution PNG rendering.

The script logs progress by default. For a fast diagnostic run that only writes CSV tables:

```bash
python scripts/sample_dw_lulc_change_processor.py \
  --input-dir /mnt/KSA-Oasis/houcine/field_delineation_data/field_delineation_runs \
  --years 2021,2022,2023 \
  --seasons february_april,june_august \
  --filename-template "kursh_{year}_{season}__dw_lulc" \
  --recursive \
  --pair-mode same-season-yearly \
  --skip-figures \
  --skip-rasters \
  --output-dir data/processed/kursh_2021_2023_dw_lulc_sample_fast_check
```

Use `--debug` for extra detail or `--quiet` to show only warnings/errors.

## Outputs

```text
data/processed/kursh_2021_2023_dw_lulc_sample/
  manifest.json
  tables/
    snapshot_class_counts.csv
    pair_summary.csv
    transition_matrix_long.csv
  rasters/
    *_changed_mask.tif
    *_crop_change_state.tif
    *_transition_code.tif
  figures/
    snapshots/*.png
    pairs/*.png
    change_trend.png
```
