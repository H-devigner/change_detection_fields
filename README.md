# Kursh Agricultural Field Change Detection

Standalone sample processor for agricultural field detection rasters. It compares
field presence through time and exports field area, stable field, field gain, and
field loss products.

By default, every non-zero valid pixel is treated as field. This supports binary
field masks and rasters where each delineated field has a separate positive
object ID. If your field raster uses a specific code, use `--field-values`.

This raster workflow tracks field extent changes. Exact field-object tracking,
splits, and merges require vector polygon matching, because a binary raster mask
does not preserve individual field identity across snapshots.

## Expected Inputs

Place field rasters in:

```text
data/raw/kursh_fields/
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
python scripts/sample_field_change_processor.py \
  --input-dir data/raw/kursh_fields \
  --years 2021,2022,2023 \
  --seasons february_april,june_august \
  --filename-template "kursh_{year}_{season}__dw_lulc" \
  --pair-mode adjacent \
  --output-dir data/processed/kursh_2021_2023_field_sample
```

If your snapshots live as run directories under another folder:

```bash
python scripts/sample_field_change_processor.py \
  --input-dir /mnt/KSA-Oasis/houcine/field_delineation_data/field_delineation_runs \
  --years 2021,2022,2023 \
  --seasons february_april,june_august \
  --filename-template "kursh_{year}_{season}__dw_lulc" \
  --recursive \
  --pair-mode adjacent \
  --preview-max-size 2048 \
  --output-dir data/processed/kursh_2021_2023_field_sample
```

Default comparisons with `--pair-mode adjacent`:

```text
2021_february_april -> 2021_june_august
2021_june_august -> 2022_february_april
2022_february_april -> 2022_june_august
2022_june_august -> 2023_february_april
2023_february_april -> 2023_june_august
```

Other pair modes:

```text
adjacent              chronological snapshot order; default
same-season-yearly    consecutive years for each season
same-season-all-years all cross-year combinations within each season
all                   every from-to pair in snapshot order
```

GeoTIFF outputs are written as tiled BigTIFF files, so large rasters above the classic
4 GB TIFF limit are supported. PNG figures are downsampled previews by default with
`--preview-max-size 2048`; use `--preview-max-size 0` only if you explicitly need
full-resolution PNG rendering.

If only value `1` should be treated as field:

```bash
--field-values 1
```

The script logs progress by default. For a fast diagnostic run that only writes CSV tables:

```bash
python scripts/sample_field_change_processor.py \
  --input-dir /mnt/KSA-Oasis/houcine/field_delineation_data/field_delineation_runs \
  --years 2021,2022,2023 \
  --seasons february_april,june_august \
  --filename-template "kursh_{year}_{season}__dw_lulc" \
  --recursive \
  --pair-mode adjacent \
  --skip-figures \
  --skip-rasters \
  --output-dir data/processed/kursh_2021_2023_field_sample_fast_check
```

Use `--debug` for extra detail or `--quiet` to show only warnings/errors.

The older `scripts/sample_dw_lulc_change_processor.py` entry point is still kept
for compatibility, but the field-specific wrapper above is preferred.

## Outputs

```text
data/processed/kursh_2021_2023_field_sample/
  manifest.json
  tables/
    snapshot_field_summary.csv
    pair_summary.csv
    field_transition_matrix_long.csv
    transition_matrix_long.csv
  rasters/
    *_changed_mask.tif
    *_field_change_state.tif
    *_field_transition_code.tif
  figures/
    snapshots/*.png
    pairs/*.png
    change_trend.png
```
