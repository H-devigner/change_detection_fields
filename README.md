# Kursh Dynamic World LULC Change Detection

Standalone sample processor for seasonal Dynamic World LULC raster change detection.

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
pip install -e .
```

## Run

```bash
python scripts/sample_dw_lulc_change_processor.py \
  --input-dir data/raw/kursh_dw_lulc \
  --years 2021,2022,2023 \
  --seasons february_april,june_august \
  --filename-template "kursh_{year}_{season}__dw_lulc" \
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
  --output-dir data/processed/kursh_2021_2023_dw_lulc_sample
```

## Outputs

```text
data/processed/kursh_2021_2023_dw_lulc_sample/
  manifest.json
  tables/
    snapshot_class_counts.csv
    adjacent_pair_summary.csv
    adjacent_transition_matrix_long.csv
  rasters/
    *_changed_mask.tif
    *_crop_change_state.tif
    *_transition_code.tif
  figures/
    snapshots/*.png
    pairs/*.png
    adjacent_change_trend.png
```
