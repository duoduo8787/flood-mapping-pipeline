# Data Assets — Not Included in This Package

The following large data files are excluded from this submission package
due to size constraints. They are available in the FullData version.

## Zarr Stores (cloud-native, pre-built)

| File | Size | Description |
|------|------|-------------|
| `data/analogue_results.zarr` | ~132 MB | Analogue search results (Step 3) |
| `data/satellite_images.zarr` | ~803 MB | Raw satellite imagery (Step 4) |
| `data/flood_predictions.zarr` | ~1.4 GB | EDL model outputs (Step 5) |
| `data/flood_extent_merged.zarr` | ~604 MB | Final flood extent (Step 6) |

## Rebuild Instructions

```bash
uv sync
python3 scripts/step3_build_analogue_zarr.py
python3 scripts/step4_build_satellite_zarr.py
python3 scripts/step5_build_predictions_zarr.py
python3 scripts/step6_data_merge.py
```
