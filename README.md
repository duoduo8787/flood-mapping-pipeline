# Flood Mapping Pipeline

End-to-end flood extent mapping using multi-sensor satellite imagery, GloFAS analogue search, and EDL-based segmentation with uncertainty quantification. Covers 6 HydroBASINS watersheds in South Asia.

---

## Team & Division of Labor

| Member | Role | Responsibilities |
|--------|------|-----------------|
| 陳麒如 (Chi-Ju Chen) | Data Engineering | Steps 1–2: GloFAS discharge retrieval & analogue search (RMSE) |
| 李碩宸 (Shuo-Chen Lee) | ML Pipeline | Steps 3–7: Zarr ETL pipeline, EDL inference integration, data fusion, chunking benchmark & STAC catalog |
| 丁俊瑋 (Junwei Ding) | Analysis & Docs | Step 8: Downstream analysis, result visualization & documentation |

---

## Pipeline Overview

```
[Step 1-2]  GloFAS discharge data  ──►  Analogue search (RMSE)
                                             │
                                             ▼
[Step 3]   analogue_results.zarr   ◄── build_zarr_step3.py
             (matched dates, RMSE scores, RMSE maps)
                                             │
                                             ▼
[Step 4]   satellite_images.zarr   ◄── build_zarr_step4.py
             (Sentinel-1/2 + Landsat, 6 basins × RP × Rank)
                                             │
                                             ▼
[Step 5]   flood_predictions.zarr  ◄── EDL inference (NAS server)
             (classification, water_probability, uncertainty × 7 vars)
                                             │
                                             ▼
[Step 6]   flood_extent_merged.zarr ◄── data_merge.py
             (optical+SAR fusion via 1/uncertainty weighting)
```

---

## Setup

### Prerequisites

- Python 3.13 (via [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [uv](https://github.com/astral-sh/uv) package manager
- Node.js (for STAC Browser)

### Install dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync

# Or install directly
uv pip install -r pyproject.toml
```

---

## Data Layout

```
EnvBigData/
├── flood_predictions/all/         # EDL inference TIF outputs (Step 5 input)
│   └── {basinID}/{RP}/{sensor}/   # *_output_EDL.tif (7 bands)
├── satellite_images/              # Raw satellite TIFs (Step 4 input)
│   └── {basinID}/{RP2|5|10}/      # *_{L|S1|S2}_*.tif
├── rmse_maps/                     # RMSE spatial maps (Step 3 input)
│   └── basin_{ID}_rp{N}_rmse.tif
├── analogue_search_results.json   # Top-K analogue metadata (Step 3 input)
├── hybas_si_lev01-12_v1c/         # HydroBASINS shapefiles
│   └── hybas_si_lev08_v1c.shp     # Level-8 basin boundaries
│
├── analogue_results.zarr          # Step 3 output
├── satellite_images.zarr          # Step 4 output
├── flood_predictions.zarr         # Step 5 output
└── flood_extent_merged.zarr       # Step 6 final output
```

Six basin IDs: `3080572620`, `3080572630`, `3080576250`, `3080576260`, `3080580980`, `3080585700`

Return periods: RP=2, RP=5, RP=10 (years) | Top-K analogues: rank 1–5

---

## Scripts

### Step 3 — Analogue Results Zarr

```bash
python3 scripts/step3_build_analogue_zarr.py
```

Reads `analogue_search_results.json` and RMSE map TIFs. Outputs `analogue_results.zarr` with variables:
- `matched_datetime` — matched flood event date (basin × RP × rank)
- `rmse_score` — similarity score (basin × RP × rank)
- `rmse_map` — spatial RMSE map (basin × RP × y × x)

### Step 4 — Satellite Images Zarr

```bash
python3 scripts/step4_build_satellite_zarr.py
```

Clips and mosaics raw satellite TIFs to a unified 30 m grid. Outputs `satellite_images.zarr`:
- `optical`: (RP=3, rank=5, channel=6, y=2508, x=3188) — bands: blue/green/red/nir/swir1/swir2
- `sar`: (RP=3, rank=5, channel=2, y=2508, x=3188) — bands: VV/VH

### Step 5 — EDL Flood Predictions Zarr

```bash
python3 scripts/step5_build_predictions_zarr.py
```

Mosaics EDL TIF outputs into `flood_predictions.zarr`:
- Variables: `classification`, `water_probability`, `dst_uncertainty`, `evidence_neg`, `evidence_pos`, `aleatoric`, `epistemic`
- Dimensions: (sensor_type=2, RP=3, rank=5, y=2508, x=3188)

### Step 6 — Data Merge (Final Flood Map)

```bash
python3 scripts/step6_data_merge.py
```

Fuses optical + SAR predictions using uncertainty-weighted average (weight = 1/dst_uncertainty). Outputs `flood_extent_merged.zarr`:
- `water_probability` — fused probability (RP=3, rank=5, y=2508, x=3188)
- `classification` — binary flood map (threshold = 0.5)
- `uncertainty` — fused uncertainty

---

## Performance Benchmarking

```bash
python3 scripts/benchmark_chunking.py
```

Compares three chunking strategies across three query patterns:

| Strategy | Chunk Shape | Best For |
|----------|-------------|---------|
| A | (1,1,1,512,512) | Single scene reads, basin bbox queries |
| B | (2,3,5,128,128) | Full temporal slice (all sensors/RPs) |
| C | (1,1,1,2508,3188) | Data Merge (all ranks in one shot) |

**Selected strategy: A** — best balance for primary use cases (scene reads + spatial queries). 512×512 aligns with smallest basin footprint (~652×716 px), ensuring each chunk covers a semantically meaningful area.

Results saved to `results/benchmark_results.png` and `results/benchmark_results.csv`.

---

## Downstream Analysis

```bash
python3 scripts/downstream_analysis.py
```

Generates `results/`:
- `fig1_speed_comparison.png` — Read speed: raw TIF (30 files) vs zarr (~6× speedup)
- `fig2_flood_maps.png` — Flood agreement and uncertainty maps for RP=2/5/10

---

## STAC Catalog

### Build catalog

```bash
python3 scripts/build_stac.py
```

Generates `stac/` directory with STAC 1.1.0 catalog describing all 4 zarr assets.

### Serve and browse

```bash
# Terminal 1: start CORS-enabled file server (port 8888)
python3 scripts/serve_stac.py

# Terminal 2: start STAC Browser
cd stac-browser
npm install     # first time only
npm run dev
```

Open `http://localhost:5173` in your browser. The catalog URL is pre-configured in `stac-browser/config.js`.

---

## Notes

- **Missing imagery**: RP=2 rank=4 and RP=10 rank=3/4 have no valid pixels (no satellite pass during flood). These ranks are excluded from area statistics.
- **Coordinate system**: All zarr stores use EPSG:4326 at 30 m resolution (TARGET_RES = 0.0002694946°).
- **EDL inference** was run on a NAS server (`/home/NAS/homes/cjchen-10025/ML4FloodsUncertainty/`); the output TIFs are the starting point for this repository.
