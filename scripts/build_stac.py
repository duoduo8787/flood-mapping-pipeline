"""
build_stac.py
=============
建立符合 STAC 1.0 規範的 micro catalog，描述本專案所有 zarr 資產。

輸出結構：
  stac/
    catalog.json                       ← 根目錄
    flood-mapping-pipeline/
      collection.json                  ← 主集合
      satellite-images/item.json
      flood-predictions/item.json
      analogue-results/item.json
      flood-extent-merged/item.json
"""

import json
import datetime
from pathlib import Path
import pystac
from pystac import (
    Catalog, Collection, Item, Asset,
    Extent, SpatialExtent, TemporalExtent,
    MediaType, Link,
)

BASE_DIR  = Path(__file__).parent
STAC_DIR  = BASE_DIR / "stac"
STAC_DIR.mkdir(exist_ok=True)

# ── 共用空間範圍（全 6 集水區 mosaic）──────────────────────────────────────
BBOX      = [68.9710, 54.6037, 69.8298, 55.2793]   # [west, south, east, north]
GEOMETRY  = {
    "type": "Polygon",
    "coordinates": [[
        [BBOX[0], BBOX[1]],
        [BBOX[2], BBOX[1]],
        [BBOX[2], BBOX[3]],
        [BBOX[0], BBOX[3]],
        [BBOX[0], BBOX[1]],
    ]],
}

# ── 時間範圍（類比搜尋涵蓋的歷史區間）──────────────────────────────────────
T_START = datetime.datetime(2002, 4, 26, tzinfo=datetime.timezone.utc)
T_END   = datetime.datetime(2025, 4, 11, tzinfo=datetime.timezone.utc)


# ══════════════════════════════════════════════════════════════════════════
# 建立四個 STAC Item
# ══════════════════════════════════════════════════════════════════════════
def make_item_satellite_images():
    item = Item(
        id="satellite-images",
        geometry=GEOMETRY,
        bbox=BBOX,
        datetime=T_START,
        properties={
            "title": "Historical Satellite Imagery (Optical + SAR)",
            "description": (
                "Multi-sensor historical satellite imagery for 6 HydroBASINS watersheds. "
                "optical: best-available Sentinel-2 or Landsat (6 bands: blue/green/red/nir/swir1/swir2). "
                "sar: Sentinel-1 (VV, VH). "
                "Dimensions: (return_period=3) × (rank=5) × (channel) × (y=2508) × (x=3188)."
            ),
            "datetime": T_START.isoformat(),
            "end_datetime": T_END.isoformat(),
            "platform": ["Sentinel-2", "Landsat-8", "Landsat-9", "Sentinel-1"],
            "instruments": ["MSI", "OLI", "C-SAR"],
            "return_periods": [2, 5, 10],
            "top_k_analogues": 5,
            "crs": "EPSG:4326",
            "resolution_m": 30,
            "zarr_dims": "(return_period=3) x (rank=5) x (channel) x (y=2508) x (x=3188)",
            "optical_channels": ["blue", "green", "red", "nir", "swir1", "swir2"],
            "sar_channels": ["VV", "VH"],
            "chunk_shape": "(1,1,1,512,512)",
            "pipeline_step": "Step 4",
        },
    )
    item.add_asset("data", Asset(
        href="../data/satellite_images.zarr",
        title="satellite_images.zarr",
        media_type="application/vnd+zarr",
        roles=["data"],
        extra_fields={"zarr:format": 3, "zarr:variables": ["optical", "sar"]},
    ))
    return item


def make_item_flood_predictions():
    item = Item(
        id="flood-predictions",
        geometry=GEOMETRY,
        bbox=BBOX,
        datetime=T_START,
        properties={
            "title": "EDL Flood Prediction Outputs (Optical + SAR)",
            "description": (
                "Evidential Deep Learning (EDL) flood segmentation outputs with uncertainty quantification. "
                "Two sensor types (optical, SAR), 3 return periods, 5 analogue ranks. "
                "Dimensions: (sensor_type=2) × (return_period=3) × (rank=5) × (y=2508) × (x=3188)."
            ),
            "datetime": T_START.isoformat(),
            "end_datetime": T_END.isoformat(),
            "model": "EDL (Evidential Deep Learning)",
            "sensor_types": ["optical", "sar"],
            "return_periods": [2, 5, 10],
            "top_k_analogues": 5,
            "crs": "EPSG:4326",
            "resolution_m": 30,
            "zarr_dims": "(sensor_type=2) x (return_period=3) x (rank=5) x (y=2508) x (x=3188)",
            "variables": [
                "classification", "water_probability", "dst_uncertainty",
                "evidence_neg", "evidence_pos", "aleatoric", "epistemic"
            ],
            "chunk_shape": "(1,1,1,512,512)",
            "pipeline_step": "Step 5",
        },
    )
    item.add_asset("data", Asset(
        href="../data/flood_predictions.zarr",
        title="flood_predictions.zarr",
        media_type="application/vnd+zarr",
        roles=["data"],
        extra_fields={
            "zarr:format": 3,
            "zarr:variables": [
                "classification", "water_probability", "dst_uncertainty",
                "evidence_neg", "evidence_pos", "aleatoric", "epistemic"
            ],
        },
    ))
    return item


def make_item_analogue_results():
    item = Item(
        id="analogue-results",
        geometry=GEOMETRY,
        bbox=BBOX,
        datetime=T_START,
        properties={
            "title": "Analogue Search Results + RMSE Maps",
            "description": (
                "Top-K historically analogous flood events per basin/return period, "
                "retrieved via RMSE-based similarity search against GloFAS historical discharge. "
                "Includes matched datetimes, RMSE scores, and spatial RMSE maps. "
                "Metadata dims: (basin=6) × (return_period=3) × (rank=5). "
                "Spatial dims: (basin=6) × (return_period=3) × (y=2508) × (x=3188)."
            ),
            "datetime": T_START.isoformat(),
            "end_datetime": T_END.isoformat(),
            "method": "RMSE-based analogue search",
            "basins": [
                3080572620, 3080572630, 3080576250,
                3080576260, 3080580980, 3080585700
            ],
            "return_periods": [2, 5, 10],
            "top_k": 5,
            "crs": "EPSG:4326",
            "resolution_m": 30,
            "variables": ["matched_datetime", "rmse_score", "rmse_map"],
            "pipeline_step": "Step 3",
        },
    )
    item.add_asset("data", Asset(
        href="../data/analogue_results.zarr",
        title="analogue_results.zarr",
        media_type="application/vnd+zarr",
        roles=["data"],
        extra_fields={
            "zarr:format": 3,
            "zarr:variables": ["matched_datetime", "rmse_score", "rmse_map"],
        },
    ))
    item.add_asset("metadata", Asset(
        href="../analogue_search_results.json",
        title="analogue_search_results.json",
        media_type=MediaType.JSON,
        roles=["metadata"],
    ))
    return item


def make_item_flood_extent_merged():
    item = Item(
        id="flood-extent-merged",
        geometry=GEOMETRY,
        bbox=BBOX,
        datetime=T_START,
        properties={
            "title": "Merged Flood Extent Map (Final Output)",
            "description": (
                "Final flood extent map produced by uncertainty-weighted fusion of optical and SAR EDL outputs. "
                "Provides water probability, binary classification, and combined uncertainty "
                "per return period and analogue rank. "
                "Dimensions: (return_period=3) × (rank=5) × (y=2508) × (x=3188)."
            ),
            "datetime": T_START.isoformat(),
            "end_datetime": T_END.isoformat(),
            "fusion_method": "uncertainty-weighted average (1/dst_uncertainty)",
            "th_water": 0.5,
            "return_periods": [2, 5, 10],
            "top_k_analogues": 5,
            "crs": "EPSG:4326",
            "resolution_m": 30,
            "zarr_dims": "(return_period=3) x (rank=5) x (y=2508) x (x=3188)",
            "variables": ["water_probability", "classification", "uncertainty"],
            "chunk_shape": "(1,1,512,512)",
            "pipeline_step": "Step 6 – Data Merge (Final Output)",
        },
    )
    item.add_asset("data", Asset(
        href="../data/flood_extent_merged.zarr",
        title="flood_extent_merged.zarr",
        media_type="application/vnd+zarr",
        roles=["data"],
        extra_fields={
            "zarr:format": 3,
            "zarr:variables": ["water_probability", "classification", "uncertainty"],
        },
    ))
    return item


# ══════════════════════════════════════════════════════════════════════════
# 組裝 Catalog + Collection
# ══════════════════════════════════════════════════════════════════════════
def build_catalog():
    catalog = Catalog(
        id="flood-mapping-pipeline-catalog",
        description=(
            "STAC Catalog for the End-to-End Flood Mapping Pipeline. "
            "Covers 6 HydroBASINS watersheds in South Asia using "
            "GloFAS analogue search, Sentinel-1/2 & Landsat imagery, "
            "and EDL-based flood segmentation with uncertainty quantification."
        ),
        title="Flood Mapping Pipeline Catalog",
    )

    extent = Extent(
        SpatialExtent(bboxes=[BBOX]),
        TemporalExtent(intervals=[[T_START, T_END]]),
    )

    collection = Collection(
        id="flood-mapping-pipeline",
        description=(
            "Multi-sensor flood extent dataset for 6 HydroBASINS watersheds. "
            "Pipeline: GloFAS analogue search → satellite imagery retrieval → "
            "EDL AI segmentation → optical+SAR fusion → final flood extent map."
        ),
        title="Flood Mapping Pipeline – South Asia",
        extent=extent,
        license="proprietary",
        extra_fields={
            "keywords": [
                "flood", "SAR", "Sentinel-1", "Sentinel-2", "Landsat",
                "HydroBASINS", "GloFAS", "EDL", "uncertainty", "zarr"
            ],
            "providers": [{"name": "CIE 5158 Research Team", "roles": ["producer"]}],
            "basin_ids": [
                3080572620, 3080572630, 3080576250,
                3080576260, 3080580980, 3080585700
            ],
            "return_periods": [2, 5, 10],
            "resolution_m": 30,
            "crs": "EPSG:4326",
        },
    )

    items = [
        make_item_satellite_images(),
        make_item_flood_predictions(),
        make_item_analogue_results(),
        make_item_flood_extent_merged(),
    ]
    for item in items:
        collection.add_item(item)

    catalog.add_child(collection)
    return catalog


# ══════════════════════════════════════════════════════════════════════════
# 儲存
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("建立 STAC catalog ...")
    catalog = build_catalog()

    if STAC_DIR.exists():
        import shutil
        shutil.rmtree(STAC_DIR)

    catalog.normalize_hrefs(str(STAC_DIR))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    print(f"✓ 儲存完成：{STAC_DIR}")
    print()

    # 列出產生的檔案
    for f in sorted(STAC_DIR.rglob("*.json")):
        rel = f.relative_to(STAC_DIR)
        size = f.stat().st_size
        print(f"  {rel}  ({size} bytes)")


if __name__ == "__main__":
    main()
