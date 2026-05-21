"""
build_zarr_step3.py
===================
將 Analogue 搜尋結果儲存為 zarr，包含兩個部分：

1. analogue_metadata zarr
   維度：(basin=6, return_period=3, rank=5)
   變數：matched_datetime (string), rmse_score (float32)

2. rmse_maps zarr
   維度：(basin=6, return_period=3, y, x)
   變數：rmse (float32)  — 6個集水區 mosaic 至統一 30m grid
"""

import json
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from pathlib import Path
from shapely.geometry import mapping

# ── 路徑設定 ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
ANALOGUE_JSON  = BASE_DIR / "analogue_search_results.json"
RMSE_DIR       = BASE_DIR / "rmse_maps"
SHP_PATH       = BASE_DIR / "hybas_si_lev01-12_v1c" / "hybas_si_lev08_v1c.shp"
OUTPUT_ZARR    = BASE_DIR / "data" / "analogue_results.zarr"

# ── 常數 ──────────────────────────────────────────────────────────────────
BASIN_IDS  = [3080572620, 3080572630, 3080576250, 3080576260, 3080580980, 3080585700]
RPS        = [2, 5, 10]
K_MAX      = 5
TARGET_RES = 0.0002694946  # ~30m


# ══════════════════════════════════════════════════════════════════════════
# Part 1: analogue metadata (datetime + score)
# ══════════════════════════════════════════════════════════════════════════
def build_analogue_metadata():
    print("Part 1: 建立 analogue metadata arrays...")
    with open(ANALOGUE_JSON) as f:
        data = json.load(f)

    n_basin = len(BASIN_IDS)
    n_rp    = len(RPS)
    # datetime 以字串儲存（U25 = 最多25字元）
    datetimes = np.full((n_basin, n_rp, K_MAX), "", dtype="U25")
    scores    = np.full((n_basin, n_rp, K_MAX), np.nan, dtype=np.float32)

    for bi, basin_id in enumerate(BASIN_IDS):
        basin_str = str(basin_id)
        if basin_str not in data:
            continue
        for ri, rp in enumerate(RPS):
            rp_str = str(rp)
            if rp_str not in data[basin_str]:
                continue
            for ki, entry in enumerate(data[basin_str][rp_str][:K_MAX]):
                datetimes[bi, ri, ki] = entry.get("datetime", "")
                scores[bi, ri, ki]    = float(entry.get("score", np.nan))

    print(f"  填入 {(datetimes != '').sum()} 筆有效時間點")
    return datetimes, scores


# ══════════════════════════════════════════════════════════════════════════
# Part 2: RMSE 空間圖 mosaic
# ══════════════════════════════════════════════════════════════════════════
def load_basins():
    gdf = gpd.read_file(SHP_PATH)
    return gdf[gdf["HYBAS_ID"].isin(BASIN_IDS)].set_index("HYBAS_ID")


def build_target_grid(basins):
    minx, miny, maxx, maxy = basins.total_bounds
    width  = int(np.ceil((maxx - minx) / TARGET_RES))
    height = int(np.ceil((maxy - miny) / TARGET_RES))
    transform = from_bounds(minx, miny, minx + width * TARGET_RES,
                            miny + height * TARGET_RES, width, height)
    xs = np.linspace(minx + TARGET_RES/2, minx + (width  - 0.5) * TARGET_RES, width)
    ys = np.linspace(miny + (height - 0.5) * TARGET_RES, miny + TARGET_RES/2, height)
    print(f"  全域 grid: {height}r × {width}c")
    return transform, width, height, xs, ys


def build_rmse_mosaic(basins, target_transform, width, height):
    print("Part 2: mosaic RMSE 空間圖...")
    n_basin, n_rp = len(BASIN_IDS), len(RPS)
    rmse_arr = np.full((n_basin, n_rp, height, width), np.nan, dtype=np.float32)

    for bi, basin_id in enumerate(BASIN_IDS):
        basin_geom = basins.loc[basin_id, "geometry"]
        for ri, rp in enumerate(RPS):
            tif_path = RMSE_DIR / f"basin_{basin_id}_rp{rp}_rmse.tif"
            if not tif_path.exists():
                print(f"  ⚠ 找不到 {tif_path.name}，跳過")
                continue

            with rasterio.open(tif_path) as src:
                try:
                    clipped, clipped_tf = rio_mask(
                        src,
                        [mapping(basin_geom)],
                        crop=True,
                        filled=False,
                        all_touched=True,
                    )
                except Exception as e:
                    print(f"  ⚠ clip 失敗 ({basin_id}, rp{rp}): {e}")
                    continue

                band = clipped[0].astype(np.float32)
                if hasattr(band, "fill_value"):
                    band = band.filled(np.nan)

                out = np.full((height, width), np.nan, dtype=np.float32)
                reproject(
                    source=band,
                    destination=out,
                    src_transform=clipped_tf,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=rasterio.crs.CRS.from_epsg(4326),
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )

                mask_new  = ~np.isnan(out)
                mask_both = ~np.isnan(rmse_arr[bi, ri]) & mask_new
                rmse_arr[bi, ri][mask_new & ~mask_both] = out[mask_new & ~mask_both]
                rmse_arr[bi, ri][mask_both] = (rmse_arr[bi, ri][mask_both] + out[mask_both]) / 2

            print(f"  [{bi+1}/6 basin, rp{rp}] done")

    return rmse_arr


# ══════════════════════════════════════════════════════════════════════════
# Save zarr
# ══════════════════════════════════════════════════════════════════════════
def save_zarr(datetimes, scores, rmse_arr, xs, ys):
    print("儲存 analogue_results.zarr...")

    basin_coords = [str(b) for b in BASIN_IDS]

    # ── Part 1: metadata variables ──
    ds = xr.Dataset(
        {
            "matched_datetime": xr.DataArray(
                datetimes,
                dims=("basin", "return_period", "rank"),
                attrs={"description": "ISO datetime of matched historical event"},
            ),
            "rmse_score": xr.DataArray(
                scores,
                dims=("basin", "return_period", "rank"),
                attrs={"units": "m³/s", "description": "RMSE score (lower = more similar)"},
            ),
            "rmse_map": xr.DataArray(
                rmse_arr,
                dims=("basin", "return_period", "y", "x"),
                attrs={"description": "Spatial RMSE map (flood discharge similarity)"},
            ),
        },
        coords={
            "basin":         basin_coords,
            "return_period": RPS,
            "rank":          list(range(1, K_MAX + 1)),
            "y":             ys,
            "x":             xs,
        },
    )

    ds.attrs = {
        "description": "Analogue search results: top-K matched historical flood events per basin/RP",
        "crs":         "EPSG:4326",
        "resolution":  f"{TARGET_RES:.10f} deg (~30m)",
        "source_json": "analogue_search_results.json",
        "source_rmse": "rmse_maps/basin_{basinID}_rp{2|5|10}_rmse.tif",
        "step3_dims":  "(basin=6) x (return_period=3) x (rank=5)  +  (basin=6) x (return_period=3) x (y) x (x)",
    }

    encoding = {
        "rmse_map": {"chunks": (1, 1, 256, 256), "compressor": None},
        "rmse_score":        {"chunks": (6, 3, 5)},
        "matched_datetime":  {"chunks": (6, 3, 5)},
    }

    import shutil
    if OUTPUT_ZARR.exists():
        shutil.rmtree(OUTPUT_ZARR)

    ds.to_zarr(OUTPUT_ZARR, mode="w", encoding=encoding)
    print(f"  ✓ 儲存完成：{OUTPUT_ZARR}")
    print(f"  Dataset:\n{ds}")
    return ds


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    datetimes, scores = build_analogue_metadata()

    print("載入集水區邊界與目標 grid...")
    basins = load_basins()
    target_tf, W, H, xs, ys = build_target_grid(basins)

    rmse_arr = build_rmse_mosaic(basins, target_tf, W, H)

    save_zarr(datetimes, scores, rmse_arr, xs, ys)
    print("\n完成！")
