"""
build_zarr.py
=============
將 flood_predictions/all/ 下各集水區的 EDL 輸出 TIF：
  1. 依 hybas_si_lev08 shapefile 裁切至集水區邊界
  2. 統一重採樣至 ~30m (0.000269°) 解析度
  3. 拼接（mosaic）6 個集水區
  4. 組裝成 xarray Dataset 並存成 zarr

最終 zarr 維度（依 HW1 spec）：
  (sensor_type=2, rp=3, rank=5, y, x)
  sensor_type: ["optical", "sar"]
  return_period: [2, 5, 10]
  rank: [1, 2, 3, 4, 5]
  variables: classification, water_probability, dst_uncertainty,
             evidence_neg, evidence_pos, aleatoric, epistemic
"""

import re
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from pathlib import Path
from shapely.geometry import mapping

# ── 路徑設定 ──────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
PRED_DIR     = BASE_DIR / "flood_predictions" / "all"
SHP_PATH     = BASE_DIR / "hybas_si_lev01-12_v1c" / "hybas_si_lev08_v1c.shp"
OUTPUT_ZARR  = BASE_DIR / "data" / "flood_predictions.zarr"

# ── 常數 ──────────────────────────────────────────────────────────────────
BASIN_IDS    = [3080572620, 3080572630, 3080576250, 3080576260, 3080580980, 3080585700]
RPS          = [2, 5, 10]
K_MAX        = 5         # 最大 Rank
TARGET_RES   = 0.0002694946   # ≈30m，Landsat 原生解析度
NODATA       = np.nan
BAND_NAMES   = [
    "classification",
    "water_probability",
    "dst_uncertainty",
    "evidence_neg",
    "evidence_pos",
    "aleatoric",
    "epistemic",
]
# 每個 TIF band 對應的 index（1-based）
BAND_IDX     = {name: i+1 for i, name in enumerate(BAND_NAMES)}

# 解析檔名的 pattern
FNAME_PAT = re.compile(
    r"(?P<basin>\d+)_(?P<rp>RP\d+)_Rank(?P<rank>\d+)_(?P<sensor>L|S1|S2)"
    r"_(?P<date>\d{8})_dev(?P<dev>[+-]?\d+)days_(?P<valid>\d+)pctvalid"
)


# ══════════════════════════════════════════════════════════════════════════
# Step 1: 載入 shapefile，取得 6 個集水區邊界
# ══════════════════════════════════════════════════════════════════════════
def load_basins():
    print("Step 1: 載入集水區邊界...")
    gdf = gpd.read_file(SHP_PATH)
    basins = gdf[gdf["HYBAS_ID"].isin(BASIN_IDS)].set_index("HYBAS_ID")
    print(f"  找到 {len(basins)} 個集水區（CRS: {basins.crs}）")
    return basins


# ══════════════════════════════════════════════════════════════════════════
# Step 2: 建立全域 mosaic 目標 grid
# ══════════════════════════════════════════════════════════════════════════
def build_target_grid(basins):
    print("Step 2: 計算全域 grid 範圍...")
    total_bounds = basins.total_bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = total_bounds

    # 依 TARGET_RES 建立整齊的 grid
    width  = int(np.ceil((maxx - minx) / TARGET_RES))
    height = int(np.ceil((maxy - miny) / TARGET_RES))
    transform = from_bounds(minx, miny, minx + width * TARGET_RES,
                            miny + height * TARGET_RES, width, height)
    xs = np.linspace(minx + TARGET_RES/2, minx + (width  - 0.5) * TARGET_RES, width)
    ys = np.linspace(miny + (height - 0.5) * TARGET_RES, miny + TARGET_RES/2, height)

    print(f"  全域 grid: {height} rows × {width} cols  "
          f"(lon {minx:.4f}~{maxx:.4f}, lat {miny:.4f}~{maxy:.4f})")
    return transform, width, height, xs, ys


# ══════════════════════════════════════════════════════════════════════════
# Step 3: 收集所有 TIF，挑出每個 (basin, rp, sensor_type, rank) 的最佳影像
#         best = pctvalid 最高；同分取 |dev| 最小
# ══════════════════════════════════════════════════════════════════════════
def collect_best_tifs():
    print("Step 3: 掃描 TIF，每個 (basin, rp, sensor_type, rank) 選最佳影像...")
    best = {}   # key: (basin_id, rp_int, sensor_type_str, rank_int) → Path

    for tif in sorted(PRED_DIR.rglob("*_output_EDL*.tif")):
        m = FNAME_PAT.search(tif.name)
        if not m:
            continue
        basin   = int(m.group("basin"))
        rp      = int(m.group("rp").replace("RP", ""))
        rank    = int(m.group("rank"))
        sensor  = m.group("sensor")
        valid   = int(m.group("valid"))
        dev     = int(m.group("dev"))
        stype   = "sar" if sensor == "S1" else "optical"
        key     = (basin, rp, stype, rank)

        if key not in best:
            best[key] = (valid, abs(dev), tif)
        else:
            prev_valid, prev_dev, _ = best[key]
            # 優先選 pctvalid 高的；相同則選 |dev| 小的
            if (valid, -abs(dev)) > (prev_valid, -prev_dev):
                best[key] = (valid, abs(dev), tif)

    print(f"  共選出 {len(best)} 個 (basin, rp, sensor_type, rank) 組合")
    return {k: v[2] for k, v in best.items()}   # key → Path


# ══════════════════════════════════════════════════════════════════════════
# Step 4: 裁切單張 TIF → 重採樣 → 投影到全域 grid
# ══════════════════════════════════════════════════════════════════════════
def clip_and_reproject(tif_path, basin_geom, target_transform, width, height):
    """
    裁切到集水區邊界，重採樣後寫進 (7, height, width) 的 numpy array。
    集水區外的像素保持 NaN。
    """
    da = rxr.open_rasterio(tif_path, masked=True)

    # 裁切
    da_clipped = da.rio.clip(
        [mapping(basin_geom)],
        crs="EPSG:4326",
        drop=True,
        all_touched=True,
    )

    out = np.full((7, height, width), np.nan, dtype=np.float32)

    with rasterio.open(tif_path) as src:
        target_crs = rasterio.crs.CRS.from_epsg(4326)
        # 先做 clip 的 mask
        from rasterio.mask import mask as rio_mask
        clipped_data, clipped_transform = rio_mask(
            src,
            [mapping(basin_geom)],
            crop=True,
            filled=False,
            all_touched=True,
        )

        for band_i in range(7):
            band_data = clipped_data[band_i].astype(np.float32)
            if hasattr(band_data, 'fill_value'):
                band_data = band_data.filled(np.nan)

            # 重投影到全域 grid
            reproject(
                source=band_data,
                destination=out[band_i],
                src_transform=clipped_transform,
                src_crs=target_crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

    return out


# ══════════════════════════════════════════════════════════════════════════
# Step 5: 組裝完整的 5D array
# ══════════════════════════════════════════════════════════════════════════
def build_arrays(best_tifs, basins, target_transform, width, height):
    print("Step 5: 裁切 + 重採樣 + mosaic 所有 TIF...")

    sensor_types = ["optical", "sar"]
    n_st, n_rp, n_rank = len(sensor_types), len(RPS), K_MAX

    # 7 個變數，每個 shape: (n_st, n_rp, n_rank, height, width)
    arrays = {name: np.full((n_st, n_rp, n_rank, height, width), np.nan, dtype=np.float32)
              for name in BAND_NAMES}

    total = len(best_tifs)
    for idx, ((basin_id, rp, stype, rank), tif_path) in enumerate(best_tifs.items(), 1):
        print(f"  [{idx:3d}/{total}] {tif_path.name[:60]}")

        st_i   = sensor_types.index(stype)
        rp_i   = RPS.index(rp)
        rank_i = rank - 1   # 0-indexed

        basin_geom = basins.loc[basin_id, "geometry"]
        try:
            data = clip_and_reproject(tif_path, basin_geom,
                                      target_transform, width, height)
        except Exception as e:
            print(f"    ⚠ 跳過（{e}）")
            continue

        for b_i, name in enumerate(BAND_NAMES):
            existing = arrays[name][st_i, rp_i, rank_i]
            new      = data[b_i]
            # mosaic：優先保留已有值；若重疊則取平均
            mask_new  = ~np.isnan(new)
            mask_both = ~np.isnan(existing) & mask_new
            arrays[name][st_i, rp_i, rank_i][mask_new & ~mask_both] = \
                new[mask_new & ~mask_both]
            arrays[name][st_i, rp_i, rank_i][mask_both] = \
                (existing[mask_both] + new[mask_both]) / 2

    return arrays


# ══════════════════════════════════════════════════════════════════════════
# Step 6: 包成 xarray Dataset → 存 zarr
# ══════════════════════════════════════════════════════════════════════════
def save_zarr(arrays, xs, ys):
    print("Step 6: 包成 xarray Dataset 並存成 zarr...")

    sensor_types = ["optical", "sar"]
    dims = ("sensor_type", "return_period", "rank", "y", "x")
    coords = {
        "sensor_type":   sensor_types,
        "return_period": RPS,
        "rank":          list(range(1, K_MAX + 1)),
        "y":             ys,
        "x":             xs,
    }

    data_vars = {}
    for name in BAND_NAMES:
        data_vars[name] = xr.DataArray(
            arrays[name],
            dims=dims,
            attrs={"long_name": name},
        )

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs = {
        "description": "EDL flood prediction mosaic for 6 HydroBASINS watersheds",
        "crs":         "EPSG:4326",
        "resolution":  f"{TARGET_RES:.10f} deg (~30m)",
        "sensor_type_0": "optical (best of Landsat / Sentinel-2)",
        "sensor_type_1": "SAR (Sentinel-1)",
        "hw1_dims":    "(sensor_type=2) x (return_period=3) x (rank=5) x (y) x (x)",
    }

    # 設定 chunk 大小（每個 chunk ~1MB），透過 encoding 傳入，不依賴 dask
    chunk_shape = (1, 1, 1, 256, 256)
    encoding = {
        name: {"chunks": chunk_shape, "compressor": None}
        for name in BAND_NAMES
    }

    if OUTPUT_ZARR.exists():
        import shutil
        shutil.rmtree(OUTPUT_ZARR)

    ds.to_zarr(OUTPUT_ZARR, mode="w", encoding=encoding)
    print(f"  ✓ 儲存完成：{OUTPUT_ZARR}")
    print(f"  Dataset 內容：\n{ds}")
    return ds


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    basins                      = load_basins()
    transform, W, H, xs, ys    = build_target_grid(basins)
    best_tifs                   = collect_best_tifs()
    arrays                      = build_arrays(best_tifs, basins, transform, W, H)
    ds                          = save_zarr(arrays, xs, ys)
    print("\n完成！")
