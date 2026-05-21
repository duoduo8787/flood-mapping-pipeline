"""
build_zarr_step4.py — HW1 Step 4
=================================
將 satellite_images/ 下各集水區的原始衛星影像：
  1. 依 hybas_si_lev08 shapefile 裁切至集水區邊界
  2. 統一重採樣至 ~30m (0.000269°) 解析度
  3. 拼接（mosaic）6 個集水區
  4. 組裝成 xarray Dataset 並存成 zarr

最終 zarr 維度（依 HW1 spec Step 4）：
  optical: (return_period=3, rank=5, channel=6, y, x)
    channel: ["blue", "green", "red", "nir", "swir1", "swir2"]
    來源: S2 (B2,B3,B4,B8,B11,B12) 或 L (SR_B2~B7)，統一波段順序
  sar:     (return_period=3, rank=5, channel=2, y, x)
    channel: ["VV", "VH"]
    來源: S1
"""

import re
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.mask import mask as rio_mask
from pathlib import Path
from shapely.geometry import mapping

# ── 路徑設定 ──────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
IMG_DIR     = BASE_DIR / "satellite_images"
SHP_PATH    = BASE_DIR / "hybas_si_lev01-12_v1c" / "hybas_si_lev08_v1c.shp"
OUTPUT_ZARR = BASE_DIR / "data" / "satellite_images.zarr"

# ── 常數 ──────────────────────────────────────────────────────────────────
BASIN_IDS  = [3080572620, 3080572630, 3080576250, 3080576260, 3080580980, 3080585700]
RPS        = [2, 5, 10]
K_MAX      = 5
TARGET_RES = 0.0002694946   # ≈30m

# 波段順序對照表（統一為 blue/green/red/nir/swir1/swir2）
# S2 檔案內的順序: [B4=red, B3=green, B2=blue, B8=nir, B11=swir1, B12=swir2]
#   → 重排到 [blue,green,red,nir,swir1,swir2] = 0-indexed [2,1,0,3,4,5]
S2_REORDER = [2, 1, 0, 3, 4, 5]
# L 檔案內的順序: [SR_B2=blue, SR_B3=green, SR_B4=red, SR_B5=nir, SR_B6=swir1, SR_B7=swir2]
#   → 已符合順序，不需重排
L_REORDER  = [0, 1, 2, 3, 4, 5]

OPT_CHANNELS = ["blue", "green", "red", "nir", "swir1", "swir2"]
SAR_CHANNELS = ["VV", "VH"]

FNAME_PAT = re.compile(
    r"(?P<basin>\d+)_(?P<rp>RP\d+)_Rank(?P<rank>\d+)_(?P<sensor>L|S1|S2)"
    r"_(?P<date>\d{8})_dev(?P<dev>[+-]?\d+)days_(?P<valid>\d+)pctvalid"
)


# ══════════════════════════════════════════════════════════════════════════
def load_basins():
    print("Step 1: 載入集水區邊界...")
    gdf = gpd.read_file(SHP_PATH)
    basins = gdf[gdf["HYBAS_ID"].isin(BASIN_IDS)].set_index("HYBAS_ID")
    print(f"  找到 {len(basins)} 個集水區")
    return basins


def build_target_grid(basins):
    print("Step 2: 計算全域 grid 範圍...")
    minx, miny, maxx, maxy = basins.total_bounds
    width  = int(np.ceil((maxx - minx) / TARGET_RES))
    height = int(np.ceil((maxy - miny) / TARGET_RES))
    transform = from_bounds(minx, miny, minx + width * TARGET_RES,
                            miny + height * TARGET_RES, width, height)
    xs = np.linspace(minx + TARGET_RES/2, minx + (width  - 0.5) * TARGET_RES, width)
    ys = np.linspace(miny + (height - 0.5) * TARGET_RES, miny + TARGET_RES/2, height)
    print(f"  全域 grid: {height} rows × {width} cols")
    return transform, width, height, xs, ys


def collect_best_tifs():
    """每個 (basin, rp, sensor_type, rank) 選 pctvalid 最高的影像"""
    print("Step 3: 掃描 satellite_images/，選最佳影像...")
    best = {}
    for tif in sorted(IMG_DIR.rglob("*.tif")):
        m = FNAME_PAT.search(tif.name)
        if not m:
            continue
        basin  = int(m.group("basin"))
        rp     = int(m.group("rp").replace("RP", ""))
        rank   = int(m.group("rank"))
        sensor = m.group("sensor")
        valid  = int(m.group("valid"))
        dev    = int(m.group("dev"))
        stype  = "sar" if sensor == "S1" else "optical"
        key    = (basin, rp, stype, rank)

        if key not in best:
            best[key] = (valid, abs(dev), tif)
        else:
            pv, pd, _ = best[key]
            if (valid, -abs(dev)) > (pv, -pd):
                best[key] = (valid, abs(dev), tif)

    print(f"  共選出 {len(best)} 個組合")
    return {k: v[2] for k, v in best.items()}


def clip_and_reproject_bands(tif_path, basin_geom, target_transform,
                              width, height, band_reorder, n_bands, out_dtype):
    """裁切 + 重採樣 → (n_bands, height, width) numpy array"""
    crs = rasterio.crs.CRS.from_epsg(4326)

    with rasterio.open(tif_path) as src:
        clipped, clipped_transform = rio_mask(
            src, [mapping(basin_geom)],
            crop=True, filled=False, all_touched=True,
        )

    out = np.full((n_bands, height, width), np.nan, dtype=np.float32)

    for out_i, src_i in enumerate(band_reorder):
        band = clipped[src_i].astype(np.float32)
        if hasattr(band, "fill_value"):
            band = band.filled(np.nan)

        reproject(
            source=band,
            destination=out[out_i],
            src_transform=clipped_transform,
            src_crs=crs,
            dst_transform=target_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return out


def build_arrays(best_tifs, basins, target_transform, width, height):
    print("Step 4: 裁切 + 重採樣 + mosaic...")

    n_rp, n_rank = len(RPS), K_MAX
    # optical: (rp, rank, 6, H, W) ; sar: (rp, rank, 2, H, W)
    opt = np.full((n_rp, n_rank, 6, height, width), np.nan, dtype=np.float32)
    sar = np.full((n_rp, n_rank, 2, height, width), np.nan, dtype=np.float32)

    total = len(best_tifs)
    for idx, ((basin_id, rp, stype, rank), tif_path) in enumerate(best_tifs.items(), 1):
        print(f"  [{idx:3d}/{total}] {tif_path.name[:65]}")

        rp_i   = RPS.index(rp)
        rank_i = rank - 1
        basin_geom = basins.loc[basin_id, "geometry"]

        try:
            if stype == "optical":
                # 判斷是 S2 還是 L
                sensor = re.search(r"_(L|S2)_", tif_path.name).group(1)
                reorder = S2_REORDER if sensor == "S2" else L_REORDER
                data = clip_and_reproject_bands(
                    tif_path, basin_geom, target_transform,
                    width, height, reorder, 6, np.float32
                )
                target = opt[rp_i, rank_i]   # shape (6, H, W)
                for ch in range(6):
                    mask_new  = ~np.isnan(data[ch])
                    mask_both = ~np.isnan(target[ch]) & mask_new
                    target[ch][mask_new & ~mask_both] = data[ch][mask_new & ~mask_both]
                    target[ch][mask_both] = (target[ch][mask_both] + data[ch][mask_both]) / 2

            else:  # SAR
                data = clip_and_reproject_bands(
                    tif_path, basin_geom, target_transform,
                    width, height, [0, 1], 2, np.float32
                )
                target = sar[rp_i, rank_i]   # shape (2, H, W)
                for ch in range(2):
                    mask_new  = ~np.isnan(data[ch])
                    mask_both = ~np.isnan(target[ch]) & mask_new
                    target[ch][mask_new & ~mask_both] = data[ch][mask_new & ~mask_both]
                    target[ch][mask_both] = (target[ch][mask_both] + data[ch][mask_both]) / 2

        except Exception as e:
            print(f"    ⚠ 跳過（{e}）")

    return opt, sar


def save_zarr(opt, sar, xs, ys):
    print("Step 5: 包成 xarray Dataset 並存成 zarr...")

    ds = xr.Dataset(
        {
            "optical": xr.DataArray(
                opt,
                dims=("return_period", "rank", "opt_channel", "y", "x"),
                coords={
                    "return_period": RPS,
                    "rank":          list(range(1, K_MAX + 1)),
                    "opt_channel":   OPT_CHANNELS,
                    "y": ys, "x": xs,
                },
                attrs={"long_name": "optical satellite imagery (best of S2 / Landsat)",
                       "units": "surface reflectance (uint16 scaled)"},
            ),
            "sar": xr.DataArray(
                sar,
                dims=("return_period", "rank", "sar_channel", "y", "x"),
                coords={
                    "return_period": RPS,
                    "rank":          list(range(1, K_MAX + 1)),
                    "sar_channel":   SAR_CHANNELS,
                    "y": ys, "x": xs,
                },
                attrs={"long_name": "SAR imagery (Sentinel-1)",
                       "units": "backscatter coefficient (linear)"},
            ),
        }
    )
    ds.attrs = {
        "description": "Historical satellite imagery mosaic for 6 HydroBASINS watersheds",
        "crs":         "EPSG:4326",
        "resolution":  f"{TARGET_RES:.10f} deg (~30m)",
        "hw1_dims":    "(return_period=3) x (rank=5) x (channel) x (y) x (x)",
        "optical_channels": str(OPT_CHANNELS),
        "sar_channels":     str(SAR_CHANNELS),
    }

    encoding = {
        "optical": {"chunks": (1, 1, 1, 256, 256), "compressor": None},
        "sar":     {"chunks": (1, 1, 1, 256, 256), "compressor": None},
        # coordinate arrays（不需要 chunk）
    }

    if OUTPUT_ZARR.exists():
        import shutil
        shutil.rmtree(OUTPUT_ZARR)

    ds.to_zarr(OUTPUT_ZARR, mode="w", encoding=encoding)
    print(f"  ✓ 儲存完成：{OUTPUT_ZARR}")
    print(f"  Dataset 內容：\n{ds}")
    return ds


if __name__ == "__main__":
    basins                   = load_basins()
    transform, W, H, xs, ys = build_target_grid(basins)
    best_tifs                = collect_best_tifs()
    opt, sar                 = build_arrays(best_tifs, basins, transform, W, H)
    ds                       = save_zarr(opt, sar, xs, ys)
    print("\n完成！")
