"""
data_merge.py
=============
融合 optical 與 SAR 的 EDL 預測結果，產出每個重現期的最終淹水範圍圖。

輸入：flood_predictions.zarr
  (sensor_type=2, return_period=3, rank=5, y, x) × 7 variables

融合策略（逐像素）：
  - optical + SAR 都有值 → 以 1/dst_uncertainty 加權平均
  - 只有一個有值         → 直接使用
  - 都是 NaN            → 保持 NaN

輸出：flood_extent_merged.zarr
  (return_period=3, rank=5, y, x)
  variables: water_probability, classification, uncertainty
"""

import numpy as np
import xarray as xr
from pathlib import Path

BASE_DIR    = Path(__file__).parent
SRC_ZARR    = BASE_DIR / "data" / "flood_predictions.zarr"
OUTPUT_ZARR = BASE_DIR / "data" / "flood_extent_merged.zarr"
TH_WATER    = 0.5   # 水體分類門檻


# ══════════════════════════════════════════════════════════════════════════
# 融合單一 (RP, rank) 的 optical + SAR
# ══════════════════════════════════════════════════════════════════════════
def merge_sensor_pair(wp_opt, wp_sar, unc_opt, unc_sar):
    """
    wp_opt, wp_sar:   water_probability  (y, x) float32
    unc_opt, unc_sar: dst_uncertainty    (y, x) float32
    回傳: merged_wp, merged_unc  (y, x)
    """
    has_opt = ~np.isnan(wp_opt)
    has_sar = ~np.isnan(wp_sar)
    both    = has_opt & has_sar

    merged_wp  = np.full_like(wp_opt, np.nan)
    merged_unc = np.full_like(unc_opt, np.nan)

    # 只有 optical
    only_opt = has_opt & ~has_sar
    merged_wp[only_opt]  = wp_opt[only_opt]
    merged_unc[only_opt] = unc_opt[only_opt]

    # 只有 SAR
    only_sar = has_sar & ~has_opt
    merged_wp[only_sar]  = wp_sar[only_sar]
    merged_unc[only_sar] = unc_sar[only_sar]

    # 兩者都有 → 1/uncertainty 加權平均
    if both.any():
        # 避免除以零
        w_opt = 1.0 / np.where(unc_opt[both] > 1e-6, unc_opt[both], 1e-6)
        w_sar = 1.0 / np.where(unc_sar[both] > 1e-6, unc_sar[both], 1e-6)
        total_w = w_opt + w_sar
        merged_wp[both]  = (w_opt * wp_opt[both] + w_sar * wp_sar[both]) / total_w
        merged_unc[both] = 1.0 / total_w   # 加權後不確定性

    return merged_wp, merged_unc


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("載入 flood_predictions.zarr ...")
    ds = xr.open_zarr(SRC_ZARR)

    rps   = ds.return_period.values   # [2, 5, 10]
    ranks = ds["rank"].values          # [1, 2, 3, 4, 5]
    ys    = ds.y.values
    xs    = ds.x.values
    H, W  = len(ys), len(xs)
    n_rp, n_rank = len(rps), len(ranks)

    wp_merged  = np.full((n_rp, n_rank, H, W), np.nan, dtype=np.float32)
    cls_merged = np.full((n_rp, n_rank, H, W), np.nan, dtype=np.float32)
    unc_merged = np.full((n_rp, n_rank, H, W), np.nan, dtype=np.float32)

    print("開始融合 optical + SAR ...")
    for ri, rp in enumerate(rps):
        for ki, rank in enumerate(ranks):
            print(f"  RP={rp:2d}  rank={rank} ...", end=" ")

            wp_opt  = ds["water_probability"].sel(
                sensor_type="optical", return_period=rp, rank=rank).values
            wp_sar  = ds["water_probability"].sel(
                sensor_type="sar",     return_period=rp, rank=rank).values
            unc_opt = ds["dst_uncertainty"].sel(
                sensor_type="optical", return_period=rp, rank=rank).values
            unc_sar = ds["dst_uncertainty"].sel(
                sensor_type="sar",     return_period=rp, rank=rank).values

            merged_wp, merged_unc = merge_sensor_pair(
                wp_opt, wp_sar, unc_opt, unc_sar)

            wp_merged[ri, ki]  = merged_wp
            unc_merged[ri, ki] = merged_unc
            cls_merged[ri, ki] = (merged_wp >= TH_WATER).astype(np.float32)
            # 無資料區域保持 NaN
            cls_merged[ri, ki][np.isnan(merged_wp)] = np.nan

            valid_pct = (~np.isnan(merged_wp)).mean() * 100
            print(f"有效像素 {valid_pct:.1f}%")

    print("\n組裝 xarray Dataset ...")
    dims   = ("return_period", "rank", "y", "x")
    coords = {
        "return_period": rps,
        "rank":          ranks,
        "y":             ys,
        "x":             xs,
    }

    ds_out = xr.Dataset(
        {
            "water_probability": xr.DataArray(
                wp_merged, dims=dims,
                attrs={"description": "Uncertainty-weighted merged water probability (optical + SAR)",
                       "range": "[0, 1]"}),
            "classification": xr.DataArray(
                cls_merged, dims=dims,
                attrs={"description": f"Binary flood classification (threshold={TH_WATER})",
                       "values": "1=water, 0=land, NaN=no data"}),
            "uncertainty": xr.DataArray(
                unc_merged, dims=dims,
                attrs={"description": "Combined uncertainty after sensor fusion"}),
        },
        coords=coords,
    )
    ds_out.attrs = {
        "description":    "Merged flood extent map (optical + SAR sensor fusion)",
        "crs":            "EPSG:4326",
        "resolution":     "0.0002694946 deg (~30m)",
        "fusion_method":  "uncertainty-weighted average (1/dst_uncertainty)",
        "th_water":       str(TH_WATER),
        "source":         "flood_predictions.zarr",
        "pipeline_step":  "Step 6 – Data Merge",
        "output_dims":    "(return_period=3) x (rank=5) x (y) x (x)",
    }

    encoding = {
        "water_probability": {"chunks": (1, 1, 512, 512), "compressor": None},
        "classification":    {"chunks": (1, 1, 512, 512), "compressor": None},
        "uncertainty":       {"chunks": (1, 1, 512, 512), "compressor": None},
    }

    if OUTPUT_ZARR.exists():
        import shutil
        shutil.rmtree(OUTPUT_ZARR)

    ds_out.to_zarr(OUTPUT_ZARR, mode="w", encoding=encoding)
    print(f"\n✓ 儲存完成：{OUTPUT_ZARR}")
    print(ds_out)


if __name__ == "__main__":
    main()
