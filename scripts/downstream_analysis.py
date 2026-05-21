"""
downstream_analysis.py
======================
Downstream Task Demo — Task A: Data Analysis

展示內容：
  1. 速度比較：原始 TIF 逐檔讀取 vs zarr 直讀（計算各 RP 平均水體機率）
  2. 各 RP 淹水面積 bar chart（RP=2/5/10）
  3. 各 RP 淹水範圍地圖（3張並排，取 rank 中位數）
  4. 不確定性分布圖

輸出：downstream_results.png
"""

import time
import warnings
import numpy as np
import xarray as xr
import rasterio
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent
ZARR_PATH   = BASE_DIR / "data" / "flood_extent_merged.zarr"
PRED_DIR    = BASE_DIR / "flood_predictions" / "all"
SHP_PATH    = BASE_DIR / "hybas_si_lev01-12_v1c" / "hybas_si_lev08_v1c.shp"
TARGET_RES  = 0.0002694946   # ~30m per pixel
PIXEL_AREA_KM2 = (TARGET_RES * 111) ** 2  # 1度≈111km

RPS = [2, 5, 10]
BASIN_IDS = [3080572620, 3080572630, 3080576250, 3080576260, 3080580980, 3080585700]


# ══════════════════════════════════════════════════════════════════════════
# Part 1: 速度比較
# ══════════════════════════════════════════════════════════════════════════
def benchmark_raw_vs_zarr():
    print("Part 1: 速度比較 (raw TIF vs zarr) ...")

    # ── Raw TIF：逐檔讀取所有 optical TIF，計算全域平均水體機率 ──
    tif_files = list(PRED_DIR.rglob("*_output_EDL*.tif"))
    optical_tifs = [f for f in tif_files if "_S1_" not in f.name][:30]  # 取前30個代表

    t0 = time.perf_counter()
    raw_vals = []
    for tif in optical_tifs:
        with rasterio.open(tif) as src:
            band = src.read(2).astype(np.float32)   # band2 = water_probability
            band[band == src.nodata] = np.nan if src.nodata else band
            valid = band[~np.isnan(band)]
            if len(valid) > 0:
                raw_vals.append(valid.mean())
    raw_mean = np.mean(raw_vals)
    t_raw = time.perf_counter() - t0

    # ── Zarr：直接讀取 flood_extent_merged.zarr ──
    t0 = time.perf_counter()
    ds = xr.open_zarr(ZARR_PATH)
    zarr_mean = float(ds["water_probability"].mean().values)
    t_zarr = time.perf_counter() - t0

    speedup = t_raw / t_zarr
    print(f"  Raw TIF ({len(optical_tifs)} files): {t_raw*1000:.1f} ms → mean={raw_mean:.4f}")
    print(f"  Zarr:                          {t_zarr*1000:.1f} ms → mean={zarr_mean:.4f}")
    print(f"  速度提升：{speedup:.1f}x")

    return t_raw, t_zarr, speedup, ds


# ══════════════════════════════════════════════════════════════════════════
# Part 2: 各 RP 淹水面積統計
# ══════════════════════════════════════════════════════════════════════════
def calc_flood_area(ds):
    print("Part 2: 計算各 RP 淹水面積 ...")
    areas = {}        # rp → list of valid areas (NaN for missing imagery)
    valid_counts = {} # rp → 有實際影像的 rank 數
    for rp in RPS:
        rank_areas = []
        for rank in range(1, 6):
            cls = ds["classification"].sel(return_period=rp, rank=rank).values
            n_valid_px = int(np.sum(~np.isnan(cls)))
            if n_valid_px == 0:
                rank_areas.append(np.nan)   # 無影像 → 排除於統計
            else:
                rank_areas.append(np.nansum(cls == 1) * PIXEL_AREA_KM2)
        areas[rp] = rank_areas
        valid_counts[rp] = int(np.sum(~np.isnan(rank_areas)))
        valid_areas = [a for a in rank_areas if not np.isnan(a)]
        print(f"  RP={rp:2d}: mean={np.nanmean(rank_areas):.1f} km²  "
              f"valid ranks={valid_counts[rp]}/5  "
              f"({[f'{a:.0f}' if not np.isnan(a) else 'N/A' for a in rank_areas]})")
    return areas, valid_counts


# ══════════════════════════════════════════════════════════════════════════
# Part 3 & 4: 視覺化
# ══════════════════════════════════════════════════════════════════════════
def load_basin_outlines(ds):
    """載入集水區邊界，轉換為像素座標用於疊加在地圖上"""
    gdf = gpd.read_file(SHP_PATH)
    basins = gdf[gdf["HYBAS_ID"].isin(BASIN_IDS)]

    xs = ds.x.values
    ys = ds.y.values
    x_min, x_res = xs[0], xs[1] - xs[0]
    y_min, y_res = ys[-1], ys[0] - ys[-1]   # ys 是遞減

    outlines = []
    for _, row in basins.iterrows():
        geom = row.geometry
        # 取外環座標轉像素
        if geom.geom_type == "Polygon":
            coords = list(geom.exterior.coords)
        else:
            coords = list(geom.convex_hull.exterior.coords)
        px = [(c[0] - x_min) / x_res for c in coords]
        py = [(ys[0] - c[1]) / (ys[0] - ys[-1]) * len(ys) for c in coords]
        outlines.append((px, py))
    return outlines


def plot_results(t_raw, t_zarr, speedup, ds):
    print("Part 3: 繪圖 ...")

    OUT_DIR = BASE_DIR / "downstream_figures"
    OUT_DIR.mkdir(exist_ok=True)

    basin_outlines = load_basin_outlines(ds)

    # ── 圖1：速度比較（單獨儲存）─────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    bars = ax1.bar(["Raw TIFs\n(30 files)", "Zarr\n(optimized)"],
                   [t_raw * 1000, t_zarr * 1000],
                   color=["#e74c3c", "#2ecc71"], width=0.5, alpha=0.85)
    for bar, val in zip(bars, [t_raw * 1000, t_zarr * 1000]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f} ms", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax1.set_title(f"Read Speed Comparison\n(Speedup: {speedup:.1f}×)", fontsize=13)
    ax1.set_ylabel("Time (ms)")
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    p1 = OUT_DIR / "fig1_speed_comparison.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  ✓ {p1.name}")

    # ── 圖2～7：地圖（2列×3欄，合併儲存）───────────────────────────
    fig2 = plt.figure(figsize=(15, 9))
    fig2.suptitle("Flood Mapping Pipeline — Flood Extent & Uncertainty\n"
                  "6 HydroBASINS Watersheds · Sensor Fusion (Optical + SAR) · EDL Model",
                  fontsize=13, fontweight="bold")
    gs = fig2.add_gridspec(2, 4, hspace=0.35, wspace=0.3,
                           width_ratios=[1, 1, 1, 0.05])

    for i, rp in enumerate(RPS):
        ax = fig2.add_subplot(gs[0, i])
        cls_mean = np.nanmean(ds["classification"].sel(return_period=rp).values, axis=0)
        im = ax.imshow(cls_mean, cmap="Blues", vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")
        for px, py in basin_outlines:
            ax.plot(px, py, color="black", linewidth=0.8, alpha=0.7)
        ax.set_title(f"Flood Agreement — RP={rp}\n(mean across top-5 analogues)", fontsize=10)
        ax.set_xlabel("X (pixels)", fontsize=8)
        ax.set_ylabel("Y (pixels)", fontsize=8)
        ax.tick_params(labelsize=7)
        area = float(np.nansum(cls_mean >= 0.5)) * PIXEL_AREA_KM2
        ax.text(0.02, 0.02, f"Area (≥50%): {area:.0f} km²",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    cbar_ax1 = fig2.add_subplot(gs[0, 3])
    cbar = fig2.colorbar(im, cax=cbar_ax1, ticks=[0, 0.5, 1])
    cbar.set_ticklabels(["0%\n(no analogue)", "50%", "100%\n(all analogues)"])
    cbar.set_label("Flood Agreement", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for i, rp in enumerate(RPS):
        ax = fig2.add_subplot(gs[1, i])
        unc_mean = np.nanmean(ds["uncertainty"].sel(return_period=rp).values, axis=0)
        im2 = ax.imshow(unc_mean, cmap="YlOrRd", aspect="auto",
                        interpolation="nearest", vmin=0, vmax=0.5)
        for px, py in basin_outlines:
            ax.plot(px, py, color="black", linewidth=0.8, alpha=0.7)
        ax.set_title(f"Uncertainty — RP={rp}\n(mean across top-5 analogues)", fontsize=10)
        ax.set_xlabel("X (pixels)", fontsize=8)
        ax.set_ylabel("Y (pixels)", fontsize=8)
        ax.tick_params(labelsize=7)
        mean_unc = float(np.nanmean(unc_mean))
        ax.text(0.02, 0.02, f"Mean unc: {mean_unc:.3f}",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    cbar_ax2 = fig2.add_subplot(gs[1, 3])
    cbar2 = fig2.colorbar(im2, cax=cbar_ax2)
    cbar2.set_label("Uncertainty", fontsize=9)
    cbar2.ax.tick_params(labelsize=8)

    p2 = OUT_DIR / "fig2_flood_maps.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ {p2.name}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_raw, t_zarr, speedup, ds = benchmark_raw_vs_zarr()
    areas, valid_counts = calc_flood_area(ds)
    plot_results(t_raw, t_zarr, speedup, ds)
    print("\n完成！輸出：downstream_figures/")
