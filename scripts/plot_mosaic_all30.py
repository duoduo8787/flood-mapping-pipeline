"""
plot_mosaic_all30.py
====================
重新生成 mosaic_all30.png，並將所有文字顏色改為黑色（解決白字在白底不可見的問題）。

圖表配置：
  列（6 列）：3 RP × 2 sensor_type = RP2/optical, RP2/SAR, RP5/optical, RP5/SAR, RP10/optical, RP10/SAR
  欄（5 欄）：Rank 1–5
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from pathlib import Path

# ── 中文字型設定 ──────────────────────────────────────────────────────────
_cjk_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_cjk_font)
plt.rcParams["font.family"] = "Noto Sans CJK JP"   # TTC 載入後名稱為 JP，但支援全部 CJK
plt.rcParams["axes.unicode_minus"] = False

# ── 路徑設定 ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
ZARR_PATH  = BASE_DIR / "flood_predictions.zarr"
SHP_PATH   = BASE_DIR / "hybas_si_lev01-12_v1c" / "hybas_si_lev08_v1c.shp"
OUTPUT_PNG = BASE_DIR / "mosaic_all30.png"

BASIN_IDS  = [3080572620, 3080572630, 3080576250, 3080576260, 3080580980, 3080585700]
RPS        = [2, 5, 10]
RANKS      = [1, 2, 3, 4, 5]
SENSORS    = ["optical", "sar"]

# ── 載入資料 ──────────────────────────────────────────────────────────────
print("載入 zarr...")
ds = xr.open_zarr(ZARR_PATH)
clf = ds["classification"]   # (sensor_type, return_period, rank, y, x)
xs = ds.coords["x"].values
ys = ds.coords["y"].values

print("載入集水區 shapefile...")
gdf = gpd.read_file(SHP_PATH)
basins = gdf[gdf["HYBAS_ID"].isin(BASIN_IDS)]

# ── 色彩設定 ──────────────────────────────────────────────────────────────
# classification: 0 = 非水體, 1 = 水體, NaN = 無資料
cmap_clf = mcolors.ListedColormap(["#f0f0f0", "#1a3c8f"])  # 淺灰/深藍
norm_clf = mcolors.BoundaryNorm([0, 0.5, 1.5], cmap_clf.N)

# ── 繪圖 ──────────────────────────────────────────────────────────────────
n_rows = len(RPS) * len(SENSORS)   # 6
n_cols = len(RANKS)                # 5

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(n_cols * 3, n_rows * 3.2),
    constrained_layout=True,
)

# 全圖標題（黑色）
fig.suptitle(
    "全部 30 張組合（3 RP × 5 Rank × 2 Sensor）的個別水道圖統計\n"
    "數字: 紅字=HydroBasin邊界 / 藍色=洪水區域",
    fontsize=13,
    color="black",      # ← 黑色
    fontweight="bold",
)

row_labels = []
for rp in RPS:
    for sensor in SENSORS:
        row_labels.append(f"RP{rp}\n{'光學' if sensor == 'optical' else 'SAR'}")

for row_i, (rp, sensor) in enumerate(
    [(rp, s) for rp in RPS for s in SENSORS]
):
    rp_idx     = RPS.index(rp)
    sensor_idx = SENSORS.index(sensor)

    for col_i, rank in enumerate(RANKS):
        rank_idx = rank - 1
        ax = axes[row_i, col_i]

        # 取得分類資料
        data = clf.isel(
            sensor_type=sensor_idx,
            return_period=rp_idx,
            rank=rank_idx,
        ).values   # (y, x)

        # 繪製分類圖
        ax.imshow(
            data,
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            origin="upper",
            cmap=cmap_clf,
            norm=norm_clf,
            interpolation="nearest",
        )

        # 疊加 HydroBasin 邊界（紅色）
        basins.boundary.plot(ax=ax, color="red", linewidth=0.6)

        # ── 計算水體統計數字 ──────────────────────────────────────────────
        valid_px = int(np.sum(~np.isnan(data)))
        water_px = int(np.sum(data == 1))
        pct      = water_px / valid_px * 100 if valid_px > 0 else 0.0

        if water_px == 0:
            stats_txt = "無資料"
            txt_color = "gray"
        else:
            stats_txt = f"洪水: {pct:.1f}%\n({water_px/1000:.0f}k px)"
            txt_color = "red"

        # 左上角標注統計數字
        ax.text(
            0.03, 0.97, stats_txt,
            transform=ax.transAxes,
            fontsize=6,
            color=txt_color,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.75, edgecolor="none"),
        )

        # 副標題（黑色）
        sensor_label = "Optical" if sensor == "optical" else "SAR"
        ax.set_title(
            f"RP={rp} | {sensor_label} | Rank{rank}",
            fontsize=6.5,
            color="black",   # ← 黑色
            pad=2,
        )

        ax.axis("off")

    # 最左欄加列標籤（黑色）
    axes[row_i, 0].set_ylabel(
        row_labels[row_i],
        fontsize=8,
        color="black",      # ← 黑色
        rotation=90,
        labelpad=4,
    )
    axes[row_i, 0].yaxis.set_visible(True)
    axes[row_i, 0].set_yticks([])



# 圖例
legend_elements = [
    Patch(facecolor="#1a3c8f", label="洪水 (Water)"),
    Patch(facecolor="#f0f0f0", label="非洪水 (Non-Water)", edgecolor="gray"),
    Patch(facecolor="none",    edgecolor="red", label="HydroBasin 邊界"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    fontsize=8,
    framealpha=0.9,
    labelcolor="black",   # ← 黑色
    bbox_to_anchor=(0.5, -0.01),
)

print(f"儲存圖片：{OUTPUT_PNG}")
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("完成！")
