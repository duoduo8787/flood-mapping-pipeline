"""
benchmark_chunking.py
=====================
比較三種 Zarr chunking 策略的讀取效能。
以 flood_predictions.zarr 的 water_probability 變數為基準。

三種策略：
  A: (1, 1, 1, 512, 512)   — 對應最小集水區尺度（652×716），空間查詢優化
  B: (2, 3, 5, 128, 128)   — 完整時序+rank軸，Data Merge 優化
  C: (1, 1, 1, 2508, 3188) — 完整場景單一 chunk，全圖讀取 baseline

三種查詢模式：
  Q1: 單場景讀取   water_probability[sensor=0, rp=0, rank=0, :, :]
  Q2: Data Merge   water_probability[sensor=0, rp=0, :, :, :]  (所有 rank)
  Q3: 集水區查詢   water_probability[0, 0, 0, 900:1412, 900:1556]  (~最小集水區)

輸出：
  benchmark_results.csv     — 數值結果
  benchmark_results.png     — 長條圖視覺化
"""

import time
import shutil
import numpy as np
import xarray as xr
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).parent
SRC_ZARR   = BASE_DIR / "data" / "flood_predictions.zarr"
BENCH_DIR  = BASE_DIR / "data" / "benchmark_zarr"
BENCH_DIR.mkdir(exist_ok=True)

N_TRIALS = 5   # 每個查詢重複次數（取平均）

# ── 三種 chunking 策略 ─────────────────────────────────────────────────────
STRATEGIES = {
    "A: Basin-Scale\n(1,1,1,512,512)":  (1, 1, 1, 512, 512),
    "B: Full-Temporal\n(2,3,5,128,128)": (2, 3, 5, 128, 128),
    "C: Full-Scene\n(1,1,1,full,full)": (1, 1, 1, 2508, 3188),
}

# ══════════════════════════════════════════════════════════════════════════
# Step 1: 依三種策略建立測試 zarr（僅存 water_probability）
# ══════════════════════════════════════════════════════════════════════════
def build_test_zarrs():
    print("=== Step 1: 建立三份測試 zarr ===")
    ds = xr.open_zarr(SRC_ZARR)[["water_probability"]]

    paths = {}
    for label, chunks in STRATEGIES.items():
        short_name = label.split("\n")[0].split(":")[0].strip()  # "A", "B", "C"
        out_path = BENCH_DIR / f"strategy_{short_name}.zarr"
        paths[label] = out_path

        if out_path.exists():
            print(f"  {short_name}: 已存在，跳過重建")
            continue

        print(f"  建立策略 {short_name} {chunks} ...")
        encoding = {
            "water_probability": {
                "chunks": chunks,
                "compressor": None,
            }
        }
        ds.to_zarr(out_path, mode="w", encoding=encoding)
        print(f"    ✓ 儲存至 {out_path}")

    return paths


# ══════════════════════════════════════════════════════════════════════════
# Step 2: 執行查詢，測量 Latency 與 Throughput
# ══════════════════════════════════════════════════════════════════════════
def run_query(ds, query_fn):
    """回傳 (latency_sec, throughput_MBps, nbytes_MB)"""
    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        result = query_fn(ds).values   # 強制載入到記憶體
        t1 = time.perf_counter()
        times.append(t1 - t0)

    nbytes_MB = result.nbytes / 1e6
    latency   = np.mean(times)
    throughput = nbytes_MB / latency
    return latency, throughput, nbytes_MB


QUERIES = {
    "Q1: Single Scene\n(sensor=0,rp=0,rank=0)": lambda ds: (
        ds["water_probability"].isel(sensor_type=0, return_period=0, rank=0)
    ),
    "Q2: Data Merge\n(all ranks, rp=0)": lambda ds: (
        ds["water_probability"].isel(sensor_type=0, return_period=0)
    ),
    "Q3: Basin-Scale\n(~最小集水區)": lambda ds: (
        ds["water_probability"].isel(
            sensor_type=0, return_period=0, rank=0,
            y=slice(900, 1552), x=slice(900, 1616)
        )
    ),
}


def benchmark_all(zarr_paths):
    print("\n=== Step 2: 執行 Benchmark ===")
    records = []

    for strategy_label, zarr_path in zarr_paths.items():
        short = strategy_label.split("\n")[0]
        ds = xr.open_zarr(zarr_path)

        z = zarr.open(str(zarr_path))
        chunk_shape = z["water_probability"].chunks
        # chunks 可能是 tuple of ints 或 tuple of tuples
        first_chunks = [c[0] if hasattr(c, '__len__') else c for c in chunk_shape]
        chunk_MB = np.prod(first_chunks) * 4 / 1e6

        print(f"\n  策略 {short} | chunk={chunk_shape} | 單 chunk={chunk_MB:.2f} MB")

        for query_label, query_fn in QUERIES.items():
            q_short = query_label.split("\n")[0]
            latency, throughput, nbytes = run_query(ds, query_fn)
            print(f"    {q_short}: latency={latency*1000:.1f} ms  "
                  f"throughput={throughput:.1f} MB/s  data={nbytes:.2f} MB")
            records.append({
                "strategy": short,
                "query":    q_short,
                "latency_ms":    round(latency * 1000, 2),
                "throughput_MBps": round(throughput, 2),
                "data_MB":       round(nbytes, 2),
                "chunk_shape":   str(chunk_shape),
            })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# Step 3: 視覺化並儲存結果
# ══════════════════════════════════════════════════════════════════════════
def plot_results(df):
    print("\n=== Step 3: 繪製結果圖 ===")

    strategies = df["strategy"].unique()
    queries    = df["query"].unique()
    x          = np.arange(len(queries))
    width      = 0.25
    colors     = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Zarr Chunking Strategy Benchmark\n(water_probability, flood_predictions.zarr)",
                 fontsize=13, fontweight="bold")

    # ── Latency ──
    ax = axes[0]
    for i, (strat, color) in enumerate(zip(strategies, colors)):
        vals = [df[(df["strategy"] == strat) & (df["query"] == q)]["latency_ms"].values[0]
                for q in queries]
        bars = ax.bar(x + i * width, vals, width, label=strat, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Latency (ms)  — lower is better", fontsize=11)
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(queries, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Throughput ──
    ax = axes[1]
    for i, (strat, color) in enumerate(zip(strategies, colors)):
        vals = [df[(df["strategy"] == strat) & (df["query"] == q)]["throughput_MBps"].values[0]
                for q in queries]
        bars = ax.bar(x + i * width, vals, width, label=strat, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Throughput (MB/s)  — higher is better", fontsize=11)
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(queries, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_png = BASE_DIR / "benchmark_results.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  ✓ 圖表儲存：{out_png}")

    out_csv = BASE_DIR / "benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"  ✓ 數值儲存：{out_csv}")
    return out_png, out_csv


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    zarr_paths = build_test_zarrs()
    df         = benchmark_all(zarr_paths)

    print("\n=== 結果摘要 ===")
    print(df.to_string(index=False))

    plot_results(df)
    print("\n完成！")
