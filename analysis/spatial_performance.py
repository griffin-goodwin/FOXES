"""
Flux-Weighted Error Heatmap on Solar Disk
==========================================
For each matched flux map, accumulates:
    mae_sum[i,j]  += flux[i,j] * |log10 error|
    bias_sum[i,j] += flux[i,j] *  log10 error
    weight[i,j]   += flux[i,j]

Then normalizes to get flux-weighted mean error per patch.

Usage
-----
    python analysis/spatial_performance.py

Outputs
-------
    analysis/flux_weighted_errors_t0.npz     — accumulation cache
    analysis/performance_heatmap_all.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from cmap import Colormap

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from forecasting.inference.evaluation import setup_barlow_font

# ---------------------------------------------------------------------------
# Paths — edit here
# ---------------------------------------------------------------------------
FLUX_DIR        = "/Volumes/T9/FOXES_Data/flux/"
PREDICTIONS_CSV = "/Volumes/T9/FOXES_Misc/batch_results/vit/vit_predictions_test.csv"
OUT_DIR         = Path(__file__).parent
GRID_SIZE       = 64    # 512px / 8px patch size
BIN_SIZE        = 1     # downsample factor (1 = full 64×64 resolution)
CROP_FACTOR     = 1.1   # AIA images cropped at 1.1 solar radii
SOLAR_RADIUS_PATCHES = (GRID_SIZE / 2) / CROP_FACTOR   # ≈ 29.1 patches

# Only patches above this percentile (per flux map) contribute.
# 0 = include all non-zero patches.
FLUX_THRESHOLD_PERCENTILE = 0

# Percentile cap for colorbar scaling (applied to non-NaN values).
# e.g. 99 clips the top 1% of values so detail in the bulk is visible.
VMAX_PERCENTILE = 99.9

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype(str).str.replace("_", ":", regex=False), utc=False,
    ).dt.floor("s")


def _ts_key(fpath: str) -> str:
    raw = os.path.basename(fpath).replace('.npy', '').replace('_', ':')
    return pd.Timestamp(raw).floor('s').isoformat()


def load_predictions(predictions_csv: str) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    df["timestamp"] = normalize_ts(df["timestamp"])
    df["log_pred"]      = np.log10(df["predictions"])
    df["log_gt"]        = np.log10(df["groundtruth"])
    df["log_error"]     = df["log_pred"] - df["log_gt"]
    df["log_abs_error"] = df["log_error"].abs()
    print(f"Loaded {len(df)} predictions")
    return df


# ---------------------------------------------------------------------------
# Heatmap accumulation
# ---------------------------------------------------------------------------

# NOTE: module-level for ProcessPoolExecutor (spawn on macOS)
def _accumulate_flux_map(args):
    fpath, log_abs_error, log_error, threshold_pct = args
    fmap = np.load(fpath).astype(np.float64)
    active = fmap[fmap > 0]
    if active.size == 0:
        return None
    if threshold_pct > 0:
        thresh = np.percentile(active, threshold_pct)
        fmap = np.where(fmap >= thresh, fmap, 0.0)
    else:
        fmap = np.where(fmap > 0, fmap, 0.0)
    total = fmap.sum()
    if total == 0:
        return None
    fmap = fmap / total  # normalise: each timestamp contributes equal total weight
    return fmap * log_abs_error, fmap * log_error, fmap


def compute_flux_weighted_errors(flux_dir: str, df: pd.DataFrame, cache_path: Path,
                                  threshold_pct: int = FLUX_THRESHOLD_PERCENTILE) -> dict:
    cache_path = cache_path.with_stem(f"{cache_path.stem}_t{threshold_pct}")

    if cache_path.exists():
        print(f"Loading cached flux-weighted error maps from {cache_path}")
        data = np.load(cache_path)
        result = {}
        n = float(data['count'])
        w = data['weight']
        mae  = np.where(w > 0, data['mae_sum']  / n, np.nan) if n > 0 else np.full_like(w, np.nan)
        bias = np.where(w > 0, data['bias_sum'] / n, np.nan) if n > 0 else np.full_like(w, np.nan)
        return mae, bias, w

    lookup = {}
    for _, row in df.iterrows():
        key = pd.Timestamp(row['timestamp']).floor('s').isoformat()
        lookup[key] = (float(row['log_abs_error']), float(row['log_error']))

    shape = (GRID_SIZE, GRID_SIZE)
    mae_sum  = np.zeros(shape)
    bias_sum = np.zeros(shape)
    weight   = np.zeros(shape)
    count    = 0

    files = sorted([os.path.join(flux_dir, f)
                    for f in os.listdir(flux_dir) if f.endswith('.npy')])
    args_list = []
    for fpath in files:
        try:
            ts_key = _ts_key(fpath)
        except Exception:
            continue
        if ts_key not in lookup:
            continue
        abs_err, err = lookup[ts_key]
        args_list.append((fpath, abs_err, err, threshold_pct))

    print(f"Matched {len(args_list)} / {len(files)} flux maps")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(_accumulate_flux_map, a): i
                   for i, a in enumerate(args_list)}
        for future in tqdm(as_completed(futures), total=len(args_list),
                           desc="Accumulating flux-weighted errors"):
            result = future.result()
            if result is None:
                continue
            mae_c, bias_c, flux_c = result
            mae_sum  += mae_c
            bias_sum += bias_c
            weight   += flux_c
            count    += 1

    np.savez(cache_path, mae_sum=mae_sum, bias_sum=bias_sum,
             weight=weight, count=np.array(count))
    print(f"Saved → {cache_path}")

    mae  = np.where(weight > 0, mae_sum  / count, np.nan) if count > 0 else np.full(shape, np.nan)
    bias = np.where(weight > 0, bias_sum / count, np.nan) if count > 0 else np.full(shape, np.nan)
    return mae, bias, weight


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _bin_grid(grid: np.ndarray, bin_size: int) -> np.ndarray:
    if bin_size == 1:
        return grid
    h, w = grid.shape
    bh, bw = h // bin_size, w // bin_size
    cropped = grid[:bh * bin_size, :bw * bin_size]
    return np.nanmean(cropped.reshape(bh, bin_size, bw, bin_size), axis=(1, 3))


def plot_flux_weighted_heatmap(mae_grid: np.ndarray, bias_grid: np.ndarray,
                                weight_grid: np.ndarray, out_path: Path,
                                subtitle: str = "", bin_size: int = BIN_SIZE,
                                vmax_pct: int = VMAX_PERCENTILE):
    setup_barlow_font()
    text_color = "#111111"
    theta      = np.linspace(0, 2 * np.pi, 300)

    mae_b    = _bin_grid(mae_grid, bin_size)
    bias_b   = _bin_grid(bias_grid, bin_size)
    weight_b = _bin_grid(weight_grid, bin_size)
    n_bins   = mae_b.shape[0]
    cy, cx   = n_bins / 2, n_bins / 2

    r_limb       = SOLAR_RADIUS_PATCHES / bin_size

    mae_vmax  = np.nanpercentile(mae_b, vmax_pct)
    mae_norm  = plt.Normalize(vmin=0, vmax=mae_vmax)

    bias_cap  = np.nanpercentile(np.abs(bias_b), vmax_pct)
    bias_norm = plt.Normalize(vmin=-bias_cap, vmax=bias_cap)

    panels = [
        (mae_b,   r"Flux-Weighted MAE (log$_{10}$ Space)",  Colormap('cmocean:thermal').to_mpl(),  mae_norm),
        (bias_b,  r"Flux-Weighted MBE (log$_{10}$ Space)",  Colormap('cmasher:fusion_r').to_mpl(), bias_norm),
        # (np.log10(np.where(weight_b > 0, weight_b, np.nan)),
        #           r"log$_{10}$ Accumulated Flux",                 "viridis", None),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("white")

    for ax, (grid, title, cmap, norm) in zip(axes, panels):
        im = ax.imshow(grid, origin="lower", cmap=cmap, norm=norm,
                       interpolation="bilinear", extent=[0, n_bins, 0, n_bins])
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.ax.tick_params(labelsize=9, colors=text_color)
        def _fmt(x, _):
            m, e = f"{x:.2e}".split("e")
            return f"{m}e{int(e)}"
        cbar.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_fmt))
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontfamily("Barlow")
            lbl.set_fontsize(9)
            lbl.set_color(text_color)
        #cbar.set_label(title, fontsize=9, color=text_color, fontfamily="Barlow")

        ax.plot(cx + r_limb       * np.cos(theta), cy + r_limb       * np.sin(theta),
                color="#4488FF", linestyle="--", linewidth=1.2, alpha=0.8,
                label=f"Solar Limb")

        tick_bins   = np.linspace(0, n_bins, 7)
        tick_labels = [f"{int((t - n_bins / 2) * bin_size)}" for t in tick_bins]
        ax.set_xticks(tick_bins); ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_bins); ax.set_yticklabels(tick_labels)

        ax.set_title(title, fontsize=10, color=text_color, fontfamily="Barlow")
        ax.set_xlabel("Solar X (ViT Patches From Center)", fontsize=9,
                      color=text_color, fontfamily="Barlow")
        ax.set_ylabel("Solar Y (ViT Patches From Center)", fontsize=9,
                      color=text_color, fontfamily="Barlow")
        ax.tick_params(labelsize=8, colors=text_color)
        ax.legend(fontsize=7, facecolor="white", edgecolor="grey", loc="upper right",)
        for spine in ax.spines.values():
            spine.set_color(text_color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux_dir",        default=FLUX_DIR)
    parser.add_argument("--predictions_csv", default=PREDICTIONS_CSV)
    parser.add_argument("--out_dir",         default=str(OUT_DIR))
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_predictions(args.predictions_csv)

    mae, bias, weight = compute_flux_weighted_errors(
        args.flux_dir, df, out / "flux_weighted_errors.npz"
    )
    plot_flux_weighted_heatmap(mae, bias, weight,
                               out / "performance_heatmap_all.png",
                               subtitle="All flares")
