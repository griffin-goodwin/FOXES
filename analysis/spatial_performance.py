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

from matplotlib.colors import LogNorm
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
BIN_SIZE        = 1    # downsample factor (1 = full 64×64 resolution)
CROP_FACTOR     = 1.1   # AIA images cropped at 1.1 solar radii
SOLAR_RADIUS_PATCHES = (GRID_SIZE / 2) / CROP_FACTOR   # ≈ 29.1 patches

# Patches beyond ±PATCH_CROP_RADIUS from center (in original 64×64 patch units) are masked.
PATCH_CROP_RADIUS = 24

# Percentile cap for colorbar scaling (applied to non-NaN values).
# e.g. 99 clips the top 1% of values so detail in the bulk is visible.
VMAX_PERCENTILE = 99

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
    fpath, log_abs_error, log_error, bin_size = args
    fmap = np.load(fpath).astype(np.float64)

    active = fmap[fmap > 0]
    if active.size == 0:
        return None

    fmap = np.where(fmap > 0, fmap, 0.0)

    # Spatially bin before normalization — sum preserves relative log-flux within each bin
    if bin_size > 1:
        h, w = fmap.shape
        bh, bw = h // bin_size, w // bin_size
        fmap = fmap[:bh * bin_size, :bw * bin_size].reshape(bh, bin_size, bw, bin_size).sum(axis=(1, 3))

    total = fmap.sum()
    if total == 0:
        return None
    fmap = fmap / total  # normalise: each timestamp contributes equal total weight
    return fmap * log_abs_error, fmap * log_error, fmap


def _crop_mask(shape: tuple, bin_size: int, radius: int = PATCH_CROP_RADIUS) -> np.ndarray:
    """True for patches within ±radius original-grid patches from center (y-axis only)."""
    n = shape[0]
    r_binned = radius / bin_size
    cy = (n - 1) / 2
    y = np.ogrid[:n, :n][0]
    return np.abs(y - cy) <= r_binned

def compute_flux_weighted_errors(flux_dir: str, df: pd.DataFrame, cache_path: Path,
                                  bin_size: int = BIN_SIZE) -> dict:
    cache_path = cache_path.with_stem(f"{cache_path.stem}_b{bin_size}")

    if cache_path.exists():
        print(f"Loading cached flux-weighted error maps from {cache_path}")
        data = np.load(cache_path)
        n = float(data['count'])
        w    = data['flux_distribution']
        mask = _crop_mask(w.shape, bin_size)
        mae  = np.where(mask, data['mae_sum'] / w, np.nan) if n > 0 else np.full_like(w, np.nan)
        bias = np.where(mask, data['bias_sum'] / w, np.nan) if n > 0 else np.full_like(w, np.nan)
        return mae, bias, w

    lookup = {}
    for _, row in df.iterrows():
        key = pd.Timestamp(row['timestamp']).floor('s').isoformat()
        lookup[key] = (float(row['log_abs_error']), float(row['log_error']))

    binned_grid = GRID_SIZE // bin_size
    shape = (binned_grid, binned_grid)
    mae_sum           = np.zeros(shape)
    bias_sum          = np.zeros(shape)
    flux_distribution = np.zeros(shape)
    count             = 0

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
        args_list.append((fpath, abs_err, err, bin_size))

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
            mae_sum          += mae_c
            bias_sum         += bias_c
            flux_distribution += flux_c
            count            += 1

    np.savez(cache_path, mae_sum=mae_sum, bias_sum=bias_sum,
             flux_distribution=flux_distribution, count=np.array(count))
    print(f"Saved → {cache_path}")

    mask = _crop_mask(shape, bin_size)
    mae  = np.where(mask, mae_sum  / flux_distribution, np.nan) if count > 0 else np.full(shape, np.nan)
    bias = np.where(mask, bias_sum / flux_distribution, np.nan) if count > 0 else np.full(shape, np.nan)
    return mae, bias, flux_distribution


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

    # Grids are already pre-binned during accumulation — use directly
    mae_b  = mae_grid
    bias_b = bias_grid
    n_bins = mae_b.shape[0]
    cy, cx = n_bins / 2, n_bins / 2

    # Solar limb radius in binned-patch units
    r_limb = SOLAR_RADIUS_PATCHES / bin_size

    mae_vmax  = np.nanpercentile(mae_b, vmax_pct)
    mae_norm  = plt.Normalize(vmin=0, vmax=mae_vmax)

    bias_cap  = np.nanpercentile(np.abs(bias_b), vmax_pct)
    bias_norm = plt.Normalize(vmin=-bias_cap, vmax=bias_cap)

    panels = [
        (mae_b,   r"Normalized Flux-Weighted MAE",  Colormap('cmocean:thermal').to_mpl(),  mae_norm),
        (bias_b,  r"Normalized Flux-Weighted MBE",  Colormap('cmasher:fusion_r').to_mpl(), bias_norm),
        # (np.log10(np.where(weight_b > 0, weight_b, np.nan)),
        #           r"log$_{10}$ Accumulated Flux",                 "viridis", None),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("white")

    for ax, (grid, title, cmap, norm) in zip(axes, panels):
        im = ax.imshow(grid, origin="lower", cmap=cmap, norm=norm,
                       interpolation="bicubic", extent=[0, n_bins, 0, n_bins])
        cbar = fig.colorbar(im, ax=ax, shrink=0.82,norm=LogNorm(vmin=0, vmax=mae_vmax),)
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

        ax.set_title(title, fontsize=10, color=text_color, fontfamily="Barlow",)
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
