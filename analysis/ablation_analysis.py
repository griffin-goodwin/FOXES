import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from forecasting.inference.evaluation import setup_barlow_font

setup_barlow_font()

DATA_DIR = "/Users/griffingoodwin/Documents/gitrepos/FOXES/Untracked/data"
WAVELENGTHS = ["94", "131", "171", "193", "211", "304", "335", "stereo", "all"]
LABELS = ["Ablate 94 Å", "Ablate 131 Å", "Ablate 171 Å", "Ablate 193 Å",
          "Ablate 211 Å", "Ablate 304 Å", "Ablate 335 Å", "Ablate STEREO", "Ablate All"]

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Barlow', 'Arial', 'DejaVu Sans']

FLARE_CLASSES = {
    'A1.0': (1e-8, 1e-7),
    'B1.0': (1e-7, 1e-6),
    'C1.0': (1e-6, 1e-5),
    'M1.0': (1e-5, 1e-4),
    'X1.0': (1e-4, 1e-3),
}

text_color  = 'black'
grid_color  = '#CCCCCC'

VMIN_GLOBAL = 1e-9
VMAX_GLOBAL = 1e-2


def add_flare_class_axes(ax, vmin, vmax):
    def identity(x):
        return x

    ax_top   = ax.secondary_xaxis('top',   functions=(identity, identity))
    ax_right = ax.secondary_yaxis('right', functions=(identity, identity))

    positions, labels = [], []
    for cls, (lo, hi) in FLARE_CLASSES.items():
        if vmin <= lo <= vmax:
            positions.append(lo)
            labels.append(cls)

    ax_top.set_xticks(positions)
    ax_top.set_xticklabels(labels, fontsize=6, color=text_color, rotation=45, ha='left')
    ax_top.grid(False)
    ax_top.tick_params(length=3)

    ax_right.set_yticks(positions)
    ax_right.set_yticklabels(labels, fontsize=6, color=text_color)
    ax_right.grid(False)
    ax_right.tick_params(length=3)


fig, axes = plt.subplots(3, 3, figsize=(16, 14), layout='constrained')
axes = axes.flatten()

hb_last = None  # for shared colorbar

for i, (wav, label) in enumerate(zip(WAVELENGTHS, LABELS)):
    ab  = pd.read_csv(f"{DATA_DIR}/ablate_{wav}_global_1.csv")
    gt  = ab["groundtruth"].values
    pred = ab["predictions"].values

    mask = (gt > 0) & (pred > 0)
    gt, pred = gt[mask], pred[mask]

    log_mae = np.mean(np.abs(np.log10(gt) - np.log10(pred)))

    vmin = max(VMIN_GLOBAL, min(gt.min(), pred.min()))
    vmax = min(VMAX_GLOBAL, max(gt.max(), pred.max()))

    ax = axes[i]
    ax.set_facecolor("#FFFFFF")

    hb = ax.hexbin(gt, pred, gridsize=80, xscale='log', yscale='log',
                   cmap='bone', mincnt=1, bins='log',
                   extent=(np.log10(vmin), np.log10(vmax),
                           np.log10(vmin), np.log10(vmax)))
    hb_last = hb

    # 1:1 line
    ax.plot([vmin, vmax], [vmin, vmax], ls='--', c='red', alpha=0.85, lw=1.2)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xscale('log')
    ax.set_yscale('log')

    #ax.set_title(label, fontsize=11, fontweight='bold', color=text_color)
    ax.set_xlabel(r'Ground Truth (W/m$^2$)', fontsize=8, color=text_color)
    ax.set_ylabel(r'Prediction (W/m$^2$)', fontsize=8, color=text_color)
    ax.tick_params(labelsize=7, colors=text_color)
    ax.grid(True, alpha=0.5, color=grid_color, linewidth=0.5)
    ax.set_axisbelow(True)

    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily('Barlow')
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily('Barlow')

    ax.text(0.04, 0.96, f"Log MAE = {log_mae:.3f}",
            transform=ax.transAxes, fontsize=8, va='top', color=text_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.85))

    add_flare_class_axes(ax, vmin, vmax)

# Shared colorbar
cbar = fig.colorbar(hb_last, ax=axes.tolist(), orientation='vertical', shrink=0.6, pad=0.01)
cbar.set_label("Count (log)", fontsize=11, color=text_color)
cbar.ax.tick_params(labelsize=9, colors=text_color)
cbar.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs='auto', numticks=10))
cbar.ax.tick_params(which='minor', colors=text_color)

#fig.suptitle("Ablation Study: Channel Masking vs. Baseline", fontsize=14, fontweight='bold')
plt.savefig("/Users/griffingoodwin/Documents/gitrepos/FOXES/analysis/ablation_3x3.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: analysis/ablation_3x3.png")
