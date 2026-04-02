import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib import rcParams


def setup_barlow_font():
    try:
        barlow_fonts = [f.name for f in fm.fontManager.ttflist
                        if 'barlow' in f.name.lower() or 'barlow' in f.fname.lower()]
        if barlow_fonts:
            rcParams['font.family'] = 'Barlow'
        else:
            for path in ['/usr/share/fonts/truetype/barlow/Barlow-Regular.ttf',
                         '/Users/griffingoodwin/Library/Fonts/Barlow-Regular.otf']:
                if os.path.exists(path):
                    fm.fontManager.addfont(path)
                    rcParams['font.family'] = 'Barlow'
                    break
            else:
                rcParams['font.family'] = 'sans-serif'
    except Exception:
        rcParams['font.family'] = 'sans-serif'

setup_barlow_font()

DATA_DIR     = "/Users/griffingoodwin/Documents/gitrepos/FOXES/Untracked/data"
BASELINE_CSV = "/Volumes/T9/FOXES_Misc/batch_results/vit/vit_predictions_test.csv"

WAVELENGTHS = ["94", "131", "171", "193", "211", "304", "335","STEREO"]
LABELS = {
    "94":     "Ablate 94 Å",
    "131":    "Ablate 131 Å",
    "171":    "Ablate 171 Å",
    "193":    "Ablate 193 Å",
    "211":    "Ablate 211 Å",
    "304":    "Ablate 304 Å",
    "335":    "Ablate 335 Å",
    "STEREO": "Ablate 94, 131, 335 Å\n(STEREO)",
}

FLARE_CLASSES = {
    '< C': (1e-15, 1e-6),
    'C':  (1e-6, 1e-5),
    'M':  (1e-5, 1e-4),
    'X':  (1e-4, 1e-2),
}

CLASS_COLORS = {
    '< C': '#4C9BE8',
    'C':  '#56C490',
    'M':  '#F5A623',
    'X':  '#E84C4C',
}

# ── Compute metrics ────────────────────────────────────────────────────────────
def compute_row(label, gt, pred, is_baseline=False):
    mask = (gt > 0) & (pred > 0)
    gt, pred = gt[mask], pred[mask]
    overall = np.mean(np.abs(np.log10(gt) - np.log10(pred)))
    row = {"label": label, "overall": overall, "is_baseline": is_baseline}
    for cls, (lo, hi) in FLARE_CLASSES.items():
        m = (gt >= lo) & (gt < hi)
        row[cls] = np.mean(np.abs(np.log10(gt[m]) - np.log10(pred[m]))) if m.sum() > 5 else np.nan
    return row

records = []

# Baseline
bl = pd.read_csv(BASELINE_CSV)
records.append(compute_row("FOXES (no ablation)",
                           bl["groundtruth"].values, bl["predictions"].values,
                           is_baseline=True))

for wav in WAVELENGTHS:
    ab = pd.read_csv(f"{DATA_DIR}/ablate_{wav}_global_1.csv")
    records.append(compute_row(LABELS[wav], ab["groundtruth"].values, ab["predictions"].values))

# Sort ablation rows by overall MAE (worst first), keep baseline pinned at bottom
ablation_df = pd.DataFrame([r for r in records if not r["is_baseline"]])
ablation_df = ablation_df.sort_values("overall", ascending=False).reset_index(drop=True)
baseline_df = pd.DataFrame([r for r in records if r["is_baseline"]])
df = pd.concat([ablation_df, baseline_df], ignore_index=True)

# ── Plot ───────────────────────────────────────────────────────────────────────
n_rows = len(df)
fig, ax = plt.subplots(figsize=(11, 0.6 * n_rows + 1.5))
#ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("#FFFFFF")

y_positions = np.arange(n_rows)

# Separator line between ablations and baseline
ax.axhline(y=n_rows - 1.5, color="#BBBBBB", linewidth=1, linestyle=":", zorder=1)

for i, row in df.iterrows():
    y = y_positions[i]
    is_bl = row["is_baseline"]

    # Highlight baseline row
    if is_bl:
        ax.axhspan(y - 0.45, y + 0.45, color="#EEF6FF", zorder=0)

    # Span line across per-class range
    class_vals = [row[c] for c in FLARE_CLASSES if not np.isnan(row[c])]
    if class_vals:
        ax.hlines(y, min(class_vals), max(class_vals),
                  color="#CCCCCC", linewidth=2, zorder=1)

    # Stem from 0 to overall
    ax.hlines(y, 0, row["overall"],
              color="#AAAAAA", linewidth=1.2, linestyle="--", zorder=0, alpha=0.6)

    # Per-class dots
    for cls in FLARE_CLASSES:
        val = row[cls]
        if not np.isnan(val):
            ax.scatter(val, y, color=CLASS_COLORS[cls], s=80, zorder=4,
                       edgecolors="white", linewidths=0.6, alpha=0.75)

    # Overall dot
    outline_color = "#1A6BBF" if is_bl else "black"
    ax.scatter(row["overall"], y, color="white", s=190, zorder=3,
               edgecolors=outline_color, linewidths=2.0 if is_bl else 1.5, alpha=0.75)
    ax.scatter(row["overall"], y, color=outline_color, s=75, zorder=3,
               marker="|", linewidths=1.5, alpha=0.75)

tick_colors = ["black"] * n_rows
tick_colors[-1] = "#1A6BBF"  # baseline label in blue
ax.set_yticks(y_positions)
ax.set_yticklabels(df["label"], fontsize=12)
for ticklabel, color in zip(ax.get_yticklabels(), tick_colors):
    ticklabel.set_color(color)
    if color != "black":
        ticklabel.set_fontweight("bold")
ax.set_xlabel("MAE (log$_{10}$ scale)", fontsize=12)
ax.grid(True, axis="x", alpha=0.4, color="#CCCCCC", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=11)
ax.tick_params(axis="x", labelsize=10)

# Legend
class_patches = [
    mpatches.Patch(color=CLASS_COLORS[c], label=f"{c}-class") for c in FLARE_CLASSES
]
overall_patch   = mpatches.Patch(facecolor="white", edgecolor="black",  label="Overall")
#baseline_patch  = mpatches.Patch(facecolor="white", edgecolor="#1A6BBF", label="Baseline (overall)")
ax.legend(handles=class_patches + [overall_patch],
          loc="upper right", fontsize=10, framealpha=0.9,
          edgecolor="#CCCCCC")

# ax.set_title("Ablation Study — Log MAE by Channel & Flare Class",
#              fontsize=14, fontweight="bold", pad=14)
plt.xlim(0, .85)
plt.tight_layout()
plt.savefig("ablation_lollipop.png", dpi=450, bbox_inches="tight")
plt.show()
print("Saved: analysis/ablation_lollipop.png")
