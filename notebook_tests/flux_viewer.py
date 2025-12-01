import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- configure these paths as needed ---
# Directory where flux contribution files were saved by inference
flux_dir = Path("/data/FOXES_Data/batch_results/vit/flux")  # adjust if your run dir is different

# Timestamp of interest
timestamp = "2012-02-06T21:17:00"

# Full path to the flux file (saved as CSV by FOXES inference)
flux_file = flux_dir / timestamp

# Load flux map
flux_map = np.loadtxt(flux_file, delimiter=",")

# Plot
plt.figure(figsize=(6, 5))
im = plt.imshow(
    flux_map,
    origin="lower",
    cmap="magma",
    interpolation="nearest",
)
plt.colorbar(im, label="Flux contribution (arbitrary units)")
plt.title(f"Flux map {timestamp}")
plt.xlabel("Patch X")
plt.ylabel("Patch Y")
plt.tight_layout()
plt.savefig(f"flux_map_{timestamp}.png")
plt.show()
