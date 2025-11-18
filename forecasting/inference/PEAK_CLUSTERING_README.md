# Peak-Based Region Clustering

## Overview

This implementation adds a new **peak-based clustering** method for detecting solar active regions. This approach addresses the issue of regions spuriously merging and unmerging by ensuring each region has a clear flux peak with flux decreasing outward.

## Problem It Solves

Standard DBSCAN clustering can merge distinct active regions that are spatially close, even when they have separate flux peaks. This causes:
- Regions incorrectly merging when they get close
- Regions splitting apart when they move slightly  
- Unstable region tracking over time

## How It Works

The peak-based clustering method works in 4 steps:

### 1. **Peak Detection**
- Identifies local maxima (peaks) in the flux map using a maximum filter
- Only considers peaks above a threshold: `base_threshold * peak_min_flux_multiplier`
- Each peak represents a potential active region center

### 2. **Patch Assignment**
For each significant patch, assigns it to the nearest peak IF:
- The peak is within `peak_assignment_max_distance` patches
- The peak has higher flux than the patch (patch is "downhill" from peak)
- The flux generally decreases along the path from peak to patch

### 3. **Flux Decrease Check**
Samples points along the line from peak to patch and verifies:
- At least `flux_decrease_strictness` fraction of consecutive points show flux decrease
- Allows small increases (10%) to handle noise
- This ensures patches belong to regions with radial flux decrease

### 4. **Region Formation**
- Groups patches assigned to the same peak into a region
- Applies morphological closing to fill small gaps
- Calculates region properties (size, flux, centroid, etc.)

## Configuration Parameters

Add these to your `flare_detection` section in the config YAML:

```yaml
flare_detection:
  # Enable the new method
  use_peak_clustering: true
  
  # Peak detection parameters
  peak_neighborhood_size: 3          # Size of window for local max (3 = 3x3)
  peak_min_flux_multiplier: 1.5      # Peaks must be 1.5x above base threshold
  
  # Assignment parameters  
  peak_assignment_max_distance: 10   # Max distance (patches) to assign to peak
  flux_decrease_strictness: 0.7      # 70% of path must show flux decrease
```

## Parameter Tuning Guide

### `peak_neighborhood_size` (default: 3)
- **Smaller (1-3)**: Detects more peaks, splits regions more aggressively
- **Larger (5-7)**: Detects fewer peaks, allows larger unified regions
- **Recommended**: 3 for typical solar active regions

### `peak_min_flux_multiplier` (default: 1.5)
- **Lower (1.2-1.5)**: More sensitive, detects weaker peaks
- **Higher (2.0-3.0)**: Only strong peaks, may miss small regions
- **Recommended**: 1.5 for balanced detection

### `peak_assignment_max_distance` (default: 10)
- **Smaller (5-8)**: Tighter regions, may fragment large regions
- **Larger (12-20)**: Larger regions, may merge close regions
- **Recommended**: 10 patches (~80 pixels for patch_size=8)

### `flux_decrease_strictness` (default: 0.7)
- **Lower (0.5-0.6)**: More lenient, tolerates flux irregularities
- **Higher (0.8-0.9)**: Stricter, requires smooth flux decrease
- **Recommended**: 0.7 for noisy data, 0.8 for clean data

## Comparison with DBSCAN

| Feature | DBSCAN | Peak Clustering |
|---------|--------|-----------------|
| **Basis** | Spatial density | Physical peak structure |
| **Region merging** | Can merge nearby regions | Keeps regions with separate peaks distinct |
| **Stability** | Can fluctuate | More stable over time |
| **Physics** | Agnostic | Respects radial flux decrease |
| **Speed** | Fast | Slightly slower (path checking) |

## Usage

To enable peak-based clustering, simply set in your config:

```yaml
use_peak_clustering: true
```

To revert to DBSCAN:

```yaml
use_peak_clustering: false
```

## Expected Results

With peak-based clustering, you should observe:
- **Fewer spurious merges**: Regions with separate peaks stay separate
- **More stable tracking**: Region IDs persist more consistently
- **Better flare attribution**: Each region maintains its identity through events
- **Physically meaningful boundaries**: Regions correspond to flux peaks

## Debugging

Each detected region now includes:
- `peak_y`, `peak_x`: Patch coordinates of the region's flux peak
- `peak_flux`: Flux value at the peak
- `bright_patch_y`, `bright_patch_x`: Brightest patch (should match peak)

Check these values to verify peaks are being detected correctly.

## Example

For a scene with two close active regions:
- **DBSCAN**: May merge them into one region if they're within `eps` distance
- **Peak clustering**: Keeps them separate as long as each has a distinct flux peak

This is particularly important for tracking regions through close encounters or partial overlaps.

