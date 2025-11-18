# Peak-Based Clustering Quick Start

## What Was Implemented

I've added a new **peak-based clustering** method that solves the region merging/unmerging problem by:

1. **Finding flux peaks** (local maxima) in each frame
2. **Assigning patches to peaks** only if:
   - They're within a maximum distance
   - The flux decreases from peak to patch (radial decrease pattern)
3. **Creating stable regions** around each peak

This ensures regions with distinct peaks stay separate, preventing spurious merging.

## Quick Start

### 1. Configuration (Already Set!)

Your config file has been updated with the new method enabled:

```yaml
# In patch_analysis_config_v6.yaml
flare_detection:
  use_peak_clustering: true    # NEW: Uses peak-based clustering
  
  # Peak detection parameters
  peak_neighborhood_size: 3
  peak_min_flux_multiplier: 1.5
  peak_assignment_max_distance: 10
  flux_decrease_strictness: 0.7
```

### 2. Run Your Analysis

Just run your normal analysis - it will automatically use peak clustering:

```bash
cd /home/ggoodwin5/FOXES/forecasting/inference

python patch_analysis_v6.py \
    --config patch_analysis_config_v6.yaml
```

### 3. Compare Methods (Optional)

Visualize the difference between peak clustering and DBSCAN:

```bash
python visualize_peak_detection.py \
    --config patch_analysis_config_v6.yaml \
    --timestamp "2024-08-01T11:39:00" \
    --output-dir ./peak_comparison/
```

This creates a 4-panel comparison showing:
- Flux map with detected peaks marked
- Regions from peak clustering
- Regions from DBSCAN
- Difference map

## Key Parameters to Tune

If you need to adjust behavior:

### `peak_neighborhood_size: 3`
Controls how close peaks can be:
- Smaller → More peaks detected (splits regions more)
- Larger → Fewer peaks (allows larger regions)

### `peak_min_flux_multiplier: 1.5`
Controls peak sensitivity:
- Lower (1.2) → Detects weaker peaks
- Higher (2.0) → Only strong peaks

### `peak_assignment_max_distance: 10`
Maximum distance to assign patches to peaks:
- Smaller (5-8) → Tighter, more compact regions
- Larger (15-20) → Larger regions

### `flux_decrease_strictness: 0.7`
How strictly to enforce flux decrease (0-1):
- Lower (0.5) → More lenient (tolerates noisy data)
- Higher (0.9) → Stricter (requires smooth decrease)

## What You Should See

### Improvements:
✓ **Fewer spurious merges**: Two close regions with separate peaks stay separate  
✓ **More stable tracking**: Region IDs persist longer  
✓ **Better flare attribution**: Each peak = one region  
✓ **Physical validity**: Matches expected radial flux decrease pattern

### New Region Data:
Each detected region now includes:
- `peak_y`, `peak_x`: Location of flux peak
- `peak_flux`: Flux value at peak

## Switching Back to DBSCAN

If you want to compare or switch back:

```yaml
use_peak_clustering: false  # Uses original DBSCAN
```

## Troubleshooting

### "Too many regions detected"
→ Increase `peak_min_flux_multiplier` or `peak_neighborhood_size`

### "Regions are too small/fragmented"
→ Increase `peak_assignment_max_distance` or decrease `flux_decrease_strictness`

### "Missing weak regions"
→ Decrease `peak_min_flux_multiplier` or `threshold_std_multiplier`

### "Regions still merging"
→ Check that peaks are being detected correctly using the visualization script

## Files Modified

1. **patch_analysis_v6.py**: Added peak clustering implementation
2. **patch_analysis_config_v6.yaml**: Added new parameters, enabled by default
3. **PEAK_CLUSTERING_README.md**: Full documentation
4. **visualize_peak_detection.py**: Comparison tool (new)

## Questions?

The peak clustering algorithm:
- Finds local maxima in flux using maximum filters
- Assigns patches by checking spatial distance + flux monotonicity
- Each region guaranteed to have exactly one flux peak
- Prevents merging by keeping peak-centered regions distinct

This should significantly reduce the merge/unmerge flickering you were seeing!

