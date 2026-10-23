# K-Nearest Inference Visualization Guide

## Quick Start

Generate predictions with visualizations:

```bash
python predict_knearest.py \
    --route data/realistic_flight_route.csv \
    --output predictions_realistic_knearest.csv \
    --K 10 \
    --visualize
```

## Generated Visualizations

### 1. Spatiotemporal Evolution (`*_spatiotemporal_evolution.png`)
**Size:** 589KB | **Dimensions:** 18×12 inches

Shows how weather variables evolve over time across the spatial domain.

**Layout:** 3 rows × 6 columns (18 panels)
- **Row 1:** Temperature predictions at 6 time periods
- **Row 2:** Wind speed predictions at 6 time periods  
- **Row 3:** Temperature error (or pressure if no ground truth)

**Use case:** Understanding temporal dynamics and spatial patterns

---

### 2. 3D Trajectory (`*_3d_trajectory.png`)
**Size:** 4.4MB | **Dimensions:** 18×25 inches

3D flight path colored by weather variable values.

**Layout (with ground truth):** 5 rows × 3 columns (15 panels)
- **Column 1:** Ground truth values along trajectory
- **Column 2:** Predicted values along trajectory
- **Column 3:** Absolute prediction error

**Variables covered:**
1. Temperature (°C)
2. Wind Speed (knots)
3. Visibility (miles)
4. Pressure (hPa)
5. Wind Direction (degrees)

**Layout (no ground truth):** 5 rows × 1 column (5 panels)
- Just predictions for each variable

**Use case:** Understanding vertical atmospheric structure and 3D spatial patterns

---

### 3. Scatter Plots (`*_scatter_plots.png`)
**Size:** 626KB | **Dimensions:** 18×12 inches

Predicted vs ground truth scatter plots with metrics.

**Layout:** 2 rows × 3 columns (5 panels + 1 empty)

Each panel shows:
- **X-axis:** Ground truth values
- **Y-axis:** Predicted values
- **Color:** Time (hours)
- **Red dashed line:** Perfect prediction (y=x)
- **Metrics:** R² and MAE in title

**Variables:**
1. Temperature
2. Wind Speed
3. Visibility
4. Pressure
5. Wind Direction

**Use case:** Quantifying prediction accuracy and identifying biases

---

### 4. Distance Analysis (`*_distance_analysis.png`)
**Size:** 188KB | **Dimensions:** 14×5 inches

Analyzes relationship between K-nearest station distances and prediction quality.

**Layout:** 1 row × 2 columns (2 panels)

**Panel 1: Distance Distribution**
- Histogram of distances to nearest stations
- Separate distributions for 1st, 2nd, 3rd nearest
- Shows spatial coverage quality

**Panel 2: Error vs Distance** (if ground truth available)
- Scatter plot: nearest station distance vs temperature error
- Colored by time
- Trend line showing correlation
- Formula: `y = slope*x + intercept`

**Use case:** 
- Understanding spatial interpolation quality
- Identifying areas with poor station coverage
- Validating distance-weighted blending

---

## Visualization Features

### Color Schemes
- **Temperature:** Red-Yellow-Blue reversed (`RdYlBu_r`)
- **Wind Speed:** Yellow-Orange-Red (`YlOrRd`)
- **Visibility:** Viridis (`viridis`)
- **Pressure:** Plasma (`plasma`)
- **Wind Direction:** Twilight (`twilight`)
- **Time:** Viridis (`viridis`)
- **Errors:** Red scale (`Reds`)

### Resolution
- All plots saved at **300 DPI** for publication quality
- Format: PNG with `bbox_inches='tight'`

---

## Example Results

From realistic flight route (30 waypoints):

### Prediction Metrics
| Variable        | MAE    | RMSE   | R²    |
|----------------|--------|--------|-------|
| Temperature    | 6.29°C | 6.90°C | 0.739 |
| Wind Speed     | 10.06  | 12.39  | 0.513 |
| Pressure       | 8.98   | 15.63  | 0.994 |
| Wind Direction | 29.37° | 34.13° | -     |

### Station Coverage
- **Nearest station:** 14-105 km
- **Average distance:** ~50 km
- **K=10 stations** provide robust spatial coverage

---

## Command Line Options

```bash
python predict_knearest.py \
    --route data/route.csv \          # Input route file (required)
    --output predictions.csv \        # Output predictions (optional)
    --K 10 \                          # Number of nearest stations (default: 10)
    --max-waypoints 1000 \            # Max waypoints to process (default: 1000)
    --visualize \                     # Enable visualizations (flag)
    --model best_model.pt \           # Model path (default: best_spatiotemporal_model.pt)
    --data data/stations.csv \        # Station database (default: data/jan_sorted_cleaned.csv)
    --device cpu                      # Device (default: cpu)
```

---

## Output Directory Structure

```
project/
├── predictions_realistic_knearest.csv          # Numerical predictions
├── visualizations/
│   ├── realistic_flight_route_knearest_spatiotemporal_evolution.png
│   ├── realistic_flight_route_knearest_3d_trajectory.png
│   ├── realistic_flight_route_knearest_scatter_plots.png
│   └── realistic_flight_route_knearest_distance_analysis.png
```

---

## Comparison: Old vs New Visualizations

| Aspect | quick_predict.py | predict_knearest.py |
|--------|-----------------|---------------------|
| **3D Trajectory** | ✅ 5 vars × 3 cols | ✅ 5 vars × 3 cols (same) |
| **Spatiotemporal Evolution** | ✅ 6 time bins | ✅ 6 time bins (same) |
| **Scatter Plots** | ✅ With R²/MAE | ✅ With R²/MAE (same) |
| **Altitude Profiles** | ✅ Included | ⚠️ Skipped (small routes) |
| **Time Series** | ✅ Multiple altitudes | ⚠️ Skipped (small routes) |
| **Distance Analysis** | ❌ Not available | ✅ **NEW FEATURE** |

### New Feature: Distance Analysis
- Shows K-nearest station distances
- Correlates distance with prediction error
- Validates spatial interpolation quality
- Unique to K-nearest architecture

---

## Tips for Best Visualizations

1. **Large routes (>100 waypoints):** Enable all visualizations
   ```bash
   --visualize --max-waypoints 1000
   ```

2. **Small routes (<50 waypoints):** Works but altitude/time series skipped
   - Spatiotemporal evolution still shows patterns
   - 3D trajectory captures full flight path
   - Distance analysis shows coverage

3. **No ground truth:** Still generates predictions + distance analysis
   - Spatiotemporal evolution shows predictions
   - 3D trajectory shows predictions only
   - Scatter plots skipped

4. **Custom output location:**
   ```bash
   --output results/flight_predictions.csv
   ```
   Visualizations go to `visualizations/{route_name}_knearest_*.png`

---

## Performance

- **Realistic flight (30 waypoints):** ~30 seconds total
  - Prediction: ~15 seconds
  - Visualization: ~15 seconds

- **Large route (1000 waypoints):** ~5-10 minutes
  - Subsampling handled automatically

---

## Troubleshooting

### "No visualizations created"
- Check `--visualize` flag is set
- Verify `visualizations/` directory exists (auto-created)

### "Altitude profiles skipped"
- Normal for routes <50 waypoints
- Time series needs sufficient data

### "ODE integration warnings"
- Normal for early times (t < 1 hour)
- Model returns z0 fallback
- Predictions still valid

### "Distance analysis empty"
- Check `nearest_distances_km` in CSV
- Should contain comma-separated values

---

## Next Steps

1. **Compare K values:** Test K=5, K=10, K=15, K=20
2. **Route evaluation:** Run on route1_subset.csv and route2_subset.csv
3. **Error analysis:** Study distance vs error correlation
4. **Publication figures:** All plots are 300 DPI publication-ready
