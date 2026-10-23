# Spatiotemporal Model Visualization Guide

## Overview
This guide explains the 6 types of paper-quality visualizations generated for the spatiotemporal weather prediction model. Each visualization highlights different aspects of the 4D (latitude, longitude, altitude, time) predictions.

---

## Generated Visualizations

### 1. **Spatiotemporal Evolution** (`*_spatiotemporal_evolution.png`)
**Purpose**: Shows how predictions evolve through TIME and SPACE simultaneously

**What it shows**:
- **Top row (6 panels)**: Temperature predictions at 6 different time windows
- **Middle row (6 panels)**: Wind speed predictions at 6 different time windows  
- **Bottom row (6 panels)**: Temperature error at 6 different time windows

**Key insights**:
- Observe spatial patterns in predictions across the flight route
- See how weather variables change over time
- Identify if errors are concentrated at specific times/locations

**For paper**: Use this to demonstrate temporal consistency and spatial coverage

---

### 2. **3D Trajectory** (`*_3d_trajectory.png`)
**Purpose**: Visualizes the ALTITUDE dimension - critical for aviation weather

**What it shows** (6 subplots):
- Ground truth Temperature in 3D (lat, lon, altitude)
- Predicted Temperature in 3D
- Temperature error in 3D
- Ground truth Wind Speed in 3D
- Predicted Wind Speed in 3D
- Wind Speed error in 3D

**Key insights**:
- See how atmospheric variables change with altitude (vertical structure)
- Identify if model captures temperature inversions or wind shear
- Check if errors are altitude-dependent

**For paper**: Essential for showing your model understands 3D atmospheric physics

---

### 3. **Altitude Profiles** (`*_altitude_profiles.png`)
**Purpose**: Shows vertical atmospheric structure (like traditional weather soundings)

**What it shows**:
- Each weather variable plotted against altitude
- Points colored by TIME to show temporal evolution
- Ground truth vs predictions overlaid

**Key insights**:
- Does the model capture atmospheric layering?
- Temperature should generally decrease with altitude
- Pressure should decrease exponentially with altitude
- Wind patterns vary with altitude

**For paper**: Demonstrates physical realism of predictions

---

### 4. **Time Series at Different Altitudes** (`*_timeseries_altitudes.png`)
**Purpose**: Shows temporal evolution stratified by altitude

**What it shows**:
- Top-left: Temperature over time at low/mid/high altitudes
- Top-right: Wind speed over time at low/mid/high altitudes
- Bottom-left: Pressure over time at low/mid/high altitudes
- Bottom-right: How prediction error varies over time

**Key insights**:
- Are predictions better at certain altitudes?
- Does model maintain temporal coherence?
- When do predictions degrade (beginning/end of route)?

**For paper**: Shows model handles multi-altitude scenarios

---

### 5. **Scatter Plots** (`*_scatter.png`)
**Purpose**: Standard predicted vs ground truth validation

**What it shows**:
- 6 scatter plots (one per weather variable)
- Points colored by ALTITUDE (spatiotemporal context)
- Perfect prediction line (y=x) for reference
- Statistics: MAE, RMSE, R²

**Key insights**:
- Overall prediction accuracy
- Systematic biases (points consistently above/below line)
- R² score indicates explained variance
- Altitude coloring reveals altitude-dependent patterns

**For paper**: Standard validation plot expected by reviewers

---

### 6. **Error Distribution** (`*_error_dist.png`)
**Purpose**: Statistical analysis of prediction errors

**What it shows**:
- Histogram of errors for each weather variable
- Mean error (should be near zero for unbiased predictions)
- Standard deviation (spread of errors)

**Key insights**:
- Are errors normally distributed (Gaussian)?
- Systematic bias (mean ≠ 0)?
- Heavy tails indicate occasional large errors

**For paper**: Demonstrates statistical properties of errors

---

## Key Metrics Explained

### MAE (Mean Absolute Error)
- Average magnitude of errors
- **Lower is better**
- Same units as the predicted variable
- Example: MAE = 8.45°C for temperature

### RMSE (Root Mean Square Error)
- Emphasizes larger errors more than MAE
- **Lower is better**
- Always ≥ MAE
- RMSE >> MAE indicates occasional large errors

### R² (Coefficient of Determination)
- Fraction of variance explained by the model
- **Range**: -∞ to 1.0 (1.0 is perfect)
- **Interpretation**:
  - R² = 1.0: Perfect predictions
  - R² = 0.5: Model explains 50% of variance
  - R² = 0.0: Model no better than predicting the mean
  - R² < 0: Model worse than predicting the mean

---

## Results Summary

### Route 1
- Temperature: MAE=8.45°C, RMSE=15.41°C, R²=0.501
- Wind Speed: MAE=7.36 m/s, RMSE=17.31 m/s, R²=0.847
- Pressure: MAE=10.30 hPa, RMSE=13.15 hPa, R²=0.999
- **18,581 prediction points**

### Route 2
- Temperature: MAE=8.09°C, RMSE=14.81°C, R²=0.464
- Wind Speed: MAE=7.21 m/s, RMSE=17.38 m/s, R²=0.846
- Pressure: MAE=10.89 hPa, RMSE=13.88 hPa, R²=0.998
- **18,110 prediction points**

---

## Recommendations for Paper

### Main Figures:
1. **Figure 1**: 3D Trajectory (shows spatiotemporal nature)
2. **Figure 2**: Scatter plots (standard validation)
3. **Figure 3**: Spatiotemporal evolution (unique to your approach)
4. **Figure 4**: Time series at altitudes (demonstrates multi-scale capability)

### Supplementary:
- Altitude profiles (physical realism)
- Error distributions (statistical properties)

### In Text:
- Emphasize the **4D nature** (lat, lon, altitude, time)
- Highlight **excellent pressure prediction** (R² ≈ 0.999)
- Discuss **wind speed performance** (R² ≈ 0.847)
- Address **temperature challenges** (R² ≈ 0.5) - still useful for aviation
- Note: Wind U/V are zero because route data uses wind speed/direction instead

---

## File Outputs

All visualizations are saved at **300 DPI** (publication quality) in PNG format:

- `predictions_route1_spatiotemporal_evolution.png`
- `predictions_route1_3d_trajectory.png`
- `predictions_route1_altitude_profiles.png`
- `predictions_route1_timeseries_altitudes.png`
- `predictions_route1_scatter.png`
- `predictions_route1_error_dist.png`
- `predictions_route1.csv` (raw data)

(Same for route 2)

---

## Citation Suggestions

When describing your visualizations in the paper:

> "To validate the spatiotemporal prediction capability, we visualize the results across four dimensions (Figure X). The 3D trajectory plots demonstrate the model's ability to capture vertical atmospheric structure, while temporal evolution heatmaps show consistent prediction accuracy across different flight segments and time windows."

> "Scatter plots (Figure Y) reveal strong performance for pressure (R²=0.999) and wind speed (R²=0.847), with altitude-stratified coloring indicating the model successfully handles multi-altitude scenarios."
