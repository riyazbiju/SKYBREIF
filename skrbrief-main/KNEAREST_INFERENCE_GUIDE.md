# K-Nearest Spatial Aggregation Inference Guide

## Overview

The new `predict_knearest.py` script enables weather predictions using the spatiotemporal model with **K=10 nearest station aggregation** and **real time ODE integration**.

## Key Features

### ✅ K-Nearest Spatial Aggregation
- Finds 10 nearest weather stations to any query location
- Uses haversine distance (great circle distance on Earth)
- Applies distance-weighted blending: `softmax(-distance/100km)`
- Closer stations have higher influence on predictions

### ✅ Real Temporal Integration  
- Integrates ODE through real time (not dummy 0→1)
- Uses observations from dataset start up to query time
- Integrates from observation time → query time
- Handles variable prediction horizons

### ✅ Arbitrary Query Points
- Predict at any `(latitude, longitude, altitude, time)` location
- Not limited to existing station locations
- Works anywhere within the station network coverage

## Usage

### Basic Command
```bash
python predict_knearest.py \
    --route data/realistic_flight_route.csv \
    --output predictions_knearest.csv \
    --K 10
```

### Command Line Arguments
- `--model`: Path to trained model (default: `best_spatiotemporal_model.pt`)
- `--data`: Path to station database CSV (default: `data/jan_sorted_cleaned.csv`)
- `--route`: Path to route CSV file (required)
- `--output`: Output CSV path (auto-generated if not specified)
- `--K`: Number of nearest stations (default: 10)
- `--max-waypoints`: Maximum waypoints to process (default: 1000)
- `--device`: Device to use (default: `cpu`)

### Route CSV Format

Required columns (flexible naming):
```csv
step_order,Latitude,Longitude,Altitude,Temperature,Wind_Speed,Visiblity,Pressure,Wind_Direction
0,49.25,-123.25,1000,6.76,8.28,10.0,900,122.5
1,49.25,-123.0,5000,-12.72,27.08,10.0,550,232.8
...
```

Column mapping:
- `Latitude` → latitude
- `Longitude` → longitude  
- `Altitude` → altitude (feet)
- `Temperature` → temperature (°C)
- `Wind_Speed` → windspeed (knots)
- `Visiblity` → visibility (miles)
- `Pressure` → pressure (hPa)
- `Wind_Direction` → wind_direction (degrees)
- `step_order` → time_hours (converted to hours)

## Output Format

The prediction CSV contains:

**Query Information:**
- `latitude`, `longitude`, `altitude`: Query location
- `time_hours`: Query time (hours from dataset start)

**Predictions:**
- `pred_temperature`: Predicted temperature (°C)
- `pred_windspeed`: Predicted wind speed (knots)
- `pred_visibility`: Predicted visibility (miles)
- `pred_pressure`: Predicted pressure (hPa)
- `pred_wind_direction`: Predicted wind direction (degrees)

**K-Nearest Info:**
- `nearest_stations`: Top 3 nearest station codes (comma-separated)
- `nearest_distances_km`: Distances to top 3 stations (km)

**Ground Truth (if available):**
- `true_temperature`, `true_windspeed`, etc.

## Performance Metrics

Example results from realistic flight route (30 waypoints):

| Variable        | MAE    | RMSE   | R²     |
|----------------|--------|--------|--------|
| Temperature    | 6.34°C | 6.98°C | 0.733  |
| Wind Speed     | 10.90  | 13.57  | 0.415  |
| Pressure       | 6.58   | 8.91   | 0.998  |
| Wind Direction | 28.73° | 33.77° | -      |

## Architecture Details

### Model: `SpatioTemporalLatentODE`
- **Input**: K sequences from nearest stations `[1, K, seq_len, 6]`
- **Spatial aggregation**: Distance-weighted attention over K stations
- **ODE integration**: From observation time to query time through real hours
- **Output**: Weather prediction at query point `[1, 6]`

### ODE Function: `SpatioTemporalODEFunc`
- **Spatial encoding**: Fixed `(lat, lon, alt)` → 32D embedding
- **Temporal encoding**: Dynamic `time` → 16D embedding  
- **Integration parameter**: Real normalized time (not dummy 0→1)
- **Returns**: `dz/dt` for latent state evolution

### Dataset: 230 Weather Stations
- **Coverage**: Western North America (primarily Canada)
- **Time range**: 0-743 hours (31 days)
- **Features**: temperature, windspeed, visibility, pressure, wind_u, wind_v
- **Spatial scale**: 100km for distance weighting

## Limitations & Warnings

### ⚠️ Early Time Queries (t < 1 hour)
- Queries near dataset start (t≈0) may fail ODE integration
- Model returns initial state (z0) when integration fails
- Recommendation: Query times ≥ 1 hour from dataset start

### ⚠️ Large Route Files  
- Routes with >1000 waypoints are automatically subsampled
- Use `--max-waypoints` to control subsampling
- Example: 946K waypoints → 1000 waypoints (every 946th point)

### ⚠️ Coverage Area
- Predictions work best within station network coverage
- Extrapolating far outside network may be unreliable
- Check `nearest_distances_km` to assess coverage quality

## Comparison with Old System

| Feature | Old (`quick_predict.py`) | New (`predict_knearest.py`) |
|---------|--------------------------|----------------------------|
| Spatial | Single station | K=10 nearest stations |
| Temporal | Dummy time (0→1) | Real time integration |
| Blending | None | Distance-weighted softmax |
| Coverage | Station locations only | Arbitrary (lat,lon,alt) |
| Horizons | Fixed | Variable (1-24 hours) |

## Examples

### Predict Realistic Flight Route
```bash
python predict_knearest.py \
    --route data/realistic_flight_route.csv \
    --output predictions_flight.csv \
    --K 10
```

### Predict Large Route (with subsampling)
```bash
python predict_knearest.py \
    --route data/route1_subset.csv \
    --output predictions_route1.csv \
    --max-waypoints 500 \
    --K 10
```

### Use Different Model
```bash
python predict_knearest.py \
    --model checkpoints/checkpoint_epoch_45.pt \
    --route data/realistic_flight_route.csv \
    --output predictions_epoch45.csv
```

## Next Steps

1. **Evaluate on route1 and route2** (with subsampling)
2. **Test different K values** (K=5, K=15, K=20)
3. **Analyze spatial distance vs. error** 
4. **Compare with baseline (single nearest station)**
5. **Create visualization** of predictions vs ground truth

## Files

- `predict_knearest.py`: Main inference script
- `spatiotemporal_latent_ode.py`: Model architecture
- `best_spatiotemporal_model.pt`: Trained model checkpoint
- `config_multi_station.py`: Configuration file
- `data/jan_sorted_cleaned.csv`: Station database (230 stations)
- `data/realistic_flight_route.csv`: Example route (30 waypoints)
- `data/route1_subset.csv`: Route 1 (946K waypoints)
- `data/route2_subset.csv`: Route 2 (varies)

## Training Details

The model was trained with:
- **Epochs**: 50  
- **Best validation loss**: 16.2732 (epoch 40)
- **Training examples**: 3000 (variable horizons: 1, 3, 6, 12, 24 hours)
- **Validation examples**: 4783 (1-hour horizon)
- **K-nearest**: 10 stations per example
- **Optimizer**: AdamW with ReduceLROnPlateau scheduler
- **Device**: CPU with RK4 ODE solver
