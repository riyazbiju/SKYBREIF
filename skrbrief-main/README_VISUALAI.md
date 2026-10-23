# VisualAI Weather Prediction System

## Overview

This system connects a trained SpatioTemporalLatentODE deep learning model with a React-based web interface for real-time weather predictions along flight routes.

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  React Frontend │ ◄─────► │   Flask API      │ ◄─────► │  PyTorch Model   │
│   (Port 5173)   │  CORS   │   (Port 5001)    │         │  (best_model.pt) │
└─────────────────┘         └──────────────────┘         └──────────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │ Station Database │
                            │  (230 stations)  │
                            └──────────────────┘
```

## Components

### 1. Backend API (`api_server.py`)

**Flask REST API** that serves weather predictions using the trained model.

**Endpoints:**
- `GET /health` - Server health check
- `GET /info` - Model and dataset information
- `POST /predict/point` - Single point prediction
- `POST /predict/route` - Route prediction (multiple waypoints)

**Model Features:**
- K=10 nearest station spatial aggregation
- Distance-weighted blending (softmax over haversine distances)
- Real-time ODE integration
- Predicts: temperature, windspeed, visibility, pressure, wind_direction

**Dataset:**
- 230 weather stations
- Coverage: Western North America (primarily Canada)
- Time range: January 2026 (0-743 hours, ~31 days)
- Features: temperature, windspeed, visibility, pressure, wind components

### 2. Frontend UI (`visualai/`)

**React + TypeScript + Vite** web application with aviation-themed UI.

**Pages:**
- `/` - Home page
- `/point` - Single point query form
- `/point/results` - Display point prediction results
- `/route` - Route CSV upload
- `/route/results` - Display route predictions

**Features:**
- Interactive map visualization
- Real-time weather data display
- Aviation-style metrics (feet, knots, statute miles)
- Responsive design with shadcn/ui components

### 3. API Client (`visualai/src/utils/api.ts`)

TypeScript client for making API calls from the frontend.

**Functions:**
- `predictPoint(query)` - Single point prediction
- `predictRoute(waypoints)` - Route prediction
- `getHealth()` - Health check
- `getInfo()` - Model information

## Setup & Installation

### Backend Setup

1. **Install Python dependencies:**
```bash
cd /Users/harshasathish/Desktop/MAI/latent_ode_new
pip install -r requirements.txt
```

2. **Start the API server:**
```bash
python api_server.py
```

The server will start on `http://localhost:5001`

### Frontend Setup

1. **Install Node.js dependencies:**
```bash
cd visualai
npm install
```

2. **Fix permissions (if needed on macOS):**
```bash
chmod +x node_modules/.bin/vite
```

3. **Start the development server:**
```bash
npm run dev
```

The frontend will start on `http://localhost:5173`

## Usage

### Testing the API

Use the provided test script:

```bash
python test_api.py
```

This will test all endpoints and display sample predictions.

### Single Point Prediction

**API Request:**
```bash
curl -X POST http://localhost:5001/predict/point \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 56.1304,
    "lon": -106.3468,
    "alt": 10000,
    "datetime": "2026-01-15T12:00:00Z"
  }'
```

**Response:**
```json
{
  "temperature": -50.5,
  "windspeed": 71.9,
  "wind_direction": 311,
  "pressure": 272.6,
  "visibility": 18.3,
  "nearest_stations": ["CYVC", "CYVT", "CYKJ"],
  "distances_km": [128.4, 132.8, 132.8]
}
```

### Route Prediction

**API Request:**
```bash
curl -X POST http://localhost:5001/predict/route \
  -H "Content-Type: application/json" \
  -d '{
    "waypoints": [
      {"lat": 56.1304, "lon": -106.3468, "alt": 10000, "datetime": "2026-01-15T12:00:00Z"},
      {"lat": 56.5, "lon": -106.0, "alt": 15000, "datetime": "2026-01-15T13:00:00Z"}
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "temperature": -50.5,
      "windspeed": 78.1,
      "wind_direction": 307,
      "pressure": 271.7,
      "visibility": 15.9
    },
    ...
  ],
  "count": 2
}
```

## Frontend Usage

### Point Query

1. Navigate to `http://localhost:5173/point`
2. Enter coordinates (latitude, longitude)
3. Enter altitude (feet)
4. Select date and time
5. Click "Analyze Weather"
6. View results with map visualization

### Route Query

1. Navigate to `http://localhost:5173/route`
2. Upload a CSV file with columns: `latitude`, `longitude`, `altitude`, `datetime`
3. Click "Analyze Route"
4. View route predictions with visualizations

## Data Format

### Coordinates
- Latitude: Decimal degrees (-90 to 90)
- Longitude: Decimal degrees (-180 to 180)
- Altitude: Feet (frontend) / Meters (backend conversion)

### Time
- Format: ISO 8601 (`2026-01-15T12:00:00Z`)
- Range: January 2026 (dataset coverage)
- Must be at least 1 hour after 2026-01-01 00:00:00 UTC

### Weather Units
- Temperature: °C
- Wind Speed: knots
- Wind Direction: degrees (0-360, meteorological convention)
- Pressure: hPa
- Visibility: km (backend) / statute miles (frontend display)

## Model Details

### Architecture
- **Type:** SpatioTemporalLatentODE
- **Spatial Aggregation:** K=10 nearest stations
- **Weighting:** softmax(-distance/100km)
- **ODE Solver:** RK4 (Runge-Kutta 4th order)
- **Latent Dimension:** 64
- **Hidden Dimension:** 128

### Training
- **Epochs:** 50
- **Best Validation Loss:** 16.27
- **Variable Horizons:** 1, 3, 6, 12, 24 hours
- **Training Data:** 230 stations × 31 days

### Performance Metrics

**Realistic Flight Route (30 waypoints):**
- Temperature MAE: 6.29°C, R²: 0.739
- Pressure MAE: 8.98 hPa, R²: 0.994

**Route 1 (1000 waypoints):**
- Temperature MAE: 9.09°C, R²: 0.578
- Pressure MAE: 20.35 hPa, R²: 0.994

**Route 2 (999 waypoints):**
- Temperature MAE: 9.24°C, R²: 0.472
- Pressure MAE: 22.14 hPa, R²: 0.993

## Files

### Backend
- `api_server.py` - Flask API server
- `spatiotemporal_latent_ode.py` - Model architecture
- `best_spatiotemporal_model.pt` - Trained model checkpoint
- `data/jan_sorted_cleaned.csv` - Station database
- `test_api.py` - API test script

### Frontend
- `visualai/src/App.tsx` - Main app with routing
- `visualai/src/pages/PointInput.tsx` - Point query form
- `visualai/src/pages/PointResults.tsx` - Point results display
- `visualai/src/pages/RouteInput.tsx` - Route upload form
- `visualai/src/pages/RouteResults.tsx` - Route results display
- `visualai/src/utils/api.ts` - API client

### Documentation
- `README_VISUALAI.md` - This file
- `KNEAREST_INFERENCE_GUIDE.md` - K-nearest inference guide
- `VISUALIZATION_GUIDE.md` - Visualization guide

## Troubleshooting

### Port 5000 in use (macOS AirPlay)
The API server uses port 5001 to avoid conflicts with AirPlay Receiver on macOS.

### CORS Issues
The API server has CORS enabled for all origins. If you encounter CORS errors:
1. Check that the API server is running
2. Verify the API_BASE_URL in `visualai/src/utils/api.ts` is correct
3. Check browser console for specific CORS errors

### Time Range Errors
Predictions are limited to the dataset time range (January 2026):
- Minimum: 2026-01-01 01:00:00 UTC (1 hour after start)
- Maximum: 2026-01-31 23:00:00 UTC (~743 hours)

### ODE Integration Warnings
Warnings like "ODE integration failed: t must be strictly increasing" are normal and handled gracefully:
- Occurs when query time equals observation time
- Model falls back to spatial interpolation only
- Predictions remain valid

## API Response Times

- Single point: ~50-100ms
- Route (10 waypoints): ~500ms-1s
- Route (100 waypoints): ~5-10s

**Note:** First request after server start may take longer due to model initialization.

## Production Deployment

For production use, replace Flask development server with a production WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 api_server:app
```

For the frontend:

```bash
cd visualai
npm run build
# Serve the dist/ folder with nginx or similar
```

## Contact & Support

For issues or questions:
1. Check the troubleshooting section
2. Review the API test output: `python test_api.py`
3. Check server logs in the terminal running `api_server.py`

## License

This project uses the trained weather prediction model for research and educational purposes.
