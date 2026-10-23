"""
Flask API Server for Spatiotemporal Weather Prediction
Connects the VisualAI frontend with the trained K-nearest model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Import your model
from spatiotemporal_latent_ode import (
    SpatioTemporalLatentODE,
    haversine_distance
)
from config_multi_station import MODEL_CONFIG, DATA_CONFIG

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for model and data
model = None
scaler = None
station_data = None
dataset_start_time = None
min_time = None
max_time = None

ROUTE_GRID_FILES = ['data/route1_subset.csv', 'data/route2_subset.csv']


def load_route_grid_as_stations():
    """Fallback when the training CSV is missing: use the gridded route data,
    treating each (lat, lon) grid point as a station."""
    print("data/jan_sorted_cleaned.csv not found - using route grid data as stations")
    frames = [pd.read_csv(p) for p in ROUTE_GRID_FILES if Path(p).exists()]
    if not frames:
        raise FileNotFoundError("No station data: need data/jan_sorted_cleaned.csv or route subset CSVs")
    df = pd.concat(frames, ignore_index=True).drop(columns=['step_order'], errors='ignore')
    df = df.rename(columns={
        'Date_Time': 'datetime', 'Latitude': 'latitude', 'Longitude': 'longitude',
        'Altitude': 'altitude', 'Pressure': 'pressure', 'Temperature': 'temperature',
        'Wind_Speed': 'windspeed', 'Wind_Direction': 'winddirection', 'Visiblity': 'visibility',
    }).drop_duplicates()
    df['station'] = df['latitude'].round(2).astype(str) + ',' + df['longitude'].round(2).astype(str)
    return df


def load_model_and_data():
    """Load trained model and station database"""
    global model, scaler, station_data, dataset_start_time, min_time, max_time
    
    print("Loading model and station database...")
    
    # Load model
    device = 'cpu'
    model_path = 'best_spatiotemporal_model.pt'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = SpatioTemporalLatentODE(MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    dataset_start_time = checkpoint.get('dataset_start_time', 0.0)
    
    print(f"✓ Model loaded (dataset start time: {dataset_start_time:.1f} hours)")
    
    # Load station database
    csv_path = Path('data/jan_sorted_cleaned.csv')
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = load_route_grid_as_stations()

    # Convert datetime
    if 'datetime' not in df.columns and 'Date_Time' in df.columns:
        df = df.rename(columns={'Date_Time': 'datetime'})
    # Dates come as both "2026-01-01" and "2026-01-01 01:00:00"
    df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601', errors='coerce')
    
    # Add time_hours
    if 'time_hours' not in df.columns:
        df = df.sort_values('datetime')
        min_datetime = df['datetime'].min()
        df['time_hours'] = (df['datetime'] - min_datetime).dt.total_seconds() / 3600
    
    # Convert wind direction to u/v
    if 'wind_u' not in df.columns:
        if 'winddirection' in df.columns and 'windspeed' in df.columns:
            df['wind_direction_original'] = df['winddirection'].copy()
            wind_dir_rad = np.radians(df['winddirection'])
            df['wind_u'] = -df['windspeed'] * np.sin(wind_dir_rad)
            df['wind_v'] = -df['windspeed'] * np.cos(wind_dir_rad)
    
    # Drop NaN
    required = ['temperature', 'windspeed', 'visibility', 'pressure', 
                'latitude', 'longitude', 'altitude', 'station', 'wind_u', 'wind_v']
    df = df.dropna(subset=required)
    
    # Scale features
    feat_cols = ['temperature', 'windspeed', 'visibility', 'pressure', 'wind_u', 'wind_v']
    df[feat_cols] = scaler.transform(df[feat_cols])
    
    # Build station database
    station_data = {}
    for station, group in df.groupby('station'):
        group = group.sort_values('datetime')
        lat = group['latitude'].iloc[0]
        lon = group['longitude'].iloc[0]
        station_data[station] = {
            'lat': lat,
            'lon': lon,
            'data': group
        }
    
    min_time = df['time_hours'].min()
    max_time = df['time_hours'].max()
    
    print(f"✓ Loaded {len(station_data)} stations")
    print(f"Dataset time range: {min_time:.1f} to {max_time:.1f} hours")


def find_k_nearest_stations(query_lat, query_lon, K=10):
    """Find K nearest stations to query location"""
    distances = []
    for station, info in station_data.items():
        dist = haversine_distance(query_lat, query_lon, info['lat'], info['lon'])
        distances.append((dist, station))
    
    distances.sort()
    k_nearest = [s for _, s in distances[:K]]
    k_distances = [d for d, _ in distances[:K]]
    
    return k_nearest, k_distances


def prepare_k_station_observations(k_stations, query_time, max_seq=100):
    """Get observation sequences from K nearest stations before query time"""
    feat_cols = ['temperature', 'windspeed', 'visibility', 'pressure', 'wind_u', 'wind_v']
    altitude_scale = DATA_CONFIG['altitude_scale']
    total_hrs = DATA_CONFIG['total_hours']
    
    obs_seqs = []
    obs_spacetimes = []
    
    for station in k_stations:
        station_df = station_data[station]['data']
        
        # Get observations before query time
        past_obs = station_df[station_df['time_hours'] <= query_time].copy()
        
        if len(past_obs) == 0:
            # No past observations - use earliest available
            past_obs = station_df.iloc[:1].copy()
        
        # Take up to max_seq most recent observations
        if len(past_obs) > max_seq:
            past_obs = past_obs.iloc[-max_seq:].copy()
        
        # Extract features
        obs = torch.tensor(past_obs[feat_cols].values.astype(np.float32), dtype=torch.float32)
        spacetime = torch.tensor(np.column_stack([
            past_obs['latitude'].values.astype(np.float32),
            past_obs['longitude'].values.astype(np.float32),
            (past_obs['altitude'].values / altitude_scale).astype(np.float32),
            (past_obs['time_hours'].values / total_hrs).astype(np.float32)
        ]), dtype=torch.float32)
        
        obs_seqs.append(obs)
        obs_spacetimes.append(spacetime)
    
    return obs_seqs, obs_spacetimes


def pad_sequences(obs_seqs, obs_spacetimes, K):
    """Pad sequences to same length"""
    max_len = max([obs.shape[0] for obs in obs_seqs])
    
    obs_padded = []
    spacetime_padded = []
    
    for i in range(K):
        obs = obs_seqs[i]
        spacetime = obs_spacetimes[i]
        seq_len = obs.shape[0]
        
        if seq_len < max_len:
            pad_len = max_len - seq_len
            obs_pad = torch.zeros(pad_len, obs.shape[1])
            spacetime_pad = torch.zeros(pad_len, 4)
            
            obs_padded.append(torch.cat([obs, obs_pad], dim=0))
            spacetime_padded.append(torch.cat([spacetime, spacetime_pad], dim=0))
        else:
            obs_padded.append(obs)
            spacetime_padded.append(spacetime)
    
    obs_batch = torch.stack(obs_padded).unsqueeze(0)
    spacetime_batch = torch.stack(spacetime_padded).unsqueeze(0)
    
    return obs_batch, spacetime_batch


def predict_point(lat, lon, alt, query_time, K=10):
    """Predict weather at a single point"""
    altitude_scale = DATA_CONFIG['altitude_scale']
    total_hrs = DATA_CONFIG['total_hours']
    
    # Find K nearest stations
    k_stations, k_distances = find_k_nearest_stations(lat, lon, K)
    
    # Get observations
    obs_seqs, obs_spacetimes = prepare_k_station_observations(k_stations, query_time)
    
    # Pad and batch
    obs_batch, spacetime_batch = pad_sequences(obs_seqs, obs_spacetimes, K)
    
    # Query spacetime
    query_spacetime = torch.tensor([[
        lat,
        lon,
        alt / altitude_scale,
        query_time / total_hrs
    ]], dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        pred, _, _, _ = model(obs_batch, spacetime_batch, query_spacetime, integrate_ode=True)
    
    # Inverse transform
    pred_unscaled = scaler.inverse_transform(pred.cpu().numpy())[0]
    
    # Convert wind u/v to direction
    wind_u, wind_v = pred_unscaled[4], pred_unscaled[5]
    wind_dir = (np.degrees(np.arctan2(-wind_u, -wind_v)) + 360) % 360
    
    return {
        'temperature': float(pred_unscaled[0]),
        'windspeed': float(pred_unscaled[1]),
        'visibility': float(pred_unscaled[2]),
        'pressure': float(pred_unscaled[3]),
        'wind_direction': float(wind_dir),
        'nearest_stations': k_stations[:3],
        'distances_km': [float(d) for d in k_distances[:3]]
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'stations': len(station_data) if station_data else 0
    })


@app.route('/predict/point', methods=['POST'])
def predict_point_endpoint():
    """
    Predict weather at a single point
    
    Request JSON:
    {
        "lat": 56.1304,
        "lon": -106.3468,
        "alt": 10000,
        "datetime": "2026-01-15T12:00:00Z"
    }
    """
    try:
        data = request.json
        
        # Parse inputs
        lat = float(data['lat'])
        lon = float(data['lon'])
        alt = float(data['alt'])
        datetime_str = data['datetime']
        
        # Convert datetime to hours from dataset start
        # For now, we'll use a simple mapping based on the dataset
        # The dataset covers Jan 2026, so we map input datetime to hours
        from datetime import timezone
        query_dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        
        # Map to dataset time (0-743 hours)
        # Assuming dataset starts at 2026-01-01 00:00:00 UTC
        dataset_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        time_diff = query_dt - dataset_start
        query_time = time_diff.total_seconds() / 3600
        
        # Validate time range
        if query_time < 1.0:
            return jsonify({
                'error': 'Query time too early. Must be at least 1 hour after dataset start (2026-01-01 00:00).'
            }), 400
        
        if query_time > max_time:
            return jsonify({
                'error': f'Query time too late. Dataset covers up to {max_time:.1f} hours (~{int(max_time/24)} days).'
            }), 400
        
        # Make prediction
        result = predict_point(lat, lon, alt, query_time, K=10)
        
        # Add metadata
        result['query'] = {
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'datetime': datetime_str,
            'time_hours': query_time
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/route', methods=['POST'])
def predict_route_endpoint():
    """
    Predict weather along a route
    
    Request JSON:
    {
        "waypoints": [
            {"lat": 56.1304, "lon": -106.3468, "alt": 10000, "datetime": "2026-01-15T12:00:00Z"},
            {"lat": 56.5, "lon": -106.0, "alt": 15000, "datetime": "2026-01-15T13:00:00Z"},
            ...
        ]
    }
    """
    try:
        data = request.json
        waypoints = data['waypoints']
        
        if not waypoints:
            return jsonify({'error': 'No waypoints provided'}), 400
        
        if len(waypoints) > 100:
            return jsonify({'error': 'Too many waypoints. Maximum 100 allowed.'}), 400
        
        results = []
        
        for wp in waypoints:
            lat = float(wp['lat'])
            lon = float(wp['lon'])
            alt = float(wp['alt'])
            datetime_str = wp['datetime']
            
            # Convert datetime to hours
            from datetime import timezone
            query_dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            dataset_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            time_diff = query_dt - dataset_start
            query_time = time_diff.total_seconds() / 3600
            
            # Skip if out of range
            if query_time < 1.0 or query_time > max_time:
                results.append({
                    'query': {
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': alt,
                        'datetime': datetime_str
                    },
                    'error': 'Time out of dataset range'
                })
                continue
            
            # Make prediction
            result = predict_point(lat, lon, alt, query_time, K=10)
            result['query'] = {
                'latitude': lat,
                'longitude': lon,
                'altitude': alt,
                'datetime': datetime_str,
                'time_hours': query_time
            }
            results.append(result)
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/info', methods=['GET'])
def info():
    """Get model and dataset information"""
    return jsonify({
        'model': {
            'type': 'SpatioTemporalLatentODE',
            'k_neighbors': 10,
            'features': ['temperature', 'windspeed', 'visibility', 'pressure', 'wind_direction']
        },
        'dataset': {
            'stations': len(station_data) if station_data else 0,
            'time_range_hours': [float(min_time), float(max_time)],
            'time_range_days': [0, int(max_time / 24)],
            'coverage': 'Western North America (primarily Canada)',
            'period': 'January 2026'
        }
    })


if __name__ == '__main__':
    print("="*70)
    print("SPATIOTEMPORAL WEATHER PREDICTION API SERVER")
    print("="*70)
    
    # Load model and data
    load_model_and_data()
    
    print("\n" + "="*70)
    print("Starting Flask server...")
    print("API endpoints:")
    print("  - GET  /health           - Health check")
    print("  - GET  /info             - Model information")
    print("  - POST /predict/point    - Single point prediction")
    print("  - POST /predict/route    - Route prediction")
    print("="*70 + "\n")
    
    # Start server
    app.run(host='0.0.0.0', port=5001, debug=True)
