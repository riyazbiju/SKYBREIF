"""
K-Nearest Spatial Aggregation Inference Script
===============================================
Predict weather at arbitrary (lat, lon, alt, time) queries using K=10 nearest stations
"""

import torch
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from spatiotemporal_latent_ode import (
    SpatioTemporalLatentODE,
    haversine_distance
)
from config_multi_station import MODEL_CONFIG, DATA_CONFIG


def load_model_and_data(model_path='best_spatiotemporal_model.pt', device='cpu'):
    """Load trained model, scaler, and station database"""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load model
    model = SpatioTemporalLatentODE(MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler and dataset info
    scaler = checkpoint['scaler']
    dataset_start_time = checkpoint.get('dataset_start_time', 0.0)
    
    print(f"✓ Model loaded (dataset start time: {dataset_start_time:.1f} hours)")
    
    return model, scaler, dataset_start_time


def load_station_database(csv_path='data/jan_sorted_cleaned.csv', scaler=None):
    """Load all stations and their observations for K-nearest queries"""
    print(f"Loading station database from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Convert datetime
    if 'datetime' not in df.columns and 'Date_Time' in df.columns:
        df = df.rename(columns={'Date_Time': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Add time_hours
    if 'time_hours' not in df.columns:
        df = df.sort_values('datetime')
        min_time = df['datetime'].min()
        df['time_hours'] = (df['datetime'] - min_time).dt.total_seconds() / 3600
    
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
    
    # Scale features if scaler provided
    if scaler is not None:
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
    
    print(f"✓ Loaded {len(station_data)} stations")
    return station_data, df['time_hours'].min(), df['time_hours'].max()


def find_k_nearest_stations(query_lat, query_lon, station_data, K=10):
    """Find K nearest stations to query location"""
    distances = []
    for station, info in station_data.items():
        dist = haversine_distance(query_lat, query_lon, info['lat'], info['lon'])
        distances.append((dist, station))
    
    distances.sort()
    k_nearest = [s for _, s in distances[:K]]
    k_distances = [d for d, _ in distances[:K]]
    
    return k_nearest, k_distances


def prepare_k_station_observations(k_stations, station_data, query_time, max_seq=100):
    """Get observation sequences from K nearest stations before query time"""
    # Already scaled columns
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
        
        # Extract features (already scaled during load_station_database)
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
    """Pad sequences to same length for batch processing"""
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
    
    # Stack: [K, max_len, features]
    obs_batch = torch.stack(obs_padded)
    spacetime_batch = torch.stack(spacetime_padded)
    
    # Add batch dimension: [1, K, max_len, features]
    obs_batch = obs_batch.unsqueeze(0)
    spacetime_batch = spacetime_batch.unsqueeze(0)
    
    return obs_batch, spacetime_batch


def predict_at_query(model, station_data, query_lat, query_lon, query_alt, query_time, 
                     K=10, max_seq=100, device='cpu'):
    """
    Predict weather at arbitrary query point
    
    Args:
        model: Trained SpatioTemporalLatentODE
        station_data: Dictionary of all stations
        query_lat: Latitude in degrees
        query_lon: Longitude in degrees
        query_alt: Altitude in feet
        query_time: Time in hours from dataset start
        K: Number of nearest stations to use
        max_seq: Maximum observation sequence length
        device: torch device
    
    Returns:
        prediction: [temperature, windspeed, visibility, pressure, wind_u, wind_v]
        k_stations: List of K nearest station names
        k_distances: Distances to each station (km)
    """
    altitude_scale = DATA_CONFIG['altitude_scale']
    total_hrs = DATA_CONFIG['total_hours']
    
    # Find K nearest stations
    k_stations, k_distances = find_k_nearest_stations(query_lat, query_lon, station_data, K)
    
    # Get observations from K stations
    obs_seqs, obs_spacetimes = prepare_k_station_observations(
        k_stations, station_data, query_time, max_seq
    )
    
    # Pad and batch
    obs_batch, spacetime_batch = pad_sequences(obs_seqs, obs_spacetimes, K)
    
    # Query spacetime
    query_spacetime = torch.tensor([[
        query_lat,
        query_lon,
        query_alt / altitude_scale,
        query_time / total_hrs
    ]], dtype=torch.float32)
    
    # Move to device
    obs_batch = obs_batch.to(device)
    spacetime_batch = spacetime_batch.to(device)
    query_spacetime = query_spacetime.to(device)
    
    # Predict
    with torch.no_grad():
        pred, _, _, _ = model(obs_batch, spacetime_batch, query_spacetime, integrate_ode=True)
    
    prediction = pred.cpu().numpy()[0]
    
    return prediction, k_stations, k_distances


def predict_route(model, scaler, station_data, route_csv, output_csv='predictions_knearest.csv',
                  K=10, max_seq=100, device='cpu', max_waypoints=1000):
    """Predict weather along a route using K-nearest spatial aggregation"""
    print(f"\nPredicting route: {route_csv}")
    
    # Load route
    route_df = pd.read_csv(route_csv)
    route_df = route_df.dropna()
    
    # Map column names
    col_map = {
        'Temperature': 'temperature',
        'Wind_Speed': 'windspeed',
        'Visiblity': 'visibility',
        'Pressure': 'pressure',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Altitude': 'altitude',
        'Wind_Direction': 'wind_direction',
        'step_order': 'step_order'
    }
    route_df = route_df.rename(columns=col_map)
    
    # Add time_hours from step_order
    if 'step_order' in route_df.columns:
        route_df['time_hours'] = route_df['step_order'].astype(float)
    else:
        route_df['time_hours'] = np.arange(len(route_df), dtype=float)
    
    # Subsample if too many waypoints
    if len(route_df) > max_waypoints:
        print(f"⚠ Route has {len(route_df)} waypoints, subsampling to {max_waypoints}...")
        step = len(route_df) // max_waypoints
        route_df = route_df.iloc[::step].reset_index(drop=True)
        print(f"  Using every {step}th waypoint")
    
    predictions = []
    feat_cols = ['temperature', 'windspeed', 'visibility', 'pressure', 'wind_u', 'wind_v']
    
    print(f"Processing {len(route_df)} waypoints...")
    
    for idx, row in route_df.iterrows():
        query_lat = row['latitude']
        query_lon = row['longitude']
        query_alt = row['altitude']
        query_time = row['time_hours']
        
        # Skip very early times where ODE integration may fail
        # (need some historical data for integration)
        if query_time < 1.0:
            print(f"  Skipping waypoint {idx+1} (time={query_time:.2f}h too early)")
            continue
        
        # Predict
        try:
            pred, k_stations, k_distances = predict_at_query(
                model, station_data, query_lat, query_lon, query_alt, query_time,
                K=K, max_seq=max_seq, device=device
            )
        except Exception as e:
            print(f"  ⚠ Warning: Failed at waypoint {idx+1} (time={query_time:.2f}h): {e}")
            continue
        
        # Inverse transform prediction
        pred_unscaled = scaler.inverse_transform(pred.reshape(1, -1))[0]
        
        # Convert wind u/v back to direction
        wind_u, wind_v = pred_unscaled[4], pred_unscaled[5]
        wind_dir = (np.degrees(np.arctan2(-wind_u, -wind_v)) + 360) % 360
        
        # Store result
        result = {
            'latitude': query_lat,
            'longitude': query_lon,
            'altitude': query_alt,
            'time_hours': query_time,
            'pred_temperature': pred_unscaled[0],
            'pred_windspeed': pred_unscaled[1],
            'pred_visibility': pred_unscaled[2],
            'pred_pressure': pred_unscaled[3],
            'pred_wind_direction': wind_dir,
            'nearest_stations': ','.join(k_stations[:3]),  # Top 3
            'nearest_distances_km': ','.join([f'{d:.1f}' for d in k_distances[:3]])
        }
        
        # Add ground truth if available
        if 'temperature' in row:
            result['true_temperature'] = row['temperature']
        if 'windspeed' in row:
            result['true_windspeed'] = row['windspeed']
        if 'visibility' in row:
            result['true_visibility'] = row['visibility']
        if 'pressure' in row:
            result['true_pressure'] = row['pressure']
        if 'wind_direction' in row:
            result['true_wind_direction'] = row['wind_direction']
        
        predictions.append(result)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(route_df)} waypoints")
    
    # Save results
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_csv, index=False)
    print(f"✓ Saved predictions to {output_csv}")
    
    # Compute metrics if ground truth available
    if 'true_temperature' in pred_df.columns:
        print("\n" + "="*60)
        print("PREDICTION METRICS")
        print("="*60)
        
        metrics = {}
        for var in ['temperature', 'windspeed', 'visibility', 'pressure', 'wind_direction']:
            true_col = f'true_{var}'
            pred_col = f'pred_{var}'
            
            if true_col in pred_df.columns:
                true_vals = pred_df[true_col].values
                pred_vals = pred_df[pred_col].values
                
                mae = np.mean(np.abs(true_vals - pred_vals))
                rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
                
                # R² for non-circular variables
                if var != 'wind_direction':
                    ss_res = np.sum((true_vals - pred_vals)**2)
                    ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
                    r2 = 1 - (ss_res / ss_tot)
                    print(f"{var.capitalize():20s}: MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:7.3f}")
                else:
                    print(f"{var.capitalize():20s}: MAE={mae:7.2f}°  RMSE={rmse:7.2f}°")
    
    return pred_df


def create_visualizations(pred_df, output_name='knearest'):
    """Create comprehensive visualizations similar to quick_predict.py"""
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Create visualizations directory
    Path('visualizations').mkdir(exist_ok=True)
    
    # Extract data
    lats = pred_df['latitude'].values
    lons = pred_df['longitude'].values
    alts = pred_df['altitude'].values
    times = pred_df['time_hours'].values
    
    # Predictions
    pred_temp = pred_df['pred_temperature'].values
    pred_wind = pred_df['pred_windspeed'].values
    pred_vis = pred_df['pred_visibility'].values
    pred_press = pred_df['pred_pressure'].values
    pred_wind_dir = pred_df['pred_wind_direction'].values
    
    # Check if ground truth available
    has_gt = 'true_temperature' in pred_df.columns
    
    if has_gt:
        # Ground truth
        gt_temp = pred_df['true_temperature'].values
        gt_wind = pred_df['true_windspeed'].values
        gt_vis = pred_df['true_visibility'].values
        gt_press = pred_df['true_pressure'].values
        gt_wind_dir = pred_df['true_wind_direction'].values
        
        # Errors
        error_temp = pred_temp - gt_temp
        error_wind = pred_wind - gt_wind
        error_vis = pred_vis - gt_vis
        error_press = pred_press - gt_press
        
        # Wind direction error (circular)
        error_wind_dir = pred_wind_dir - gt_wind_dir
        error_wind_dir = np.where(error_wind_dir > 180, error_wind_dir - 360, error_wind_dir)
        error_wind_dir = np.where(error_wind_dir < -180, error_wind_dir + 360, error_wind_dir)
    
    feature_names = ['Temperature (°C)', 'Wind Speed (knots)', 'Visibility (miles)', 
                     'Pressure (hPa)', 'Wind Direction (°)']
    
    # ========================================================================
    # FIGURE 1: 4D SPATIOTEMPORAL VISUALIZATION - Time Evolution
    # ========================================================================
    print("Creating spatiotemporal evolution plot...")
    fig1 = plt.figure(figsize=(18, 12))
    
    # Divide time into bins to show evolution
    time_bins = np.linspace(times.min(), times.max(), 7)  # 6 time periods
    time_labels = [f't={t:.1f}h' for t in time_bins[:-1]]
    
    # Temperature evolution over time
    for idx in range(6):
        ax = plt.subplot(3, 6, idx + 1)
        mask = (times >= time_bins[idx]) & (times < time_bins[idx + 1])
        if mask.sum() > 0:
            scatter = ax.scatter(lons[mask], lats[mask], 
                               c=pred_temp[mask], 
                               cmap='RdYlBu_r', s=30, alpha=0.7,
                               vmin=pred_temp.min(), 
                               vmax=pred_temp.max())
            ax.set_title(f'Pred Temp {time_labels[idx]}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Lon', fontsize=8)
            ax.set_ylabel('Lat', fontsize=8)
            plt.colorbar(scatter, ax=ax, label='°C')
            ax.grid(True, alpha=0.3)
    
    # Wind speed evolution over time
    for idx in range(6):
        ax = plt.subplot(3, 6, idx + 7)
        mask = (times >= time_bins[idx]) & (times < time_bins[idx + 1])
        if mask.sum() > 0:
            scatter = ax.scatter(lons[mask], lats[mask], 
                               c=pred_wind[mask], 
                               cmap='YlOrRd', s=30, alpha=0.7,
                               vmin=pred_wind.min(), 
                               vmax=pred_wind.max())
            ax.set_title(f'Pred Wind {time_labels[idx]}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Lon', fontsize=8)
            ax.set_ylabel('Lat', fontsize=8)
            plt.colorbar(scatter, ax=ax, label='knots')
            ax.grid(True, alpha=0.3)
    
    # Temperature error evolution over time (if GT available)
    if has_gt:
        for idx in range(6):
            ax = plt.subplot(3, 6, idx + 13)
            mask = (times >= time_bins[idx]) & (times < time_bins[idx + 1])
            if mask.sum() > 0:
                temp_error_abs = np.abs(error_temp)
                scatter = ax.scatter(lons[mask], lats[mask], 
                                   c=temp_error_abs[mask], 
                                   cmap='Reds', s=30, alpha=0.7,
                                   vmin=0, vmax=np.percentile(temp_error_abs, 95))
                ax.set_title(f'Temp Error {time_labels[idx]}', fontsize=9, fontweight='bold')
                ax.set_xlabel('Lon', fontsize=8)
                ax.set_ylabel('Lat', fontsize=8)
                plt.colorbar(scatter, ax=ax, label='|Error|')
                ax.grid(True, alpha=0.3)
    else:
        # Just show pressure if no GT
        for idx in range(6):
            ax = plt.subplot(3, 6, idx + 13)
            mask = (times >= time_bins[idx]) & (times < time_bins[idx + 1])
            if mask.sum() > 0:
                scatter = ax.scatter(lons[mask], lats[mask], 
                                   c=pred_press[mask], 
                                   cmap='viridis', s=30, alpha=0.7)
                ax.set_title(f'Pred Press {time_labels[idx]}', fontsize=9, fontweight='bold')
                ax.set_xlabel('Lon', fontsize=8)
                ax.set_ylabel('Lat', fontsize=8)
                plt.colorbar(scatter, ax=ax, label='hPa')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    temporal_path = f'visualizations/{output_name}_spatiotemporal_evolution.png'
    plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {temporal_path}")
    plt.close()
    
    # ========================================================================
    # FIGURE 2: 3D Trajectory with Altitude
    # ========================================================================
    print("Creating 3D trajectory plots...")
    fig2 = plt.figure(figsize=(18, 25))
    
    cmaps = ['RdYlBu_r', 'YlOrRd', 'viridis', 'plasma', 'twilight']
    pred_vars = [pred_temp, pred_wind, pred_vis, pred_press, pred_wind_dir]
    
    if has_gt:
        gt_vars = [gt_temp, gt_wind, gt_vis, gt_press, gt_wind_dir]
        error_vars = [error_temp, error_wind, error_vis, error_press, error_wind_dir]
    
    # Create 3D plots for all 5 variables
    for var_idx in range(5):
        cmap = cmaps[var_idx]
        
        if has_gt:
            # Column 1: Ground Truth
            ax1 = fig2.add_subplot(5, 3, var_idx * 3 + 1, projection='3d')
            scatter = ax1.scatter(lons, lats, alts, c=gt_vars[var_idx], 
                                 cmap=cmap, s=10, alpha=0.6)
            ax1.set_xlabel('Longitude', fontweight='bold', fontsize=9)
            ax1.set_ylabel('Latitude', fontweight='bold', fontsize=9)
            ax1.set_zlabel('Altitude (ft)', fontweight='bold', fontsize=9)
            ax1.set_title(f'Ground Truth\n{feature_names[var_idx]}', fontweight='bold', fontsize=10)
            plt.colorbar(scatter, ax=ax1, shrink=0.5, pad=0.1)
            
            # Column 2: Predictions
            ax2 = fig2.add_subplot(5, 3, var_idx * 3 + 2, projection='3d')
            scatter = ax2.scatter(lons, lats, alts, c=pred_vars[var_idx], 
                                 cmap=cmap, s=10, alpha=0.6,
                                 vmin=gt_vars[var_idx].min(), 
                                 vmax=gt_vars[var_idx].max())
            ax2.set_xlabel('Longitude', fontweight='bold', fontsize=9)
            ax2.set_ylabel('Latitude', fontweight='bold', fontsize=9)
            ax2.set_zlabel('Altitude (ft)', fontweight='bold', fontsize=9)
            ax2.set_title(f'Predicted\n{feature_names[var_idx]}', fontweight='bold', fontsize=10)
            plt.colorbar(scatter, ax=ax2, shrink=0.5, pad=0.1)
            
            # Column 3: Absolute Error
            ax3 = fig2.add_subplot(5, 3, var_idx * 3 + 3, projection='3d')
            error_abs = np.abs(error_vars[var_idx])
            scatter = ax3.scatter(lons, lats, alts, c=error_abs, 
                                 cmap='Reds', s=10, alpha=0.6,
                                 vmin=0, vmax=np.percentile(error_abs, 95))
            ax3.set_xlabel('Longitude', fontweight='bold', fontsize=9)
            ax3.set_ylabel('Latitude', fontweight='bold', fontsize=9)
            ax3.set_zlabel('Altitude (ft)', fontweight='bold', fontsize=9)
            ax3.set_title(f'Absolute Error\n{feature_names[var_idx]}', fontweight='bold', fontsize=10)
            plt.colorbar(scatter, ax=ax3, shrink=0.5, pad=0.1)
        else:
            # Just predictions if no GT
            ax = fig2.add_subplot(5, 1, var_idx + 1, projection='3d')
            scatter = ax.scatter(lons, lats, alts, c=pred_vars[var_idx], 
                               cmap=cmap, s=10, alpha=0.6)
            ax.set_xlabel('Longitude', fontweight='bold', fontsize=9)
            ax.set_ylabel('Latitude', fontweight='bold', fontsize=9)
            ax.set_zlabel('Altitude (ft)', fontweight='bold', fontsize=9)
            ax.set_title(f'Predicted {feature_names[var_idx]}', fontweight='bold', fontsize=10)
            plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
    
    plt.tight_layout()
    trajectory_3d_path = f'visualizations/{output_name}_3d_trajectory.png'
    plt.savefig(trajectory_3d_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {trajectory_3d_path}")
    plt.close()
    
    # ========================================================================
    # FIGURE 3: Scatter Plots (Predicted vs Ground Truth)
    # ========================================================================
    if has_gt:
        print("Creating scatter plots...")
        fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
        axes3 = axes3.flatten()
        
        for i in range(5):
            ax = axes3[i]
            name = feature_names[i]
            
            # Scatter plot
            ax.scatter(gt_vars[i], pred_vars[i], c=times, cmap='viridis', 
                      s=10, alpha=0.5)
            
            # Perfect prediction line
            min_val = min(gt_vars[i].min(), pred_vars[i].min())
            max_val = max(gt_vars[i].max(), pred_vars[i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                   linewidth=2, label='Perfect Prediction')
            
            # Calculate R²
            ss_res = np.sum((gt_vars[i] - pred_vars[i])**2)
            ss_tot = np.sum((gt_vars[i] - np.mean(gt_vars[i]))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0
            
            # Calculate MAE
            mae = np.mean(np.abs(gt_vars[i] - pred_vars[i]))
            
            ax.set_xlabel(f'Ground Truth {name}', fontweight='bold', fontsize=11)
            ax.set_ylabel(f'Predicted {name}', fontweight='bold', fontsize=11)
            ax.set_title(f'{name}\nR²={r2:.3f}, MAE={mae:.2f}', 
                        fontweight='bold', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Remove 6th subplot
        axes3[5].remove()
        
        plt.tight_layout()
        scatter_path = f'visualizations/{output_name}_scatter_plots.png'
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {scatter_path}")
        plt.close()
    
    # ========================================================================
    # FIGURE 4: K-Nearest Station Distance Analysis
    # ========================================================================
    if 'nearest_distances_km' in pred_df.columns:
        print("Creating K-nearest distance analysis...")
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
        
        # Parse distances
        distances = []
        for dist_str in pred_df['nearest_distances_km']:
            dists = [float(d) for d in dist_str.split(',')]
            distances.append(dists)
        distances = np.array(distances)
        
        # Plot 1: Distance distribution
        ax = axes4[0]
        ax.hist(distances[:, 0], bins=30, alpha=0.7, label='Nearest', color='blue')
        if distances.shape[1] > 1:
            ax.hist(distances[:, 1], bins=30, alpha=0.7, label='2nd Nearest', color='orange')
        if distances.shape[1] > 2:
            ax.hist(distances[:, 2], bins=30, alpha=0.7, label='3rd Nearest', color='green')
        ax.set_xlabel('Distance (km)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
        ax.set_title('Distribution of K-Nearest Station Distances', fontweight='bold', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error vs Distance (if GT available)
        if has_gt:
            ax = axes4[1]
            nearest_dist = distances[:, 0]
            ax.scatter(nearest_dist, np.abs(error_temp), c=times, cmap='viridis',
                      s=20, alpha=0.5)
            ax.set_xlabel('Distance to Nearest Station (km)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Temperature Error (°C)', fontweight='bold', fontsize=11)
            ax.set_title('Prediction Error vs Station Distance', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(nearest_dist, np.abs(error_temp), 1)
            p = np.poly1d(z)
            ax.plot(sorted(nearest_dist), p(sorted(nearest_dist)), 
                   "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        distance_path = f'visualizations/{output_name}_distance_analysis.png'
        plt.savefig(distance_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {distance_path}")
        plt.close()
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='K-Nearest Spatial Aggregation Inference')
    parser.add_argument('--model', type=str, default='best_spatiotemporal_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='data/jan_sorted_cleaned.csv',
                       help='Path to full dataset (for station database)')
    parser.add_argument('--route', type=str, required=True,
                       help='Path to route CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path')
    parser.add_argument('--K', type=int, default=10,
                       help='Number of nearest stations to use')
    parser.add_argument('--max-waypoints', type=int, default=1000,
                       help='Maximum number of waypoints to process (subsample if larger)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        route_name = Path(args.route).stem
        args.output = f'results/predictions_{route_name}_knearest.csv'
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("K-NEAREST SPATIAL AGGREGATION INFERENCE")
    print("="*70)
    
    # Load model
    model, scaler, dataset_start_time = load_model_and_data(args.model, args.device)
    
    # Load station database
    station_data, min_time, max_time = load_station_database(args.data, scaler)
    print(f"Dataset time range: {min_time:.1f} to {max_time:.1f} hours")
    
    # Predict route
    pred_df = predict_route(
        model, scaler, station_data, args.route, args.output,
        K=args.K, max_waypoints=args.max_waypoints, device=args.device
    )
    
    # Create visualizations if requested
    if args.visualize:
        route_name = Path(args.route).stem
        create_visualizations(pred_df, output_name=f'{route_name}_knearest')
    
    print("\n" + "="*70)
    print("✓ PREDICTION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
