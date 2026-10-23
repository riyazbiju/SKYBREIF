"""
Quick Route Prediction - Command Line Interface
Usage: python quick_predict.py --route 1
       python quick_predict.py --route 2
       python quick_predict.py --csv myroute.csv
"""

import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config_multi_station import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG
from spatiotemporal_latent_ode import (
    SpatioTemporalLatentODE, 
    prepare_route_dataset, 
    collate_fn,
    prepare_data_loaders
)


def load_model(device):
    """Load trained model"""
    model_candidates = (
        Path("models/best_spatiotemporal_model.pt"),
        Path("best_spatiotemporal_model.pt"),
    )
    model_path = next((path for path in model_candidates if path.exists()), None)
    if model_path is None:
        locations = ", ".join(str(path) for path in model_candidates)
        raise FileNotFoundError(f"Model not found in any expected location: {locations}")
    
    print(f"Loading model from {model_path}...")
    model = SpatioTemporalLatentODE(config=MODEL_CONFIG).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint.get('scaler')
    else:
        model.load_state_dict(checkpoint)
        scaler = None
    
    model.eval()
    print("✓ Model loaded")
    return model, scaler


def create_scaler(config):
    """Create scaler from training data"""
    print("Creating scaler from training data...")
    _, _, scaler = prepare_data_loaders(config)
    return scaler


def predict_route(model, route_path, scaler, config, device):
    """Make predictions for a route"""
    print(f"\nPredicting for: {route_path}")
    
    # Prepare dataset (this handles dropna and filtering)
    route_ds = prepare_route_dataset(route_path, scaler, config)
    if route_ds is None:
        print(f"❌ Failed to load route data")
        return None, None, None, None, None
    
    route_loader = DataLoader(route_ds, batch_size=16, collate_fn=collate_fn)
    
    # Collect predictions
    all_preds = []
    all_targets = []
    all_coords = []
    all_times = []
    all_wind_dirs = []
    
    with torch.no_grad():
        for batch in route_loader:
            obs_seq = batch['obs_seq'].to(device)
            obs_spacetime = batch['obs_spacetime'].to(device)
            query_spacetime = batch['query_spacetime'].to(device)
            targets = batch['target'].to(device)
            wind_dirs = batch['wind_dir_original']
            
            try:
                # Forward pass through spatiotemporal model
                predictions, _, _, _ = model(obs_seq, obs_spacetime, query_spacetime, integrate_ode=True)
                
                if not torch.isnan(predictions).any():
                    pred_orig = scaler.inverse_transform(predictions.cpu().numpy())
                    target_orig = scaler.inverse_transform(targets.cpu().numpy())
                    
                    all_preds.append(pred_orig)
                    all_targets.append(target_orig)
                    all_wind_dirs.extend(wind_dirs)
                    
                    # Extract coordinates from query_spacetime
                    query_np = query_spacetime.cpu().numpy()
                    
                    # Get scaling factors from config
                    altitude_scale = config['data'].get('altitude_scale', 40000.0)
                    total_hours = config['data'].get('total_hours', 744)
                    
                    for i in range(len(query_np)):
                        lat, lon, alt_norm, time_norm = query_np[i]
                        all_coords.append([lat, lon, alt_norm * altitude_scale])  # Denormalize altitude correctly
                        all_times.append(time_norm * total_hours)  # Denormalize time correctly
            except Exception as e:
                print(f"Warning: Batch error - {e}")
    
    if len(all_preds) > 0:
        predictions = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        coords = np.array(all_coords)
        times = np.array(all_times)
        
        # Convert wind_dirs list to numpy array, replacing None with NaN
        wind_dirs_array = np.array([w if w is not None else np.nan for w in all_wind_dirs])
        
        print(f"✓ Collected {len(predictions)} predictions")
        return predictions, targets, coords, times, wind_dirs_array
    else:
        return None, None, None, None, None


def save_and_plot(predictions, targets, coords, times, output_name, original_wind_dir=None):
    """Save predictions and create visualization for spatiotemporal model"""
    
    # Convert predicted wind components (u, v) to wind direction (degrees)
    # u = -speed * sin(dir), v = -speed * cos(dir)
    # Therefore: dir = atan2(-u, -v) in radians, convert to degrees
    pred_wind_dir = (np.degrees(np.arctan2(-predictions[:, 4], -predictions[:, 5])) + 360) % 360
    
    # Use original wind direction from CSV if available, otherwise compute from target u/v
    if original_wind_dir is not None and len(original_wind_dir) == len(predictions):
        # Filter out NaN values for comparison
        valid_mask = ~np.isnan(original_wind_dir)
        if valid_mask.sum() > 0:
            gt_wind_dir = original_wind_dir.copy()
        else:
            # If all are NaN, compute from u/v components
            gt_wind_dir = (np.degrees(np.arctan2(-targets[:, 4], -targets[:, 5])) + 360) % 360
    else:
        gt_wind_dir = (np.degrees(np.arctan2(-targets[:, 4], -targets[:, 5])) + 360) % 360
    
    # Calculate wind direction error (accounting for circular nature)
    wind_dir_error = pred_wind_dir - gt_wind_dir
    # Adjust for circular nature (e.g., 359° and 1° are only 2° apart)
    wind_dir_error = np.where(wind_dir_error > 180, wind_dir_error - 360, wind_dir_error)
    wind_dir_error = np.where(wind_dir_error < -180, wind_dir_error + 360, wind_dir_error)
    
    # Save CSV with errors
    errors = predictions - targets
    df = pd.DataFrame({
        'latitude': coords[:, 0],
        'longitude': coords[:, 1],
        'altitude_m': coords[:, 2],
        'time_hours': times,
        'pred_temperature': predictions[:, 0],
        'gt_temperature': targets[:, 0],
        'error_temperature': errors[:, 0],
        'pred_windspeed': predictions[:, 1],
        'gt_windspeed': targets[:, 1],
        'error_windspeed': errors[:, 1],
        'pred_visibility': predictions[:, 2],
        'gt_visibility': targets[:, 2],
        'error_visibility': errors[:, 2],
        'pred_pressure': predictions[:, 3],
        'gt_pressure': targets[:, 3],
        'error_pressure': errors[:, 3],
        'pred_wind_u': predictions[:, 4],
        'gt_wind_u': targets[:, 4],
        'error_wind_u': errors[:, 4],
        'pred_wind_v': predictions[:, 5],
        'gt_wind_v': targets[:, 5],
        'error_wind_v': errors[:, 5],
        'pred_wind_direction': pred_wind_dir,
        'gt_wind_direction': gt_wind_dir,
        'error_wind_direction': wind_dir_error,
    })
    
    csv_path = f'results/{output_name}.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Calculate and print statistics
    feature_names = ['Temperature (°C)', 'Wind Speed (knots)', 'Visibility (SM)', 
                    'Pressure (hPa)', 'Wind Direction (°)']
    
    print("\n" + "="*60)
    print("PREDICTION METRICS")
    print("="*60)
    
    error_list = []
    # Add regular errors
    for i in range(4):  # temp, wind_speed, visibility, pressure
        error_list.append(predictions[:, i] - targets[:, i])
    # Add wind direction error (already computed with circular correction)
    error_list.append(wind_dir_error)
    
    # Calculate metrics
    for i, name in enumerate(feature_names):
        if i < 4:
            # Regular metrics
            mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
            rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i])**2))
            var_target = np.var(targets[:, i])
            if var_target > 1e-6:
                r2 = 1 - np.sum((targets[:, i] - predictions[:, i])**2) / np.sum((targets[:, i] - np.mean(targets[:, i]))**2)
            else:
                r2 = 0.0
        else:
            # Wind direction metrics (circular) - handle NaN values
            valid_mask = ~np.isnan(wind_dir_error)
            if valid_mask.sum() > 0:
                mae = np.mean(np.abs(wind_dir_error[valid_mask]))
                rmse = np.sqrt(np.mean(wind_dir_error[valid_mask]**2))
                # For circular data, R² is not meaningful, use circular correlation instead
                # For simplicity, we'll compute a pseudo-R² based on circular error
                r2 = 1 - np.mean(wind_dir_error[valid_mask]**2) / (180**2)  # Normalized by max possible error
            else:
                mae = rmse = r2 = np.nan
        
        print(f"{name:20s}: MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:6.3f}")
    
    # Import 3D plotting
    from mpl_toolkits.mplot3d import Axes3D
    
    # ========================================================================
    # FIGURE 1: 4D SPATIOTEMPORAL VISUALIZATION - Time Evolution
    # ========================================================================
    fig1 = plt.figure(figsize=(18, 12))
    
    # Divide time into bins to show evolution
    time_bins = np.linspace(times.min(), times.max(), 7)  # 6 time periods
    time_labels = [f't={t:.1f}h' for t in time_bins[:-1]]
    
    lats, lons = coords[:, 0], coords[:, 1]
    alts = coords[:, 2]
    
    # Temperature evolution over time
    for idx in range(6):
        ax = plt.subplot(3, 6, idx + 1)
        mask = (times >= time_bins[idx]) & (times < time_bins[idx + 1])
        if mask.sum() > 0:
            scatter = ax.scatter(lons[mask], lats[mask], 
                               c=predictions[mask, 0], 
                               cmap='RdYlBu_r', s=30, alpha=0.7,
                               vmin=predictions[:, 0].min(), 
                               vmax=predictions[:, 0].max())
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
                               c=predictions[mask, 1], 
                               cmap='YlOrRd', s=30, alpha=0.7,
                               vmin=predictions[:, 1].min(), 
                               vmax=predictions[:, 1].max())
            ax.set_title(f'Pred Wind {time_labels[idx]}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Lon', fontsize=8)
            ax.set_ylabel('Lat', fontsize=8)
            plt.colorbar(scatter, ax=ax, label='m/s')
            ax.grid(True, alpha=0.3)
    
    # Temperature error evolution over time
    for idx in range(6):
        ax = plt.subplot(3, 6, idx + 13)
        mask = (times >= time_bins[idx]) & (times < time_bins[idx + 1])
        if mask.sum() > 0:
            temp_error = np.abs(error_list[0])
            scatter = ax.scatter(lons[mask], lats[mask], 
                               c=temp_error[mask], 
                               cmap='Reds', s=30, alpha=0.7,
                               vmin=0, vmax=np.percentile(temp_error, 95))
            ax.set_title(f'Temp Error {time_labels[idx]}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Lon', fontsize=8)
            ax.set_ylabel('Lat', fontsize=8)
            plt.colorbar(scatter, ax=ax, label='|Error|')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    temporal_path = f'visualizations/{output_name}_spatiotemporal_evolution.png'
    plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {temporal_path}")
    plt.close()
    
    # ========================================================================
    # FIGURE 2: 3D Trajectory with Altitude - Key Spatiotemporal Feature
    # ========================================================================
    fig2 = plt.figure(figsize=(18, 25))  # Increased height for 5 rows
    
    # Define colormaps for each variable
    cmaps = ['RdYlBu_r', 'YlOrRd', 'viridis', 'plasma', 'twilight']
    var_labels = ['Temperature (°C)', 'Wind Speed (m/s)', 'Visibility (m)', 
                  'Pressure (hPa)', 'Wind Direction (°)']
    
    # Create 3D plots for all 5 variables (5 rows x 3 columns)
    for var_idx in range(5):
        cmap = cmaps[var_idx]
        label = var_labels[var_idx]
        
        # Column 1: Ground Truth
        ax1 = fig2.add_subplot(5, 3, var_idx * 3 + 1, projection='3d')
        if var_idx < 4:
            # Regular variables
            scatter = ax1.scatter(lons, lats, alts, c=targets[:, var_idx], 
                                 cmap=cmap, s=10, alpha=0.6)
        else:
            # Wind Direction
            scatter = ax1.scatter(lons, lats, alts, c=gt_wind_dir, 
                                 cmap=cmap, s=10, alpha=0.6, vmin=0, vmax=360)
        
        ax1.set_xlabel('Longitude', fontweight='bold', fontsize=9)
        ax1.set_ylabel('Latitude', fontweight='bold', fontsize=9)
        ax1.set_zlabel('Altitude (m)', fontweight='bold', fontsize=9)
        ax1.set_title(f'Ground Truth\n{feature_names[var_idx]}', fontweight='bold', fontsize=10)
        plt.colorbar(scatter, ax=ax1, label=label, shrink=0.5, pad=0.1)
        
        # Column 2: Predictions
        ax2 = fig2.add_subplot(5, 3, var_idx * 3 + 2, projection='3d')
        if var_idx < 4:
            # Regular variables
            scatter = ax2.scatter(lons, lats, alts, c=predictions[:, var_idx], 
                                 cmap=cmap, s=10, alpha=0.6,
                                 vmin=targets[:, var_idx].min(), 
                                 vmax=targets[:, var_idx].max())
        else:
            # Wind Direction
            scatter = ax2.scatter(lons, lats, alts, c=pred_wind_dir, 
                                 cmap=cmap, s=10, alpha=0.6, vmin=0, vmax=360)
        
        ax2.set_xlabel('Longitude', fontweight='bold', fontsize=9)
        ax2.set_ylabel('Latitude', fontweight='bold', fontsize=9)
        ax2.set_zlabel('Altitude (m)', fontweight='bold', fontsize=9)
        ax2.set_title(f'Predicted\n{feature_names[var_idx]}', fontweight='bold', fontsize=10)
        plt.colorbar(scatter, ax=ax2, label=label, shrink=0.5, pad=0.1)
        
        # Column 3: Absolute Error
        ax3 = fig2.add_subplot(5, 3, var_idx * 3 + 3, projection='3d')
        error_abs = np.abs(error_list[var_idx])
        scatter = ax3.scatter(lons, lats, alts, c=error_abs, 
                             cmap='Reds', s=10, alpha=0.6,
                             vmin=0, vmax=np.percentile(error_abs, 95))
        
        ax3.set_xlabel('Longitude', fontweight='bold', fontsize=9)
        ax3.set_ylabel('Latitude', fontweight='bold', fontsize=9)
        ax3.set_zlabel('Altitude (m)', fontweight='bold', fontsize=9)
        ax3.set_title(f'Absolute Error\n{feature_names[var_idx]}', fontweight='bold', fontsize=10)
        plt.colorbar(scatter, ax=ax3, label=f'|Error|', shrink=0.5, pad=0.1)
    
    plt.tight_layout()
    trajectory_3d_path = f'visualizations/{output_name}_3d_trajectory.png'
    plt.savefig(trajectory_3d_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {trajectory_3d_path}")
    plt.close()
    
    # ========================================================================
    # FIGURE 3: Altitude vs Variables - Show vertical atmospheric structure
    # ========================================================================
    # FIGURE 3: Altitude vs Variables - Show vertical atmospheric structure
    # ========================================================================
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
    axes3 = axes3.flatten()
    
    # Subsample for clarity
    step = max(1, len(alts) // 2000)
    idx_sub = np.arange(0, len(alts), step)
    
    for i in range(5):
        ax = axes3[i]
        name = feature_names[i]
        
        if i < 4:
            # Regular features: Altitude vs Variable, colored by time
            scatter = ax.scatter(targets[idx_sub, i], alts[idx_sub], 
                               c=times[idx_sub], cmap='viridis', 
                               s=15, alpha=0.5, label='Ground Truth')
            ax.scatter(predictions[idx_sub, i], alts[idx_sub], 
                      c=times[idx_sub], cmap='plasma', 
                      s=15, alpha=0.5, marker='x', label='Prediction')
            
            ax.set_xlabel(name, fontweight='bold', fontsize=10)
        else:
            # Wind direction: Altitude vs Wind Direction
            scatter = ax.scatter(gt_wind_dir[idx_sub], alts[idx_sub], 
                               c=times[idx_sub], cmap='viridis', 
                               s=15, alpha=0.5, label='Ground Truth')
            ax.scatter(pred_wind_dir[idx_sub], alts[idx_sub], 
                      c=times[idx_sub], cmap='plasma', 
                      s=15, alpha=0.5, marker='x', label='Prediction')
            
            ax.set_xlabel('Wind Direction (°)', fontweight='bold', fontsize=10)
            ax.set_xlim(0, 360)
        
        ax.set_ylabel('Altitude (m)', fontweight='bold', fontsize=10)
        ax.set_title(f'{name} vs Altitude', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for time
        if i == 2:  # Add one colorbar
            plt.colorbar(scatter, ax=ax, label='Time (hours)')
    
    # Remove the 6th subplot
    axes3[5].remove()
    
    plt.tight_layout()
    altitude_path = f'visualizations/{output_name}_altitude_profiles.png'
    plt.savefig(altitude_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {altitude_path}")
    plt.close()
    
    # ========================================================================
    # FIGURE 4: Time Series at Different Altitudes
    # ========================================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))
    
    # Divide altitude into quantiles
    alt_quantiles = np.percentile(alts, [0, 33, 66, 100])
    alt_labels = ['Low Alt', 'Mid Alt', 'High Alt']
    
    # Temperature at different altitudes
    ax = axes4[0, 0]
    for i in range(3):
        mask = (alts >= alt_quantiles[i]) & (alts < alt_quantiles[i+1])
        if mask.sum() > 10:
            time_sorted = np.argsort(times[mask])
            ax.plot(times[mask][time_sorted][:200], targets[mask, 0][time_sorted][:200], 
                   'o-', alpha=0.6, markersize=3, linewidth=1, label=f'GT {alt_labels[i]}')
            ax.plot(times[mask][time_sorted][:200], predictions[mask, 0][time_sorted][:200], 
                   's--', alpha=0.6, markersize=3, linewidth=1, label=f'Pred {alt_labels[i]}')
    ax.set_xlabel('Time (hours)', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('Temperature Evolution at Different Altitudes', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Wind Speed at different altitudes
    ax = axes4[0, 1]
    for i in range(3):
        mask = (alts >= alt_quantiles[i]) & (alts < alt_quantiles[i+1])
        if mask.sum() > 10:
            time_sorted = np.argsort(times[mask])
            ax.plot(times[mask][time_sorted][:200], targets[mask, 1][time_sorted][:200], 
                   'o-', alpha=0.6, markersize=3, linewidth=1, label=f'GT {alt_labels[i]}')
            ax.plot(times[mask][time_sorted][:200], predictions[mask, 1][time_sorted][:200], 
                   's--', alpha=0.6, markersize=3, linewidth=1, label=f'Pred {alt_labels[i]}')
    ax.set_xlabel('Time (hours)', fontweight='bold')
    ax.set_ylabel('Wind Speed (m/s)', fontweight='bold')
    ax.set_title('Wind Speed Evolution at Different Altitudes', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Pressure at different altitudes
    ax = axes4[1, 0]
    for i in range(3):
        mask = (alts >= alt_quantiles[i]) & (alts < alt_quantiles[i+1])
        if mask.sum() > 10:
            time_sorted = np.argsort(times[mask])
            ax.plot(times[mask][time_sorted][:200], targets[mask, 3][time_sorted][:200], 
                   'o-', alpha=0.6, markersize=3, linewidth=1, label=f'GT {alt_labels[i]}')
            ax.plot(times[mask][time_sorted][:200], predictions[mask, 3][time_sorted][:200], 
                   's--', alpha=0.6, markersize=3, linewidth=1, label=f'Pred {alt_labels[i]}')
    ax.set_xlabel('Time (hours)', fontweight='bold')
    ax.set_ylabel('Pressure (hPa)', fontweight='bold')
    ax.set_title('Pressure Evolution at Different Altitudes', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Error vs Time
    ax = axes4[1, 1]
    temp_mae_by_time = []
    time_bins_plot = np.linspace(times.min(), times.max(), 20)
    for i in range(len(time_bins_plot)-1):
        mask = (times >= time_bins_plot[i]) & (times < time_bins_plot[i+1])
        if mask.sum() > 0:
            temp_mae_by_time.append(np.mean(np.abs(error_list[0][mask])))
        else:
            temp_mae_by_time.append(np.nan)
    
    ax.plot(time_bins_plot[:-1], temp_mae_by_time, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Time (hours)', fontweight='bold')
    ax.set_ylabel('Temperature MAE (°C)', fontweight='bold')
    ax.set_title('Prediction Error Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timeseries_alt_path = f'visualizations/{output_name}_timeseries_altitudes.png'
    plt.savefig(timeseries_alt_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {timeseries_alt_path}")
    plt.close()
    
    # ========================================================================
    # FIGURE 5: Scatter Plots (Predicted vs Ground Truth) - PAPER QUALITY
    # ========================================================================
    fig5, axes5 = plt.subplots(2, 3, figsize=(15, 10))
    axes5 = axes5.flatten()
    
    # Plot first 4 features normally, then wind direction
    for i in range(5):
        ax = axes5[i]
        name = feature_names[i]
        
        if i < 4:
            # Regular scatter plot colored by altitude
            scatter = ax.scatter(targets[:, i], predictions[:, i], 
                               c=alts, cmap='viridis', alpha=0.4, s=8, edgecolors='none')
            
            # Perfect prediction line
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Statistics
            mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
            rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i])**2))
            var_target = np.var(targets[:, i])
            if var_target > 1e-6:
                r2 = 1 - np.sum((targets[:, i] - predictions[:, i])**2) / np.sum((targets[:, i] - np.mean(targets[:, i]))**2)
            else:
                r2 = 0.0
            
            ax.set_xlabel(f'Ground Truth {name}', fontsize=10, fontweight='bold')
            ax.set_ylabel(f'Predicted {name}', fontsize=10, fontweight='bold')
        else:
            # Wind direction scatter plot
            scatter = ax.scatter(gt_wind_dir, pred_wind_dir, 
                               c=alts, cmap='viridis', alpha=0.4, s=8, edgecolors='none')
            
            # Perfect prediction line
            ax.plot([0, 360], [0, 360], 'r--', linewidth=2, label='Perfect Prediction')
            ax.set_xlim(0, 360)
            ax.set_ylim(0, 360)
            
            # Statistics
            mae = np.mean(np.abs(wind_dir_error))
            rmse = np.sqrt(np.mean(wind_dir_error**2))
            r2 = 1 - np.mean(wind_dir_error**2) / (180**2)
            
            ax.set_xlabel('Ground Truth Wind Direction (°)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted Wind Direction (°)', fontsize=10, fontweight='bold')
        
        ax.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
        
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add colorbar for altitude
        if i == 2:
            plt.colorbar(scatter, ax=ax, label='Altitude (m)')
    
    # Remove the 6th subplot
    axes5[5].remove()
    
    plt.tight_layout()
    scatter_path = f'visualizations/{output_name}_scatter.png'
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {scatter_path}")
    plt.close()
    
    # ========================================================================
    # FIGURE 6: Error Distribution - PAPER QUALITY
    # ========================================================================
    fig6, axes6 = plt.subplots(2, 3, figsize=(15, 10))
    axes6 = axes6.flatten()
    
    for i, name in enumerate(feature_names):
        ax = axes6[i]
        error = error_list[i]
        
        # Histogram with KDE
        ax.hist(error, bins=50, alpha=0.7, edgecolor='black', density=True)
        
        # Statistics
        mean_err = np.mean(error)
        std_err = np.std(error)
        
        # Add vertical lines
        ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}')
        ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero Error')
        
        ax.text(0.05, 0.95, f'Mean: {mean_err:.2f}\nStd: {std_err:.2f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               fontsize=9)
        
        ax.set_xlabel(f'Prediction Error {name}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Density', fontsize=10, fontweight='bold')
        ax.set_title(f'Error Distribution: {name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Remove the 6th subplot
    axes6[5].remove()
    
    plt.tight_layout()
    error_path = f'visualizations/{output_name}_error_dist.png'
    plt.savefig(error_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {error_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Weather Prediction on Routes')
    parser.add_argument('--route', type=int, choices=[1, 2], 
                       help='Route number (1 or 2)')
    parser.add_argument('--csv', type=str, 
                       help='Path to custom CSV file')
    parser.add_argument('--output', type=str, default='predictions',
                       help='Output prefix for CSV and plot')
    
    args = parser.parse_args()
    
    if args.route is None and args.csv is None:
        parser.error("Must specify --route or --csv")
    
    print("="*60)
    print("WEATHER ROUTE PREDICTION")
    print("="*60)
    
    # Setup
    device = torch.device(TRAINING_CONFIG['device'])
    config = {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'training': TRAINING_CONFIG
    }
    
    # Load model
    model, scaler = load_model(device)
    
    # Create scaler if needed
    if scaler is None:
        scaler = create_scaler(config)
    
    # Determine route path
    if args.route:
        route_path = DATA_CONFIG[f'route{args.route}_path']
        output_name = f'predictions_route{args.route}'
    else:
        route_path = args.csv
        output_name = args.output
    
    # Make predictions
    predictions, targets, coords, times, original_wind_dir = predict_route(
        model, route_path, scaler, config, device
    )
    
    if predictions is not None:
        save_and_plot(predictions, targets, coords, times, output_name, original_wind_dir)
        print("\n" + "="*60)
        print("✓ DONE!")
        print("="*60)
    else:
        print("\n❌ No predictions made")


if __name__ == "__main__":
    main()
