import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spatiotemporal_latent_ode import (
    SpatioTemporalLatentODE,
    SpatioTemporalWeatherDataset
)
from config_multi_station import MODEL_CONFIG, DATA_CONFIG
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler(checkpoint_path):
    """Load trained model and scaler"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = SpatioTemporalLatentODE(config=MODEL_CONFIG)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    return model, scaler

def predict_route(model, scaler, route_file):
    """Generate predictions for a route"""
    df = pd.read_csv(route_file)
    
    # Normalize column names
    col_mapping = {
        'Date_Time': 'datetime',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Altitude': 'altitude',
        'Temperature': 'temperature',
        'Wind_Speed': 'windspeed',
        'Wind_Direction': 'winddirection',
        'Visiblity': 'visibility',
        'Pressure': 'pressure'
    }
    df = df.rename(columns=col_mapping)
    
    # Add station column if missing
    if 'station' not in df.columns:
        df['station'] = 'route'
    
    # Convert to datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Add time_hours if not present
    if 'time_hours' not in df.columns:
        df = df.sort_values('datetime')
        min_time = df['datetime'].min()
        df['time_hours'] = (df['datetime'] - min_time).dt.total_seconds() / 3600
    
    # Compute wind components if needed
    if 'wind_u' not in df.columns or 'wind_v' not in df.columns:
        if 'winddirection' in df.columns and 'windspeed' in df.columns:
            # Convert wind direction (degrees) to u,v components
            wind_dir_rad = np.deg2rad(df['winddirection'])
            df['wind_u'] = -df['windspeed'] * np.sin(wind_dir_rad)
            df['wind_v'] = -df['windspeed'] * np.cos(wind_dir_rad)
        else:
            df['wind_u'] = 0.0
            df['wind_v'] = 0.0
    
    # Store original values for later
    df_original = df.copy()
    
    # Scale the features
    feat_cols = ['temperature', 'windspeed', 'visibility', 'pressure', 'wind_u', 'wind_v']
    df[feat_cols] = scaler.transform(df[feat_cols])
    
    # Create config for dataset
    config = {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
    }
    
    # Create dataset
    dataset = SpatioTemporalWeatherDataset(df, config, scaler, mode='test')
    
    all_predictions = []
    all_actuals = []
    all_query_coords = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            
            # Add batch dimension
            obs_seq = data['obs_seq'].unsqueeze(0)
            obs_spacetime = data['obs_spacetime'].unsqueeze(0)
            query_spacetime = data['query_spacetime'].unsqueeze(0)
            target = data['target']
            
            # Get prediction
            pred, _, _, _ = model(obs_seq, obs_spacetime, query_spacetime)
            
            # Convert to numpy
            pred_np = pred.squeeze(0).numpy()
            target_np = target.numpy()
            query_np = query_spacetime.squeeze(0).numpy()
            
            # Inverse transform
            pred_original = scaler.inverse_transform(pred_np.reshape(1, -1))
            actual_original = scaler.inverse_transform(target_np.reshape(1, -1))
            
            all_predictions.append(pred_original[0])
            all_actuals.append(actual_original[0])
            all_query_coords.append(query_np)
    
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    query_coords = np.array(all_query_coords)
    
    return predictions, actuals, query_coords, df_original

def create_visualizations(route_name, predictions, actuals, query_coords, df):
    """Create comprehensive visualizations for a route"""
    
    variable_names = ['Temperature (°C)', 'Dew Point (°C)', 'Wind Speed (m/s)', 
                      'Visibility (m)', 'Pressure (hPa)', 'Wind Direction (°)']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Altitude Profile (top row, full width)
    ax1 = plt.subplot(3, 3, (1, 3))
    
    # Extract altitude from query coordinates (index 2, scaled by altitude_scale from config)
    altitude_scale = DATA_CONFIG.get('altitude_scale', 40000.0)
    altitudes = query_coords[:, 2] * altitude_scale  # Rescale to meters using correct scale
    waypoint_indices = range(len(altitudes))
    
    ax1.plot(waypoint_indices, altitudes, 
             'o-', linewidth=2, markersize=6, color='navy', label='Flight Altitude')
    ax1.fill_between(waypoint_indices, altitudes, 
                      alpha=0.3, color='skyblue')
    ax1.set_xlabel('Waypoint Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Altitude (meters)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{route_name} - Altitude Profile Along Route', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add altitude statistics
    alt_text = f'Min: {altitudes.min():.0f}m | Max: {altitudes.max():.0f}m | Mean: {altitudes.mean():.0f}m'
    ax1.text(0.5, 0.95, alt_text, transform=ax1.transAxes, 
             fontsize=10, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2-7. Prediction vs True for each variable (2 rows, 3 columns each)
    for i, var_name in enumerate(variable_names):
        ax = plt.subplot(3, 3, i + 4)
        
        pred_values = predictions[:, i]
        actual_values = actuals[:, i]
        
        # Scatter plot
        ax.scatter(actual_values, pred_values, alpha=0.5, s=20, edgecolors='none')
        
        # Perfect prediction line
        min_val = min(actual_values.min(), pred_values.min())
        max_val = max(actual_values.max(), pred_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate metrics
        mae = np.mean(np.abs(pred_values - actual_values))
        rmse = np.sqrt(np.mean((pred_values - actual_values) ** 2))
        r2 = 1 - (np.sum((actual_values - pred_values) ** 2) / 
                  np.sum((actual_values - actual_values.mean()) ** 2))
        
        ax.set_xlabel(f'True {var_name}', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Predicted {var_name}', fontsize=10, fontweight='bold')
        ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{route_name}_visualizations.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {route_name}_visualizations.png")
    plt.close()
    
    # Create a second figure with time series plots
    fig2 = plt.figure(figsize=(20, 10))
    
    for i, var_name in enumerate(variable_names):
        ax = plt.subplot(2, 3, i + 1)
        
        pred_values = predictions[:, i]
        actual_values = actuals[:, i]
        indices = range(len(pred_values))
        
        ax.plot(indices, actual_values, 'o-', label='True', linewidth=2, markersize=4, alpha=0.7)
        ax.plot(indices, pred_values, 's-', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
        
        ax.set_xlabel('Sample Index', fontsize=10, fontweight='bold')
        ax.set_ylabel(var_name, fontsize=10, fontweight='bold')
        ax.set_title(f'{var_name} - Time Series', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{route_name}_timeseries.png', dpi=300, bbox_inches='tight')
    print(f"Saved time series: {route_name}_timeseries.png")
    plt.close()

def main():
    print("Loading model and generating visualizations...")
    print("=" * 80)
    
    # Load model
    model, scaler = load_model_and_scaler('best_spatiotemporal_model.pt')
    print("✓ Model loaded successfully")
    
    # Process Route 1
    print("\nProcessing Route 1...")
    pred1, actual1, coords1, df1 = predict_route(model, scaler, 'route1_subset.csv')
    print(f"  - Generated {len(pred1)} predictions")
    create_visualizations('Route1', pred1, actual1, coords1, df1)
    
    # Process Route 2
    print("\nProcessing Route 2...")
    pred2, actual2, coords2, df2 = predict_route(model, scaler, 'route2_subset.csv')
    print(f"  - Generated {len(pred2)} predictions")
    create_visualizations('Route2', pred2, actual2, coords2, df2)
    
    print("\n" + "=" * 80)
    print("✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - Route1_visualizations.png (altitude + prediction vs true)")
    print("  - Route1_timeseries.png (time series for all variables)")
    print("  - Route2_visualizations.png (altitude + prediction vs true)")
    print("  - Route2_timeseries.png (time series for all variables)")

if __name__ == "__main__":
    main()
