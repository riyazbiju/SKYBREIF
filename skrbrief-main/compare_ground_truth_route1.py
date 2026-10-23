"""
Compare ground truth from route1_subset.csv with model predictions.
Creates 3D visualization showing ground truth vs predicted trajectories with error metrics.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Import model components and prediction functions
from spatiotemporal_latent_ode import SpatioTemporalLatentODE
from predict_knearest import (
    load_model_and_data,
    load_station_database,
    predict_at_query
)
from config_multi_station import DATA_CONFIG

def convert_prediction_to_original_space(pred, scaler):
    """Convert predictions from scaled space back to original units."""
    # pred is [temp, windspeed, visibility, pressure, wind_u, wind_v]
    pred_denorm = scaler.inverse_transform(pred.reshape(1, -1))[0]
    
    temp = pred_denorm[0]
    windspeed = pred_denorm[1]
    visibility = pred_denorm[2]
    pressure = pred_denorm[3]
    wind_u = pred_denorm[4]
    wind_v = pred_denorm[5]
    
    # Convert wind components back to direction
    wind_direction = np.degrees(np.arctan2(-wind_u, -wind_v)) % 360
    
    return {
        'temperature': temp,
        'wind_speed': windspeed,
        'wind_direction': wind_direction,
        'pressure': pressure,
        'visibility': visibility
    }

def compute_errors(ground_truth, predictions):
    """Compute error metrics between ground truth and predictions."""
    errors = {}
    
    # Map ground truth columns to prediction columns
    column_map = {
        'temperature': 'Temperature',
        'wind_speed': 'Wind_Speed',
        'wind_direction': 'Wind_Direction',
        'pressure': 'Pressure',
        'visibility': 'Visiblity'  # Note: typo in ground truth column name
    }
    
    for pred_key, gt_key in column_map.items():
        gt = np.array(ground_truth[gt_key])
        pred = np.array(predictions[pred_key])
        
        mae = np.mean(np.abs(gt - pred))
        rmse = np.sqrt(np.mean((gt - pred)**2))
        
        errors[pred_key] = {
            'mae': mae,
            'rmse': rmse,
            'mean_gt': np.mean(gt),
            'mean_pred': np.mean(pred)
        }
    
    return errors

def create_3d_comparison_plot(ground_truth_df, predictions_df, errors, output_path):
    """Create 3D visualization comparing ground truth vs predictions."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 3D Trajectory plot
    ax1 = fig.add_subplot(231, projection='3d')
    
    # Ground truth trajectory
    ax1.plot(ground_truth_df['Longitude'], 
             ground_truth_df['Latitude'],
             ground_truth_df['Altitude'] / 1000,  # Convert to thousands of feet
             'b-', linewidth=2, label='Ground Truth', alpha=0.7)
    
    # Predicted trajectory (same lat/lon, but showing altitude profile)
    ax1.plot(predictions_df['Longitude'],
             predictions_df['Latitude'],
             predictions_df['Altitude'] / 1000,
             'r--', linewidth=2, label='Predicted', alpha=0.7)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_zlabel('Altitude (1000 ft)')
    ax1.set_title('3D Flight Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temperature comparison
    ax2 = fig.add_subplot(232)
    waypoints = range(len(ground_truth_df))
    ax2.plot(waypoints, ground_truth_df['Temperature'], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax2.plot(waypoints, predictions_df['temperature'], 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax2.fill_between(waypoints, ground_truth_df['Temperature'], predictions_df['temperature'], alpha=0.2, color='gray')
    ax2.set_xlabel('Waypoint')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title(f"Temperature (MAE: {errors['temperature']['mae']:.2f}°C, RMSE: {errors['temperature']['rmse']:.2f}°C)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Wind Speed comparison
    ax3 = fig.add_subplot(233)
    ax3.plot(waypoints, ground_truth_df['Wind_Speed'], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax3.plot(waypoints, predictions_df['wind_speed'], 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax3.fill_between(waypoints, ground_truth_df['Wind_Speed'], predictions_df['wind_speed'], alpha=0.2, color='gray')
    ax3.set_xlabel('Waypoint')
    ax3.set_ylabel('Wind Speed (km/h)')
    ax3.set_title(f"Wind Speed (MAE: {errors['wind_speed']['mae']:.2f} km/h, RMSE: {errors['wind_speed']['rmse']:.2f} km/h)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Wind Direction comparison
    ax4 = fig.add_subplot(234)
    ax4.plot(waypoints, ground_truth_df['Wind_Direction'], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax4.plot(waypoints, predictions_df['wind_direction'], 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax4.set_xlabel('Waypoint')
    ax4.set_ylabel('Wind Direction (°)')
    ax4.set_title(f"Wind Direction (MAE: {errors['wind_direction']['mae']:.2f}°, RMSE: {errors['wind_direction']['rmse']:.2f}°)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Pressure comparison
    ax5 = fig.add_subplot(235)
    ax5.plot(waypoints, ground_truth_df['Pressure'], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax5.plot(waypoints, predictions_df['pressure'], 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax5.fill_between(waypoints, ground_truth_df['Pressure'], predictions_df['pressure'], alpha=0.2, color='gray')
    ax5.set_xlabel('Waypoint')
    ax5.set_ylabel('Pressure (hPa)')
    ax5.set_title(f"Pressure (MAE: {errors['pressure']['mae']:.2f} hPa, RMSE: {errors['pressure']['rmse']:.2f} hPa)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Visibility comparison
    ax6 = fig.add_subplot(236)
    ax6.plot(waypoints, ground_truth_df['Visiblity'], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax6.plot(waypoints, predictions_df['visibility'], 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax6.fill_between(waypoints, ground_truth_df['Visiblity'], predictions_df['visibility'], alpha=0.2, color='gray')
    ax6.set_xlabel('Waypoint')
    ax6.set_ylabel('Visibility (km)')
    ax6.set_title(f"Visibility (MAE: {errors['visibility']['mae']:.2f} km, RMSE: {errors['visibility']['rmse']:.2f} km)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison visualization to {output_path}")
    plt.close()

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("GROUND TRUTH vs PREDICTED COMPARISON - ROUTE 1")
    print("="*70 + "\n")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'best_spatiotemporal_model.pt'
    station_file = 'data/jan_sorted_cleaned.csv'
    ground_truth_file = 'data/route1_subset.csv'
    output_csv = 'predictions_route1_comparison.csv'
    output_plot = 'visualizations/route1_ground_truth_vs_predicted_3d.png'
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Load model and station database
    model, scaler, dataset_start_time = load_model_and_data(model_path, device)
    station_data, time_min, time_max = load_station_database(station_file, scaler)
    
    # Load ground truth data
    print(f"\nLoading ground truth from {ground_truth_file}...")
    print("(This may take a moment for large files...)")
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    # Sample every N waypoints to get exactly 50 waypoints
    total_waypoints = len(ground_truth_df)
    sample_indices = np.linspace(0, total_waypoints - 1, 50, dtype=int)
    ground_truth_df = ground_truth_df.iloc[sample_indices].reset_index(drop=True)
    
    print(f"✓ Loaded and sampled {len(ground_truth_df)} waypoints from ground truth")
    print(f"  Route: ({ground_truth_df.iloc[0]['Latitude']:.2f}, {ground_truth_df.iloc[0]['Longitude']:.2f}) → "
          f"({ground_truth_df.iloc[-1]['Latitude']:.2f}, {ground_truth_df.iloc[-1]['Longitude']:.2f})")
    print(f"  Altitude range: {ground_truth_df['Altitude'].min():.0f} - {ground_truth_df['Altitude'].max():.0f} ft")
    
    # Convert datetime to hours since dataset start
    dataset_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    # Make predictions for each waypoint
    print("\nGenerating predictions for 50 waypoints...")
    predictions_list = []
    
    for idx, row in ground_truth_df.iterrows():
        # Parse datetime
        dt = pd.to_datetime(row['Date_Time'])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        time_hours = (dt - dataset_start).total_seconds() / 3600.0
        
        # Make prediction using K-nearest aggregation
        pred_scaled, k_stations, k_distances = predict_at_query(
            model,
            station_data,
            row['Latitude'],
            row['Longitude'],
            row['Altitude'],
            time_hours,
            K=10,
            max_seq=100,
            device=device
        )
        
        # Convert prediction back to original space
        pred = convert_prediction_to_original_space(pred_scaled, scaler)
        
        predictions_list.append({
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Altitude': row['Altitude'],
            'time_hours': time_hours,
            'temperature': pred['temperature'],
            'wind_speed': pred['wind_speed'],
            'wind_direction': pred['wind_direction'],
            'pressure': pred['pressure'],
            'visibility': pred['visibility'],
            'nearest_stations': ','.join(k_stations),
            'station_distances': ','.join([f"{d:.1f}" for d in k_distances])
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/50 waypoints...")
    
    predictions_df = pd.DataFrame(predictions_list)
    print(f"✓ Generated predictions for all waypoints")
    
    # Compute errors
    print("\nComputing error metrics...")
    errors = compute_errors(ground_truth_df, predictions_df)
    
    print("\n" + "="*70)
    print("ERROR METRICS SUMMARY")
    print("="*70)
    for key, metrics in errors.items():
        print(f"\n{key.upper()}:")
        print(f"  MAE:       {metrics['mae']:.4f}")
        print(f"  RMSE:      {metrics['rmse']:.4f}")
        print(f"  Mean GT:   {metrics['mean_gt']:.4f}")
        print(f"  Mean Pred: {metrics['mean_pred']:.4f}")
    
    # Save predictions
    predictions_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved predictions to {output_csv}")
    
    # Create visualization
    print("\nCreating 3D comparison visualization...")
    create_3d_comparison_plot(ground_truth_df, predictions_df, errors, output_plot)
    
    print("\n" + "="*70)
    print("✓ COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Predictions CSV: {output_csv}")
    print(f"  - 3D Visualization: {output_plot}")
    print()

if __name__ == '__main__':
    main()
