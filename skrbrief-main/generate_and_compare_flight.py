"""
Generate realistic flight plan with ground truth and compare with predictions.
Creates 3D visualization showing ground truth vs predicted trajectories with error metrics.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Import model components
from spatiotemporal_latent_ode import SpatioTemporalLatentODE

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def generate_realistic_flight_plan(start_lat, start_lon, start_elev_ft,
                                   end_lat, end_lon, end_elev_ft,
                                   cruise_altitude, cruise_speed,
                                   start_datetime, num_waypoints=50):
    """
    Generate realistic flight plan with climb, cruise, descent phases.
    
    Args:
        start_lat, start_lon: Starting coordinates
        start_elev_ft: Starting station elevation in feet
        end_lat, end_lon: Ending coordinates
        end_elev_ft: Ending station elevation in feet
        cruise_altitude: Cruise altitude in feet
        cruise_speed: Cruise speed in knots
        start_datetime: Flight start time
        num_waypoints: Number of waypoints to generate
    """
    
    # Constants
    climb_rate = 2000  # ft/min
    descent_rate = 2000  # ft/min
    kts_to_kmh = 1.852
    
    # Calculate total distance
    total_distance_km = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    
    # Calculate climb and descent distances
    climb_altitude_change = cruise_altitude - start_elev_ft
    descent_altitude_change = cruise_altitude - end_elev_ft
    
    climb_time_min = climb_altitude_change / climb_rate
    descent_time_min = descent_altitude_change / descent_rate
    climb_time_hr = climb_time_min / 60
    descent_time_hr = descent_time_min / 60
    
    # Distance covered during climb/descent (at cruise speed)
    speed_kmh = cruise_speed * kts_to_kmh
    climb_distance_km = climb_time_hr * speed_kmh
    descent_distance_km = descent_time_hr * speed_kmh
    
    # Cruise distance
    cruise_distance_km = total_distance_km - climb_distance_km - descent_distance_km
    cruise_time_hr = cruise_distance_km / speed_kmh
    
    # Total flight time
    total_time_hr = climb_time_hr + cruise_time_hr + descent_time_hr
    
    # Distribute waypoints across flight phases
    waypoints_data = []
    
    for i in range(num_waypoints):
        fraction = i / (num_waypoints - 1)
        
        # Calculate position along great circle
        lat = start_lat + (end_lat - start_lat) * fraction
        lon = start_lon + (end_lon - start_lon) * fraction
        
        # Calculate altitude based on flight phase
        distance_fraction = fraction
        distance_km = total_distance_km * distance_fraction
        
        if distance_km <= climb_distance_km:
            # CLIMB phase
            climb_fraction = distance_km / climb_distance_km if climb_distance_km > 0 else 0
            altitude = start_elev_ft + climb_fraction * (cruise_altitude - start_elev_ft)
            phase = "CLIMB"
            elapsed_time_hr = climb_time_hr * climb_fraction
        elif distance_km >= (total_distance_km - descent_distance_km):
            # DESCENT phase
            descent_start_dist = total_distance_km - descent_distance_km
            descent_fraction = (distance_km - descent_start_dist) / descent_distance_km if descent_distance_km > 0 else 1
            altitude = cruise_altitude - descent_fraction * (cruise_altitude - end_elev_ft)
            phase = "DESCENT"
            elapsed_time_hr = climb_time_hr + cruise_time_hr + (descent_time_hr * descent_fraction)
        else:
            # CRUISE phase
            altitude = cruise_altitude
            phase = "CRUISE"
            cruise_dist = distance_km - climb_distance_km
            elapsed_time_hr = climb_time_hr + (cruise_dist / speed_kmh)
        
        # Calculate timestamp
        timestamp = start_datetime + timedelta(hours=elapsed_time_hr)
        
        waypoints_data.append({
            'Latitude': lat,
            'Longitude': lon,
            'Altitude': altitude,
            'Date_Time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Flight_Phase': phase,
            'Waypoint_Number': i + 1
        })
    
    df = pd.DataFrame(waypoints_data)
    
    print(f"\n{'='*70}")
    print(f"FLIGHT PLAN GENERATED")
    print(f"{'='*70}")
    print(f"Route: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    print(f"Distance: {total_distance_km:.1f} km ({total_distance_km/1.852:.1f} NM)")
    print(f"Start Elevation: {start_elev_ft:.0f} ft")
    print(f"End Elevation: {end_elev_ft:.0f} ft")
    print(f"Cruise: FL{int(cruise_altitude/100)} ({cruise_altitude:.0f} ft) at {cruise_speed} kts")
    print(f"Flight Time: {int(total_time_hr)}h {int((total_time_hr % 1) * 60)}m")
    print(f"Waypoints: {num_waypoints}")
    print(f"Start: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {waypoints_data[-1]['Date_Time']}")
    print(f"Phases:")
    print(f"  - Climb: {climb_time_min:.0f} min, {climb_distance_km:.1f} km")
    print(f"  - Cruise: {cruise_time_hr:.2f}h, {cruise_distance_km:.1f} km")
    print(f"  - Descent: {descent_time_min:.0f} min, {descent_distance_km:.1f} km")
    print(f"{'='*70}\n")
    
    return df, total_time_hr

def load_model_and_stations(model_path, station_csv):
    """Load trained model and station database."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Get config - it's nested under 'model' key
    config = checkpoint.get('config', {}).get('model', {})
    
    # If config is still empty, try the checkpoint's config directly
    if not config:
        config = checkpoint.get('config', {})
    
    model = SpatioTemporalLatentODE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset_start_time = checkpoint.get('dataset_start_time', 0.0)
    print(f"✓ Model loaded (dataset start time: {dataset_start_time} hours)")
    
    print(f"Loading station database from {station_csv}...")
    station_df = pd.read_csv(station_csv)
    print(f"✓ Loaded {len(station_df)} stations")
    
    return model, station_df, dataset_start_time

def find_k_nearest_stations(lat, lon, station_df, k=10):
    """Find K nearest stations to given coordinates."""
    # Get unique stations only (station CSV has multiple time entries)
    unique_stations = station_df.drop_duplicates(subset=['station'])
    
    distances = []
    for _, row in unique_stations.iterrows():
        dist = haversine_distance(lat, lon, row['latitude'], row['longitude'])
        distances.append((row['station'], dist, row))
    
    distances.sort(key=lambda x: x[1])
    return distances[:k]

def predict_with_knearest(model, lat, lon, altitude, timestamp, station_df, 
                         dataset_start_time, k=10):
    """Make prediction using K-nearest spatial aggregation."""
    
    # Find K nearest stations
    nearest = find_k_nearest_stations(lat, lon, station_df, k)
    
    # Get station features
    station_features = []
    weights = []
    
    for station_code, dist, row in nearest:
        # Spatial features
        spatial = torch.tensor([row['latitude'], row['longitude'], row['altitude']], 
                              dtype=torch.float32)
        station_features.append(spatial)
        
        # Distance weight (inverse distance weighting)
        weight = 1.0 / (dist + 0.1)  # Add small constant to avoid division by zero
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Stack features
    station_features = torch.stack(station_features)  # [K, 3]
    
    # Time in hours from dataset start
    time_hours = (timestamp - datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)).total_seconds() / 3600
    time_from_start = time_hours - dataset_start_time
    
    # Create query point
    query_spatial = torch.tensor([[lat, lon, altitude]], dtype=torch.float32)
    query_time = torch.tensor([[time_from_start]], dtype=torch.float32)
    
    # Get latent states for all stations
    with torch.no_grad():
        try:
            # Encode stations at time 0
            t_0 = torch.zeros(k, 1)
            z_stations = model.encoder(station_features.unsqueeze(1), t_0)  # [K, latent_dim]
            
            # Evolve to query time
            t_query = query_time.expand(k, 1)
            z_evolved = model.ode_func.integrate(z_stations, t_0, t_query)
            
            # Aggregate using distance weights
            z_agg = torch.sum(z_evolved * torch.tensor(weights, dtype=torch.float32).unsqueeze(1), dim=0, keepdim=True)
            
            # Decode to predictions
            pred = model.decoder(z_agg)
            
        except Exception as e:
            # Fallback: spatial-only (no temporal evolution)
            z_stations = model.encoder(station_features.unsqueeze(1), torch.zeros(k, 1))
            z_agg = torch.sum(z_stations * torch.tensor(weights, dtype=torch.float32).unsqueeze(1), dim=0, keepdim=True)
            pred = model.decoder(z_agg)
    
    return pred.squeeze().numpy()

def generate_ground_truth(waypoint):
    """
    Generate synthetic ground truth weather data.
    In practice, you would use actual observed weather data.
    """
    # Simple synthetic model for demonstration
    lat = waypoint['Latitude']
    lon = waypoint['Longitude']
    alt = waypoint['Altitude']
    
    # Temperature: decreases with altitude and latitude
    temp_base = 15 - (lat - 45) * 0.5  # Colder at higher latitudes
    temp_alt_effect = -0.0065 * (alt * 0.3048)  # Standard lapse rate (converted ft to m)
    temperature = temp_base + temp_alt_effect + np.random.normal(0, 2)
    
    # Wind speed: increases with altitude
    windspeed = 20 + (alt / 1000) + np.random.normal(0, 5)
    
    # Wind direction: prevailing westerlies
    wind_direction = 270 + np.random.normal(0, 30)
    
    # Pressure: decreases with altitude
    pressure_sea_level = 1013.25
    pressure = pressure_sea_level * np.exp(-alt * 0.3048 / 8400) + np.random.normal(0, 5)
    
    # Visibility: better at altitude
    visibility = min(10000, 5000 + alt * 0.5) + np.random.normal(0, 500)
    
    return {
        'temperature': temperature,
        'windspeed': windspeed,
        'wind_direction': wind_direction % 360,
        'pressure': pressure,
        'visibility': max(0, visibility)
    }

def create_comparison_visualization(flight_df, predictions_df, ground_truth_df, output_file):
    """Create 3D visualization comparing ground truth vs predicted trajectories."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Main 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Plot ground truth trajectory
    ax1.plot(ground_truth_df['Longitude'], 
             ground_truth_df['Latitude'],
             ground_truth_df['Altitude'],
             'g-', linewidth=2, alpha=0.8, label='Flight Path')
    
    # Plot start and end points
    ax1.scatter([ground_truth_df.iloc[0]['Longitude']], 
                [ground_truth_df.iloc[0]['Latitude']],
                [ground_truth_df.iloc[0]['Altitude']],
                c='green', s=200, marker='o', label='Start', edgecolors='black', linewidths=2)
    ax1.scatter([ground_truth_df.iloc[-1]['Longitude']], 
                [ground_truth_df.iloc[-1]['Latitude']],
                [ground_truth_df.iloc[-1]['Altitude']],
                c='red', s=200, marker='s', label='End', edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Altitude (ft)', fontsize=12, fontweight='bold')
    ax1.set_title('3D Flight Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Calculate errors
    temp_error = np.abs(predictions_df['pred_temperature'] - ground_truth_df['gt_temperature'])
    wind_error = np.abs(predictions_df['pred_windspeed'] - ground_truth_df['gt_windspeed'])
    pressure_error = np.abs(predictions_df['pred_pressure'] - ground_truth_df['gt_pressure'])
    
    # Temperature comparison
    ax2 = fig.add_subplot(2, 2, 2)
    waypoints = range(len(ground_truth_df))
    ax2.plot(waypoints, ground_truth_df['gt_temperature'], 'g-', linewidth=2, label='Ground Truth', marker='o')
    ax2.plot(waypoints, predictions_df['pred_temperature'], 'b--', linewidth=2, label='Predicted', marker='x')
    ax2.fill_between(waypoints, ground_truth_df['gt_temperature'], predictions_df['pred_temperature'], 
                     alpha=0.3, color='red', label=f'Error (MAE: {temp_error.mean():.2f}°C)')
    ax2.set_xlabel('Waypoint Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax2.set_title('Temperature: Ground Truth vs Predicted', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Wind speed comparison
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(waypoints, ground_truth_df['gt_windspeed'], 'g-', linewidth=2, label='Ground Truth', marker='o')
    ax3.plot(waypoints, predictions_df['pred_windspeed'], 'b--', linewidth=2, label='Predicted', marker='x')
    ax3.fill_between(waypoints, ground_truth_df['gt_windspeed'], predictions_df['pred_windspeed'],
                     alpha=0.3, color='red', label=f'Error (MAE: {wind_error.mean():.2f} m/s)')
    ax3.set_xlabel('Waypoint Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Wind Speed (m/s)', fontsize=11, fontweight='bold')
    ax3.set_title('Wind Speed: Ground Truth vs Predicted', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Pressure comparison
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(waypoints, ground_truth_df['gt_pressure'], 'g-', linewidth=2, label='Ground Truth', marker='o')
    ax4.plot(waypoints, predictions_df['pred_pressure'], 'b--', linewidth=2, label='Predicted', marker='x')
    ax4.fill_between(waypoints, ground_truth_df['gt_pressure'], predictions_df['pred_pressure'],
                     alpha=0.3, color='red', label=f'Error (MAE: {pressure_error.mean():.2f} hPa)')
    ax4.set_xlabel('Waypoint Number', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Pressure (hPa)', fontsize=11, fontweight='bold')
    ax4.set_title('Pressure: Ground Truth vs Predicted', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Print error statistics
    print(f"\n{'='*70}")
    print(f"ERROR STATISTICS")
    print(f"{'='*70}")
    print(f"Temperature MAE: {temp_error.mean():.2f}°C (RMSE: {np.sqrt((temp_error**2).mean()):.2f}°C)")
    print(f"Wind Speed MAE: {wind_error.mean():.2f} m/s (RMSE: {np.sqrt((wind_error**2).mean()):.2f} m/s)")
    print(f"Pressure MAE: {pressure_error.mean():.2f} hPa (RMSE: {np.sqrt((pressure_error**2).mean()):.2f} hPa)")
    print(f"{'='*70}\n")

def main():
    # Flight parameters
    start_lat = 60.75
    start_lon = -135.00
    start_elev_ft = 2317  # CYXY elevation
    
    end_lat = 50.00
    end_lon = -96.00
    end_elev_ft = 783  # CYWG elevation (approximate)
    
    cruise_altitude = 45000  # feet
    cruise_speed = 450  # knots
    num_waypoints = 50
    start_datetime = datetime(2026, 1, 15, 18, 0, 0, tzinfo=timezone.utc)
    
    # Generate flight plan
    flight_df, flight_time = generate_realistic_flight_plan(
        start_lat, start_lon, start_elev_ft,
        end_lat, end_lon, end_elev_ft,
        cruise_altitude, cruise_speed,
        start_datetime, num_waypoints
    )
    
    # Save flight plan
    output_flight_plan = 'data/route1_realistic_50wp_45kft.csv'
    flight_df.to_csv(output_flight_plan, index=False)
    print(f"✓ Saved flight plan: {output_flight_plan}\n")
    
    # Load model and stations
    model, station_df, dataset_start_time = load_model_and_stations(
        'best_spatiotemporal_model.pt',
        'data/jan_sorted_cleaned.csv'
    )
    
    print(f"\n{'='*70}")
    print(f"GENERATING PREDICTIONS AND GROUND TRUTH")
    print(f"{'='*70}")
    
    # Generate predictions and ground truth for each waypoint
    predictions = []
    ground_truths = []
    
    for idx, row in flight_df.iterrows():
        timestamp = datetime.strptime(row['Date_Time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        
        # Generate ground truth
        gt = generate_ground_truth(row)
        ground_truths.append(gt)
        
        # Generate prediction
        pred = predict_with_knearest(
            model, row['Latitude'], row['Longitude'], row['Altitude'],
            timestamp, station_df, dataset_start_time, k=10
        )
        
        predictions.append({
            'pred_temperature': pred[0],
            'pred_windspeed': pred[1],
            'pred_wind_direction': pred[2],
            'pred_pressure': pred[3],
            'pred_visibility': pred[4]
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(flight_df)} waypoints...")
    
    print(f"✓ Completed all {len(flight_df)} waypoints\n")
    
    # Create dataframes
    predictions_df = pd.DataFrame(predictions)
    ground_truth_df = pd.DataFrame(ground_truths)
    ground_truth_df.columns = ['gt_temperature', 'gt_windspeed', 'gt_wind_direction', 
                                'gt_pressure', 'gt_visibility']
    
    # Combine with flight data
    result_df = pd.concat([flight_df, ground_truth_df, predictions_df], axis=1)
    
    # Save combined results
    output_csv = 'predictions_route1_50wp_45kft_comparison.csv'
    result_df.to_csv(output_csv, index=False)
    print(f"✓ Saved predictions: {output_csv}\n")
    
    # Create visualization
    output_viz = 'visualizations/route1_50wp_45kft_ground_truth_vs_predicted.png'
    create_comparison_visualization(flight_df, predictions_df, ground_truth_df, output_viz)
    
    print(f"\n{'='*70}")
    print(f"✓ ANALYSIS COMPLETE!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
