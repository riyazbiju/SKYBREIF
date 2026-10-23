"""
Create a modified route1 with 450 knots cruise speed and 12:00 start time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Loading original route1_subset.csv...")
df = pd.read_csv('data/route1_subset.csv')

print(f"Original route: {len(df)} waypoints")

# Subsample to reasonable size (every 1000th point to get ~946 points)
df_sampled = df.iloc[::1000].copy().reset_index(drop=True)
print(f"Subsampled to: {len(df_sampled)} waypoints")

# Update step_order to be sequential
df_sampled['step_order'] = range(len(df_sampled))

# Set start time to 12:00 (noon) on 2026-01-01
start_time = datetime(2026, 1, 1, 12, 0, 0)

# Calculate cruise speed: 450 knots = 450 nautical miles per hour
# Approximate: 1 nautical mile ≈ 1.852 km
cruise_speed_kmh = 450 * 1.852  # ~833.4 km/h

# Calculate distances between consecutive waypoints
lats = df_sampled['Latitude'].values
lons = df_sampled['Longitude'].values

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in km"""
    R = 6371  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

# Calculate cumulative flight time based on distances
times = [start_time]
cumulative_time = 0

for i in range(1, len(df_sampled)):
    dist = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
    time_hours = dist / cruise_speed_kmh
    cumulative_time += time_hours
    times.append(start_time + timedelta(hours=cumulative_time))

# Update Date_Time column
df_sampled['Date_Time'] = times

# Set Wind_Speed to 450 knots (cruise speed for the aircraft)
# Note: This represents the aircraft speed, not actual wind speed
# Keep original wind conditions from the dataset
print("\nModified route details:")
print(f"  Start time: {times[0]}")
print(f"  End time: {times[-1]}")
print(f"  Duration: {cumulative_time:.2f} hours")
print(f"  Total distance: ~{cumulative_time * cruise_speed_kmh:.1f} km")
print(f"  Cruise speed: 450 knots ({cruise_speed_kmh:.1f} km/h)")
print(f"  Waypoints: {len(df_sampled)}")

# Save modified route
output_path = 'data/route1_modified_450kts.csv'
df_sampled.to_csv(output_path, index=False)
print(f"\n✓ Saved modified route to: {output_path}")

# Show first few waypoints
print("\nFirst 5 waypoints:")
print(df_sampled[['step_order', 'Date_Time', 'Latitude', 'Longitude', 'Altitude', 'Wind_Speed']].head())

print("\nLast 5 waypoints:")
print(df_sampled[['step_order', 'Date_Time', 'Latitude', 'Longitude', 'Altitude', 'Wind_Speed']].tail())
