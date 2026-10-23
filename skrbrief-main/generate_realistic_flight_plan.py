#!/usr/bin/env python3
"""
Generate Realistic Flight Plans
Creates realistic flight routes with proper climb, cruise, and descent phases
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def generate_realistic_flight_plan(
    start_lat, start_lon, 
    end_lat, end_lon,
    cruise_altitude=40000,  # feet
    cruise_speed=450,       # knots
    start_datetime="2026-01-15 08:00:00",
    climb_rate=2000,        # ft/min
    descent_rate=2000,      # ft/min
    waypoint_interval=5     # minutes
):
    """
    Generate a realistic flight plan with proper climb, cruise, and descent
    
    Parameters:
    - start_lat, start_lon: Starting coordinates
    - end_lat, end_lon: Ending coordinates
    - cruise_altitude: Cruise altitude in feet
    - cruise_speed: Cruise speed in knots
    - start_datetime: Flight start time
    - climb_rate: Climb rate in ft/min
    - descent_rate: Descent rate in ft/min
    - waypoint_interval: Minutes between waypoints
    """
    
    # Convert speed from knots to km/h (1 knot = 1.852 km/h)
    speed_kmh = cruise_speed * 1.852
    
    # Calculate total distance
    total_distance_km = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    
    # Calculate phase distances and times
    # Climb phase
    climb_time_hours = (cruise_altitude / climb_rate) / 60  # hours
    climb_distance_km = speed_kmh * climb_time_hours
    
    # Descent phase
    descent_time_hours = (cruise_altitude / descent_rate) / 60  # hours
    descent_distance_km = speed_kmh * descent_time_hours
    
    # Cruise phase
    cruise_distance_km = total_distance_km - climb_distance_km - descent_distance_km
    cruise_time_hours = cruise_distance_km / speed_kmh
    
    # Total flight time
    total_flight_hours = climb_time_hours + cruise_time_hours + descent_time_hours
    
    print(f"\n{'='*70}")
    print(f"FLIGHT PLAN GENERATION")
    print(f"{'='*70}")
    print(f"Route: ({start_lat:.2f}, {start_lon:.2f}) → ({end_lat:.2f}, {end_lon:.2f})")
    print(f"Total Distance: {total_distance_km:.1f} km ({total_distance_km/1.852:.1f} NM)")
    print(f"Cruise Altitude: {cruise_altitude:,} ft (FL{cruise_altitude//100})")
    print(f"Cruise Speed: {cruise_speed} kts")
    print(f"\nPhase Breakdown:")
    print(f"  Climb:   {climb_distance_km:.1f} km in {climb_time_hours*60:.1f} min (rate: {climb_rate:,} ft/min)")
    print(f"  Cruise:  {cruise_distance_km:.1f} km in {cruise_time_hours:.2f} hours")
    print(f"  Descent: {descent_distance_km:.1f} km in {descent_time_hours*60:.1f} min (rate: {descent_rate:,} ft/min)")
    print(f"\nTotal Flight Time: {total_flight_hours:.2f} hours ({int(total_flight_hours)}h {int((total_flight_hours % 1)*60)}m)")
    
    # Generate waypoints
    waypoints = []
    current_time = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
    time_step = timedelta(minutes=waypoint_interval)
    
    num_waypoints = int(total_flight_hours * 60 / waypoint_interval) + 1
    
    for i in range(num_waypoints):
        # Calculate fraction of journey completed
        fraction = i / (num_waypoints - 1) if num_waypoints > 1 else 0
        
        # Interpolate position
        lat = start_lat + (end_lat - start_lat) * fraction
        lon = start_lon + (end_lon - start_lon) * fraction
        
        # Calculate distance traveled
        distance_traveled_km = total_distance_km * fraction
        
        # Determine altitude based on phase
        if distance_traveled_km < climb_distance_km:
            # Climb phase
            climb_fraction = distance_traveled_km / climb_distance_km
            altitude = climb_fraction * cruise_altitude
            phase = "CLIMB"
        elif distance_traveled_km > (total_distance_km - descent_distance_km):
            # Descent phase
            descent_start = total_distance_km - descent_distance_km
            descent_fraction = (distance_traveled_km - descent_start) / descent_distance_km
            altitude = cruise_altitude * (1 - descent_fraction)
            phase = "DESCENT"
        else:
            # Cruise phase
            altitude = cruise_altitude
            phase = "CRUISE"
        
        # Add some realistic variation (±100 ft for turbulence)
        altitude += np.random.uniform(-100, 100)
        altitude = max(0, altitude)  # Ensure non-negative
        
        waypoints.append({
            'Latitude': lat,
            'Longitude': lon,
            'Altitude': altitude,
            'Date_Time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
            'Flight_Phase': phase,
            'Waypoint_Number': i + 1
        })
        
        current_time += time_step
    
    # Create DataFrame
    df = pd.DataFrame(waypoints)
    
    print(f"\nGenerated {len(waypoints)} waypoints")
    print(f"Start: {waypoints[0]['Date_Time']} at ({start_lat:.2f}, {start_lon:.2f})")
    print(f"End:   {waypoints[-1]['Date_Time']} at ({end_lat:.2f}, {end_lon:.2f})")
    
    return df


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REALISTIC FLIGHT PLAN GENERATOR")
    print("="*70)
    
    # Route 1: Northwest Canada route
    print("\n" + "="*70)
    print("GENERATING ROUTE 1 FLIGHT PLAN")
    print("="*70)
    
    route1_df = generate_realistic_flight_plan(
        start_lat=49.25,
        start_lon=-123.25,
        end_lat=68.25,
        end_lon=-89.75,
        cruise_altitude=40000,
        cruise_speed=450,
        start_datetime="2026-01-15 08:00:00",
        climb_rate=2000,
        descent_rate=2000,
        waypoint_interval=10  # Every 10 minutes
    )
    
    output_file1 = "data/route1_realistic_flight_plan.csv"
    route1_df.to_csv(output_file1, index=False)
    print(f"\n✓ Saved to: {output_file1}")
    
    # Route 2: Northern Canada route
    print("\n" + "="*70)
    print("GENERATING ROUTE 2 FLIGHT PLAN")
    print("="*70)
    
    route2_df = generate_realistic_flight_plan(
        start_lat=60.75,
        start_lon=-135.0,
        end_lat=50.0,
        end_lon=-96.0,
        cruise_altitude=40000,
        cruise_speed=450,
        start_datetime="2026-01-15 18:00:00",  # 6 PM start
        climb_rate=2000,
        descent_rate=2000,
        waypoint_interval=10  # Every 10 minutes
    )
    
    output_file2 = "data/route2_realistic_flight_plan.csv"
    route2_df.to_csv(output_file2, index=False)
    print(f"\n✓ Saved to: {output_file2}")
    
    print("\n" + "="*70)
    print("✓ ALL FLIGHT PLANS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Run predictions:")
    print(f"   python predict_knearest.py --route {output_file1} --visualize")
    print(f"   python predict_knearest.py --route {output_file2} --visualize")
    print(f"\n2. View the CSV files to verify the flight plans")
    print("="*70 + "\n")
