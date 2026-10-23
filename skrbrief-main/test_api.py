#!/usr/bin/env python3
"""
Test script for the API server
"""

import requests
import json
from datetime import datetime

API_BASE = 'http://localhost:5001'

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint...")
    print("="*60)
    
    response = requests.get(f'{API_BASE}/health')
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_info():
    """Test info endpoint"""
    print("\n" + "="*60)
    print("Testing /info endpoint...")
    print("="*60)
    
    response = requests.get(f'{API_BASE}/info')
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_point_prediction():
    """Test point prediction"""
    print("\n" + "="*60)
    print("Testing /predict/point endpoint...")
    print("="*60)
    
    # Test location: Near Edmonton, Alberta
    query = {
        "lat": 56.1304,
        "lon": -106.3468,
        "alt": 10000,
        "datetime": "2026-01-15T12:00:00Z"
    }
    
    print(f"\nQuery:")
    print(json.dumps(query, indent=2))
    
    response = requests.post(
        f'{API_BASE}/predict/point',
        json=query,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
        
        print(f"\n{'='*60}")
        print("PREDICTION SUMMARY:")
        print(f"{'='*60}")
        print(f"Temperature:    {result['temperature']:.1f}°C")
        print(f"Wind Speed:     {result['windspeed']:.1f} kts")
        print(f"Wind Direction: {result['wind_direction']:.0f}°")
        print(f"Pressure:       {result['pressure']:.1f} hPa")
        print(f"Visibility:     {result['visibility']:.1f} km")
        if 'nearest_stations' in result:
            print(f"\nNearest Stations:")
            for i, (station, dist) in enumerate(zip(result['nearest_stations'], result['distances_km'])):
                print(f"  {i+1}. {station} ({dist:.1f} km)")
    else:
        print(f"Error: {response.text}")

def test_route_prediction():
    """Test route prediction"""
    print("\n" + "="*60)
    print("Testing /predict/route endpoint...")
    print("="*60)
    
    # Test route: 3 waypoints
    waypoints = [
        {
            "lat": 56.1304,
            "lon": -106.3468,
            "alt": 10000,
            "datetime": "2026-01-15T12:00:00Z"
        },
        {
            "lat": 56.5,
            "lon": -106.0,
            "alt": 15000,
            "datetime": "2026-01-15T13:00:00Z"
        },
        {
            "lat": 57.0,
            "lon": -105.5,
            "alt": 20000,
            "datetime": "2026-01-15T14:00:00Z"
        }
    ]
    
    print(f"\nRoute with {len(waypoints)} waypoints")
    
    response = requests.post(
        f'{API_BASE}/predict/route',
        json={"waypoints": waypoints},
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPredictions: {result['count']}")
        
        for i, pred in enumerate(result['predictions']):
            if 'error' in pred:
                print(f"\nWaypoint {i+1}: ERROR - {pred['error']}")
            else:
                print(f"\nWaypoint {i+1}:")
                print(f"  Temperature:    {pred['temperature']:.1f}°C")
                print(f"  Wind Speed:     {pred['windspeed']:.1f} kts")
                print(f"  Wind Direction: {pred['wind_direction']:.0f}°")
                print(f"  Pressure:       {pred['pressure']:.1f} hPa")
                print(f"  Visibility:     {pred['visibility']:.1f} km")
    else:
        print(f"Error: {response.text}")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("API SERVER TEST SUITE")
    print("="*60)
    
    try:
        test_health()
        test_info()
        test_point_prediction()
        test_route_prediction()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("Make sure the API server is running on http://localhost:5001")
