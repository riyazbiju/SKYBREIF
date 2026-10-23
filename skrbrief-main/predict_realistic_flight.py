"""
Predict weather for realistic flight route
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from config_multi_station import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG
from latent_ode_weather_multi_station import (
    MultiStationLatentODE,
    prepare_data_loaders,
    prepare_route_dataset,
    collate_fn
)
from simple_predict import (
    load_model_and_scaler,
    predict_and_collect,
    print_statistics,
    save_predictions_csv,
    plot_results
)


def main():
    print("="*70)
    print("REALISTIC FLIGHT ROUTE WEATHER PREDICTION")
    print("="*70)
    
    device = torch.device(TRAINING_CONFIG['device'])
    
    # Load model
    model_path = Path("final_model.pt")
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        return
    
    model, scaler = load_model_and_scaler(model_path, device)
    
    # Load data config
    config = {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'training': TRAINING_CONFIG
    }
    
    # If no scaler in checkpoint, create one from training data
    if scaler is None:
        print("Creating scaler from training data...")
        _, _, scaler = prepare_data_loaders(config)
    
    # Process realistic flight route
    route_path = 'realistic_flight_route.csv'
    route_name = 'realistic_flight'
    
    print("\n" + "="*70)
    print(f"PROCESSING REALISTIC FLIGHT ROUTE")
    print("="*70)
    
    # Get route dataset and loader
    route_ds = prepare_route_dataset(route_path, scaler, config)
    if route_ds is None:
        print(f"❌ Failed to load {route_name}")
        return
        
    route_loader = DataLoader(route_ds, batch_size=16, collate_fn=collate_fn)
    
    # Make predictions
    print(f"Making predictions for {route_name}...")
    predictions, targets, coords, times = predict_and_collect(
        model, route_loader, device, scaler
    )
    
    if predictions is not None:
        print(f"✓ Collected {len(predictions)} predictions")
        
        # Print statistics
        print_statistics(predictions, targets, route_name)
        
        # Save CSV
        save_predictions_csv(predictions, targets, coords, times, route_name)
        
        # Plot
        print(f"\nGenerating visualization for {route_name}...")
        plot_path = f'predictions_{route_name}.png'
        plot_results(predictions, targets, coords, times, route_name, plot_path)
    else:
        print(f"❌ No valid predictions for {route_name}")
    
    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
