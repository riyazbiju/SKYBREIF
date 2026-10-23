"""
Predict with Spatio-Temporal Model
===================================
Make predictions using the 4D spatio-temporal latent ODE
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from spatiotemporal_latent_ode import (
    SpatioTemporalLatentODE,
    prepare_route_dataset,
    collate_fn
)
from config_multi_station import MODEL_CONFIG, DATA_CONFIG


def load_model_and_scaler(model_path, device):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = SpatioTemporalLatentODE(config=MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint.get('scaler')
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    
    print(f"✓ Model loaded (Epoch {epoch}, Val Loss: {val_loss})")
    
    return model, scaler


def predict_route(model, loader, device, scaler):
    """Make predictions on route"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_coords = []
    all_times = []
    
    with torch.no_grad():
        for batch in loader:
            obs_seq = batch['obs_seq'].to(device)
            obs_spacetime = batch['obs_spacetime'].to(device)
            query_spacetime = batch['query_spacetime'].to(device)
            target = batch['target'].to(device)
            
            try:
                pred, _, _, _ = model(obs_seq, obs_spacetime, query_spacetime, integrate_ode=True)
                
                if not torch.isnan(pred).any():
                    # Convert to original units
                    pred_np = scaler.inverse_transform(pred.cpu().numpy())
                    target_np = scaler.inverse_transform(target.cpu().numpy())
                    
                    all_predictions.append(pred_np)
                    all_targets.append(target_np)
                    
                    # Extract coordinates and times with correct denormalization
                    altitude_scale = DATA_CONFIG.get('altitude_scale', 40000.0)
                    total_hours = DATA_CONFIG.get('total_hours', 744)
                    
                    for i in range(query_spacetime.shape[0]):
                        coords = query_spacetime[i].cpu().numpy()
                        all_coords.append([coords[0], coords[1], coords[2] * altitude_scale])  # Unscale altitude correctly
                        all_times.append(coords[3] * total_hours)  # Unscale time correctly
                        
            except Exception as e:
                print(f"Warning: Batch error - {e}")
                continue
    
    if len(all_predictions) > 0:
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        coords = np.array(all_coords)
        times = np.array(all_times)
        
        return predictions, targets, coords, times
    else:
        return None, None, None, None


def print_statistics(predictions, targets, route_name):
    """Print detailed statistics"""
    
    feature_names = ['Temperature', 'Wind Speed', 'Visibility', 
                    'Pressure', 'Wind U', 'Wind V']
    units = ['°C', 'm/s', 'm', 'hPa', 'm/s', 'm/s']
    
    print("\n" + "="*70)
    print(f"PREDICTION STATISTICS - {route_name}")
    print("="*70)
    
    for i, (name, unit) in enumerate(zip(feature_names, units)):
        pred = predictions[:, i]
        gt = targets[:, i]
        
        mae = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(np.mean((pred - gt)**2))
        mape = np.mean(np.abs((pred - gt) / (np.abs(gt) + 1e-8))) * 100
        
        print(f"\n{name} ({unit}):")
        print(f"  MAE:        {mae:.3f}")
        print(f"  RMSE:       {rmse:.3f}")
        print(f"  MAPE:       {mape:.2f}%")
        print(f"  GT Range:   [{gt.min():.2f}, {gt.max():.2f}]")
        print(f"  Pred Range: [{pred.min():.2f}, {pred.max():.2f}]")


def save_predictions_csv(predictions, targets, coords, times, route_name):
    """Save predictions to CSV"""
    
    df = pd.DataFrame({
        'latitude': coords[:, 0],
        'longitude': coords[:, 1],
        'altitude_m': coords[:, 2],
        'time_hours': times,
        'pred_temperature': predictions[:, 0],
        'gt_temperature': targets[:, 0],
        'pred_windspeed': predictions[:, 1],
        'gt_windspeed': targets[:, 1],
        'pred_visibility': predictions[:, 2],
        'gt_visibility': targets[:, 2],
        'pred_pressure': predictions[:, 3],
        'gt_pressure': targets[:, 3],
        'pred_wind_u': predictions[:, 4],
        'gt_wind_u': targets[:, 4],
        'pred_wind_v': predictions[:, 5],
        'gt_wind_v': targets[:, 5],
    })
    
    output_path = f'predictions_{route_name}_spatiotemporal.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to {output_path}")


def plot_results(predictions, targets, coords, times, route_name, save_path=None):
    """Create visualization"""
    
    feature_names = ['Temperature (°C)', 'Wind Speed (m/s)', 'Visibility (m)', 
                    'Pressure (hPa)', 'Wind U (m/s)', 'Wind V (m/s)']
    
    n_features = predictions.shape[1]
    fig, axes = plt.subplots(n_features + 1, 1, figsize=(14, 3 * (n_features + 1)))
    
    # Plot each feature
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        # Plot against time
        ax.plot(times, targets[:, i], 'o-', label='Ground Truth', alpha=0.7, markersize=4, linewidth=2)
        ax.plot(times, predictions[:, i], 's-', label='Prediction', alpha=0.7, markersize=4, linewidth=2)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i])**2))
        
        ax.set_ylabel(feature_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (hours)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{feature_name}: MAE={mae:.2f}, RMSE={rmse:.2f}', fontsize=10)
    
    # Plot route path
    ax = axes[-1]
    lats = coords[:, 0]
    lons = coords[:, 1]
    
    scatter = ax.scatter(lons, lats, c=times, 
                        cmap='viridis', s=60, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.plot(lons, lats, 'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end
    ax.plot(lons[0], lats[0], 'go', markersize=15, label='Start', 
           markeredgecolor='black', markeredgewidth=2)
    ax.plot(lons[-1], lats[-1], 'ro', markersize=15, label='End',
           markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    ax.set_title(f'Route Path: {route_name} (colored by time)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Time (hours)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_spatiotemporal.py <route_csv_path> [model_path]")
        print("\nExample:")
        print("  python predict_spatiotemporal.py realistic_flight_route.csv")
        print("  python predict_spatiotemporal.py route1_subset.csv best_spatiotemporal_model.pt")
        sys.exit(1)
    
    route_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'best_spatiotemporal_model.pt'
    
    print("="*70)
    print("SPATIO-TEMPORAL ROUTE PREDICTION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    if not Path(model_path).exists():
        print(f"❌ Error: Model file not found at {model_path}")
        return
    
    model, scaler = load_model_and_scaler(model_path, device)
    
    # Config
    from config_multi_station import DATA_CONFIG, TRAINING_CONFIG
    config = {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'training': TRAINING_CONFIG
    }
    
    # Load route
    route_name = Path(route_path).stem
    print(f"\nProcessing: {route_name}")
    print("-" * 70)
    
    route_ds = prepare_route_dataset(route_path, scaler, config)
    if route_ds is None:
        print(f"❌ Failed to load route")
        return
    
    route_loader = DataLoader(route_ds, batch_size=16, collate_fn=collate_fn)
    
    # Predict
    print(f"Making predictions...")
    predictions, targets, coords, times = predict_route(model, route_loader, device, scaler)
    
    if predictions is not None:
        print(f"✓ Collected {len(predictions)} predictions")
        
        # Statistics
        print_statistics(predictions, targets, route_name)
        
        # Save
        save_predictions_csv(predictions, targets, coords, times, route_name)
        
        # Plot
        print(f"\nGenerating visualization...")
        plot_path = f'predictions_{route_name}_spatiotemporal.png'
        plot_results(predictions, targets, coords, times, route_name, plot_path)
    else:
        print(f"❌ No valid predictions")
    
    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
