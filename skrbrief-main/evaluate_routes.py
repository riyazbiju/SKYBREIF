"""
Evaluate Spatio-Temporal Model on Route Subsets
================================================
Compute MAE for each variable on route1 and route2
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from spatiotemporal_latent_ode import (
    SpatioTemporalLatentODE,
    prepare_route_dataset,
    collate_fn
)
from config_multi_station import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG


def load_model(model_path, device):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = SpatioTemporalLatentODE(config=MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint.get('scaler')
    epoch = checkpoint.get('epoch', '?')
    
    print(f"✓ Model loaded (Epoch {epoch})")
    
    return model, scaler


def evaluate_route(model, loader, device, scaler, route_name):
    """Evaluate model on route and compute MAE per variable"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
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
                        
            except Exception as e:
                print(f"Warning: Batch error - {e}")
                continue
    
    if len(all_predictions) > 0:
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        return predictions, targets
    else:
        return None, None


def compute_metrics(predictions, targets):
    """Compute MAE, RMSE, MAPE for each variable"""
    feature_names = ['Temperature', 'Wind Speed', 'Visibility', 
                    'Pressure', 'Wind U', 'Wind V']
    units = ['°C', 'm/s', 'm', 'hPa', 'm/s', 'm/s']
    
    metrics = {}
    
    for i, (name, unit) in enumerate(zip(feature_names, units)):
        pred = predictions[:, i]
        gt = targets[:, i]
        
        mae = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(np.mean((pred - gt)**2))
        mape = np.mean(np.abs((pred - gt) / (np.abs(gt) + 1e-8))) * 100
        
        metrics[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'unit': unit,
            'gt_min': gt.min(),
            'gt_max': gt.max(),
            'pred_min': pred.min(),
            'pred_max': pred.max()
        }
    
    return metrics


def print_route_results(route_name, metrics, n_predictions):
    """Print formatted results for a route"""
    print("\n" + "="*80)
    print(f"ROUTE: {route_name.upper()}")
    print(f"Predictions: {n_predictions}")
    print("="*80)
    
    # Print table header
    print(f"\n{'Variable':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12} {'Range (GT)':<20}")
    print("-" * 80)
    
    for var_name, m in metrics.items():
        print(f"{var_name:<20} "
              f"{m['MAE']:<12.3f} "
              f"{m['RMSE']:<12.3f} "
              f"{m['MAPE']:<12.2f}% "
              f"[{m['gt_min']:.1f}, {m['gt_max']:.1f}]")


def save_summary_csv(results, output_path='route_evaluation_summary.csv'):
    """Save evaluation summary to CSV"""
    rows = []
    
    for route_name, data in results.items():
        metrics = data['metrics']
        n_pred = data['n_predictions']
        
        for var_name, m in metrics.items():
            rows.append({
                'route': route_name,
                'variable': var_name,
                'unit': m['unit'],
                'MAE': m['MAE'],
                'RMSE': m['RMSE'],
                'MAPE': m['MAPE'],
                'n_predictions': n_pred,
                'gt_min': m['gt_min'],
                'gt_max': m['gt_max'],
                'pred_min': m['pred_min'],
                'pred_max': m['pred_max']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Summary saved to {output_path}")


def main():
    print("="*80)
    print("SPATIO-TEMPORAL MODEL EVALUATION - ROUTE SUBSETS")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model_path = 'best_spatiotemporal_model.pt'
    if not Path(model_path).exists():
        print(f"\n❌ Error: Model file not found at {model_path}")
        print("Please train the model first using: python train_spatiotemporal.py")
        return
    
    model, scaler = load_model(model_path, device)
    
    # Config
    config = {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'training': TRAINING_CONFIG
    }
    
    # Routes to evaluate
    routes = {
        'route1': 'route1_subset.csv',
        'route2': 'route2_subset.csv'
    }
    
    results = {}
    
    # Evaluate each route
    for route_name, route_path in routes.items():
        print(f"\n{'='*80}")
        print(f"Evaluating {route_name.upper()}: {route_path}")
        print('='*80)
        
        if not Path(route_path).exists():
            print(f"❌ Route file not found: {route_path}")
            continue
        
        # Load route dataset
        route_ds = prepare_route_dataset(route_path, scaler, config)
        if route_ds is None:
            print(f"❌ Failed to load {route_name}")
            continue
        
        route_loader = DataLoader(route_ds, batch_size=16, collate_fn=collate_fn)
        
        # Make predictions
        print(f"Making predictions...")
        predictions, targets = evaluate_route(model, route_loader, device, scaler, route_name)
        
        if predictions is not None:
            n_pred = len(predictions)
            print(f"✓ Collected {n_pred} predictions")
            
            # Compute metrics
            metrics = compute_metrics(predictions, targets)
            
            # Store results
            results[route_name] = {
                'metrics': metrics,
                'n_predictions': n_pred,
                'predictions': predictions,
                'targets': targets
            }
            
            # Print results
            print_route_results(route_name, metrics, n_pred)
        else:
            print(f"❌ No valid predictions for {route_name}")
    
    # Print comparison summary
    if len(results) == 2:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY - MAE")
        print("="*80)
        print(f"\n{'Variable':<20} {'Route1 MAE':<15} {'Route2 MAE':<15} {'Difference':<15}")
        print("-" * 80)
        
        for var_name in ['Temperature', 'Wind Speed', 'Visibility', 'Pressure', 'Wind U', 'Wind V']:
            mae1 = results['route1']['metrics'][var_name]['MAE']
            mae2 = results['route2']['metrics'][var_name]['MAE']
            diff = mae2 - mae1
            print(f"{var_name:<20} {mae1:<15.3f} {mae2:<15.3f} {diff:<+15.3f}")
    
    # Save summary
    save_summary_csv(results)
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
