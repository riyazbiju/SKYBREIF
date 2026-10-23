"""
Training Script
===============
Trains on 'jan_sorted_cleaned.csv'
Evaluates on 'route1_subset.csv' and 'route2_subset.csv'
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from config_multi_station import get_config_dict
from latent_ode_weather_multi_station import (
    MultiStationLatentODE, prepare_data_loaders, prepare_route_dataset, collate_fn
)

def train_one_epoch(model, loader, optimizer, device, scaler, beta=1.0):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    # MAE per feature in original units
    total_mae_per_feature = np.zeros(6)
    
    for batch in loader:
        optimizer.zero_grad()
        
        targets = batch['targets'].to(device)
        queries_list = batch['queries']
        neighbors = batch['neighbors']
        
        # Convert list of query dicts to dict of lists (as model expects)
        query_dict = {
            'lat': [q['lat'] for q in queries_list],
            'lon': [q['lon'] for q in queries_list],
            'altitude_m': [q['altitude_m'] for q in queries_list],
            'time_hours': [q['time_hours'] for q in queries_list]
        }
        
        # Prepare query tensors
        q_spatial = torch.stack([
            torch.tensor([q['lat'], q['lon'], q['altitude_m']]) for q in queries_list
        ]).float().to(device)
        q_time = torch.stack([
            torch.tensor([q['time_hours']]) for q in queries_list
        ]).float().to(device)
        
        mean, logvar, z_mu, z_logvar = model(neighbors, query_dict, q_spatial, q_time)
        
        # Reconstruction Loss: Negative Log-Likelihood (Gaussian)
        squared_error = (mean - targets)**2
        recon_loss = torch.mean(0.5 * logvar + 0.5 * squared_error / torch.exp(logvar))
        
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        
        # Total ELBO loss with beta weighting for KL
        loss = recon_loss + beta * kl_loss
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected, skipping batch")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics in original units
        with torch.no_grad():
            # Inverse transform to original units
            mean_original = scaler.inverse_transform(mean.cpu().numpy())
            targets_original = scaler.inverse_transform(targets.cpu().numpy())
            
            # MAE per feature
            mae_per_feature = np.mean(np.abs(mean_original - targets_original), axis=0)
            total_mae_per_feature += mae_per_feature
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
    n = len(loader)
    avg_mae_per_feature = total_mae_per_feature / n
    return total_loss / n, total_recon / n, total_kl / n, avg_mae_per_feature

def evaluate_route(model, loader, device, scaler, route_name):
    model.eval()
    total_recon = 0
    total_mae_per_feature = np.zeros(6)
    count = 0
    predictions = []
    
    print(f"Evaluating Route: {route_name}...")
    
    with torch.no_grad():
        for batch in loader:
            targets = batch['targets'].to(device)
            queries_list = batch['queries']
            neighbors = batch['neighbors']
            
            # Convert list of query dicts to dict of lists
            query_dict = {
                'lat': [q['lat'] for q in queries_list],
                'lon': [q['lon'] for q in queries_list],
                'altitude_m': [q['altitude_m'] for q in queries_list],
                'time_hours': [q['time_hours'] for q in queries_list]
            }
            
            q_spatial = torch.stack([
                torch.tensor([q['lat'], q['lon'], q['altitude_m']]) for q in queries_list
            ]).float().to(device)
            q_time = torch.stack([
                torch.tensor([q['time_hours']]) for q in queries_list
            ]).float().to(device)
            
            try:
                mean, logvar, _, _ = model(neighbors, query_dict, q_spatial, q_time)
                
                # NLL Reconstruction loss
                squared_error = (mean - targets)**2
                recon_loss = torch.mean(0.5 * logvar + 0.5 * squared_error / torch.exp(logvar))
                
                if not torch.isnan(recon_loss) and not torch.isinf(recon_loss):
                    # Inverse transform to original units
                    mean_original = scaler.inverse_transform(mean.cpu().numpy())
                    targets_original = scaler.inverse_transform(targets.cpu().numpy())
                    
                    # MAE per feature in original units
                    mae_per_feature = np.mean(np.abs(mean_original - targets_original), axis=0)
                    
                    total_recon += recon_loss.item()
                    total_mae_per_feature += mae_per_feature
                    count += 1
                    predictions.append(mean_original)
                else:
                    print(f"Warning: NaN/Inf in batch, skipping...")
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
    
    if count > 0:
        avg_recon = total_recon / count
        avg_mae_per_feature = total_mae_per_feature / count
        feature_names = ['Temp(°C)', 'WindSpd(m/s)', 'Vis(m)', 'Press(hPa)', 'Wind_U(m/s)', 'Wind_V(m/s)']
        
        print(f"Route {route_name} | NLL: {avg_recon:.6f}")
        print(f"  MAE: ", end="")
        for name, mae_val in zip(feature_names, avg_mae_per_feature):
            print(f"{name}={mae_val:.3f}  ", end="")
        print()
        
        return avg_recon, avg_mae_per_feature
    else:
        print(f"Route {route_name} - No valid predictions")
        return float('nan'), np.full(6, float('nan'))

def main():
    config = get_config_dict()
    device = torch.device('cpu') # FORCE CPU
    print(f"Using Device: {device}")
    
    # 1. Prepare Main Data
    train_ds, val_ds, scaler = prepare_data_loaders(config)
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], collate_fn=collate_fn)
    
    # 2. Prepare Route Data for Evaluation
    route1_ds = prepare_route_dataset(config['data']['route1_path'], scaler, config)
    route2_ds = prepare_route_dataset(config['data']['route2_path'], scaler, config)
    
    route1_loader = DataLoader(route1_ds, batch_size=1, collate_fn=collate_fn) if route1_ds else None
    route2_loader = DataLoader(route2_ds, batch_size=1, collate_fn=collate_fn) if route2_ds else None

    # 3. Model
    model = MultiStationLatentODE(config['model']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 4. KL Annealing Schedule
    kl_warmup = config['training']['kl_warmup_epochs']
    kl_anneal = config['training']['kl_annealing_epochs']
    max_beta = config['training']['max_beta']
    
    # 5. Training Loop
    feature_names = ['Temp(°C)', 'WindSpd(m/s)', 'Vis(m)', 'Press(hPa)', 'Wind_U(m/s)', 'Wind_V(m/s)']
    
    for epoch in range(config['training']['num_epochs']):
        # Calculate beta for KL annealing
        if epoch < kl_warmup:
            beta = 0.0
        elif epoch < kl_warmup + kl_anneal:
            beta = max_beta * (epoch - kl_warmup) / kl_anneal
        else:
            beta = max_beta
        
        train_loss, recon_loss, kl_loss, mae_per_feature = train_one_epoch(
            model, train_loader, optimizer, device, scaler, beta
        )
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} | "
              f"ELBO: {train_loss:.4f} | NLL: {recon_loss:.4f} | KL: {kl_loss:.4f} | Beta: {beta:.3f}")
        print(f"  Train MAE: ", end="")
        for name, mae_val in zip(feature_names, mae_per_feature):
            print(f"{name}={mae_val:.3f}  ", end="")
        print()
        
        # Validate periodically
        if (epoch+1) % 5 == 0:
            if route1_loader: evaluate_route(model, route1_loader, device, scaler, "Route 1")
            if route2_loader: evaluate_route(model, route2_loader, device, scaler, "Route 2")

    print("Training Complete.")
    torch.save(model.state_dict(), "final_model.pt")

if __name__ == "__main__":
    main()