"""
Training Script for Spatio-Temporal Latent ODE
===============================================
Train model that evolves through 4D space-time
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path

from spatiotemporal_latent_ode import (
    SpatioTemporalLatentODE,
    prepare_data_loaders
)
from config_multi_station import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG


def kl_divergence(mu, logvar, free_bits=0.5):
    """
    KL divergence between N(mu, var) and N(0, 1) with free bits
    Free bits prevents KL collapse by setting a minimum KL per dimension
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Apply free bits threshold
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=-1).mean()


def train_epoch(model, loader, optimizer, device, beta=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        obs_seq = batch['obs_seq'].to(device)
        obs_spacetime = batch['obs_spacetime'].to(device)
        query_spacetime = batch['query_spacetime'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred, mu, logvar, _ = model(obs_seq, obs_spacetime, query_spacetime, integrate_ode=True)
        
        # Loss
        recon_loss = nn.MSELoss()(pred, target)
        kl_loss = kl_divergence(mu, logvar)
        loss = recon_loss + beta * kl_loss
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


def validate(model, loader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_recon = 0
    
    with torch.no_grad():
        for batch in loader:
            obs_seq = batch['obs_seq'].to(device)
            obs_spacetime = batch['obs_spacetime'].to(device)
            query_spacetime = batch['query_spacetime'].to(device)
            target = batch['target'].to(device)
            
            pred, mu, logvar, _ = model(obs_seq, obs_spacetime, query_spacetime, integrate_ode=True)
            
            recon_loss = nn.MSELoss()(pred, target)
            kl_loss = kl_divergence(mu, logvar)
            loss = recon_loss + kl_loss
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
    
    n = len(loader)
    return total_loss / n, total_recon / n


def main():
    print("="*70)
    print("SPATIO-TEMPORAL LATENT ODE TRAINING")
    print("="*70)
    
    # Device
    device = torch.device(TRAINING_CONFIG['device'])
    print(f"Device: {device}")
    
    # Config
    config = {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'training': TRAINING_CONFIG
    }
    
    # Data
    print("\nPreparing data...")
    train_loader, val_loader, scaler, dataset_start_time = prepare_data_loaders(config)
    print(f"Dataset starts at time: {dataset_start_time:.1f} hours")
    
    # Model
    print("\nInitializing model...")
    model = SpatioTemporalLatentODE(config=MODEL_CONFIG).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=1e-5
    )
    
    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training
    print("\nStarting training...")
    print("="*70)
    
    best_val_loss = float('inf')
    epochs = TRAINING_CONFIG['num_epochs']
    
    # Beta schedule for KL term - start higher to prevent collapse
    # Ramp from 0.5 to 1.0 over first 10 epochs, then stay at 1.0
    warmup_epochs = 10
    beta_schedule = np.linspace(0.5, 1.0, warmup_epochs)
    beta_schedule = np.concatenate([beta_schedule, np.ones(epochs - warmup_epochs)])
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)
        
        beta = beta_schedule[epoch]
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, beta=beta
        )
        
        # Validate
        val_loss, val_recon = validate(model, val_loader, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}")
        print(f"  Beta: {beta:.3f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'dataset_start_time': dataset_start_time,
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, 'best_spatiotemporal_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'dataset_start_time': dataset_start_time,
                'val_loss': val_loss,
                'config': config
            }
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch+1}.pt')
            print(f"  ✓ Saved checkpoint")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
