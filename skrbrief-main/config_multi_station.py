"""
Configuration for Meteorological Latent ODE Model
==================================================
"""

import torch
from pathlib import Path

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    # Main training data
    'csv_path': 'data/jan_sorted_cleaned.csv',
    
    # Validation split from main data (0.0 to 1.0)
    'val_split': 0.1,
    
    # Specific Route Files for Testing/Evaluation
    'route1_path': 'data/route1_subset.csv',
    'route2_path': 'data/route2_subset.csv',
    
    # Sequence creation
    'max_seq_length': 100,
    'min_seq_length': 3,
    
    # Time configuration
    'prediction_horizon_hours': 1,    # How many hours ahead to predict
    'total_hours': 744,                # Total time span of dataset (31 days)
    'altitude_scale': 40000.0,         # Maximum altitude for normalization
}


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

MODEL_CONFIG = {
    'input_dim': 6,          # [temp, windspeed, vis, pressure, wind_u, wind_v]
    'output_dim': 6,
    
    'latent_dim': 32,
    'hidden_dim': 40,
    'spatial_emb_dim': 16,
    
    'K_neighbors': 10,       
    'spatial_scale': 100.0,  
    'altitude_scale': 3.0,   
    'time_scale': 1.0,       
    
    # SOLVER SETTINGS - CHANGED FOR STABILITY
    # 'dopri5' causes underflow on stiff data. 'rk4' is robust.
    'ode_solver': 'rk4', 
    'ode_step_size': 0.1,    # Fixed step size for RK4
}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    'batch_size': 8,
    'num_epochs': 50, # Reduced for testing, increase as needed
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    
    'lr_scheduler': 'step',
    'lr_step_size': 10,
    'lr_gamma': 0.5,
    
    'kl_annealing': True,
    'kl_warmup_epochs': 5,
    'kl_annealing_epochs': 15,
    'max_beta': 0.1,
    
    # DEVICE CONFIGURATION
    # Latent ODEs are often unstable on MPS/CUDA due to precision issues in solvers.
    # CPU is recommended for debugging "underflow" errors.
    'device': 'cpu', 
    
    'checkpoint_dir': 'checkpoints',
    'save_every': 1,
}

PATHS = {
    'data_dir': Path(__file__).parent,
    'checkpoint_dir': Path(__file__).parent / TRAINING_CONFIG['checkpoint_dir'],
    'output_dir': Path(__file__).parent / 'outputs',
}

for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

def get_config_dict():
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'paths': {k: str(v) for k, v in PATHS.items()},
    }