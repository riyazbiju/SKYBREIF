"""
Spatio-Temporal Meteorological Latent ODE Model
================================================
4D model that evolves through space (lat, lon, alt) and time
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler


# ============================================================================
# UTILS
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in km"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# ============================================================================
# SPATIO-TEMPORAL ODE
# ============================================================================

class SpatioTemporalODEFunc(nn.Module):
    """
    ODE function that models TRUE evolution through time
    dz/dt = f(z, t) where t is REAL time (hours)
    Spacetime position (lat, lon, alt) stays constant, but time advances
    """
    def __init__(self, latent_dim: int = 32, spacetime_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encode spatial position (lat, lon, alt) - STATIC
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh()
        )
        
        # Encode current time - DYNAMIC
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16)
        )
        
        # Main ODE dynamics: evolves latent state through time
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 32 + 16, 64),  # z + spatial + temporal
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )
        
        # Store spatial context (set during forward pass)
        self.spatial_context = None
        
    def set_spatial_context(self, lat, lon, alt):
        """Set the spatial position (stays constant during integration)"""
        spatial_coords = torch.cat([lat, lon, alt], dim=-1)  # [batch, 3]
        self.spatial_context = self.spatial_encoder(spatial_coords)
        
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Current REAL time (scalar, in normalized hours)
            z: [batch, latent_dim] - latent state
        Returns:
            dz/dt: [batch, latent_dim]
        """
        batch_size = z.shape[0]
        
        # Encode current time (t is real time, advances during integration)
        t_expanded = t.expand(batch_size, 1)
        time_features = self.time_encoder(t_expanded)
        
        # Combine: latent + spatial (fixed) + temporal (dynamic)
        combined = torch.cat([z, self.spatial_context, time_features], dim=-1)
        dz_dt = self.dynamics(combined)
        
        return dz_dt


# ============================================================================
# ENCODERS
# ============================================================================

class SpatioTemporalEncoder(nn.Module):
    """Encodes observation sequences in 4D space-time to latent state"""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encode 4D spacetime
        self.spacetime_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32)
        )
        
        # GRU to process temporal sequences
        self.gru = nn.GRU(input_dim + 32, hidden_dim, batch_first=True)
        
        # Map to latent distribution
        self.h_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.h_to_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, obs: torch.Tensor, spacetime: torch.Tensor):
        """
        Args:
            obs: [batch, seq_len, input_dim] - weather observations
            spacetime: [batch, seq_len, 4] - (lat, lon, alt, time) for each obs
        Returns:
            z, mu, logvar
        """
        batch_size, seq_len, _ = obs.shape
        
        # Encode spacetime coordinates
        spacetime_emb = self.spacetime_encoder(spacetime)
        
        # Concatenate observations with spacetime embeddings
        x = torch.cat([obs, spacetime_emb], dim=-1)
        
        # Process sequence with GRU (reverse order for better encoding)
        x_reversed = torch.flip(x, dims=[1])
        _, h = self.gru(x_reversed)
        h = h.squeeze(0)
        
        # Get latent distribution
        mu = self.h_to_mu(h)
        logvar = self.h_to_logvar(h)
        logvar = torch.clamp(logvar, -10, 2)  # Prevent numerical issues
        
        # Sample
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar


class SpatioTemporalDecoder(nn.Module):
    """Decodes latent state at query spacetime location"""
    def __init__(self, latent_dim: int = 32, output_dim: int = 6):
        super().__init__()
        
        # Encode query spacetime with less capacity - force reliance on latent
        self.spacetime_encoder = nn.Sequential(
            nn.Linear(4, 16),  # Reduced from 32
            nn.Tanh()
        )
        
        # Decode to weather - latent should carry most information
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 16, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, z: torch.Tensor, query_spacetime: torch.Tensor):
        """
        Args:
            z: [batch, latent_dim] - latent state
            query_spacetime: [batch, 4] - (lat, lon, alt, time)
        Returns:
            prediction: [batch, output_dim]
        """
        spacetime_emb = self.spacetime_encoder(query_spacetime)
        combined = torch.cat([z, spacetime_emb], dim=-1)
        return self.decoder(combined)


# ============================================================================
# MAIN MODEL
# ============================================================================

class SpatioTemporalLatentODE(nn.Module):
    """Main spatio-temporal latent ODE model with K-nearest spatial aggregation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['input_dim']
        output_dim = config['output_dim']
        latent_dim = config['latent_dim']
        hidden_dim = config['hidden_dim']
        
        # Components
        self.encoder = SpatioTemporalEncoder(input_dim, hidden_dim, latent_dim)
        self.ode_func = SpatioTemporalODEFunc(latent_dim, spacetime_dim=4)
        self.decoder = SpatioTemporalDecoder(latent_dim, output_dim)
        
        # Spatial aggregation settings
        self.K_neighbors = config.get('K_neighbors', 10)
        self.spatial_scale = config.get('spatial_scale', 100.0)  # km
        
        # ODE solver settings
        self.ode_solver = config.get('ode_solver', 'rk4')
        self.ode_step_size = config.get('ode_step_size', 0.1)
        
    def spatial_attention_weights(self, query_lat, query_lon, station_lats, station_lons):
        """
        Compute distance-based attention weights for K-nearest stations
        Args:
            query_lat, query_lon: [batch] query location
            station_lats, station_lons: [batch, K] K nearest station locations
        Returns:
            weights: [batch, K] normalized attention weights
        """
        batch_size = query_lat.shape[0]
        K = station_lats.shape[1]
        
        # Compute distances for all K stations
        distances = torch.zeros(batch_size, K, device=query_lat.device)
        for k in range(K):
            # Haversine distance in km
            query_lat_np = query_lat.cpu().numpy()
            query_lon_np = query_lon.cpu().numpy()
            station_lat_np = station_lats[:, k].cpu().numpy()
            station_lon_np = station_lons[:, k].cpu().numpy()
            
            dist_km = haversine_distance(query_lat_np, query_lon_np, 
                                        station_lat_np, station_lon_np)
            distances[:, k] = torch.tensor(dist_km, device=query_lat.device)
        
        # Softmax with distance scaling
        weights = torch.softmax(-distances / self.spatial_scale, dim=-1)
        return weights
        
    def forward(self, obs_seq, obs_spacetime, query_spacetime, integrate_ode=True):
        """
        Args:
            obs_seq: [batch, K, seq_len, input_dim] - K station observation sequences
            obs_spacetime: [batch, K, seq_len, 4] - spacetime coords for K stations
            query_spacetime: [batch, 4] - query spacetime point (lat, lon, alt, time)
            integrate_ode: whether to integrate ODE through time
        Returns:
            prediction, mu, logvar, z_final
        """
        batch_size = obs_seq.shape[0]
        K = obs_seq.shape[1] if len(obs_seq.shape) == 4 else 1
        
        # Handle single station case (backward compatibility)
        if len(obs_seq.shape) == 3:
            obs_seq = obs_seq.unsqueeze(1)  # [batch, 1, seq_len, features]
            obs_spacetime = obs_spacetime.unsqueeze(1)
            K = 1
        
        # Encode each station's observations
        z_stations = []
        mu_stations = []
        logvar_stations = []
        
        for k in range(K):
            z_k, mu_k, logvar_k = self.encoder(obs_seq[:, k], obs_spacetime[:, k])
            z_stations.append(z_k)
            mu_stations.append(mu_k)
            logvar_stations.append(logvar_k)
        
        # Stack: [batch, K, latent_dim]
        z_stations = torch.stack(z_stations, dim=1)
        mu_stations = torch.stack(mu_stations, dim=1)
        logvar_stations = torch.stack(logvar_stations, dim=1)
        
        # Compute spatial attention weights based on distance to query
        if K > 1:
            query_lat = query_spacetime[:, 0:1]  # [batch, 1]
            query_lon = query_spacetime[:, 1:2]
            station_lats = obs_spacetime[:, :, -1, 0]  # [batch, K] last obs lat
            station_lons = obs_spacetime[:, :, -1, 1]  # [batch, K] last obs lon
            
            weights = self.spatial_attention_weights(
                query_lat.squeeze(1), query_lon.squeeze(1),
                station_lats, station_lons
            )  # [batch, K]
            
            # Weighted aggregation of latent states
            weights_expanded = weights.unsqueeze(-1)  # [batch, K, 1]
            z0 = (z_stations * weights_expanded).sum(dim=1)  # [batch, latent_dim]
            mu = (mu_stations * weights_expanded).sum(dim=1)
            logvar = (logvar_stations * weights_expanded).sum(dim=1)
        else:
            # Single station
            z0 = z_stations[:, 0]
            mu = mu_stations[:, 0]
            logvar = logvar_stations[:, 0]
        
        if integrate_ode:
            # Get start time from last observation
            start_time = obs_spacetime[:, 0, -1, 3]  # [batch] time of last obs
            query_time = query_spacetime[:, 3]  # [batch] query time
            
            # Set spatial context for ODE (query location - stays constant)
            query_lat = query_spacetime[:, 0:1]
            query_lon = query_spacetime[:, 1:2]
            query_alt = query_spacetime[:, 2:3]
            self.ode_func.set_spatial_context(query_lat, query_lon, query_alt)
            
            # Integrate through REAL time from start_time to query_time
            t_span = torch.stack([start_time, query_time], dim=0)  # [2, batch]
            
            # Handle case where all samples have same time span
            if torch.allclose(t_span[0], t_span[0, 0]) and torch.allclose(t_span[1], t_span[1, 0]):
                t_span_scalar = torch.tensor([t_span[0, 0].item(), t_span[1, 0].item()], 
                                            device=z0.device)
                try:
                    if self.ode_solver == 'rk4':
                        # Dynamic step: always 10 steps regardless of horizon length
                        # Fixes the bug where normalized time gaps (e.g. 1hr/744 = 0.00134)
                        # were far smaller than the fixed step_size=0.1, causing 1 useless step
                        n_steps = self.config.get('ode_n_steps', 10)
                        time_gap = t_span_scalar[1].item() - t_span_scalar[0].item()
                        dynamic_step = max(time_gap / n_steps, 1e-5)
                        
                        z_trajectory = odeint(
                            self.ode_func, 
                            z0, 
                            t_span_scalar,
                            method='rk4',
                            options={'step_size': dynamic_step}
                        )
                    else:
                        z_trajectory = odeint(
                            self.ode_func,
                            z0,
                            t_span_scalar,
                            method=self.ode_solver
                        )
                    z_final = z_trajectory[-1]  # [batch, latent_dim]
                except Exception as e:
                    print(f"Warning: ODE integration failed: {e}, using z0")
                    z_final = z0
            else:
                # Different time spans per sample - use loop (slow but correct)
                z_final = []
                for i in range(batch_size):
                    t_i = torch.tensor([start_time[i].item(), query_time[i].item()], 
                                      device=z0.device)
                    try:
                        n_steps = self.config.get('ode_n_steps', 10)
                        time_gap_i = t_i[1].item() - t_i[0].item()
                        dynamic_step_i = max(time_gap_i / n_steps, 1e-5)
                        z_traj_i = odeint(self.ode_func, z0[i:i+1], t_i,
                            method='rk4',options={'step_size': dynamic_step_i})
                        z_final.append(z_traj_i[-1])
                    except Exception as e:
                        z_final.append(z0[i:i+1])
                z_final = torch.cat(z_final, dim=0)
        
        # Decode at query spacetime location
        prediction = self.decoder(z_final, query_spacetime)
        
        return prediction, mu, logvar, z_final


# ============================================================================
# DATASET
# ============================================================================

class SpatioTemporalWeatherDataset(Dataset):
    """Dataset that provides K-nearest station sequences for spatial aggregation"""
    
    def __init__(self, df, config, scaler, mode='train', dataset_start_time=None):
        self.df = df
        self.config = config
        self.scaler = scaler
        self.mode = mode
        self.K = config.get('K_neighbors', 10)
        
        # Store dataset start time for absolute time integration
        if dataset_start_time is None:
            self.dataset_start_time = df['time_hours'].min()
        else:
            self.dataset_start_time = dataset_start_time
        
        # Build station database: location + time series for each station
        self.station_data = {}
        for station, group in df.groupby('station'):
            group = group.sort_values('datetime')
            if len(group) >= config['data']['min_seq_length']:
                # Store station location (use first observation)
                lat = group['latitude'].iloc[0]
                lon = group['longitude'].iloc[0]
                self.station_data[station] = {
                    'lat': lat,
                    'lon': lon,
                    'data': group
                }
        
        self.station_list = list(self.station_data.keys())
        print(f"Loaded {len(self.station_list)} stations for {mode} dataset")
        print(f"Dataset time range: {self.dataset_start_time:.1f} to {df['time_hours'].max():.1f} hours")
        
        # Create training examples
        # Each example: pick a query point from one station, find K nearest, use their data
        self.examples = []
        
        max_seq = config['data']['max_seq_length']
        min_seq = config['data']['min_seq_length']
        
        # Variable prediction horizons for training diversity
        if mode == 'train':
            horizons = [1, 3, 6, 12, 24]  # hours ahead to predict
        else:
            horizons = [1]  # validation uses 1hr for consistency
        
        print(f"Creating examples with horizons: {horizons}...")
        
        for target_station in self.station_list:
            target_data = self.station_data[target_station]['data']
            max_time = target_data['time_hours'].max()
            
            # Create queries from this station's time series
            if mode in ['test', 'val']:
                # For test/val: predict at regular intervals
                step = max(len(target_data) // 20, 1)  # ~20 queries per station
                for i in range(min_seq, len(target_data), step):
                    horizon = 1  # Fixed 1hr for validation
                    self.examples.append((target_station, i, horizon))
            else:
                # For training: sample query points, only use horizons where future exists
                step = max_seq // 4
                for i in range(min_seq, len(target_data) - max_seq, step):
                    obs_time = target_data.iloc[i]['time_hours']
                    # Randomly pick one horizon that has future data
                    available_horizons = [h for h in horizons if obs_time + h <= max_time]
                    if available_horizons:
                        horizon = np.random.choice(available_horizons)
                        self.examples.append((target_station, i, horizon))
        
        # Subsample for training efficiency
        if mode == 'train' and len(self.examples) > 3000:
            indices = np.random.choice(len(self.examples), 3000, replace=False)
            self.examples = [self.examples[i] for i in indices]
        
        print(f"Created {len(self.examples)} training examples")
        
        # Pre-compute K-nearest neighbors for each station
        print(f"Pre-computing K={self.K} nearest neighbors for {len(self.station_list)} stations...")
        self.knn_cache = {}
        for station in self.station_list:
            self.knn_cache[station] = self._find_k_nearest(station)
    
    def _find_k_nearest(self, target_station):
        """Find K nearest stations to target station"""
        target_lat = self.station_data[target_station]['lat']
        target_lon = self.station_data[target_station]['lon']
        
        distances = []
        for other_station in self.station_list:
            if other_station == target_station:
                continue
            other_lat = self.station_data[other_station]['lat']
            other_lon = self.station_data[other_station]['lon']
            dist = haversine_distance(target_lat, target_lon, other_lat, other_lon)
            distances.append((dist, other_station))
        
        # Sort by distance and take K nearest (excluding self)
        distances.sort()
        k_nearest = [target_station] + [s for _, s in distances[:self.K-1]]
        
        # If we don't have K stations, pad with what we have
        while len(k_nearest) < self.K:
            k_nearest.append(k_nearest[-1])  # Duplicate last station
        
        return k_nearest[:self.K]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        target_station, obs_cutoff_idx, horizon = self.examples[idx]
        
        # Get K nearest stations
        k_stations = self.knn_cache[target_station]
        
        # Get configuration parameters
        total_hrs = self.config['data'].get('total_hours', 744)
        altitude_scale = self.config['data'].get('altitude_scale', 40000.0)
        max_seq = self.config['data']['max_seq_length']
        
        # Get observation point (use obs up to this index)
        target_data = self.station_data[target_station]['data']
        obs_row = target_data.iloc[obs_cutoff_idx]
        obs_time = float(obs_row['time_hours'])
        
        # Query time: predict 'horizon' hours into the future
        query_time = obs_time + horizon
        
        # Find actual target observation closest to query_time
        target_data = self.station_data[target_station]['data']
        future_obs = target_data[target_data['time_hours'] >= query_time]
        
        if len(future_obs) == 0:
            # No future observation - use last available
            target_row = target_data.iloc[-1]
        else:
            # Use closest observation to query_time
            target_row = future_obs.iloc[0]
        
        feat_cols = ['temperature', 'windspeed', 'visibility', 'pressure', 'wind_u', 'wind_v']
        # Convert to float explicitly to avoid object dtype issues
        target = torch.tensor([float(target_row[col]) for col in feat_cols], dtype=torch.float32)
        
        # Query spacetime: where/when we want prediction (FUTURE time)
        query_spacetime = torch.tensor([
            float(target_row['latitude']),
            float(target_row['longitude']),
            float(target_row['altitude']) / altitude_scale,
            query_time / total_hrs  # Use the actual query time (not target_row time)
        ], dtype=torch.float32)
        
        # Collect observation sequences from K nearest stations
        obs_seqs = []
        obs_spacetimes = []
        
        # Use observations from START of dataset up to obs_cutoff_idx time
        # This allows ODE to integrate from dataset start to query time
        for station in k_stations:
            station_data = self.station_data[station]['data']
            
            # Get observations from START up to obs_time
            past_obs = station_data[station_data['time_hours'] <= obs_time]
            
            if len(past_obs) == 0:
                # No past observations - use earliest available
                past_obs = station_data.iloc[:1]
            
            # Take up to max_seq most recent observations
            if len(past_obs) > max_seq:
                past_obs = past_obs.iloc[-max_seq:]
            
            # Extract features and spacetime
            obs = torch.tensor(past_obs[feat_cols].values, dtype=torch.float32)
            spacetime = torch.tensor(np.column_stack([
                past_obs['latitude'].values,
                past_obs['longitude'].values,
                past_obs['altitude'].values / altitude_scale,
                past_obs['time_hours'].values / total_hrs  # Absolute time from start
            ]), dtype=torch.float32)
            
            obs_seqs.append(obs)
            obs_spacetimes.append(spacetime)
        
        # Get original wind direction if available
        wind_dir_orig = None
        if 'wind_direction_original' in target_row.index:
            wind_dir_orig = target_row['wind_direction_original']
        
        return {
            'obs_seqs': obs_seqs,  # List of K sequences (variable length)
            'obs_spacetimes': obs_spacetimes,  # List of K spacetime sequences
            'query_spacetime': query_spacetime,  # Future time: obs_time + horizon
            'target': target,
            'wind_dir_original': wind_dir_orig,
            'horizon': horizon  # Store horizon for analysis
        }


def collate_fn(batch):
    """
    Collate function for DataLoader - handles K variable-length sequences per example
    """
    batch_size = len(batch)
    K = len(batch[0]['obs_seqs'])
    
    # Find max sequence length across all K stations in batch
    max_len = 0
    for b in batch:
        for k in range(K):
            max_len = max(max_len, b['obs_seqs'][k].shape[0])
    
    # Pad all sequences to max_len
    obs_seqs_batch = []
    obs_spacetimes_batch = []
    query_spacetimes = []
    targets = []
    wind_dirs = []
    horizons = []
    
    for b in batch:
        # Collect K sequences for this example
        obs_k_padded = []
        spacetime_k_padded = []
        
        for k in range(K):
            obs_k = b['obs_seqs'][k]
            spacetime_k = b['obs_spacetimes'][k]
            seq_len = obs_k.shape[0]
            
            if seq_len < max_len:
                # Pad with zeros
                pad_len = max_len - seq_len
                obs_pad = torch.zeros(pad_len, obs_k.shape[1])
                spacetime_pad = torch.zeros(pad_len, 4)
                
                obs_k_padded.append(torch.cat([obs_k, obs_pad], dim=0))
                spacetime_k_padded.append(torch.cat([spacetime_k, spacetime_pad], dim=0))
            else:
                obs_k_padded.append(obs_k)
                spacetime_k_padded.append(spacetime_k)
        
        # Stack K sequences: [K, max_len, features]
        obs_seqs_batch.append(torch.stack(obs_k_padded))
        obs_spacetimes_batch.append(torch.stack(spacetime_k_padded))
        
        query_spacetimes.append(b['query_spacetime'])
        targets.append(b['target'])
        wind_dirs.append(b['wind_dir_original'])
        horizons.append(b['horizon'])
    
    return {
        'obs_seq': torch.stack(obs_seqs_batch),  # [batch, K, max_len, features]
        'obs_spacetime': torch.stack(obs_spacetimes_batch),  # [batch, K, max_len, 4]
        'query_spacetime': torch.stack(query_spacetimes),  # [batch, 4]
        'target': torch.stack(targets),  # [batch, features]
        'wind_dir_original': wind_dirs,
        'horizons': horizons  # List of prediction horizons
    }


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data_loaders(config):
    """Prepare training and validation data loaders"""
    
    csv_path = config['data']['csv_path']
    print(f"Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Data verification
    print("\n=== DATA VERIFICATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    if 'winddirection' in df.columns:
        print(f"Wind direction: min={df['winddirection'].min():.1f}°, max={df['winddirection'].max():.1f}°, mean={df['winddirection'].mean():.1f}°")
    if 'windspeed' in df.columns:
        print(f"Wind speed: min={df['windspeed'].min():.1f} knots, max={df['windspeed'].max():.1f} knots, mean={df['windspeed'].mean():.1f} knots")
    if 'visibility' in df.columns:
        print(f"Visibility: min={df['visibility'].min():.1f} SM, max={df['visibility'].max():.1f} SM, mean={df['visibility'].mean():.1f} SM")
    print("=" * 30 + "\n")
    
    # Ensure datetime column
    if 'datetime' not in df.columns and 'Date_Time' in df.columns:
        df = df.rename(columns={'Date_Time': 'datetime'})
    
    # Convert to datetime - force conversion even if it looks like datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Add time_hours if not present (hours since start)
    if 'time_hours' not in df.columns:
        df = df.sort_values('datetime')
        min_time = df['datetime'].min()
        df['time_hours'] = (df['datetime'] - min_time).dt.total_seconds() / 3600
    
    # Ensure required columns
    required = ['temperature', 'windspeed', 'visibility', 'pressure', 
                'latitude', 'longitude', 'altitude', 'station']
    
    # Convert wind direction to u/v components if missing
    if 'wind_u' not in df.columns or 'wind_v' not in df.columns:
        if 'winddirection' in df.columns and 'windspeed' in df.columns:
            print("Converting wind direction to u/v components...")
            # Store original wind direction BEFORE conversion
            df['wind_direction_original'] = df['winddirection'].copy()
            wind_dir_rad = np.radians(df['winddirection'])
            df['wind_u'] = -df['windspeed'] * np.sin(wind_dir_rad)
            df['wind_v'] = -df['windspeed'] * np.cos(wind_dir_rad)
            print(f"  wind_u: mean={df['wind_u'].mean():.2f}, std={df['wind_u'].std():.2f}")
            print(f"  wind_v: mean={df['wind_v'].mean():.2f}, std={df['wind_v'].std():.2f}")
        else:
            # Only set to zero if no wind info at all
            print("Warning: No wind information found, setting to zero")
            df['wind_u'] = 0.0
            df['wind_v'] = 0.0
    
    # Drop NaN
    df = df.dropna(subset=required + ['wind_u', 'wind_v'])
    
    # Fit scaler
    feat_cols = ['temperature', 'windspeed', 'visibility', 'pressure', 'wind_u', 'wind_v']
    scaler = StandardScaler()
    scaler.fit(df[feat_cols])
    df[feat_cols] = scaler.transform(df[feat_cols])
    
    # Get dataset start time for absolute time integration
    dataset_start_time = df['time_hours'].min()
    
    # Split train/val
    val_split = config['data']['val_split']
    split_idx = int(len(df) * (1 - val_split))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # Create datasets with shared start time
    train_ds = SpatioTemporalWeatherDataset(train_df, config, scaler, mode='train', dataset_start_time=dataset_start_time)
    val_ds = SpatioTemporalWeatherDataset(val_df, config, scaler, mode='val', dataset_start_time=dataset_start_time)
    
    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")
    
    # Return dataset_start_time for inference use
    return train_loader, val_loader, scaler, dataset_start_time


def prepare_route_dataset(csv_path, scaler, config):
    """Prepare route dataset for evaluation"""
    print(f"Loading Route Data: {csv_path}")
    if not Path(csv_path).exists():
        print("Route file not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Drop NaN
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with NaN values ({dropped/initial_rows*100:.2f}%)")
    
    # Map column names
    route_to_expected = {
        'Temperature': 'temperature',
        'Wind_Speed': 'windspeed',
        'Visiblity': 'visibility',
        'Pressure': 'pressure',
        'Date_Time': 'datetime',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Altitude': 'altitude',
        'Wind_Direction': 'wind_direction',
        'step_order': 'step_order'
    }
    df = df.rename(columns=route_to_expected)
    
    # Add time_hours from step_order
    if 'step_order' in df.columns:
        df['time_hours'] = df['step_order'].astype(float)
    else:
        df['time_hours'] = 0.0
    
    # Add station if missing
    if 'station' not in df.columns:
        df['station'] = 'ROUTE'
    
    # Convert wind direction and speed to u/v components if missing
    if 'wind_u' not in df.columns or 'wind_v' not in df.columns:
        if 'wind_direction' in df.columns and 'windspeed' in df.columns:
            # Store original wind direction BEFORE transformation
            df['wind_direction_original'] = df['wind_direction'].copy()
            # Convert meteorological wind direction to u/v components
            # Meteorological direction: direction wind is coming FROM (0° = North, 90° = East)
            # u = -speed * sin(direction), v = -speed * cos(direction)
            wind_dir_rad = np.radians(df['wind_direction'])
            df['wind_u'] = -df['windspeed'] * np.sin(wind_dir_rad)
            df['wind_v'] = -df['windspeed'] * np.cos(wind_dir_rad)
        else:
            # If no wind direction info, set to zero
            df['wind_u'] = 0.0
            df['wind_v'] = 0.0
    
    # Transform features
    feat_cols = ['temperature', 'windspeed', 'visibility', 'pressure', 'wind_u', 'wind_v']
    df[feat_cols] = scaler.transform(df[feat_cols])
    
    return SpatioTemporalWeatherDataset(df, config, scaler, mode='test')
