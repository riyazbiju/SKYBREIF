# Spatial-Temporal Aggregation Upgrade

## Overview
Major architectural upgrade implementing:
1. **K-nearest station spatial aggregation** with distance-weighted blending
2. **True temporal ODE integration** through real time (not dummy 0→1)
3. **Variable temporal queries** - can predict at any future time

---

## Key Changes

### 1. ODE Function - TRUE Temporal Evolution

**Before:**
```python
# Dummy time integration 0→1
t_span = torch.tensor([0.0, 1.0])
# Spacetime was frozen constant
```

**After:**
```python
# Real time integration from observation time to query time
t_span = torch.stack([start_time, query_time], dim=0)  # actual hours

# ODE evolves through real time
def forward(self, t, z):
    # t is REAL normalized time (hours)
    # Spatial position (lat, lon, alt) stays constant
    # Time advances from t_obs to t_query
    time_features = self.time_encoder(t)  # encodes current time
    dz/dt = f(z, spatial_context, time_features)
```

**Impact:**
- Model now simulates atmospheric evolution through actual time
- Can predict 1 hour, 3 hours, or 24 hours ahead by changing query time
- ODE learns temporal dynamics, not just latent transformation

---

### 2. Spatial Aggregation - K-Nearest Stations

**Before:**
```python
# Single station per training example
for station, group in df.groupby('station'):
    # Use only THIS station's data
```

**After:**
```python
# K=10 nearest stations with distance weighting
def forward(self, obs_seq, obs_spacetime, query_spacetime):
    # obs_seq: [batch, K, seq_len, features]  ← K stations
    
    # Encode each station
    for k in range(K):
        z_k = encoder(obs_seq[:, k])
    
    # Distance-weighted aggregation
    distances = haversine_distance(query_location, station_locations)
    weights = softmax(-distances / spatial_scale)  # closer = higher weight
    
    z0 = weighted_sum(z_stations, weights)  # blend latent states
```

**Impact:**
- Uses observations from 10 nearest stations
- Closer stations have higher influence (via softmax weighting)
- More robust predictions using spatial context
- Addresses the "231 isolated silos" problem

---

### 3. Dataset - Multi-Station Training Examples

**Before:**
```python
# Each example = one station sequence
self.sequences = [station1_seq, station2_seq, ...]
```

**After:**
```python
# Each example = query point + K nearest station sequences
self.examples = [(target_station, query_time_idx), ...]

# Pre-compute K-nearest neighbors for each station
self.knn_cache[station] = find_k_nearest(station)

# Returns K sequences per example
obs_seqs = [station1_obs, station2_obs, ..., station10_obs]
```

**Impact:**
- Training sees spatial relationships between stations
- Each query learns from multiple nearby observations
- Dataset automatically finds K=10 nearest stations

---

### 4. Configuration Parameters NOW USED

```python
MODEL_CONFIG = {
    'K_neighbors': 10,        # ✅ NOW USED - number of stations to aggregate
    'spatial_scale': 100.0,   # ✅ NOW USED - distance weighting (km)
    'altitude_scale': 3.0,    # ❌ Still unused (conflict with DATA_CONFIG)
    'spatial_emb_dim': 16,    # ❌ Hardcoded in decoder
}
```

---

## Architecture Diagram

### Old Architecture
```
Single Station:
  CYYC obs → encode → z₀ → [ODE dummy t:0→1] → z₁ → decode → prediction
  
Problems:
  - 231 stations isolated during training
  - No spatial context from nearby stations
  - Fixed 1-hour prediction horizon
  - Dummy time integration
```

### New Architecture
```
K-Nearest Stations:
  Query: 51.0°N, -115.0°W, time=15:00
         ↓
  Find K=10 nearest stations
         ↓
  ┌─────────────────────────────────────────────┐
  │ CYYC: encode → z₁  (distance=0km, weight=0.5)   │
  │ CYYZ: encode → z₂  (distance=50km, weight=0.2)  │
  │ CYWG: encode → z₃  (distance=80km, weight=0.1)  │
  │ ...  (7 more stations)                      │
  └─────────────────────────────────────────────┘
         ↓
  z₀ = Σ(weights × z_stations)  ← distance-weighted blend
         ↓
  Set ODE spatial context: (51.0°N, -115.0°W, altitude)
         ↓
  Integrate through REAL time: t_obs=14:00 → t_query=15:00
         ↓
  z_final = ODE(z₀, t_obs → t_query)
         ↓
  decode → prediction at 15:00
```

---

## Training Changes

### Before:
```python
# Model sees: station sequences independently
# Learns: how weather evolves at EACH station
# Problem: no spatial generalization
```

### After:
```python
# Model sees: query point + 10 nearby station observations
# Learns: 
#   1. How to spatially interpolate between stations (via weighting)
#   2. How weather evolves through real time (via ODE)
#   3. Combined spatio-temporal dynamics
```

---

## Inference Changes

### Before:
```python
# Must provide EXACTLY one observation sequence
# Can only predict 1 hour ahead (hardcoded)
```

### After:
```python
# Provide query location and time
# Model finds K nearest stations from training data
# Aggregates their observations with distance weighting
# Integrates ODE through actual time gap
# Can query ANY future time (1hr, 3hr, 24hr ahead)
```

---

## Expected Improvements

1. **Better Spatial Predictions**
   - Predictions at new locations benefit from nearby station data
   - Smoother spatial interpolation
   - Handles station gaps better

2. **Variable Time Horizons**
   - Can predict 3pm, 6pm, tomorrow, etc.
   - Not limited to 1-hour ahead
   - ODE learns temporal evolution rate

3. **More Robust**
   - If one station has bad data, 9 others compensate
   - Distance weighting reduces noise impact

4. **Physically Meaningful**
   - ODE integrates through real time
   - Spatial weighting reflects atmospheric correlation distance

---

## Backward Compatibility

The model still handles single-station input:
```python
if len(obs_seq.shape) == 3:  # [batch, seq_len, features]
    obs_seq = obs_seq.unsqueeze(1)  # [batch, 1, seq_len, features]
    K = 1  # Single station mode
```

But training should use K=10 for best results.

---

## Next Steps

1. **Retrain model** with new architecture
2. **Test variable horizons**: predict 1hr, 3hr, 6hr, 24hr ahead
3. **Evaluate spatial interpolation**: query locations between stations
4. **Compare metrics** with old single-station approach

---

## Known Limitations

1. **Slower Training**: K=10× more encoding operations per example
2. **Memory Usage**: Storing K sequences per batch
3. **Requires Station Database**: Inference needs access to all training stations
4. **Time Alignment**: All K stations must have observations before query time

---

## Configuration for Best Performance

```python
MODEL_CONFIG = {
    'K_neighbors': 10,       # Use 10 nearest stations
    'spatial_scale': 100.0,  # 100km distance scale for weighting
    'ode_solver': 'rk4',     # Stable solver for temporal integration
    'ode_step_size': 0.1,    # Fine temporal resolution
}

DATA_CONFIG = {
    'prediction_horizon_hours': 1,  # Can now be 1, 3, 6, 24, etc.
    'total_hours': 744,              # Dataset time span for normalization
    'altitude_scale': 40000.0,       # Consistent altitude scaling
}
```
