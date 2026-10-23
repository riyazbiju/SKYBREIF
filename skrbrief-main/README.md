# Flight Route Weather Prediction with a Latent ODE

Predict the weather an aircraft will meet along its route — temperature, wind speed and direction, pressure and visibility — at any latitude, longitude, altitude and time.

The project has two parts:

- **A model (Python / PyTorch).** A *spatiotemporal Latent ODE*: a neural network that learns how weather changes smoothly through space and time, so it can predict at any point, not only where measurements exist.
- **A web app (React).** Plan a flight between two Canadian airports or query a single point, and see the predicted weather on a map and in a side (altitude) view.

The data covers **January 2026** over **Canada and the northern US**.

---

## How it works

```
 Weather observations          Model                          Prediction
 near the query point
                         ┌──────────────────────┐
 10 nearest stations ──▶ │ Encoder              │
 (recent history)        │   ↓ latent state     │
                         │ ODE: evolve the      │ ──▶  temperature, wind speed,
 Query point ──────────▶ │ state through space  │      wind direction, pressure,
 (lat, lon, alt, time)   │ and time             │      visibility
                         │   ↓                  │
                         │ Decoder              │
                         └──────────────────────┘
```

1. For a point you ask about, find the **10 nearest weather stations**.
2. Encode each station's recent observations into a compact *latent state*, weighting closer stations more.
3. Use an **ordinary differential equation (ODE)** to move that state forward to the exact location, altitude and time you asked about.
4. Decode it back into weather values.

More detail: [docs/MODEL_ARCHITECTURE_COMPARISON.md](docs/MODEL_ARCHITECTURE_COMPARISON.md) and [SPATIAL_TEMPORAL_UPGRADE.md](SPATIAL_TEMPORAL_UPGRADE.md).

---

## Quick start

You need **Python 3.10+** and **Node.js 18+**.

### 1. Get the code and install

```bash
git clone https://github.com/hxrshx/skrbrief.git
cd skrbrief

# Python
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Web app
cd visualai
npm install
cd ..
```

### 2. Add the data

The data files are too large for GitHub, so they are not in the repo. Put them in `data/`:

| File | What it is | Needed for |
|------|-----------|-----------|
| `data/jan_sorted_cleaned.csv` | Station observations used for training | Training; best API results |
| `data/route1_subset.csv` | Hourly gridded weather along route 1 | Route predictions; API fallback |
| `data/route2_subset.csv` | Hourly gridded weather along route 2 | Route predictions; API fallback |

If `jan_sorted_cleaned.csv` is missing, the API uses the two route files as its stations instead (see [Known limitations](#known-limitations)).

The trained model, `best_spatiotemporal_model.pt`, **is** included, so you don't need to train anything to try it.

### 3. Run the app

Open two terminals.

**Terminal 1 — the API (port 5001):**
```bash
source .venv/bin/activate
python api_server.py
```
Wait for `Running on http://127.0.0.1:5001`. Loading the data takes about a minute.

**Terminal 2 — the web app (port 8080):**
```bash
cd visualai
npm run dev
```

Open **http://localhost:8080**.

> **Mac users:** the API uses port 5001 because macOS AirPlay already uses port 5000.

---

## Using the web app

**Route analysis** — weather along a flight
1. Enter a start and end airport by ICAO code (e.g. `CYYZ` → `CYBB`). 231 Canadian aerodromes are built in.
2. Pick a date and time (UTC) in January 2026.
3. Set cruise altitude (ft) and cruise speed (knots).
4. See the route from above on a map and from the side as an altitude profile, with predicted conditions along the way.

**Point analysis** — weather at one spot
1. Enter latitude, longitude, altitude (ft), date and time (UTC).
2. See the predicted temperature, wind, pressure and visibility, plus the nearest station.

Valid dates: **2026-01-01 01:00 UTC** to **2026-01-31 23:00 UTC**.

---

## Using the model from the command line

```bash
# Predict along the two test routes; saves CSVs to results/ and plots to visualizations/
python quick_predict.py --route 1
python quick_predict.py --route 2

# Predict along your own route
python quick_predict.py --csv data/your_route.csv --output my_predictions

# Train a new model (requires data/jan_sorted_cleaned.csv)
python train_spatiotemporal.py
```

A route CSV needs these columns:

```
step_order, Date_Time, Latitude, Longitude, Altitude, Pressure, Temperature, Wind_Speed, Wind_Direction, Visiblity
```

(`Visiblity` is spelled this way in the existing data files.)

### API endpoints

You can also call the API directly:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Is the server up, and how many stations are loaded |
| GET | `/info` | Model and dataset details |
| POST | `/predict/point` | Weather at one point |
| POST | `/predict/route` | Weather at up to 100 waypoints |

```bash
curl -X POST http://localhost:5001/predict/point \
  -H "Content-Type: application/json" \
  -d '{"lat": 49.25, "lon": -120.75, "alt": 35000, "datetime": "2026-01-15T12:00:00Z"}'
```

---

## Project layout

```
├── api_server.py                  Flask API the web app talks to
├── spatiotemporal_latent_ode.py   The model and data loading
├── config_multi_station.py        All settings: model size, training, file paths
├── best_spatiotemporal_model.pt   Trained model
│
├── train_spatiotemporal.py        Train the model
├── quick_predict.py               Predict along a route + make plots
├── predict_knearest.py            More detailed K-nearest-station predictions
├── evaluate_routes.py             Score predictions against real data
├── visualize_routes.py            Plotting helpers
├── generate_*.py, create_*.py     Build realistic flight plans to test on
│
├── visualai/                      Web app (React + Vite + Tailwind)
│   └── src/pages/                 Home, Route and Point screens
├── data/                          Input CSVs (not in git — see above)
├── checkpoints/                   Saved models from training, every 5 epochs
├── results/                       Prediction CSVs
├── visualizations/                Generated plots
└── docs/                          Extra guides
```

Older experiments live in `latent_ode_weather_multi_station.py` and `train_latent_ode_multi_station.py`, a grid-based version of the model.

---

## Settings

Everything is in [config_multi_station.py](config_multi_station.py). The ones you're most likely to change:

| Setting | Default | Meaning |
|---------|---------|---------|
| `K_neighbors` | 10 | How many nearby stations feed each prediction |
| `latent_dim` | 32 | Size of the model's internal state |
| `ode_solver` | `rk4` | ODE solver (`dopri5` was unstable on this data) |
| `num_epochs` | 50 | Training length |
| `learning_rate` | 1e-3 | Training step size |

---

## Known limitations

- **Only January 2026.** The model can't predict outside the dates it was trained on.
- **High altitudes are unreliable.** Above about 30,000 ft, predictions can go out of range, for example negative pressure or visibility above 10 statute miles.
- **The fallback station data is a stand-in.** Without `jan_sorted_cleaned.csv`, the API treats each grid point in the route files as a station. That works for trying the app, but the model was trained on different station data, so expect less accurate results.
- **Route scoring uses few points.** `quick_predict.py` currently treats a whole route file as one station, so it scores only about 20 points per route.

---

## Further reading

- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) — what each plot shows
- [KNEAREST_INFERENCE_GUIDE.md](KNEAREST_INFERENCE_GUIDE.md) — how K-nearest-station prediction works
- [README_VISUALAI.md](README_VISUALAI.md) — more on the web app and API
- [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) — earlier evaluation results
