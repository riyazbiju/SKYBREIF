# SKR Brief

Check the weather along your flight before you fly. Pick two airports, a time, and your cruise altitude and speed, and SKR Brief shows the predicted temperature, wind, pressure and visibility along the way.

## Download

You need [Python](https://www.python.org/downloads/) (3.10 or newer) and [Node.js](https://nodejs.org/) (18 or newer).

```bash
git clone https://github.com/hxrshx/skrbrief.git
cd skrbrief

pip install -r requirements.txt
cd visualai && npm install && cd ..
```

## Start the app

Open two terminal windows in the `skrbrief` folder.

**Window 1:**
```bash
python api_server.py
```
Wait about a minute, until you see `Running on http://127.0.0.1:5001`.

**Window 2:**
```bash
cd visualai
npm run dev
```

Then open **http://localhost:8080** in your browser.

## Use the app

**Weather along a route**
1. Choose **Route**.
2. Enter the departure and arrival airport codes, for example `CYYZ` and `CYBB`.
3. Enter the date, the time (UTC), your cruise altitude (ft) and cruise speed (knots).
4. See your route on a map and as a side view by altitude, with the weather along it.

**Weather at one spot**
1. Choose **Point**.
2. Enter latitude, longitude, altitude (ft), date and time (UTC).
3. See the predicted conditions there and the nearest weather station.

Dates must be in **January 2026**.

## Stopping the app

Press `Ctrl + C` in both terminal windows.
