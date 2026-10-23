/**
 * API Client for Backend Weather Prediction Service
 */

const API_BASE_URL = 'http://localhost:5001';

export interface PredictionQuery {
  lat: number;
  lon: number;
  alt: number;
  datetime: string;
}

export interface PredictionResult {
  temperature: number;
  windspeed: number;
  visibility: number;
  pressure: number;
  wind_direction: number;
  nearest_stations?: string[];
  distances_km?: number[];
  query?: {
    latitude: number;
    longitude: number;
    altitude: number;
    datetime: string;
    time_hours: number;
  };
  error?: string;
}

export interface RouteWaypoint {
  lat: number;
  lon: number;
  alt: number;
  datetime: string;
}

export interface RoutePredictionResponse {
  predictions: PredictionResult[];
  count: number;
}

/**
 * Predict weather at a single point
 */
export async function predictPoint(query: PredictionQuery): Promise<PredictionResult> {
  const response = await fetch(`${API_BASE_URL}/predict/point`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(query),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Prediction failed');
  }

  return response.json();
}

/**
 * Predict weather along a route
 */
export async function predictRoute(waypoints: RouteWaypoint[]): Promise<RoutePredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/predict/route`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ waypoints }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Route prediction failed');
  }

  return response.json();
}

/**
 * Get API health status
 */
export async function getHealth(): Promise<{
  status: string;
  model_loaded: boolean;
  stations: number;
}> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
}

/**
 * Get model and dataset information
 */
export async function getInfo(): Promise<{
  model: {
    type: string;
    k_neighbors: number;
    features: string[];
  };
  dataset: {
    stations: number;
    time_range_hours: [number, number];
    time_range_days: [number, number];
    coverage: string;
    period: string;
  };
}> {
  const response = await fetch(`${API_BASE_URL}/info`);
  return response.json();
}
