// src/utils/airports.ts
export interface Airport {
  station: string;
  latitude: number;
  longitude: number;
  elevation_ft: number;
}

let AIRPORT_DB: Record<string, Airport> = {};

const loadAirportData = async (): Promise<void> => {
  try {
    const response = await fetch('/aerodromes.csv');
    const csvText = await response.text();
    const lines = csvText.split('\n');
    const db: Record<string, Airport> = {};
    
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      const [station, latStr, lonStr, elevStr] = line.split(',');
      if (station && latStr && lonStr) {
        db[station.toUpperCase()] = {
          station: station.toUpperCase(),
          latitude: parseFloat(latStr),
          longitude: parseFloat(lonStr),
          elevation_ft: parseInt(elevStr) || 0
        };
      }
    }
    AIRPORT_DB = db;
    console.log(`✓ Loaded ${Object.keys(db).length} aerodromes from CSV`);
  } catch (error) {
    console.error('Failed to load aerodromes.csv:', error);
  }
};

loadAirportData();

export const getAirport = (code: string): Airport | undefined => {
  return AIRPORT_DB[code.toUpperCase()];
};

export const getAllAirports = (): Airport[] => {
  return Object.values(AIRPORT_DB);
};
