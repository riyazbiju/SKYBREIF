// src/utils/geo.ts

// 1. Haversine Distance (in Nautical Miles)
export const calculateDistanceNM = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
  const R = 3440.065; // Earth radius in NM
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
};

// 2. Generate Great Circle Path (Curved Route)
export const getGreatCirclePath = (
  lat1: number, lon1: number, 
  lat2: number, lon2: number, 
  numPoints: number = 100
) => {
  const points = [];
  const d = calculateDistanceNM(lat1, lon1, lat2, lon2); // Distance in NM
  
  // We need distance in radians for the formula
  const earthRadiusNM = 3440.065; 
  const angularDist = d / earthRadiusNM; 

  const startLatRad = toRad(lat1);
  const startLonRad = toRad(lon1);
  const endLatRad = toRad(lat2);
  const endLonRad = toRad(lon2);

  for (let i = 0; i <= numPoints; i++) {
    const f = i / numPoints; // Fraction of the way
    
    // Spheroidal Interpolation formula (Intermediate point on Great Circle)
    const A = Math.sin((1 - f) * angularDist) / Math.sin(angularDist);
    const B = Math.sin(f * angularDist) / Math.sin(angularDist);
    
    const x = A * Math.cos(startLatRad) * Math.cos(startLonRad) + 
              B * Math.cos(endLatRad) * Math.cos(endLonRad);
    const y = A * Math.cos(startLatRad) * Math.sin(startLonRad) + 
              B * Math.cos(endLatRad) * Math.sin(endLonRad);
    const z = A * Math.sin(startLatRad) + B * Math.sin(endLatRad);
    
    const latRad = Math.atan2(z, Math.sqrt(x*x + y*y));
    const lonRad = Math.atan2(y, x);
    
    points.push({
      lat: toDeg(latRad),
      lon: toDeg(lonRad),
      fraction: f
    });
  }
  return points;
};

// 3. Color Gradient Generator (Blue -> Red)
export const getTempColor = (temp: number) => {
  // Clamp temp between -50 (Cold/Blue) and +30 (Hot/Red)
  const min = -50;
  const max = 30;
  const t = Math.max(min, Math.min(max, temp));
  
  // Normalize 0..1
  const ratio = (t - min) / (max - min);
  
  // Simple Blue to Red interpolation
  // Cold: 0,0,255 (Blue) | Hot: 255,0,0 (Red)
  // Mid: 255,255,255 (White)
  
  let r, g, b;
  
  if (ratio < 0.5) {
     // Blue to White
     // ratio 0   -> 0, 0, 255
     // ratio 0.5 -> 255, 255, 255
     const f = ratio * 2; 
     r = Math.round(f * 255);
     g = Math.round(f * 255);
     b = 255;
  } else {
     // White to Red
     // ratio 0.5 -> 255, 255, 255
     // ratio 1.0 -> 255, 0, 0
     const f = (ratio - 0.5) * 2;
     r = 255;
     g = Math.round(255 * (1 - f));
     b = Math.round(255 * (1 - f));
  }

  return `rgb(${r},${g},${b})`;
};

const toRad = (n: number) => n * (Math.PI / 180);
const toDeg = (n: number) => n * (180 / Math.PI);