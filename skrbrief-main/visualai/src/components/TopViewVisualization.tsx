import React, { useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { Plane, Flag, MapPin } from 'lucide-react';
import { renderToString } from 'react-dom/server';
import { getTempColor } from '@/utils/geo'; 

// --- Custom Icons ---
const createIcon = (IconComponent: any, color: string) => {
  const html = renderToString(<IconComponent color={color} fill={color} size={28} />);
  return L.divIcon({
    html: `<div style="filter: drop-shadow(0px 2px 2px rgba(0,0,0,0.3)); transform: translate(-14px, -28px);">${html}</div>`,
    className: 'bg-transparent',
    iconSize: [28, 28],
    iconAnchor: [0, 0] 
  });
};

const planeIcon = createIcon(Plane, "#2563eb"); // Blue Plane
const flagIcon = createIcon(Flag, "#16a34a");   // Green Flag
const redPinIcon = createIcon(MapPin, "#dc2626"); // Red Pin

export interface WeatherPoint {
  lat: number;
  lon: number;
  name?: string;
  type?: 'start' | 'end' | 'waypoint' | 'single-point';
  temp: number;
  windSpd: number;
  windDir: number;
  pressure: number;
  visibility: string;
  altitude: number;
  fraction?: number;
}

interface TopViewProps {
  points: WeatherPoint[]; 
  isRoute?: boolean;
}

// Map Controller for Auto-Zoom
const MapController = ({ points }: { points: WeatherPoint[] }) => {
  const map = useMap();
  useEffect(() => {
    if (!points || points.length === 0) return;
    if (points.length === 1) {
       map.setView([points[0].lat, points[0].lon], 8, { animate: true });
    } else {
       const bounds = L.latLngBounds(points.map(p => [p.lat, p.lon]));
       if (bounds.isValid()) map.fitBounds(bounds, { padding: [50, 50] });
    }
    setTimeout(() => map.invalidateSize(), 200);
  }, [points, map]);
  return null;
};

export const TopViewVisualization: React.FC<TopViewProps> = ({ points, isRoute = false }) => {
  const centerPos = points && points.length > 0 ? [points[0].lat, points[0].lon] : [0, 0];

  // 1. Generate Heatmap Segments (Weather along the route)
  const segments = useMemo(() => {
    if (!isRoute || points.length < 2) return [];
    const segs = [];
    for (let i = 0; i < points.length - 1; i++) {
       segs.push({
         positions: [
            [points[i].lat, points[i].lon],
            [points[i+1].lat, points[i+1].lon]
         ] as [number, number][],
         color: getTempColor(points[i].temp), // Function from utils/geo
         data: points[i]
       });
    }
    return segs;
  }, [points, isRoute]);

  // 2. Filter Markers: ONLY show Start and End (Clean View)
  const markers = useMemo(() => {
    if (!isRoute) return points; // For single point query, show it
    return points.filter(p => p.type === 'start' || p.type === 'end');
  }, [points, isRoute]);

  return (
    <div className="w-full fade-in border border-slate-200 rounded-xl overflow-hidden shadow-xl bg-white">
      {/* Header - Light Theme */}
      <div className="bg-slate-50 px-4 py-3 border-b border-slate-200 flex justify-between items-center">
         <div className="flex items-center gap-3">
            <h3 className="text-slate-800 font-bold text-sm">
              {isRoute ? "Route Weather Map" : "Location Analysis"}
            </h3>
            {/* Legend for Route Heatmap */}
            {isRoute && (
               <div className="flex items-center gap-2 text-xs font-mono bg-white px-2 py-1 rounded border border-slate-200 shadow-sm">
                  <span className="text-slate-500 mr-1">Temp:</span>
                  <span className="w-2 h-2 rounded-full bg-blue-600"></span> Cold
                  <span className="w-2 h-2 rounded-full bg-red-600"></span> Hot
               </div>
            )}
         </div>
      </div>

      <div className="relative h-[500px] w-full bg-slate-100 z-0">
        <MapContainer 
          center={centerPos as L.LatLngExpression} 
          zoom={4} 
          style={{ height: '100%', width: '100%' }}
        >
          {/* Light Mode Tiles (OpenStreetMap) */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          <MapController points={points} />

          {/* HEATMAP ROUTE SEGMENTS */}
          {isRoute && segments.map((seg, idx) => (
             <Polyline 
               key={idx}
               positions={seg.positions}
               pathOptions={{ color: seg.color, weight: 5, opacity: 0.9 }}
             >
               <Popup>
                  <div className="text-slate-800 text-xs font-sans min-w-[180px]">
                     <strong className="text-sm">Enroute Weather</strong><br/>
                     <div className="space-y-1 mt-2">
                        <div className="flex justify-between"><span>Temp:</span> <span className="font-bold">{seg.data.temp}°C</span></div>
                        <div className="flex justify-between"><span>Wind:</span> <span className="font-bold">{seg.data.windSpd} kt @ {seg.data.windDir}°</span></div>
                        <div className="flex justify-between"><span>Vis:</span> <span className="font-bold">{seg.data.visibility}</span></div>
                        <div className="flex justify-between"><span>Press:</span> <span className="font-bold">{seg.data.pressure} hPa</span></div>
                        <div className="flex justify-between"><span>Alt:</span> <span className="font-bold">{seg.data.altitude} ft</span></div>
                     </div>
                  </div>
               </Popup>
             </Polyline>
          ))}

          {/* START / END MARKERS ONLY */}
          {markers.map((point, idx) => {
            let icon = redPinIcon;
            if (point.type === 'start') icon = planeIcon;
            else if (point.type === 'end') icon = flagIcon;

            return (
              <Marker key={idx} position={[point.lat, point.lon]} icon={icon}>
                <Popup>
                   <div className="text-slate-800 text-sm min-w-[150px]">
                      <div className="font-bold border-b border-slate-200 mb-2 pb-1 text-base">
                        {point.name}
                      </div>
                      <div className="space-y-1">
                         <div className="flex justify-between"><span>Temp:</span> <span className="font-bold">{point.temp}°C</span></div>
                         <div className="flex justify-between"><span>Wind:</span> <span className="font-bold">{point.windSpd} kt @ {point.windDir}°</span></div>
                         <div className="flex justify-between"><span>Vis:</span> <span className="font-bold">{point.visibility}</span></div>
                         <div className="flex justify-between"><span>Press:</span> <span className="font-bold">{point.pressure} hPa</span></div>
                         <div className="flex justify-between"><span>Alt:</span> <span className="font-bold">{point.altitude} ft</span></div>
                      </div>
                   </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
};