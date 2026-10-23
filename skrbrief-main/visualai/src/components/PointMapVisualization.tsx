import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { CloudSun } from 'lucide-react';
import { renderToString } from 'react-dom/server';

const createIcon = () => {
  const html = renderToString(<CloudSun color="#dc2626" fill="#dc2626" size={32} />);
  return L.divIcon({
    html: `<div style="filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.4)); transform: translate(-16px, -32px);">${html}</div>`,
    className: 'bg-transparent',
    iconSize: [32, 32],
    iconAnchor: [0, 0] 
  });
};

const pointIcon = createIcon();

interface PointMapProps {
  lat: number;
  lon: number;
  label?: string;
  weather: any;
}

const MapController = ({ lat, lon }: { lat: number, lon: number }) => {
  const map = useMap();
  useEffect(() => {
    map.setView([lat, lon], 7, { animate: true });
    map.invalidateSize();
  }, [lat, lon, map]);
  return null;
};

export const PointMapVisualization: React.FC<PointMapProps> = ({ lat, lon, label, weather }) => {
  return (
    <div className="w-full h-[500px] rounded-xl overflow-hidden shadow-xl border border-slate-200 bg-slate-100">
      <MapContainer center={[lat, lon]} zoom={7} style={{ height: '100%', width: '100%' }}>
        <TileLayer attribution='&copy; CARTO' url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" />
        <MapController lat={lat} lon={lon} />
        <Marker position={[lat, lon]} icon={pointIcon}>
          <Popup minWidth={150}>
             <div className="font-bold text-red-700 border-b mb-1">{label}</div>
             <div className="text-xs">
                <div>Temp: {weather.temp}°C</div>
                <div>Wind: {weather.windSpd} kt</div>
                <div>Press: {weather.pressure} hPa</div>
             </div>
          </Popup>
        </Marker>
      </MapContainer>
    </div>
  );
};