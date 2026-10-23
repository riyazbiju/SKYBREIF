import { useMemo, useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { TopViewVisualization, WeatherPoint } from "@/components/TopViewVisualization";
import { ArrowLeft, Loader2 } from "lucide-react";
import { format } from "date-fns";
import { predictPoint, PredictionResult } from "@/utils/api";

const PointResults = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const d = location.state || {
    lat: 43.65, lon: -79.38, alt: 5000, datetime: new Date()
  };

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const result = await predictPoint({
          lat: parseFloat(d.lat),
          lon: parseFloat(d.lon),
          alt: parseFloat(d.alt),
          datetime: new Date(d.datetime).toISOString()
        });
        
        setPrediction(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch prediction');
      } finally {
        setLoading(false);
      }
    };

    fetchPrediction();
  }, [d.lat, d.lon, d.alt, d.datetime]);

  const pointData: WeatherPoint[] = useMemo(() => {
    if (!prediction) {
      return [{
        lat: parseFloat(d.lat),
        lon: parseFloat(d.lon),
        name: "Analysis Target",
        altitude: parseInt(d.alt),
        type: 'single-point',
        temp: 0,
        windSpd: 0,
        windDir: 0,
        pressure: 0,
        visibility: "N/A"
      }];
    }

    // Convert visibility from km to statute miles (SM)
    const visibilitySM = (prediction.visibility / 1.609).toFixed(1);
    
    return [{
      lat: parseFloat(d.lat),
      lon: parseFloat(d.lon),
      name: "Analysis Target",
      altitude: parseInt(d.alt),
      type: 'single-point',
      temp: Math.round(prediction.temperature),
      windSpd: Math.round(prediction.windspeed),
      windDir: Math.round(prediction.wind_direction),
      pressure: Math.round(prediction.pressure),
      visibility: `${visibilitySM} SM`
    }];
  }, [d, prediction]);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans p-6">
        <div className="container mx-auto max-w-4xl">
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-500" />
              <p className="text-slate-400">Analyzing weather conditions...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans p-6">
        <div className="container mx-auto max-w-4xl">
          <button onClick={() => navigate("/point")} className="flex items-center text-slate-400 hover:text-white mb-6 transition-colors">
            <ArrowLeft className="w-4 h-4 mr-2"/> Back to Input
          </button>
          
          <div className="bg-red-900/20 border border-red-500/30 text-red-400 p-6 rounded-lg text-center">
            <h3 className="text-xl font-bold mb-2">Prediction Error</h3>
            <p>{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!pointData) return null;

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans p-6">
      <div className="container mx-auto max-w-4xl">
         
         <button onClick={() => navigate("/point")} className="flex items-center text-slate-400 hover:text-white mb-6 transition-colors">
            <ArrowLeft className="w-4 h-4 mr-2"/> Back to Input
         </button>

         <div className="bg-[#1e293b] p-6 rounded-lg border border-slate-700 text-center mb-8 shadow-xl">
            <h3 className="text-xl font-bold text-white mb-2">Spot Weather Analysis</h3>
            <div className="inline-block bg-[#0f172a] px-3 py-1 rounded border border-slate-600 font-mono text-[#0ea5e9]">
               Lat {d.lat} / Lon {d.lon} @ {d.alt} ft
            </div>
            <div className="text-slate-500 text-sm mt-3">
               {d.datetime ? format(new Date(d.datetime), "PPP p") : "N/A"}
            </div>
         </div>

         {prediction && (
           <div className="bg-[#1e293b] p-6 rounded-lg border border-slate-700 mb-6 shadow-xl">
             <h4 className="text-lg font-semibold text-white mb-4">Predicted Conditions</h4>
             <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
               <div className="bg-[#0f172a] p-3 rounded border border-slate-600">
                 <div className="text-slate-400 text-sm">Temperature</div>
                 <div className="text-2xl font-bold text-white">{Math.round(prediction.temperature)}°C</div>
               </div>
               <div className="bg-[#0f172a] p-3 rounded border border-slate-600">
                 <div className="text-slate-400 text-sm">Wind Speed</div>
                 <div className="text-2xl font-bold text-white">{Math.round(prediction.windspeed)} kts</div>
               </div>
               <div className="bg-[#0f172a] p-3 rounded border border-slate-600">
                 <div className="text-slate-400 text-sm">Wind Direction</div>
                 <div className="text-2xl font-bold text-white">{Math.round(prediction.wind_direction)}°</div>
               </div>
               <div className="bg-[#0f172a] p-3 rounded border border-slate-600">
                 <div className="text-slate-400 text-sm">Pressure</div>
                 <div className="text-2xl font-bold text-white">{Math.round(prediction.pressure)} hPa</div>
               </div>
               <div className="bg-[#0f172a] p-3 rounded border border-slate-600">
                 <div className="text-slate-400 text-sm">Visibility</div>
                 <div className="text-2xl font-bold text-white">{(prediction.visibility / 1.609).toFixed(1)} SM</div>
               </div>
               {prediction.nearest_stations && prediction.nearest_stations.length > 0 && (
                 <div className="bg-[#0f172a] p-3 rounded border border-slate-600 col-span-2 md:col-span-1">
                   <div className="text-slate-400 text-sm">Nearest Station</div>
                   <div className="text-lg font-bold text-white truncate">{prediction.nearest_stations[0]}</div>
                   <div className="text-slate-500 text-xs">{prediction.distances_km?.[0].toFixed(1)} km</div>
                 </div>
               )}
             </div>
           </div>
         )}

         <div className="space-y-4">
            <div className="bg-blue-900/20 border border-blue-500/30 text-blue-400 p-3 rounded text-sm text-center flex items-center justify-center gap-2">
               <span>ℹ️</span> Click the red marker on the map below to view the full weather breakdown.
            </div>
            
            {/* Map Component */}
            <TopViewVisualization points={pointData} isRoute={false} />
         </div>
      </div>
    </div>
  );
};

export default PointResults;