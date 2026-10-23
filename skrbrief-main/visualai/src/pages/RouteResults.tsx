import { useState, useMemo, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { AviationButton } from "@/components/ui/aviation-button";
import { TopViewVisualization, WeatherPoint } from "@/components/TopViewVisualization";
import { SideViewVisualization, SideViewDataPoint } from "@/components/SideViewVisualization";
import { ArrowLeft, Eye, Layers } from "lucide-react";
import { format, addMinutes } from "date-fns";
import { getGreatCirclePath, calculateDistanceNM } from "@/utils/geo";
import { getAirport } from "@/utils/airports"; // Import real data
import { toast } from "sonner"; // Assuming you have this or standard alert

const RouteResults = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  const formData = location.state || {
    startStation: "CYYZ",
    endStation: "CYYR", // Default changed to match CSV availability
    cruiseAltitude: "30000",
    cruiseSpeed: "450",
  };

  const [activeView, setActiveView] = useState<"top" | "side">("top");

  // --- CALCULATION LOGIC ---
  const { topViewData, sideViewData, totalDistance, error } = useMemo(() => {
    
    // 1. Get Real Coordinates from CSV
    const startApt = getAirport(formData.startStation);
    const endApt = getAirport(formData.endStation);

    if (!startApt || !endApt) {
       return { 
         topViewData: [], sideViewData: [], totalDistance: 0, 
         error: `Airport not found: ${!startApt ? formData.startStation : formData.endStation}` 
       };
    }

    const startLat = startApt.latitude;
    const startLon = startApt.longitude;
    const endLat = endApt.latitude;
    const endLon = endApt.longitude;

    // 2. Real Distance (Haversine)
    const distNM = calculateDistanceNM(startLat, startLon, endLat, endLon);
    
    // 3. Great Circle Path (100 segments)
    const numSteps = 100;
    const pathPoints = getGreatCirclePath(startLat, startLon, endLat, endLon, numSteps);

    const points: WeatherPoint[] = [];
    const sidePoints: SideViewDataPoint[] = [];
    const cruiseAlt = parseInt(formData.cruiseAltitude);
    const speed = parseInt(formData.cruiseSpeed);

    // Get station elevations for terrain profile
    const startElevation = startApt.elevation_ft || 0; // in feet
    const endElevation = endApt.elevation_ft || 0; // in feet

    pathPoints.forEach((pt, i) => {
       const distCurrent = pt.fraction * distNM;
       const hoursFlown = distCurrent / speed;
       const timeStr = format(addMinutes(new Date(), hoursFlown * 60), "HH:mm") + "Z";
       
       // Flight Profile - Start from station elevation, climb to cruise, descend to destination elevation
       let alt = cruiseAlt;
       if (pt.fraction < 0.1) {
         // Takeoff phase: climb from start station elevation to cruise altitude
         alt = startElevation + (pt.fraction / 0.1) * (cruiseAlt - startElevation);
       } else if (pt.fraction > 0.9) {
         // Landing phase: descend from cruise altitude to end station elevation
         alt = endElevation + ((1 - pt.fraction) / 0.1) * (cruiseAlt - endElevation);
       }
       
       // --- Weather Logic ---
       const baseTemp = 20 - (Math.abs(pt.lat) * 0.8); 
       const altFactor = (alt / 1000) * 1.98; 
       const temp = Math.round(baseTemp - altFactor);

       const windSpd = Math.round(15 + (alt / 1000) + Math.random() * 20);
       const windDir = Math.round((270 + pt.fraction * 180) % 360);
       
       const pressure = alt < 5000 ? Math.round(1013 - alt/30) : Math.round(220 + Math.random()*20);

       let visRaw = 10;
       if (Math.sin(distCurrent/100) > 0.8) visRaw = 3 + Math.random() * 2; // Patchy fog
       const visLabel = visRaw >= 9 ? "10+ SM" : `${Math.round(visRaw)} SM`;
       
       // Terrain - Linear interpolation from start elevation to end elevation
       // with some variation to simulate terrain features
       const baseTerrainElevation = startElevation + (endElevation - startElevation) * pt.fraction;
       const terrainVariation = Math.sin(pt.fraction * Math.PI * 8) * 500 + Math.sin(pt.lon / 10) * 300;
       const terrain = Math.round(Math.max(0, baseTerrainElevation + terrainVariation));

       // --- Build Data ---
       let type: 'waypoint' | 'start' | 'end' = 'waypoint';
       if (i === 0) type = 'start';
       if (i === numSteps) type = 'end';

       points.push({
         lat: pt.lat,
         lon: pt.lon,
         name: type === 'start' ? formData.startStation : type === 'end' ? formData.endStation : `WP-${i}`,
         type: type,
         temp, windSpd, windDir, pressure, visibility: visLabel, altitude: Math.round(alt),
         fraction: pt.fraction
       });

       sidePoints.push({
         distance: Math.round(distCurrent),
         time: timeStr,
         altitude: Math.round(alt),
         terrain,
         temp, windSpd, windDir, pressure, visibility: visRaw, visLabel
       });
    });

    return { topViewData: points, sideViewData: sidePoints, totalDistance: Math.round(distNM), error: null };
  }, [formData]);

  // Handle errors (e.g., user entered invalid ICAO)
  useEffect(() => {
    if (error) {
      toast.error(error); // Show error message
    }
  }, [error]);

  if (error) {
    return (
       <div className="min-h-screen bg-[#0f172a] text-white flex items-center justify-center flex-col gap-4">
          <div className="text-red-500 font-bold text-xl">Error Calculation Route</div>
          <div>{error}</div>
          <AviationButton onClick={() => navigate("/route")}>Return to Input</AviationButton>
       </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans pb-10">
      <div className="container mx-auto px-4 py-8">
        
        {/* Navigation & Header */}
        <div className="mb-8">
           <button onClick={() => navigate("/route")} className="flex items-center text-slate-400 hover:text-white mb-4 transition-colors">
              <ArrowLeft className="w-4 h-4 mr-2"/> Back to Input
           </button>
           <div className="bg-[#1e293b] p-6 rounded-xl border border-slate-700 text-center shadow-lg">
              <h1 className="text-2xl font-bold text-white mb-2">{formData.startStation} ➝ {formData.endStation}</h1>
              <div className="flex flex-wrap justify-center gap-4 text-sm font-mono text-[#0ea5e9]">
                 <span className="bg-[#0f172a] px-3 py-1 rounded border border-slate-600">DIST: {totalDistance} NM</span>
                 <span className="bg-[#0f172a] px-3 py-1 rounded border border-slate-600">FL{Math.round(parseInt(formData.cruiseAltitude)/100)}</span>
                 <span className="bg-[#0f172a] px-3 py-1 rounded border border-slate-600">SPD: {formData.cruiseSpeed} KT</span>
              </div>
           </div>
        </div>

        {/* View Selection */}
        <div className="flex gap-4 mb-6">
           <AviationButton 
             onClick={() => setActiveView('top')} 
             className="flex-1"
             variant={activeView === 'top' ? 'primary' : 'secondary'}
           >
             <Eye className="w-4 h-4 mr-2"/> Top View (Map)
           </AviationButton>
           <AviationButton 
             onClick={() => setActiveView('side')} 
             className="flex-1"
             variant={activeView === 'side' ? 'primary' : 'secondary'}
           >
             <Layers className="w-4 h-4 mr-2"/> Side View (Profile)
           </AviationButton>
        </div>

        {/* Visualization Area */}
        <div className="bg-[#0f172a] rounded-xl border border-slate-800 p-1">
           {activeView === 'top' && (
              <div className="animate-in fade-in zoom-in duration-300">
                <TopViewVisualization points={topViewData} isRoute={true} />
              </div>
           )}

           {activeView === 'side' && (
              <div className="animate-in fade-in zoom-in duration-300">
                <SideViewVisualization data={sideViewData} />
              </div>
           )}
        </div>

      </div>
    </div>
  );
};

export default RouteResults;