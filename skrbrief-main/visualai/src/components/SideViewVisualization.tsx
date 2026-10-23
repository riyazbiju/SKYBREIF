import React, { useMemo, useState, useEffect, useRef } from 'react';
import { 
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue
} from "@/components/ui/select";

export interface SideViewDataPoint {
  distance: number; 
  time: string;     
  altitude: number; 
  terrain: number;  
  temp: number;     
  windSpd: number;  
  windDir: number;  
  pressure: number; 
  visibility: number; 
  visLabel: string;   
}

interface SideViewProps {
  data: SideViewDataPoint[];
}

// --- 1. Canvas Background Component (Draws the 2D Contour) ---
const WeatherBackgroundCanvas = ({ width, height, mode, maxDist, maxAlt }: any) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0 || height <= 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(width, height);
    const buffer = imgData.data;

    // Iterate over every pixel to simulate the "Meshgrid" from Python
    for (let py = 0; py < height; py++) {
      for (let px = 0; px < width; px++) {
        // Map pixel x,y to real-world Distance (nm) and Altitude (ft)
        // Note: Y is inverted in canvas (0 is top, height is bottom)
        // But in graphs, 0 is bottom. So we invert py.
        const dist = (px / width) * maxDist; 
        const alt = ((height - py) / height) * maxAlt; 

        let r = 0, g = 0, b = 0, a = 255;

        if (mode === 'temp_wind') {
          // --- TEMPERATURE MATH (Matches Python) ---
          // temp = 15 - (Y / 1000) * 2 + np.sin(X/200)*2
          const val = 15 - (alt / 1000) * 2 + Math.sin(dist / 200) * 2;
          
          // Colormap: RdBu_r (Blue -> White -> Red)
          // Range approx: -60 to +20
          const minT = -60, maxT = 20;
          const ratio = Math.max(0, Math.min(1, (val - minT) / (maxT - minT)));

          // RdBu approximation
          if (ratio < 0.5) {
             // Blue (0,0,255) to White (255,255,255)
             const step = ratio * 2; 
             r = Math.floor(0 + 255 * step);
             g = Math.floor(0 + 255 * step);
             b = Math.floor(100 + 155 * step);
          } else {
             // White (255,255,255) to Red (255,0,0)
             const step = (ratio - 0.5) * 2;
             r = Math.floor(255 - 50 * step); // slightly dark red
             g = Math.floor(255 - 255 * step);
             b = Math.floor(255 - 255 * step);
          }
          a = 200; // Opacity

        } else {
          // --- VISIBILITY MATH (Matches Python) ---
          // vis = 10 + 5 * np.sin(X/50) * np.cos(Y/5000)
          let val = 10 + 5 * Math.sin(dist / 50) * Math.cos(alt / 5000);
          val = Math.max(0, Math.min(10, val)); // Clip 0-10

          // Colormap: Bone (Dark Blue/Grey -> White)
          // High Vis (10) = Clear/Black/Blue, Low Vis (0) = White/Fog
          const ratio = val / 10; 
          
          // Bone-like: Dark Blue (#1e293b) to Light Grey
          r = Math.floor(30 + 200 * (1-ratio));
          g = Math.floor(41 + 200 * (1-ratio));
          b = Math.floor(59 + 200 * (1-ratio));
          a = 220;
        }

        const index = (py * width + px) * 4;
        buffer[index] = r;
        buffer[index + 1] = g;
        buffer[index + 2] = b;
        buffer[index + 3] = a;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [width, height, mode, maxDist, maxAlt]);

  return <canvas ref={canvasRef} width={width} height={height} className="absolute inset-0 z-0 opacity-80" />;
};


// --- 2. Custom Wind Barb ---
const WindBarb = (props: any) => {
  const { cx, cy, payload } = props;
  if (!cx || !cy) return null;
  return (
    <g transform={`rotate(${payload.windDir} ${cx} ${cy})`}>
      <line x1={0} y1={-8} x2={0} y2={8} stroke="white" strokeWidth="1.5" transform={`translate(${cx},${cy})`} />
      <line x1={0} y1={-8} x2={5} y2={-6} stroke="white" strokeWidth="1.5" transform={`translate(${cx},${cy})`} />
      <circle cx={cx} cy={cy} r={2} fill="#0ea5e9" stroke="none" />
    </g>
  );
};

// --- 3. Tooltip ---
const InspectorTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload as SideViewDataPoint;
    return (
      <div className="bg-[#0f172a]/95 backdrop-blur border-2 border-[#0ea5e9] p-3 rounded shadow-2xl text-xs font-mono z-50 min-w-[180px]">
        <div className="text-[#0ea5e9] font-bold border-b border-slate-700 pb-1 mb-2">
            TIME: {d.time}<br/>
            DIST: {Math.round(d.distance)} NM<br/>
            ALT:  {d.altitude} FT
        </div>
        <div className="text-slate-200 space-y-1">
           <div className="flex justify-between"><span>TEMP:</span> <span className="text-[#f87171] font-bold">{d.temp}°C</span></div>
           <div className="flex justify-between"><span>WIND:</span> <span className="text-white font-bold">{d.windSpd} kt @ {d.windDir}°</span></div>
           <div className="flex justify-between"><span>PRES:</span> <span className="text-[#facc15] font-bold">{d.pressure} hPa</span></div>
           <div className="flex justify-between"><span>VIS:</span>  <span className="text-[#94a3b8] font-bold">{d.visLabel}</span></div>
        </div>
      </div>
    );
  }
  return null;
};

export const SideViewVisualization: React.FC<SideViewProps> = ({ data }) => {
  const [viewMode, setViewMode] = useState<string>("temp_wind");
  const [chartDim, setChartDim] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Measure chart area for the background canvas
  useEffect(() => {
    if (containerRef.current) {
       setChartDim({ 
         width: containerRef.current.clientWidth, 
         height: containerRef.current.clientHeight 
       });
    }
  }, []);

  const maxDist = data.length > 0 ? data[data.length-1].distance : 800;
  const maxAlt = 50000; // Fixed max altitude for scale

  // Pressure Gradient for the Line (Plasma)
  const pressureGradient = useMemo(() => {
    if (data.length === 0) return [];
    const minP = Math.min(...data.map(d => d.pressure));
    const maxP = Math.max(...data.map(d => d.pressure));
    return data.map((point, index) => {
      const ratio = (point.pressure - minP) / (maxP - minP || 1);
      let color = "#0d0887"; 
      if (ratio > 0.33) color = "#cc4778"; 
      if (ratio > 0.66) color = "#f0f921"; 
      return <stop key={index} offset={`${(index / (data.length - 1)) * 100}%`} stopColor={color} />;
    });
  }, [data]);

  return (
    <div className="w-full fade-in font-sans">
      <div className="bg-[#0f172a] border border-slate-700 rounded-xl overflow-hidden shadow-2xl">
        
        {/* HEADER */}
        <div className="bg-[#1e293b] px-4 py-3 flex flex-col sm:flex-row justify-between items-center border-b border-slate-700 gap-4">
           <div className="flex items-center gap-3">
             <h3 className="text-white font-bold text-sm tracking-wide uppercase">Atmospheric Profile</h3>
             <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 text-xs font-mono border border-blue-500/20">
                {viewMode === 'temp_wind' ? 'LAYER: TEMPERATURE' : 'LAYER: VISIBILITY'}
             </span>
           </div>
           <div className="w-[220px]">
            <Select value={viewMode} onValueChange={setViewMode}>
              <SelectTrigger className="bg-[#0f172a] border-slate-600 text-slate-200 h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#0f172a] border-slate-600 text-slate-200">
                <SelectItem value="temp_wind">Temperature + Wind</SelectItem>
                <SelectItem value="vis_pres">Visibility + Pressure</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* --- CHART SECTION --- */}
        {/* We use a Ref here to size the Canvas matches the Recharts area */}
        <div ref={containerRef} className="h-[400px] w-full bg-[#0f172a] relative overflow-hidden">
          
          {/* 1. LAYER: 2D WEATHER BACKGROUND (Canvas) */}
          <WeatherBackgroundCanvas 
             width={chartDim.width} 
             height={chartDim.height} 
             mode={viewMode} 
             maxDist={maxDist}
             maxAlt={maxAlt}
          />

          {/* 2. LAYER: DATA LINES & TERRAIN (Recharts) */}
          <div className="absolute inset-0">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                <defs>
                   <linearGradient id="pressureStroke" x1="0" y1="0" x2="1" y2="0">{pressureGradient}</linearGradient>
                </defs>

                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff" vertical={false} opacity={0.15} />
                
                {/* Axes need to match the Canvas scale (0-maxDist, 0-maxAlt) */}
                <XAxis 
                  dataKey="distance" type="number" domain={[0, maxDist]}
                  tick={{fill: '#94a3b8', fontSize: 10, fontFamily: 'monospace'}}
                  tickLine={false} axisLine={{stroke: '#334155'}}
                />
                <YAxis 
                  yAxisId="alt" type="number" domain={[0, maxAlt]}
                  tick={{fill: '#94a3b8', fontSize: 10, fontFamily: 'monospace'}}
                  axisLine={false} tickLine={false}
                />

                <Tooltip content={<InspectorTooltip />} cursor={{ stroke: 'white', strokeWidth: 1 }} />

                {/* A. TERRAIN (Solid Green, Semi-Transparent) */}
                <Area 
                   yAxisId="alt" type="monotone" dataKey="terrain" 
                   fill="#22c55e" fillOpacity={0.9} stroke="none" 
                   isAnimationActive={false}
                />

                {/* B. ROUTE LINE */}
                {viewMode === 'vis_pres' ? (
                   // Pressure Mode: Line uses Plasma Gradient
                   <Line 
                     yAxisId="alt" type="monotone" dataKey="altitude" 
                     stroke="url(#pressureStroke)" strokeWidth={4} dot={false} 
                     isAnimationActive={false}
                   />
                ) : (
                   // Temp Mode: Line is Blue with Wind Barbs
                   <Line 
                      yAxisId="alt" type="monotone" dataKey="altitude"
                      stroke="#0ea5e9" strokeWidth={3}
                      dot={<WindBarb />}
                      isAnimationActive={false}
                   />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          {/* Axis Labels */}
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-white/50 font-bold tracking-wider pointer-events-none">DISTANCE (NM)</div>
          <div className="absolute top-1/2 left-2 -translate-y-1/2 -rotate-90 text-[10px] text-white/50 font-bold tracking-wider pointer-events-none">ALTITUDE (FT)</div>
        </div>

        {/* --- LEGEND PANEL --- */}
        <div className="bg-[#1e293b] border-t border-slate-700 p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
           
           {/* Left: Background Legend */}
           <div className="flex flex-col justify-center gap-4">
              <div className="text-[10px] text-slate-400 font-bold uppercase tracking-widest border-b border-slate-700 pb-1">Atmosphere</div>
              
              {viewMode === 'temp_wind' ? (
                <div className="space-y-3">
                   <div className="flex justify-between items-center text-xs text-slate-300 font-mono">
                      <span>TEMP (BACKGROUND)</span>
                      {/* RdBu Gradient */}
                      <div className="w-40 h-3 rounded bg-gradient-to-r from-blue-600 via-white to-red-600 shadow-sm border border-white/10"></div>
                   </div>
                   <div className="flex justify-between items-center text-xs text-slate-300 font-mono">
                      <span>ROUTE LINE</span>
                      <div className="w-40 h-1 bg-[#0ea5e9] flex items-center justify-center"></div>
                   </div>
                </div>
              ) : (
                <div className="space-y-3">
                   <div className="flex justify-between items-center text-xs text-slate-300 font-mono">
                      <span>VISIBILITY (BG)</span>
                      {/* Bone-like Gradient */}
                      <div className="w-40 h-3 rounded bg-gradient-to-r from-[#1e293b] to-[#94a3b8] shadow-sm border border-white/10"></div>
                   </div>
                   <div className="flex justify-between items-center text-xs text-slate-300 font-mono">
                      <span>PRESSURE (LINE)</span>
                      {/* Plasma Gradient */}
                      <div className="w-40 h-2 rounded bg-gradient-to-r from-[#0d0887] via-[#cc4778] to-[#f0f921] shadow-sm"></div>
                   </div>
                </div>
              )}
              <div className="flex justify-between items-center text-xs text-slate-300 font-mono">
                  <span>TERRAIN</span>
                  <div className="w-40 h-2 bg-[#22c55e] rounded shadow-sm"></div>
              </div>
           </div>

           {/* Right: Wind Legend & Compass */}
           <div className="relative pl-0 md:pl-8 md:border-l md:border-slate-700 flex items-center justify-around bg-[#0f172a]/50 rounded py-2 border border-slate-700/50">
              <div className="flex flex-col gap-2">
                 <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">Wind Spd</span>
                 <div className="text-xs text-white font-mono space-y-2">
                    <div className="flex items-center gap-2">
                        <svg width="24" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="white" strokeWidth="1.5"/><line x1="20" y1="5" x2="24" y2="1" stroke="white" strokeWidth="1.5"/></svg>
                        <span>10 kt</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <svg width="24" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="white" strokeWidth="1.5"/><polygon points="20,5 20,0 25,5" fill="white"/></svg>
                        <span>50 kt</span>
                    </div>
                 </div>
              </div>

              <div className="flex flex-col items-center">
                 <div className="relative w-16 h-16 flex items-center justify-center border border-slate-600/50 rounded-full bg-[#0f172a] shadow-inner">
                    <div className="absolute w-full h-[1px] bg-slate-500"></div>
                    <div className="absolute h-full w-[1px] bg-slate-500"></div>
                    <span className="absolute top-0 text-[10px] font-bold text-[#0ea5e9] bg-[#0f172a] px-1 leading-none -mt-1">N</span>
                    <span className="absolute bottom-0 text-[10px] font-bold text-[#0ea5e9] bg-[#0f172a] px-1 leading-none -mb-1">S</span>
                    <span className="absolute left-0 text-[10px] font-bold text-[#0ea5e9] bg-[#0f172a] px-1 leading-none -ml-1">W</span>
                    <span className="absolute right-0 text-[10px] font-bold text-[#0ea5e9] bg-[#0f172a] px-1 leading-none -mr-1">E</span>
                 </div>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
};