import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AviationButton } from "@/components/ui/aviation-button";
import { AviationInput } from "@/components/ui/aviation-input";
import { ArrowLeft, MapPin, Calendar, Clock, Navigation } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const PointInput = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    lat: "56.1304",
    lon: "-106.3468",
    alt: "10000",
    date: new Date().toISOString().split('T')[0],
    time: "12:00"
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Navigate with STATE
    navigate("/point/results", { 
        state: { 
            lat: formData.lat, 
            lon: formData.lon, 
            alt: formData.alt, 
            datetime: new Date(`${formData.date}T${formData.time}`).toISOString() 
        } 
    });
  };

  return (
    <div className="min-h-screen bg-background text-slate-200 font-sans p-6 flex items-center justify-center relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 grid-overlay opacity-30" />
      <div className="absolute top-20 left-1/4 w-64 h-64 bg-primary/10 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 right-1/4 w-80 h-80 bg-accent/10 rounded-full blur-3xl animate-pulse" />
      
      <Card className="w-full max-w-md shadow-xl border-slate-700 bg-[#1e293b] relative z-10">
        <CardHeader className="space-y-1">
           <button onClick={() => navigate("/")} className="flex items-center text-sm text-slate-400 hover:text-white mb-2 transition-colors">
              <ArrowLeft className="w-3 h-3 mr-1" /> Back Home
           </button>
           <CardTitle className="text-2xl font-bold text-white">Point Analysis</CardTitle>
           <p className="text-slate-400 text-sm">Calculate weather for any global coordinate</p>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
             <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                   <label className="text-xs font-bold text-slate-300 uppercase">Latitude</label>
                   <div className="relative">
                      <MapPin className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
                      <AviationInput 
                        className="pl-9 bg-[#0f172a] border-slate-600 text-white" placeholder="45.50" 
                        value={formData.lat} onChange={(e) => setFormData({...formData, lat: e.target.value})}
                      />
                   </div>
                </div>
                <div className="space-y-2">
                   <label className="text-xs font-bold text-slate-300 uppercase">Longitude</label>
                   <div className="relative">
                      <Navigation className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
                      <AviationInput 
                        className="pl-9 bg-[#0f172a] border-slate-600 text-white" placeholder="-73.56" 
                        value={formData.lon} onChange={(e) => setFormData({...formData, lon: e.target.value})}
                      />
                   </div>
                </div>
             </div>

             <div className="space-y-2">
                <label className="text-xs font-bold text-slate-300 uppercase">Altitude (ft)</label>
                <AviationInput 
                   type="number" placeholder="30000" 
                   className="bg-[#0f172a] border-slate-600 text-white"
                   value={formData.alt} onChange={(e) => setFormData({...formData, alt: e.target.value})}
                />
             </div>

             <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                   <label className="text-xs font-bold text-slate-300 uppercase">Date</label>
                   <div className="relative">
                      <Calendar className="absolute left-3 top-2.5 h-4 w-4 text-slate-500 pointer-events-none" />
                      <input 
                        type="date" className="flex h-10 w-full rounded-md border border-slate-600 bg-[#0f172a] px-3 py-2 pl-9 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        value={formData.date} onChange={(e) => setFormData({...formData, date: e.target.value})}
                      />
                   </div>
                </div>
                <div className="space-y-2">
                   <label className="text-xs font-bold text-slate-300 uppercase">Time (UTC)</label>
                   <div className="relative">
                      <Clock className="absolute left-3 top-2.5 h-4 w-4 text-slate-500 pointer-events-none" />
                      <input 
                        type="time" className="flex h-10 w-full rounded-md border border-slate-600 bg-[#0f172a] px-3 py-2 pl-9 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        value={formData.time} onChange={(e) => setFormData({...formData, time: e.target.value})}
                      />
                   </div>
                </div>
             </div>

             <AviationButton type="submit" className="w-full mt-2" variant="primary">
                Run Analysis
             </AviationButton>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default PointInput;