import { useNavigate } from "react-router-dom";
import { AviationButton } from "@/components/ui/aviation-button";
import { Plane, MapPin, Cloud, Wind } from "lucide-react";

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 grid-overlay opacity-30" />
      <div className="absolute top-20 left-1/4 w-64 h-64 bg-primary/10 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 right-1/4 w-80 h-80 bg-accent/10 rounded-full blur-3xl animate-pulse" />

      <div className="relative z-10 container mx-auto px-6 py-16 flex flex-col items-center justify-center min-h-screen">
        {/* Header */}
        <div className="text-center mb-16 fade-in">
          <div className="inline-flex items-center gap-3 mb-6">
            <div className="p-3 rounded-xl aviation-gradient">
              <Cloud className="w-8 h-8 text-primary-foreground" />
            </div>
            <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
              <span className="text-primary">SKY</span>
              <span className="text-foreground">BRIEF</span>
            </h1>
          </div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Visual Meteorological Briefing for Pilots
          </p>
          <p className="text-sm text-muted-foreground/70 mt-2 font-mono">
            Powered by Latent ODE 
          </p>
        </div>

        {/* Main Action Cards */}
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl w-full">
          {/* Weather Along Route Card */}
          <div
            onClick={() => navigate("/route")}
            className="group data-panel rounded-2xl p-8 cursor-pointer transition-all duration-300 hover:scale-[1.02] cockpit-glow hover:border-primary/50"
          >
            <div className="flex items-start gap-4 mb-6">
              <div className="p-4 rounded-xl bg-primary/20 group-hover:bg-primary/30 transition-colors">
                <Plane className="w-8 h-8 text-primary" />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-foreground mb-2">
                  Weather Along Route
                </h2>
                <p className="text-muted-foreground">
                  Get comprehensive weather forecasts for your entire flight path
                </p>
              </div>
            </div>

            <div className="space-y-3 mb-6">
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <span className="w-2 h-2 rounded-full bg-primary" />
                <span>Top view route visualization</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <span className="w-2 h-2 rounded-full bg-primary" />
                <span>Vertical cross-section analysis</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <span className="w-2 h-2 rounded-full bg-primary" />
                <span>Temperature, wind, pressure & visibility data</span>
              </div>
            </div>

            <AviationButton variant="outline" className="w-full group-hover:bg-primary group-hover:text-primary-foreground">
              Enter Route Details
            </AviationButton>
          </div>

          {/* Weather at Point Card */}
          <div
            onClick={() => navigate("/point")}
            className="group data-panel rounded-2xl p-8 cursor-pointer transition-all duration-300 hover:scale-[1.02] cockpit-glow hover:border-accent/50"
          >
            <div className="flex items-start gap-4 mb-6">
              <div className="p-4 rounded-xl bg-accent/20 group-hover:bg-accent/30 transition-colors">
                <MapPin className="w-8 h-8 text-accent" />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-foreground mb-2">
                  Weather at Unknown Point
                </h2>
                <p className="text-muted-foreground">
                  Query weather conditions at any 4D location
                </p>
              </div>
            </div>

            <div className="space-y-3 mb-6">
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <span className="w-2 h-2 rounded-full bg-accent" />
                <span>Arbitrary lat/long/alt/time query</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <span className="w-2 h-2 rounded-full bg-accent" />
                <span>Time-specific predictions</span>
              </div>
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <span className="w-2 h-2 rounded-full bg-accent" />
                <span>Location overview visualization</span>
              </div>
            </div>

            <AviationButton variant="outline" className="w-full border-accent text-accent group-hover:bg-accent group-hover:text-accent-foreground">
              Enter Location Details
            </AviationButton>
          </div>
        </div>

        {/* Footer info */}
        <div className="mt-16 flex items-center gap-8 text-sm text-muted-foreground/60 font-mono">
          <div className="flex items-center gap-2">
            <Wind className="w-4 h-4" />
            <span>231 Canadian Aerodromes</span>
          </div>
          <div className="flex items-center gap-2">
            <Cloud className="w-4 h-4" />
            <span>METAR • TAF • ERA5</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
