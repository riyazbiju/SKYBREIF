import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AviationButton } from "@/components/ui/aviation-button";
import { AviationInput } from "@/components/ui/aviation-input";
import { ArrowLeft, Plane } from "lucide-react";

const RouteInput = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    startStation: "",
    endStation: "",
    date: "",
    time: "",
    cruiseAltitude: "",
    cruiseSpeed: "",
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Pass form data to results page
    navigate("/route/results", { state: formData });
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      <div className="absolute inset-0 grid-overlay opacity-20" />
      
      <div className="relative z-10 container mx-auto px-6 py-8">
        {/* Back button */}
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-8"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back to Home</span>
        </button>

        <div className="max-w-2xl mx-auto">
          {/* Header */}
          <div className="text-center mb-10 fade-in">
            <div className="inline-flex items-center gap-3 mb-4">
              <div className="p-3 rounded-xl aviation-gradient">
                <Plane className="w-6 h-6 text-primary-foreground" />
              </div>
              <h1 className="text-3xl font-bold text-foreground">
                Weather Along Route
              </h1>
            </div>
            <p className="text-muted-foreground">
              Enter your flight details to generate weather visualization
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-8 fade-in">
            <div className="data-panel rounded-xl p-8 cockpit-glow">
              {/* Route Section */}
              <div className="mb-8">
                <h3 className="text-sm font-semibold text-primary uppercase tracking-wider mb-4">
                  Route Information
                </h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <AviationInput
                    label="Start Station (ICAO)"
                    name="startStation"
                    placeholder="CYYZ"
                    value={formData.startStation}
                    onChange={handleChange}
                    required
                  />
                  <AviationInput
                    label="End Station (ICAO)"
                    name="endStation"
                    placeholder="CYBB"
                    value={formData.endStation}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>

              {/* Time Section */}
              <div className="mb-8">
                <h3 className="text-sm font-semibold text-primary uppercase tracking-wider mb-4">
                  Date & Time
                </h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <AviationInput
                    label="Date"
                    name="date"
                    type="date"
                    value={formData.date}
                    onChange={handleChange}
                    required
                  />
                  <AviationInput
                    label="Time (UTC)"
                    name="time"
                    type="time"
                    value={formData.time}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>

              {/* Flight Parameters */}
              <div>
                <h3 className="text-sm font-semibold text-primary uppercase tracking-wider mb-4">
                  Flight Parameters
                </h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <AviationInput
                    label="Cruise Altitude"
                    name="cruiseAltitude"
                    type="number"
                    placeholder="35000"
                    unit="ft"
                    value={formData.cruiseAltitude}
                    onChange={handleChange}
                    required
                  />
                  <AviationInput
                    label="Cruise Speed"
                    name="cruiseSpeed"
                    type="number"
                    placeholder="450"
                    unit="kts"
                    value={formData.cruiseSpeed}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <AviationButton
              type="submit"
              variant="action"
              size="lg"
              className="w-full"
            >
              Generate Weather Briefing
            </AviationButton>
          </form>
        </div>
      </div>
    </div>
  );
};

export default RouteInput;
