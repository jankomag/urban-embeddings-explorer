import React, { useState, useCallback, useEffect } from 'react';
import { Button } from "./components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "./components/ui/card";
import { Separator } from "./components/ui/separator";
import { MapPin } from 'lucide-react';
import ScatterPlot from './components/ScatterPlot';
import MapboxMapComponent from './components/MapboxMapComponent';
import LocationSelector from './components/LocationSelector';
import PCSelector from './components/PCSelector';

const App = () => {
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedCity, setSelectedCity] = useState('');
  const [allPoints, setAllPoints] = useState([]);
  const [isControlsCollapsed, setIsControlsCollapsed] = useState(false);
  const [scatterPlotZoom, setScatterPlotZoom] = useState(null);
  const [selectedPCX, setSelectedPCX] = useState(1);
  const [selectedPCY, setSelectedPCY] = useState(2);

  useEffect(() => {
    const fetchAllPoints = async () => {
      try {
        const response = await fetch('http://localhost:8000/pca_data');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const result = await response.json();
        setAllPoints(result.data);
      } catch (error) {
        console.error('Error fetching all points data:', error);
      }
    };
    fetchAllPoints();
  }, []);

  const handlePointSelect = useCallback((point) => {
    if (!point) return;
    
    console.log("Selected point:", point); // For debugging
    setSelectedPoint(point);
    setSelectedCountry(point.country);
    setSelectedCity(point.city || '');
  }, []);

  const handleLocationSelect = useCallback((country, city) => {
    setSelectedCountry(country);
    setSelectedCity(city);
    setSelectedPoint(null);
  }, []);

  const handleZoomChange = useCallback((zoomState) => {
    setScatterPlotZoom(zoomState);
  }, []);

  const handleMapClick = useCallback((point) => {
    if (!point) return;
    console.log("Map click point:", point); // For debugging
    handlePointSelect(point);
  }, [handlePointSelect]);

  const handlePCChange = useCallback((axis, value) => {
    if (axis === 'x') {
      setSelectedPCX(value);
    } else {
      setSelectedPCY(value);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-2xl font-bold">
            Global Urban Embeddings Explorer
          </CardTitle>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-lg">Controls</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsControlsCollapsed(!isControlsCollapsed)}
              >
                {isControlsCollapsed ? 'Expand' : 'Collapse'}
              </Button>
            </CardHeader>
            <CardContent className={`space-y-4 transition-all duration-300 ${isControlsCollapsed ? 'hidden' : ''}`}>
              <LocationSelector onSelect={handleLocationSelect} />
              <Separator className="my-4" />
              <PCSelector 
                selectedX={selectedPCX}
                selectedY={selectedPCY}
                onChange={handlePCChange}
              />
            </CardContent>
          </Card>

          {/* Selection Info */}
          {(selectedCountry || selectedCity || selectedPoint) && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Selected Location</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center space-x-2">
                  <MapPin className="h-4 w-4 text-blue-500" />
                  <span className="font-medium">
                    {selectedCountry}
                    {selectedCity && `, ${selectedCity}`}
                    {selectedPoint && !selectedCity && `, ${selectedPoint.city}`}
                  </span>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Map Component */}
          <Card className={`transition-all duration-300 ${isControlsCollapsed ? 'h-[calc(100vh-16rem)]' : 'h-[500px]'}`}>
            <CardContent className="p-0 h-full">
              <MapboxMapComponent
                selectedPoint={selectedPoint}
                onMapClick={handleMapClick}
                allPoints={allPoints}
                selectedCity={selectedCity}
              />
            </CardContent>
          </Card>
        </div>

        {/* Scatter Plot */}
        <Card className="h-[calc(100vh-8rem)]">
          <CardHeader>
            <CardTitle className="text-lg">PCA Visualization</CardTitle>
          </CardHeader>
          <CardContent className="p-0 h-[calc(100%-5rem)]">
            <ScatterPlot
              onPointSelect={handlePointSelect}
              selectedPoint={selectedPoint}
              selectedCountry={selectedCountry}
              selectedCity={selectedCity}
              onZoomChange={handleZoomChange}
              currentZoom={scatterPlotZoom}
              selectedPCX={selectedPCX}
              selectedPCY={selectedPCY}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default App;
