import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';
import LocationSelector from './components/LocationSelector';
import PCSelector from './components/PCSelector';
import { MapPin } from 'lucide-react';
import ScatterPlot from './components/ScatterPlot';
import MapboxMapComponent from './components/MapboxMapComponent';

const App = () => {
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedCity, setSelectedCity] = useState('');
  const [selectedPCX, setSelectedPCX] = useState(1);
  const [selectedPCY, setSelectedPCY] = useState(2);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [allPoints, setAllPoints] = useState([]);

  // Fetch all points data when component mounts
  useEffect(() => {
    const fetchPoints = async () => {
      try {
        const response = await fetch('http://localhost:8000/pca_data');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const result = await response.json();
        setAllPoints(result.data);
      } catch (error) {
        console.error('Error fetching points:', error);
      }
    };
    fetchPoints();
  }, []);

  const handlePointSelect = useCallback((point) => {
    setSelectedPoint(point);
    setSelectedCountry(point.country);
    setSelectedCity(point.city);
  }, []);

  const handleLocationSelect = useCallback((country, city) => {
    setSelectedCountry(country);
    setSelectedCity(city);
    setSelectedPoint(null);
  }, []);

  const handlePCChange = useCallback((axis, value) => {
    if (axis === 'x') {
      setSelectedPCX(value);
    } else {
      setSelectedPCY(value);
    }
  }, []);

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header Section */}
      <div className="p-4">
        <Card className="mb-4">
          <CardHeader>
            <div className="flex flex-row items-center justify-between mb-4">
              <CardTitle className="text-2xl font-bold">
                Global Urban Embeddings Explorer
              </CardTitle>
              {(selectedCountry || selectedCity || selectedPoint) && (
                <div className="flex items-center space-x-2 bg-white px-4 py-2 rounded-lg shadow-sm border">
                  <MapPin className="h-4 w-4 text-blue-500" />
                  <span className="font-medium">
                    {selectedCountry}
                    {selectedCity && `, ${selectedCity}`}
                  </span>
                </div>
              )}
            </div>

            <div className="flex flex-row items-center justify-between space-x-4">
              <div className="min-w-[300px]">
                <LocationSelector onSelect={handleLocationSelect} />
              </div>
              <div className="flex items-center space-x-4">
                <span className="text-sm font-medium">Plot Controls:</span>
                <PCSelector 
                  selectedX={selectedPCX}
                  selectedY={selectedPCY}
                  onChange={handlePCChange}
                />
              </div>
            </div>
          </CardHeader>
        </Card>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 min-h-0 p-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
          {/* ScatterPlot Card */}
          <Card className="h-full overflow-hidden">
            <CardContent className="p-0 h-full">
              <ScatterPlot 
                onPointSelect={handlePointSelect}
                selectedPoint={selectedPoint} 
                selectedCountry={selectedCountry} 
                selectedCity={selectedCity} 
                selectedPCX={selectedPCX} 
                selectedPCY={selectedPCY} 
              />
            </CardContent>
          </Card>

          {/* Map Card */}
          <Card className="h-full overflow-hidden">
            <CardContent className="p-0 h-full">
              <MapboxMapComponent 
                selectedPoint={selectedPoint}
                onMapClick={handlePointSelect}
                allPoints={allPoints}
                selectedCity={selectedCity}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default App;