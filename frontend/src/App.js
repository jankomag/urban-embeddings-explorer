import React, { useState, useCallback, useEffect } from 'react';
import ScatterPlot from './components/ScatterPlot';
import MapboxMapComponent from './components/MapboxMapComponent';
import LocationSelector from './components/LocationSelector';

const App = () => {
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedCity, setSelectedCity] = useState('');
  const [allPoints, setAllPoints] = useState([]);

  useEffect(() => {
    // Fetch all points data when the component mounts
    const fetchAllPoints = async () => {
      try {
        const response = await fetch('http://localhost:8000/tsne_data');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        setAllPoints(result.data);
      } catch (error) {
        console.error('Error fetching all points data:', error);
      }
    };

    fetchAllPoints();
  }, []);

  const handlePointSelect = useCallback((point) => {
    setSelectedPoint(point);
    setSelectedCountry(point.country);
    setSelectedCity(''); // Deselect the city when a single point is selected
  }, []);

  const handleLocationSelect = useCallback((country, city) => {
    setSelectedCountry(country);
    setSelectedCity(city);
    setSelectedPoint(null); // Deselect the point when a city is selected
  }, []);

  const handleMapClick = useCallback((point) => {
    // Use the same handler as for scatter plot clicks
    handlePointSelect(point);
  }, [handlePointSelect]);

  return (
    <div className="App" style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh', 
      padding: '20px', 
      boxSizing: 'border-box',
      fontFamily: 'Arial, sans-serif'
    }}>
      <h1 style={{ 
        textAlign: 'center', 
        margin: '0 0 20px 0',
        color: '#333'
      }}>
        Global Urban Embeddings Explorer
      </h1>
      
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '20px' 
      }}>
        <LocationSelector onSelect={handleLocationSelect} />
      </div>

      {(selectedCountry || selectedCity || selectedPoint) && (
        <div style={{ 
          marginBottom: '20px', 
          padding: '10px', 
          backgroundColor: '#f0f0f0', 
          borderRadius: '5px' 
        }}>
          Selected: {selectedCountry}
          {selectedCity && `, ${selectedCity}`}
          {selectedPoint && !selectedCity && `, ${selectedPoint.city}`}
        </div>
      )}

      <div style={{ display: 'flex', flex: 1, gap: '20px', minHeight: 0 }}>
        <div style={{ flex: 1, minHeight: 0 }}>
          <ScatterPlot 
            onPointSelect={handlePointSelect} 
            selectedPoint={selectedPoint}
            selectedCountry={selectedCountry}
            selectedCity={selectedCity}
          />
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          <MapboxMapComponent 
            selectedPoint={selectedPoint} 
            onMapClick={handleMapClick}
            allPoints={allPoints}
            selectedCity={selectedCity}
          />
        </div>
      </div>
    </div>
  );
};

export default App;