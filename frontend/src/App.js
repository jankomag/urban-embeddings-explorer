import React, { useState, useCallback, useEffect } from 'react';
import ScatterPlot from './components/ScatterPlot';
import MapboxMapComponent from './components/MapboxMapComponent';
import LocationSelector from './components/LocationSelector';
import CustomUMAPControls from './components/CustomUMAPControls';
import DimensionSelector from './components/DimensionSelector';

const App = () => {
  // State for selected points and locations
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedCity, setSelectedCity] = useState('');
  const [allPoints, setAllPoints] = useState([]);

  // State for UMAP controls
  const [customUMAPData, setCustomUMAPData] = useState(null);
  const [showUMAPControls, setShowUMAPControls] = useState(false);
  const [isUMAPLoading, setIsUMAPLoading] = useState(false);

  // State for visualization mode and dimension selection
  const [visualizationMode, setVisualizationMode] = useState('umap');
  const [dimensionX, setDimensionX] = useState(0);
  const [dimensionY, setDimensionY] = useState(1);
  const [maxDimensions, setMaxDimensions] = useState(0);

  // Fetch initial data
  useEffect(() => {
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

    const fetchDimensions = async () => {
      try {
        const response = await fetch('http://localhost:8000/embedding_dimensions');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        setMaxDimensions(result.dimensions);
      } catch (error) {
        console.error('Error fetching dimensions:', error);
      }
    };

    fetchAllPoints();
    fetchDimensions();
  }, []);

  // Transform data based on visualization mode
  const transformData = useCallback((data) => {
    if (!data) return [];
    
    if (visualizationMode === 'umap') {
      if (customUMAPData) {
        return data.map((point, index) => ({
          ...point,
          x: customUMAPData[index][0],
          y: customUMAPData[index][1]
        }));
      }
      return data;
    } else {
      return data.map(point => ({
        ...point,
        x: point.original_embeddings[dimensionX],
        y: point.original_embeddings[dimensionY]
      }));
    }
  }, [visualizationMode, dimensionX, dimensionY, customUMAPData]);

  // Event handlers
  const handlePointSelect = useCallback((point) => {
    setSelectedPoint(point);
    setSelectedCountry(point.country);
    setSelectedCity('');
  }, []);

  const handleLocationSelect = useCallback((country, city) => {
    setSelectedCountry(country);
    setSelectedCity(city);
    setSelectedPoint(null);
  }, []);

  const handleMapClick = useCallback((point) => {
    handlePointSelect(point);
  }, [handlePointSelect]);

  const handleNewUMAPData = useCallback((newData) => {
    setCustomUMAPData(newData);
  }, []);

  const handleResetUMAP = useCallback(() => {
    setCustomUMAPData(null);
  }, []);

  const handleVisualizationModeChange = useCallback((mode) => {
    setVisualizationMode(mode);
    setCustomUMAPData(null); // Reset UMAP data when changing modes
  }, []);

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
        alignItems: 'start',
        marginBottom: '20px',
        gap: '20px'
      }}>
        <LocationSelector onSelect={handleLocationSelect} />
        <DimensionSelector
          mode={visualizationMode}
          setMode={handleVisualizationModeChange}
          dimensionX={dimensionX}
          setDimensionX={setDimensionX}
          dimensionY={dimensionY}
          setDimensionY={setDimensionY}
          maxDimensions={maxDimensions}
          disabled={isUMAPLoading}
        />
        {visualizationMode === 'umap' && (
          <button
            onClick={() => setShowUMAPControls(!showUMAPControls)}
            style={{
              padding: '8px 16px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {showUMAPControls ? 'Hide UMAP Controls' : 'Show UMAP Controls'}
          </button>
        )}
      </div>

      {showUMAPControls && visualizationMode === 'umap' && (
        <div style={{ marginBottom: '20px' }}>
          <CustomUMAPControls
            onNewUMAPData={handleNewUMAPData}
            onReset={handleResetUMAP}
            onLoadingChange={setIsUMAPLoading}
          />
        </div>
      )}

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

      <div style={{ 
        display: 'flex', 
        flex: 1, 
        gap: '20px', 
        minHeight: 0,
        position: 'relative'
      }}>
        <div style={{ flex: 1, minHeight: 0 }}>
          <div style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            zIndex: 1,
            backgroundColor: 'rgba(255, 255, 255, 0.8)',
            padding: '5px 10px',
            borderRadius: '4px',
            fontSize: '14px'
          }}>
            {visualizationMode === 'umap' 
              ? (customUMAPData ? 'Custom UMAP' : 'Pre-computed UMAP')
              : `Original Embeddings (Dimensions ${dimensionX + 1} vs ${dimensionY + 1})`
            }
          </div>
          <ScatterPlot
            onPointSelect={handlePointSelect}
            selectedPoint={selectedPoint}
            selectedCountry={selectedCountry}
            selectedCity={selectedCity}
            data={transformData(allPoints)}
            isLoading={isUMAPLoading}
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