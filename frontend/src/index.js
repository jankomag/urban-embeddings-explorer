import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import './App.css';
import MapView from './MapView';
import UMapView from './UMapView';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-domain.com'
  : 'http://localhost:8000';

function App() {
  // State management
  const [locations, setLocations] = useState([]);
  const [selectedLocations, setSelectedLocations] = useState(new Set());
  const [currentSelectedLocation, setCurrentSelectedLocation] = useState(null);
  const [similarResults, setSimilarResults] = useState([]);
  const [findingSimilar, setFindingSimilar] = useState(false);
  const [showSimilarResults, setShowSimilarResults] = useState(false);
  const [totalLocations, setTotalLocations] = useState('Loading...');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mapboxToken, setMapboxToken] = useState('');
  const [showUMap, setShowUMap] = useState(false);

  // Initialize app
  useEffect(() => {
    const initializeApp = async () => {
      try {
        setLoading(true);
        const config = await loadConfig();
        setMapboxToken(config.mapbox_token);
        
        await Promise.all([
          loadStats(),
          loadLocations()
        ]);
      } catch (error) {
        console.error('Error initializing app:', error);
        setError('Failed to initialize application. Please refresh the page.');
      } finally {
        setLoading(false);
      }
    };

    initializeApp();
  }, []);

  const loadConfig = async () => {
    const response = await fetch(`${API_BASE_URL}/api/config`);
    if (!response.ok) throw new Error('Failed to load config');
    return response.json();
  };

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/stats`);
      if (!response.ok) throw new Error('Failed to load stats');
      const stats = await response.json();
      setTotalLocations(stats.total_locations.toLocaleString());
    } catch (error) {
      console.error('Error loading stats:', error);
      setTotalLocations('Error loading');
    }
  };

  const loadLocations = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/locations`);
      if (!response.ok) throw new Error('Failed to load locations');
      const locationData = await response.json();
      setLocations(locationData);
      console.log(`Loaded ${locationData.length} locations`);
    } catch (error) {
      console.error('Error loading locations:', error);
      setError('Failed to load locations');
    }
  };

  const findSimilarLocations = async () => {
    if (!currentSelectedLocation) return;

    setFindingSimilar(true);
    setShowSimilarResults(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/similarity/${currentSelectedLocation.id}`);
      if (!response.ok) throw new Error('Failed to find similar locations');
      const data = await response.json();
      setSimilarResults(data.similar_locations);
    } catch (error) {
      console.error('Error finding similar locations:', error);
      setError('Error finding similar locations');
    } finally {
      setFindingSimilar(false);
    }
  };

  const clearSelection = () => {
    setSelectedLocations(new Set());
    setCurrentSelectedLocation(null);
    setShowSimilarResults(false);
    setSimilarResults([]);
  };

  const handleLocationSelect = (locationId) => {
    const newSelected = new Set();
    let newCurrentLocation = null;

    if (!selectedLocations.has(locationId)) {
      newSelected.add(locationId);
      newCurrentLocation = locations.find(loc => loc.id === locationId);
    }

    setSelectedLocations(newSelected);
    setCurrentSelectedLocation(newCurrentLocation);
    setShowSimilarResults(false);
    setSimilarResults([]);
  };

  return (
    <div className="app">
      <header>
        <h1>🛰️ Satellite Embeddings Explorer</h1>
        <div className="controls-header">
          <div className="stats">
            <span>{totalLocations}</span> locations
          </div>
          <div className="view-controls">
            <button 
              className={`view-toggle ${!showUMap ? 'active' : ''}`}
              onClick={() => setShowUMap(false)}
            >
              🗺️ Map View
            </button>
            <button 
              className={`view-toggle ${showUMap ? 'active' : ''}`}
              onClick={() => setShowUMap(true)}
            >
              📊 UMAP View
            </button>
          </div>
          <div className="selection-info">
            <span>{selectedLocations.size}</span> selected
            <button 
              className="clear-selection" 
              disabled={selectedLocations.size === 0}
              onClick={clearSelection}
            >
              Clear Selection
            </button>
          </div>
        </div>
      </header>

      <div className="main-content">
        {/* Fixed Left Panel */}
        <div className="left-panel">
          <div className="panel-header">
            <h3>Analysis Panel</h3>
          </div>
          
          <div className="panel-content">
            {currentSelectedLocation && (
              <div className="target-location">
                <h4>Selected Location</h4>
                <div className="target-info">
                  <div><strong>{currentSelectedLocation?.city}</strong></div>
                  <div>{currentSelectedLocation?.country}, {currentSelectedLocation?.continent}</div>
                  {currentSelectedLocation?.date && <div>Date: {currentSelectedLocation.date}</div>}
                </div>
                <button 
                  className="find-similar-btn"
                  onClick={findSimilarLocations}
                  disabled={findingSimilar}
                >
                  {findingSimilar ? 'Finding Similar...' : 'Find Similar Locations'}
                </button>
              </div>
            )}
            
            {showSimilarResults && (
              <div className="similar-results">
                <h4>Most Similar Locations</h4>
                <div className="similar-list">
                  {findingSimilar ? (
                    <div className="loading-similar">
                      <div className="spinner"></div>
                      <p>Analyzing embeddings...</p>
                    </div>
                  ) : (
                    similarResults.map((location, index) => {
                      const similarity = (location.similarity_score * 100).toFixed(1);
                      const desc = `#${index + 1} ${location.city}, ${location.country}${location.date ? ` (${location.date})` : ''}`;
                      
                      return (
                        <div 
                          key={location.id}
                          className="similar-item-text" 
                          onClick={() => {
                            if (!showUMap) {
                              // Handle map zoom in MapView component
                              window.dispatchEvent(new CustomEvent('zoomToLocation', {
                                detail: { longitude: location.longitude, latitude: location.latitude, id: location.id }
                              }));
                            } else {
                              // Handle UMAP highlight
                              window.dispatchEvent(new CustomEvent('highlightUmapPoint', {
                                detail: { locationId: location.id }
                              }));
                            }
                          }}
                        >
                          <div className="similarity-badge">{similarity}%</div>
                          <div className="location-desc">{desc}</div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            )}

            {!currentSelectedLocation && (
              <div className="no-selection">
                <p>Click on a location to view details and find similar places.</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Content Area */}
        <div className="content-area">
          {loading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <p>Loading locations...</p>
            </div>
          )}

          {error && (
            <div className="error-indicator">
              <p>{error}</p>
            </div>
          )}

          {!loading && !error && (
            <>
              {!showUMap ? (
                <MapView
                  locations={locations}
                  selectedLocations={selectedLocations}
                  onLocationSelect={handleLocationSelect}
                  mapboxToken={mapboxToken}
                />
              ) : (
                <UMapView
                  locations={locations}
                  selectedLocations={selectedLocations}
                  onLocationSelect={handleLocationSelect}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);