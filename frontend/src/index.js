import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import './App.css';
import MapView from './MapView';
import UMapView from './UMapView';
import polyline from '@mapbox/polyline';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-domain.com'
  : 'http://localhost:8000';

// Correct tile size: 224 pixels × 10m/pixel = 2240 meters
const TILE_SIZE_METERS = 2240;

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
  const [visibleSimilarCount, setVisibleSimilarCount] = useState(6);
  const [stats, setStats] = useState(null);

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
      const statsData = await response.json();
      setStats(statsData);
      setTotalLocations(statsData.total_locations.toLocaleString());
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

  const createTilePolygon = (centerLng, centerLat, sizeMeters) => {
    const metersPerDegreeLat = 111320;
    const metersPerDegreeLng = 111320 * Math.cos(centerLat * Math.PI / 180);

    const latOffset = (sizeMeters / 2) / metersPerDegreeLat;
    const lngOffset = (sizeMeters / 2) / metersPerDegreeLng;

    return [
      [centerLng - lngOffset, centerLat + latOffset],
      [centerLng + lngOffset, centerLat + latOffset],
      [centerLng + lngOffset, centerLat - latOffset],
      [centerLng - lngOffset, centerLat - latOffset],
      [centerLng - lngOffset, centerLat + latOffset]
    ];
  };

  const getStaticMapImage = (longitude, latitude, width = 320, height = 320, zoom = 12) => {
    if (!mapboxToken) return '';
    
    const tileBoundaryCoords = createTilePolygon(longitude, latitude, TILE_SIZE_METERS);
    const coordsForPolyline = tileBoundaryCoords.map(coord => [coord[1], coord[0]]);
    const encoded = polyline.encode(coordsForPolyline);
    const urlEncodedPolyline = encodeURIComponent(encoded);
    const tilePath = `path-3+ffffff-0.90(${urlEncodedPolyline})`;
    
    return `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/${tilePath}/${longitude},${latitude},${zoom}/${width}x${height}@2x?access_token=${mapboxToken}`;
  };

  const findSimilarLocations = async (locationId) => {
    setFindingSimilar(true);
    setShowSimilarResults(true);
    setVisibleSimilarCount(6);

    try {
      console.log(`Finding similar locations for ${locationId}`);
      
      const response = await fetch(`${API_BASE_URL}/api/similarity/${locationId}?top_k=20`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to find similar locations');
      }
      const data = await response.json();
      setSimilarResults(data.similar_locations);
    } catch (error) {
      console.error('Error finding similar locations:', error);
      setError(`Error finding similar locations: ${error.message}`);
    } finally {
      setFindingSimilar(false);
    }
  };

  const clearSelection = () => {
    setSelectedLocations(new Set());
    setCurrentSelectedLocation(null);
    setShowSimilarResults(false);
    setSimilarResults([]);
    setVisibleSimilarCount(6);
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
    
    // Automatically find similar locations when a location is selected
    if (newCurrentLocation) {
      findSimilarLocations(locationId);
    } else {
      setShowSimilarResults(false);
      setSimilarResults([]);
      setVisibleSimilarCount(6);
    }
  };

  const showMoreSimilar = () => {
    setVisibleSimilarCount(prev => Math.min(prev + 6, similarResults.length));
  };

  const handleImageClick = (location) => {
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
  };

  return (
    <div className="app">
      <header>
        <h1>🛰️ Satellite Embeddings Explorer</h1>
        <div className="controls-header">
          <div className="stats">
            <span>{totalLocations}</span> locations
            {stats && (
              <span className="embedding-stats">
                • {stats.embedding_dimension}D TerraMind Embeddings
              </span>
            )}
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
                <h4>{currentSelectedLocation?.city}, {currentSelectedLocation?.country}</h4>
                
                {/* Selected Location Image */}
                {mapboxToken && (
                  <div className="selected-image-container">
                    <div className="selected-image-square">
                      <img
                        src={getStaticMapImage(currentSelectedLocation.longitude, currentSelectedLocation.latitude, 120, 120, 11.2)}
                        alt={`${currentSelectedLocation.city} satellite view`}
                        className="selected-location-image"
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {(showSimilarResults || findingSimilar) && (
              <div className="similar-results">
                <h4>Most Similar Locations</h4>
                <div className="similarity-method-info">
                  Using TerraMind embeddings • Cosine similarity
                </div>
                
                {findingSimilar ? (
                  <div className="loading-similar">
                    <div className="spinner"></div>
                    <p>Finding similar locations...</p>
                  </div>
                ) : (
                  <>
                    <div className="similar-grid-container">
                      {similarResults.slice(0, visibleSimilarCount).map((location, index) => {
                        const similarity = (location.similarity_score * 100).toFixed(1);
                        
                        return (
                          <div 
                            key={location.id}
                            className="similar-tile"
                            onClick={() => handleImageClick(location)}
                          >
                            {mapboxToken && (
                              <img
                                src={getStaticMapImage(location.longitude, location.latitude, 160, 160, 11.3)}
                                alt={`${location.city} satellite view`}
                                className="similar-tile-image"
                                onError={(e) => {
                                  e.target.style.display = 'none';
                                }}
                              />
                            )}
                            <div className="similarity-badge">{similarity}%</div>
                            <div className="tile-text">
                              {location.city}, {location.country}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    
                    {visibleSimilarCount < similarResults.length && (
                      <button 
                        className="show-more-btn"
                        onClick={showMoreSimilar}
                      >
                        Show More ({similarResults.length - visibleSimilarCount} remaining)
                      </button>
                    )}
                  </>
                )}
              </div>
            )}

            {!currentSelectedLocation && (
              <div className="no-selection">
                <p>Click on a location to view details and find similar places.</p>
                <div className="embedding-info">
                  <h5>TerraMind Embeddings</h5>
                  <p>Using AI-powered satellite image embeddings to find visually similar urban areas based on spatial patterns, building density, and urban morphology.</p>
                </div>
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
              <button onClick={() => window.location.reload()} className="retry-btn">
                Retry
              </button>
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