import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import './ModernApp.css'; // Import the new modern styles
import MapView from './MapView';
import HighPerformanceUMapView from './HighPerformanceUMapView';
import polyline from '@mapbox/polyline';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-domain.com'
  : 'http://localhost:8000';

const TILE_SIZE_METERS = 2240;

function ModernApp() {
  // State management
  const [locations, setLocations] = useState([]);
  const [selectedLocations, setSelectedLocations] = useState(new Set());
  const [currentSelectedLocation, setCurrentSelectedLocation] = useState(null);
  const [similarResults, setSimilarResults] = useState([]);
  const [findingSimilar, setFindingSimilar] = useState(false);
  const [totalLocations, setTotalLocations] = useState('Loading...');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mapboxToken, setMapboxToken] = useState('');
  const [visibleSimilarCount, setVisibleSimilarCount] = useState(8);
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
        setError('Failed to initialize application');
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
      setTotalLocations('Error');
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

  const getStaticMapImage = (longitude, latitude, width = 160, height = 160, zoom = 12) => {
    if (!mapboxToken || !longitude || !latitude) {
      console.log('Missing mapbox token or coordinates:', { mapboxToken: !!mapboxToken, longitude, latitude });
      return '';
    }
    
    try {
      const tileBoundaryCoords = createTilePolygon(longitude, latitude, TILE_SIZE_METERS);
      const coordsForPolyline = tileBoundaryCoords.map(coord => [coord[1], coord[0]]);
      const encoded = polyline.encode(coordsForPolyline);
      const urlEncodedPolyline = encodeURIComponent(encoded);
      const tilePath = `path-3+ffffff-0.90(${urlEncodedPolyline})`;
      
      const imageUrl = `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/${tilePath}/${longitude},${latitude},${zoom}/${width}x${height}@2x?access_token=${mapboxToken}`;
      console.log('Generated image URL for', longitude, latitude); // Debug log
      return imageUrl;
    } catch (error) {
      console.error('Error generating static map image:', error);
      return '';
    }
  };

  const findSimilarLocations = async (locationId) => {
    console.log('Finding similar locations for:', locationId); // Debug log
    setFindingSimilar(true);
    setVisibleSimilarCount(8);

    try {
      const response = await fetch(`${API_BASE_URL}/api/similarity/${locationId}?top_k=20`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to find similar locations');
      }
      const data = await response.json();
      console.log('Found similar locations:', data.similar_locations.length); // Debug log
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
    setSimilarResults([]);
    setVisibleSimilarCount(8);
  };

  const handleLocationSelect = (locationId) => {
    console.log('Selecting location:', locationId); // Debug log
    
    const newSelected = new Set();
    let newCurrentLocation = null;

    // Always select the new location (don't toggle)
    newSelected.add(locationId);
    newCurrentLocation = locations.find(loc => loc.id === locationId);
    
    console.log('Found location:', newCurrentLocation); // Debug log

    setSelectedLocations(newSelected);
    setCurrentSelectedLocation(newCurrentLocation);
    
    if (newCurrentLocation) {
      findSimilarLocations(locationId);
    } else {
      setSimilarResults([]);
      setVisibleSimilarCount(8);
    }
  };

  const handleSimilarClick = (location) => {
    // Select the similar location
    handleLocationSelect(location.id);
    
    // Also trigger map/umap interactions
    window.dispatchEvent(new CustomEvent('zoomToLocation', {
      detail: { longitude: location.longitude, latitude: location.latitude, id: location.id }
    }));
    window.dispatchEvent(new CustomEvent('highlightUmapPoint', {
      detail: { locationId: location.id }
    }));
  };

  const showMoreSimilar = () => {
    setVisibleSimilarCount(prev => Math.min(prev + 8, similarResults.length));
  };

  if (loading) {
    return (
      <div className="app">
        <div className="loading-state" style={{ height: '100vh', justifyContent: 'center' }}>
          <div className="spinner"></div>
          <div className="loading-text">Loading satellite embeddings...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app">
        <div className="loading-state" style={{ height: '100vh', justifyContent: 'center', color: '#ef4444' }}>
          <div>⚠️ {error}</div>
          <button onClick={() => window.location.reload()} style={{ marginTop: '12px', padding: '8px 16px' }}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Compact Header */}
      <header className="header">
        <div className="header-content">
          <div>
            <h1>🛰️ Satellite Embeddings Explorer</h1>
            <div className="header-stats">
              {totalLocations} locations • {stats?.embedding_dimension}D TerraMind embeddings
            </div>
          </div>
          <div className="selection-info">
            <span>{selectedLocations.size} selected</span>
            <button 
              className="clear-btn" 
              disabled={selectedLocations.size === 0}
              onClick={clearSelection}
            >
              Clear
            </button>
          </div>
        </div>
      </header>

      {/* Three-Panel Layout */}
      <div className="main-container">
        {/* Map View */}
        <div className="viz-panel">
          <div className="viz-header">
            <h3>Map View</h3>
            <p>Satellite imagery with location tiles</p>
          </div>
          <div className="viz-content">
            <MapView
              locations={locations}
              selectedLocations={selectedLocations}
              onLocationSelect={handleLocationSelect}
              mapboxToken={mapboxToken}
            />
          </div>
        </div>

        {/* UMAP View */}
        <div className="viz-panel">
          <div className="viz-header">
            <h3>UMAP Embedding Space</h3>
            <p>Canvas-accelerated visualization • Drag to pan, scroll to zoom</p>
          </div>
          <div className="viz-content">
            <HighPerformanceUMapView
              locations={locations}
              selectedLocations={selectedLocations}
              onLocationSelect={handleLocationSelect}
            />
          </div>
        </div>

        {/* Analysis Panel */}
        <div className="analysis-panel">
          <div className="analysis-header">
            <h3>Analysis</h3>
          </div>
          <div className="analysis-content">
            {currentSelectedLocation ? (
              <>
                {/* Selected Location Card */}
                <div className="selected-card">
                  <h4>{currentSelectedLocation.city}, {currentSelectedLocation.country}</h4>
                  {mapboxToken && currentSelectedLocation && (
                    <img
                      src={getStaticMapImage(currentSelectedLocation.longitude, currentSelectedLocation.latitude, 160, 160, 11.2)}
                      alt={`${currentSelectedLocation.city} satellite view`}
                      className="selected-image"
                      onError={(e) => {
                        console.log('Failed to load selected image for:', currentSelectedLocation.city);
                        e.target.style.display = 'none';
                      }}
                      onLoad={(e) => {
                        console.log('Successfully loaded selected image for:', currentSelectedLocation.city);
                      }}
                    />
                  )}
                  <div className="selected-meta">
                    {currentSelectedLocation.continent}
                    {currentSelectedLocation.date && ` • ${currentSelectedLocation.date}`}
                  </div>
                </div>

                {/* Similar Results */}
                <div className="similar-section">
                  <div className="similar-header">
                    <h4>Similar Locations</h4>
                    {similarResults.length > 0 && (
                      <div className="similar-count">
                        {Math.min(visibleSimilarCount, similarResults.length)} of {similarResults.length}
                      </div>
                    )}
                  </div>
                  
                  <div className="similar-method">
                    TerraMind embeddings • Cosine similarity
                  </div>
                  
                  {findingSimilar ? (
                    <div className="loading-state">
                      <div className="spinner"></div>
                      <div className="loading-text">Finding similar locations...</div>
                    </div>
                  ) : (
                    <>
                      <div className="similarity-grid">
                        {similarResults.slice(0, visibleSimilarCount).map((location, index) => {
                          const similarity = (location.similarity_score * 100).toFixed(1);
                          const imageUrl = getStaticMapImage(location.longitude, location.latitude, 120, 120, 11.3);
                          
                          return (
                            <div 
                              key={`${location.id}-${index}`}
                              className="similarity-tile"
                              onClick={() => handleSimilarClick(location)}
                              title={`${location.city}, ${location.country} - ${similarity}% similar`}
                            >
                              {mapboxToken && imageUrl ? (
                                <img
                                  src={imageUrl}
                                  alt={`${location.city} satellite view`}
                                  className="similarity-image"
                                  onError={(e) => {
                                    console.log('Failed to load similar image for:', location.city);
                                    e.target.parentElement.innerHTML = `
                                      <div class="loading-placeholder">
                                        <div>${location.city}</div>
                                      </div>
                                      <div class="similarity-score">${similarity}%</div>
                                      <div class="similarity-label">${location.city}, ${location.country}</div>
                                    `;
                                  }}
                                  onLoad={(e) => {
                                    console.log('Successfully loaded similar image for:', location.city);
                                  }}
                                />
                              ) : (
                                <div className="loading-placeholder">
                                  {location.city}
                                </div>
                              )}
                              <div className="similarity-score">{similarity}%</div>
                              <div className="similarity-label">
                                {location.city}, {location.country}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      
                      {visibleSimilarCount < similarResults.length && (
                        <button 
                          className="show-more"
                          onClick={showMoreSimilar}
                        >
                          Show {Math.min(8, similarResults.length - visibleSimilarCount)} more
                        </button>
                      )}
                    </>
                  )}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p>Click any location to explore similarities</p>
                <div className="empty-info">
                  <h5>AI-Powered Analysis</h5>
                  <p>Using TerraMind satellite embeddings to find visually similar urban areas based on spatial patterns and building density.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ModernApp />
  </React.StrictMode>
);