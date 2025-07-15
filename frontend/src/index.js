import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import './ModernApp.css';
import MapView from './MapView';
import HighPerformanceUMapView from './HighPerformanceUMapView';
import polyline from '@mapbox/polyline';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-domain.com'
  : 'http://localhost:8000';

const TILE_SIZE_METERS = 2240;

// Enhanced Similarity Panel Component
function EnhancedSimilarityPanel({ 
  selectedLocation, 
  primarySelectionId, 
  similarResults, 
  setSimilarResults,
  findingSimilar, 
  setFindingSimilar,
  onNavigationClick,
  mapboxToken,
  locations
}) {
  const [similarityMethods, setSimilarityMethods] = useState([]);
  const [selectedMethod, setSelectedMethod] = useState('attention_weighted');
  const [methodsLoading, setMethodsLoading] = useState(true);
  const [currentMethodConfig, setCurrentMethodConfig] = useState(null);
  const [paginationInfo, setPaginationInfo] = useState(null);
  const [loadingMoreResults, setLoadingMoreResults] = useState(false);

  // Load similarity methods on component mount
  useEffect(() => {
    const loadSimilarityMethods = async () => {
      try {
        setMethodsLoading(true);
        const response = await fetch(`${API_BASE_URL}/api/similarity-methods`);
        if (response.ok) {
          const data = await response.json();
          setSimilarityMethods(data.methods);
          setSelectedMethod(data.recommended);
          
          // Find the recommended method config
          const recommendedMethod = data.methods.find(m => m.id === data.recommended);
          if (recommendedMethod) {
            setCurrentMethodConfig(recommendedMethod.config);
          }
        } else {
          console.error('Failed to load similarity methods');
        }
      } catch (error) {
        console.error('Error loading similarity methods:', error);
      } finally {
        setMethodsLoading(false);
      }
    };

    loadSimilarityMethods();
  }, []);

  // Find similar locations with selected method (initial load)
  const findSimilarLocations = async (locationId, method, offset = 0, limit = 6) => {
    console.log('Finding similar locations for:', locationId, 'using method:', method, 'offset:', offset);
    
    if (offset === 0) {
      setFindingSimilar(true);
      setSimilarResults([]); // Clear previous results for new search
      setPaginationInfo(null);
    } else {
      setLoadingMoreResults(true);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/similarity/${locationId}?offset=${offset}&limit=${limit}&method=${method}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to find similar locations');
      }
      const data = await response.json();
      console.log('Found similar locations:', data.similar_locations.length, 'offset:', offset);
      
      if (offset === 0) {
        // Initial load - replace results
        setSimilarResults(data.similar_locations);
      } else {
        // Load more - append results
        setSimilarResults(prev => [...prev, ...data.similar_locations]);
      }
      
      setPaginationInfo(data.pagination);
      
      // Update current method config
      if (data.method_config) {
        setCurrentMethodConfig(data.method_config);
      }
    } catch (error) {
      console.error('Error finding similar locations:', error);
      // Show error state but don't clear existing results if loading more
      if (offset === 0) {
        setSimilarResults([]);
        setPaginationInfo(null);
      }
    } finally {
      if (offset === 0) {
        setFindingSimilar(false);
      } else {
        setLoadingMoreResults(false);
      }
    }
  };

  // Load more similar locations
  const loadMoreSimilarLocations = () => {
    if (paginationInfo && paginationInfo.has_more && primarySelectionId && selectedMethod) {
      findSimilarLocations(primarySelectionId, selectedMethod, paginationInfo.next_offset, 6);
    }
  };

  // Handle method change
  const handleMethodChange = (newMethod) => {
    setSelectedMethod(newMethod);
    
    // Update method config display
    const methodInfo = similarityMethods.find(m => m.id === newMethod);
    if (methodInfo) {
      setCurrentMethodConfig(methodInfo.config);
    }
    
    // Re-run similarity search if we have a primary selection (start fresh)
    if (primarySelectionId) {
      findSimilarLocations(primarySelectionId, newMethod, 0, 6);
    }
  };

  // Get method indicator color based on speed
  const getMethodColor = (speed) => {
    switch (speed) {
      case 'Fast': return '#10b981'; // green
      case 'Medium': return '#f59e0b'; // yellow
      case 'Slow': return '#ef4444'; // red
      default: return '#6b7280'; // gray
    }
  };

  const getStaticMapImage = (longitude, latitude, width = 120, height = 120, zoom = 11.3) => {
    if (!mapboxToken || !longitude || !latitude) {
      return '';
    }
    
    try {
      const imageUrl = `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/${longitude},${latitude},${zoom}/${width}x${height}@2x?access_token=${mapboxToken}`;
      return imageUrl;
    } catch (error) {
      console.error('Error generating static map image:', error);
      return '';
    }
  };

  // Effect to find similar locations when primary selection changes
  useEffect(() => {
    if (primarySelectionId && selectedMethod && !methodsLoading) {
      findSimilarLocations(primarySelectionId, selectedMethod, 0, 6);
    }
  }, [primarySelectionId, selectedMethod, methodsLoading]);

  if (!selectedLocation && !primarySelectionId) {
    return (
      <div className="empty-state">
        <p>Click any location to explore similarities</p>
        <div className="empty-info">
          <h5>AI-Powered Analysis</h5>
          <p>Using TerraMind satellite embeddings to find visually similar urban areas based on spatial patterns and building density.</p>
          {!methodsLoading && similarityMethods.length > 0 && (
            <div className="method-preview">
              <small>Available methods: {similarityMethods.length}</small>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Selected Location Card */}
      {selectedLocation && (
        <div className="selected-card">
          <h4>{selectedLocation.city}, {selectedLocation.country}</h4>
          {mapboxToken && selectedLocation && (
            <img
              src={getStaticMapImage(selectedLocation.longitude, selectedLocation.latitude, 160, 160, 11.2)}
              alt={`${selectedLocation.city} satellite view`}
              className="selected-image"
              onError={(e) => {
                e.target.style.display = 'none';
              }}
            />
          )}
          <div className="selected-meta">
            {selectedLocation.continent}
            {selectedLocation.date && ` • ${selectedLocation.date}`}
          </div>
        </div>
      )}

      {/* Similar Results - only show if we have a primary selection */}
      {primarySelectionId && (
        <div className="similar-section">
          <div className="similar-header">
            <h4>
              Similar to {locations.find(l => l.id === primarySelectionId)?.city || 'Selected Location'}
            </h4>
            {paginationInfo && (
              <div className="similar-count">
                {similarResults.length} of {paginationInfo.total_results}
              </div>
            )}
          </div>
          
          {/* Similarity Method Selector */}
          {!methodsLoading && similarityMethods.length > 0 && (
            <div className="method-selector">
              <label className="method-label">Similarity Method:</label>
              <select 
                value={selectedMethod} 
                onChange={(e) => handleMethodChange(e.target.value)}
                className="method-dropdown"
                disabled={findingSimilar}
              >
                {similarityMethods.map((method) => (
                  <option key={method.id} value={method.id}>
                    {method.config.name}
                  </option>
                ))}
              </select>
              
              {/* Method Info Display */}
              {currentMethodConfig && (
                <div className="method-info">
                  <div className="method-badges">
                    <span 
                      className="method-badge speed-badge"
                      style={{ backgroundColor: getMethodColor(currentMethodConfig.speed) }}
                    >
                      {currentMethodConfig.speed}
                    </span>
                    <span className="method-badge quality-badge">
                      {currentMethodConfig.quality} Quality
                    </span>
                  </div>
                  <p className="method-description">
                    {currentMethodConfig.description}
                  </p>
                </div>
              )}
            </div>
          )}
          
          {findingSimilar ? (
            <div className="loading-state">
              <div className="spinner"></div>
              <div className="loading-text">
                {currentMethodConfig?.speed === 'Slow' 
                  ? `Computing visual similarities using ${currentMethodConfig?.name}... This may take a moment for best quality results.`
                  : `Finding similar locations using ${currentMethodConfig?.name || selectedMethod}...`
                }
              </div>
              {currentMethodConfig?.speed === 'Slow' && (
                <div className="loading-subtext">
                  💡 High-quality methods analyze all 196 image patches for the best visual similarity
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="similarity-grid">
                {similarResults.map((location, index) => {
                  const similarity = (location.similarity_score * 100).toFixed(1);
                  const imageUrl = getStaticMapImage(location.longitude, location.latitude, 120, 120, 11.3);
                  
                  return (
                    <div 
                      key={`${location.id}-${index}`}
                      className="similarity-tile"
                      onClick={() => onNavigationClick(location)}
                      title={`${location.city}, ${location.country} - ${similarity}% similar`}
                    >
                      {mapboxToken && imageUrl ? (
                        <img
                          src={imageUrl}
                          alt={`${location.city} satellite view`}
                          className="similarity-image"
                          onError={(e) => {
                            e.target.parentElement.innerHTML = `
                              <div class="loading-placeholder">
                                <div>${location.city}</div>
                              </div>
                              <div class="similarity-score">${similarity}%</div>
                              <div class="similarity-label">${location.city}, ${location.country}</div>
                            `;
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
              
              {/* Load More Button */}
              {paginationInfo && paginationInfo.has_more && (
                <button 
                  className="show-more"
                  onClick={loadMoreSimilarLocations}
                  disabled={loadingMoreResults}
                >
                  {loadingMoreResults ? (
                    <>
                      <div className="button-spinner"></div>
                      Loading more...
                    </>
                  ) : (
                    <>Show {Math.min(6, paginationInfo.total_results - similarResults.length)} more</>
                  )}
                </button>
              )}
              
              {/* Results Summary */}
              {paginationInfo && !paginationInfo.has_more && similarResults.length > 6 && (
                <div className="results-summary">
                  Showing all {similarResults.length} results using {currentMethodConfig?.name}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </>
  );
}

function ModernApp() {
  // State management
  const [locations, setLocations] = useState([]);
  const [selectedLocations, setSelectedLocations] = useState(new Set());
  const [currentSelectedLocation, setCurrentSelectedLocation] = useState(null);
  const [primarySelectionId, setPrimarySelectionId] = useState(null);
  const [similarResults, setSimilarResults] = useState([]);
  const [findingSimilar, setFindingSimilar] = useState(false);
  const [totalLocations, setTotalLocations] = useState('Loading...');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mapboxToken, setMapboxToken] = useState('');
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
      console.log('Generated image URL for', longitude, latitude);
      return imageUrl;
    } catch (error) {
      console.error('Error generating static map image:', error);
      return '';
    }
  };

  const clearSelection = () => {
    setSelectedLocations(new Set());
    setCurrentSelectedLocation(null);
    setPrimarySelectionId(null);
    setSimilarResults([]);
  };

  // Primary selection - triggers similarity search and UMAP centering
  const handlePrimarySelection = (locationId) => {
    console.log('Primary selection:', locationId);
    
    const newSelected = new Set();
    let newCurrentLocation = null;

    // Always select the new location
    newSelected.add(locationId);
    newCurrentLocation = locations.find(loc => loc.id === locationId);
    
    setSelectedLocations(newSelected);
    setCurrentSelectedLocation(newCurrentLocation);
    setPrimarySelectionId(locationId);
    
    if (newCurrentLocation) {
      // Clear previous results - the similarity panel will handle finding new ones
      setSimilarResults([]);
      
      // Center UMAP on this point
      window.dispatchEvent(new CustomEvent('zoomToUmapPoint', {
        detail: { locationId: locationId }
      }));
    } else {
      setSimilarResults([]);
    }
  };

  // Navigation only - just moves views without changing primary selection
  const handleNavigationClick = (location) => {
    console.log('Navigation to:', location.city, location.country);
    
    // Update visual selection but don't change the primary selection for similarities
    const newSelected = new Set();
    newSelected.add(location.id);
    setSelectedLocations(newSelected);
    setCurrentSelectedLocation(location);
    // Note: Don't change primarySelectionId or clear similarResults
    
    // Move both map and UMAP to show this location
    window.dispatchEvent(new CustomEvent('zoomToLocation', {
      detail: { longitude: location.longitude, latitude: location.latitude, id: location.id }
    }));
    window.dispatchEvent(new CustomEvent('zoomToUmapPoint', {
      detail: { locationId: location.id }
    }));
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
      {/* Enhanced Header */}
      <header className="header">
        <div className="header-content">
          <div>
            <h1>🛰️ Enhanced Satellite Embeddings Explorer</h1>
            <div className="header-stats">
              {totalLocations} locations • {stats?.embedding_dimension}D TerraMind embeddings
              {stats?.available_similarity_methods && ` • ${stats.available_similarity_methods} similarity methods`}
              {stats?.locations_with_full_patches && ` • ${stats.locations_with_full_patches} with full patches`}
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
              onLocationSelect={handlePrimarySelection}
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
              onLocationSelect={handlePrimarySelection}
            />
          </div>
        </div>

        {/* Enhanced Analysis Panel */}
        <div className="analysis-panel">
          <div className="analysis-header">
            <h3>Enhanced Analysis</h3>
          </div>
          <div className="analysis-content">
            <EnhancedSimilarityPanel
              selectedLocation={currentSelectedLocation}
              primarySelectionId={primarySelectionId}
              similarResults={similarResults}
              setSimilarResults={setSimilarResults}
              findingSimilar={findingSimilar}
              setFindingSimilar={setFindingSimilar}
              onNavigationClick={handleNavigationClick}
              mapboxToken={mapboxToken}
              locations={locations}
            />
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