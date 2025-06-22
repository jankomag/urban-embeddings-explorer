import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import mapboxgl from 'mapbox-gl';
import polyline from '@mapbox/polyline';
import './App.css'; // Your single CSS file

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const TILE_SIZE_METERS = 3360;

function App() {
  // State management
  const [locations, setLocations] = useState([]);
  const [selectedLocations, setSelectedLocations] = useState(new Set());
  const [currentSelectedLocation, setCurrentSelectedLocation] = useState(null);
  const [totalLocations, setTotalLocations] = useState('Loading...');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mapboxToken, setMapboxToken] = useState('');
  const [similarResults, setSimilarResults] = useState([]);
  const [showSimilarResults, setShowSimilarResults] = useState(false);
  const [findingSimilar, setFindingSimilar] = useState(false);
  const [showSimilarityPanel, setShowSimilarityPanel] = useState(false);

  // Refs
  const mapContainer = useRef(null);
  const map = useRef(null);
  const hoveredLocationId = useRef(null);

  // Initialize app
  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      // Load config first to get Mapbox token
      const config = await loadConfig();
      setMapboxToken(config.mapbox_token);

      // Initialize map
      initializeMap(config.mapbox_token);

      // Load data
      await Promise.all([loadStats(), loadLocations()]);

    } catch (error) {
      console.error('Error initializing app:', error);
      setError('Failed to initialize application. Please refresh the page.');
    }
  };

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

  const initializeMap = (token) => {
    mapboxgl.accessToken = token;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/satellite-v9',
      center: [0, 20],
      zoom: 2
    });

    map.current.addControl(new mapboxgl.NavigationControl());

    map.current.on('load', () => {
      console.log('Map loaded successfully');
      if (locations.length > 0) {
        addLocationsToMap();
      }
    });
  };

  // Add locations to map when both map and locations are ready
  useEffect(() => {
    if (map.current && map.current.isStyleLoaded() && locations.length > 0) {
      addLocationsToMap();
    }
  }, [locations]);

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

  const addLocationsToMap = () => {
    console.log('Adding locations to map...');
    
    const tileFeatures = locations.map(location => ({
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [createTilePolygon(location.longitude, location.latitude, TILE_SIZE_METERS)]
      },
      properties: {
        id: location.id,
        city: location.city,
        country: location.country,
        continent: location.continent,
        date: location.date,
        longitude: location.longitude,
        latitude: location.latitude
      }
    }));

    const pointFeatures = locations.map(location => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [location.longitude, location.latitude]
      },
      properties: {
        id: location.id,
        city: location.city,
        country: location.country,
        continent: location.continent,
        date: location.date
      }
    }));

    // Add sources
    map.current.addSource('tiles', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: tileFeatures
      }
    });

    map.current.addSource('points', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: pointFeatures
      }
    });

    // Add layers
    addMapLayers();
    addMapEventListeners();

    console.log('Successfully added all layers and event listeners');
    setLoading(false);
  };

  const addMapLayers = () => {
    map.current.addLayer({
      id: 'tiles-fill',
      type: 'fill',
      source: 'tiles',
      paint: {
        'fill-color': [
          'case',
          ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
          '#ff6b6b',
          'transparent'
        ],
        'fill-opacity': [
          'case',
          ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
          0.6,
          0
        ]
      }
    });

    map.current.addLayer({
      id: 'tiles-border',
      type: 'line',
      source: 'tiles',
      paint: {
        'line-color': [
          'case',
          ['==', ['get', 'id'], hoveredLocationId.current || -1],
          '#4ecdc4',
          ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
          '#ff6b6b',
          'rgba(255, 255, 255, 0.4)'
        ],
        'line-width': [
          'case',
          ['==', ['get', 'id'], hoveredLocationId.current || -1],
          3,
          ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
          2,
          1
        ]
      }
    });

    map.current.addLayer({
      id: 'points-layer',
      type: 'circle',
      source: 'points',
      paint: {
        'circle-color': [
          'case',
          ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
          '#ff6b6b',
          '#4ecdc4'
        ],
        'circle-radius': [
          'case',
          ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
          8,
          6
        ],
        'circle-stroke-width': 2,
        'circle-stroke-color': '#ffffff'
      }
    });
  };

  const addMapEventListeners = () => {
    // Tile interactions
    map.current.on('mouseenter', 'tiles-fill', (e) => {
      map.current.getCanvas().style.cursor = 'pointer';
      hoveredLocationId.current = e.features[0].properties.id;
      updateTileStyles();
    });

    map.current.on('mouseleave', 'tiles-fill', () => {
      map.current.getCanvas().style.cursor = '';
      hoveredLocationId.current = null;
      updateTileStyles();
    });

    map.current.on('click', 'tiles-fill', (e) => {
      const locationId = e.features[0].properties.id;
      toggleLocationSelection(locationId);
      showLocationPopup(
        [e.features[0].properties.longitude, e.features[0].properties.latitude],
        e.features[0].properties
      );
    });

    // Point interactions
    map.current.on('click', 'points-layer', (e) => {
      const locationId = e.features[0].properties.id;
      toggleLocationSelection(locationId);
      showLocationPopup(e.features[0].geometry.coordinates, e.features[0].properties);
    });
  };

  const toggleLocationSelection = (locationId) => {
    const newSelected = new Set();
    let newCurrentLocation = null;

    if (!selectedLocations.has(locationId)) {
      newSelected.add(locationId);
      newCurrentLocation = locations.find(loc => loc.id === locationId);
      setShowSimilarityPanel(true);
    } else {
      setShowSimilarityPanel(false);
    }

    setSelectedLocations(newSelected);
    setCurrentSelectedLocation(newCurrentLocation);
    setShowSimilarResults(false);
    setSimilarResults([]);
  };

  const updateTileStyles = useCallback(() => {
    if (map.current && map.current.getLayer('tiles-fill')) {
      map.current.setPaintProperty('tiles-fill', 'fill-color', [
        'case',
        ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
        '#ff6b6b',
        'transparent'
      ]);

      map.current.setPaintProperty('tiles-border', 'line-color', [
        'case',
        ['==', ['get', 'id'], hoveredLocationId.current || -1],
        '#4ecdc4',
        ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
        '#ff6b6b',
        'rgba(255, 255, 255, 0.4)'
      ]);
    }
  }, [selectedLocations]);

  useEffect(() => {
    updateTileStyles();
  }, [selectedLocations, updateTileStyles]);

  const clearSelection = () => {
    setSelectedLocations(new Set());
    setCurrentSelectedLocation(null);
    setShowSimilarityPanel(false);
    setShowSimilarResults(false);
    setSimilarResults([]);
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

  const getStaticMapImage = (longitude, latitude, width = 320, height = 320) => {
    if (!mapboxToken) return '';
    
    const tileBoundaryCoords = createTilePolygon(longitude, latitude, TILE_SIZE_METERS);
    const coordsForPolyline = tileBoundaryCoords.map(coord => [coord[1], coord[0]]);
    
    const encoded = polyline.encode(coordsForPolyline);
    const urlEncodedPolyline = encodeURIComponent(encoded);
    const tilePath = `path-3+ffffff-0.90(${urlEncodedPolyline})`;
    const zoom = 12;

    return `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/${tilePath}/${longitude},${latitude},${zoom}/${width}x${height}@2x?access_token=${mapboxToken}`;
  };

  const zoomToLocation = (longitude, latitude, locationId) => {
    map.current.flyTo({
      center: [longitude, latitude],
      zoom: 13,
      duration: 1000
    });

    setTimeout(() => {
      const location = locations.find(loc => loc.id === locationId);
      if (location) {
        showLocationPopup([longitude, latitude], location);
      }
    }, 2000);
  };

  const showLocationPopup = (coordinates, properties) => {
    const isSelected = selectedLocations.has(properties.id);
    const popupHtml = `
      <div class="popup-content">
        <h4>${properties.city} ${isSelected ? '✓' : ''}</h4>
        <div class="popup-item">
          <span class="popup-label">Country:</span> ${properties.country}
        </div>
        <div class="popup-item">
          <span class="popup-label">Continent:</span> ${properties.continent}
        </div>
        <div class="popup-item">
          <span class="popup-label">Status:</span> ${isSelected ? 'Selected' : 'Not selected'}
        </div>
        ${properties.date ? `<div class="popup-item">
          <span class="popup-label">Date:</span> ${properties.date}
        </div>` : ''}
      </div>
    `;

    new mapboxgl.Popup({
      closeButton: true,
      closeOnClick: true,
      focusAfterOpen: false
    })
      .setLngLat(coordinates)
      .setHTML(popupHtml)
      .addTo(map.current);
  };

  return (
    <div className="app">
      <header>
        <h1>🛰️ Satellite Embeddings Explorer</h1>
        <div className="controls-header">
          <div className="stats">
            <span>{totalLocations}</span> locations
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

      <div className="map-container">
        <div ref={mapContainer} className="map" />
        
        {showSimilarityPanel && (
          <div className="similarity-panel">
            <div className="similarity-header">
              <h3>Similar Locations</h3>
              <button 
                className="close-panel"
                onClick={() => setShowSimilarityPanel(false)}
              >
                ×
              </button>
            </div>
            <div className="similarity-content">
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
                        const imageUrl = getStaticMapImage(location.longitude, location.latitude, 320, 320);
                        const desc = `#${index + 1} ${location.city}, ${location.country}${location.date ? ` (${location.date})` : ''}`;
                        
                        return (
                          <div 
                            key={location.id}
                            className="similar-item" 
                            onClick={() => zoomToLocation(location.longitude, location.latitude, location.id)}
                          >
                            <div 
                              className="similar-item-image" 
                              style={{backgroundImage: `url('${imageUrl}')`}}
                            >
                              <div className="static-tile-border"></div>
                              <div className="similarity-badge">{similarity}%</div>
                              <div className="similar-item-overlay auto-height-gradient">
                                <span className="similar-item-desc">{desc}</span>
                              </div>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

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