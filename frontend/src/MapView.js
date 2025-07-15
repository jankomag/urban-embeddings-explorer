import React, { useRef, useEffect, useCallback, useState } from 'react';
import mapboxgl from 'mapbox-gl';

// Fallback tile size for locations without exact boundaries (224 pixels × 10m/pixel = 2240 meters)
const FALLBACK_TILE_SIZE_METERS = 2240;

const MapView = ({ locations, selectedLocations, onLocationSelect, mapboxToken }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const hoveredLocationId = useRef(null);
  const currentPopup = useRef(null);
  const [tileBoundsCache, setTileBoundsCache] = useState(new Map());

  const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? 'https://your-domain.com'
    : 'http://localhost:8000';

  // Initialize map
  useEffect(() => {
    if (!mapboxToken || map.current) return;

    mapboxgl.accessToken = mapboxToken;

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
        loadTileBoundsAndAddToMap();
      }
    });

    // Close popup when clicking on map (not on a feature)
    map.current.on('click', (e) => {
      const features = map.current.queryRenderedFeatures(e.point, {
        layers: ['tiles-fill', 'points-layer']
      });
      
      if (features.length === 0 && currentPopup.current) {
        currentPopup.current.remove();
        currentPopup.current = null;
      }
    });

    // Listen for zoom events from similarity panel
    const handleZoomToLocation = (event) => {
      const { longitude, latitude, id } = event.detail;
      zoomToLocation(longitude, latitude, id);
    };

    window.addEventListener('zoomToLocation', handleZoomToLocation);

    return () => {
      window.removeEventListener('zoomToLocation', handleZoomToLocation);
      if (currentPopup.current) {
        currentPopup.current.remove();
      }
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [mapboxToken]);

  // Add locations to map when both map and locations are ready
  useEffect(() => {
    if (map.current && map.current.isStyleLoaded() && locations.length > 0) {
      loadTileBoundsAndAddToMap();
    }
  }, [locations]);

  // Load tile bounds for all locations and add to map
  const loadTileBoundsAndAddToMap = async () => {
    if (!locations.length) return;

    console.log('Loading tile boundaries for', locations.length, 'locations...');
    
    // Get exact tile boundaries for all locations
    const boundsPromises = locations.map(async (location) => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/tile-bounds/${location.id}`);
        if (response.ok) {
          const data = await response.json();
          return {
            locationId: location.id,
            bounds: data.tile_bounds,
            hasExactBounds: data.has_exact_bounds
          };
        }
      } catch (error) {
        console.warn(`Failed to get tile bounds for location ${location.id}:`, error);
      }
      return {
        locationId: location.id,
        bounds: null,
        hasExactBounds: false
      };
    });

    const boundsResults = await Promise.all(boundsPromises);
    
    // Create cache of tile bounds
    const newCache = new Map();
    boundsResults.forEach(result => {
      newCache.set(result.locationId, result);
    });
    setTileBoundsCache(newCache);

    // Add locations to map with tile boundaries
    addLocationsToMap(newCache);
  };

  // Update styles when selection changes
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

      map.current.setPaintProperty('points-layer', 'circle-color', [
        'case',
        ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
        '#ff6b6b',
        '#4ecdc4'
      ]);
    }
  }, [selectedLocations]);

  useEffect(() => {
    updateTileStyles();
  }, [selectedLocations, updateTileStyles]);

  const createFallbackTilePolygon = (centerLng, centerLat, sizeMeters) => {
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

  const addLocationsToMap = (boundsCache) => {
    if (!map.current || !locations.length) return;

    console.log('Adding', locations.length, 'locations to map with tile boundaries...');

    // Create tile polygons using exact boundaries where available
    const tileFeatures = locations.map(location => {
      const boundsInfo = boundsCache.get(location.id);
      let tileCoords;

      if (boundsInfo && boundsInfo.bounds && boundsInfo.hasExactBounds) {
        // Use exact tile boundary coordinates
        tileCoords = boundsInfo.bounds;
        console.log(`Using exact bounds for ${location.city}: ${tileCoords.length} coordinates`);
      } else {
        // Fallback to estimated tile boundary around centroid
        tileCoords = createFallbackTilePolygon(
          location.longitude, 
          location.latitude, 
          FALLBACK_TILE_SIZE_METERS
        );
      }

      return {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [tileCoords]
        },
        properties: { 
          ...location,
          hasExactBounds: boundsInfo ? boundsInfo.hasExactBounds : false
        }
      };
    });

    // Create point features
    const pointFeatures = locations.map(location => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [location.longitude, location.latitude]
      },
      properties: { ...location }
    }));

    // Add sources
    if (map.current.getSource('tiles')) {
      map.current.getSource('tiles').setData({
        type: 'FeatureCollection',
        features: tileFeatures
      });
    } else {
      map.current.addSource('tiles', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: tileFeatures
        }
      });
    }

    if (map.current.getSource('points')) {
      map.current.getSource('points').setData({
        type: 'FeatureCollection',
        features: pointFeatures
      });
    } else {
      map.current.addSource('points', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: pointFeatures
        }
      });
    }

    // Add layers if they don't exist
    if (!map.current.getLayer('tiles-fill')) {
      addMapLayers();
      addMapEventListeners();
    }

    // Log summary
    const exactBoundsCount = tileFeatures.filter(f => f.properties.hasExactBounds).length;
    console.log(`Added ${tileFeatures.length} tiles to map:`);
    console.log(`- ${exactBoundsCount} with exact boundaries`);
    console.log(`- ${tileFeatures.length - exactBoundsCount} with estimated boundaries`);
  };

  const addMapLayers = () => {
    // Add exact bounds indicator layer first (bottom)
    map.current.addLayer({
      id: 'tiles-exact-indicator',
      type: 'line',
      source: 'tiles',
      paint: {
        'line-color': [
          'case',
          ['get', 'hasExactBounds'],
          '#10b981', // green for exact bounds
          'transparent'
        ],
        'line-width': 1,
        'line-dasharray': [2, 2],
        'line-opacity': 0.7
      }
    });

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
      onLocationSelect(locationId);
      showLocationPopup(
        [e.features[0].properties.longitude, e.features[0].properties.latitude],
        e.features[0].properties
      );
    });

    // Point interactions
    map.current.on('click', 'points-layer', (e) => {
      const locationId = e.features[0].properties.id;
      onLocationSelect(locationId);
      showLocationPopup(e.features[0].geometry.coordinates, e.features[0].properties);
    });
  };

  const zoomToLocation = (longitude, latitude, locationId) => {
    map.current.flyTo({
      center: [longitude, latitude],
      zoom: 13.3,
      duration: 1000
    });

    setTimeout(() => {
      const location = locations.find(loc => loc.id === locationId);
      if (location) {
        showLocationPopup([longitude, latitude], location);
      }
    }, 1200);
  };

  const showLocationPopup = (coordinates, properties) => {
    // Close existing popup
    if (currentPopup.current) {
      currentPopup.current.remove();
      currentPopup.current = null;
    }

    const isSelected = selectedLocations.has(properties.id);
    const boundsInfo = tileBoundsCache.get(properties.id);
    const hasExactBounds = boundsInfo ? boundsInfo.hasExactBounds : false;

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
        <div class="popup-item">
          <span class="popup-label">Boundary:</span> ${hasExactBounds ? 'Exact tile bounds' : 'Estimated bounds'}
        </div>
        ${properties.date ? `<div class="popup-item">
          <span class="popup-label">Date:</span> ${properties.date}
        </div>` : ''}
      </div>
    `;

    // Create new popup with smaller size and less intrusive styling
    currentPopup.current = new mapboxgl.Popup({
      closeButton: true,
      closeOnClick: false, // We handle this manually
      focusAfterOpen: false,
      maxWidth: '240px' // Make it slightly larger to accommodate boundary info
    })
      .setLngLat(coordinates)
      .setHTML(popupHtml)
      .addTo(map.current);

    // Clear reference when popup is closed
    currentPopup.current.on('close', () => {
      currentPopup.current = null;
    });
  };

  return <div ref={mapContainer} className="map-view" />;
};

export default MapView;