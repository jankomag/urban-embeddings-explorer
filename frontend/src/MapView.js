import React, { useRef, useEffect, useCallback, useState } from 'react';
import mapboxgl from 'mapbox-gl';

// Fallback tile size for locations without exact boundaries
const FALLBACK_TILE_SIZE_METERS = 2240;

const MapView = ({ locations, selectedLocations, cityFilteredLocations, onLocationSelect, mapboxToken }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const hoveredLocationId = useRef(null);
  const currentPopup = useRef(null);
  const [boundsStats, setBoundsStats] = useState({ exact: 0, fallback: 0 });

  const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? process.env.REACT_APP_API_URL || 'http://localhost:8000'
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
      if (locations.length > 0) {
        addLocationsToMapWithBounds();
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

    // Listen for zoom events
    const handleZoomToLocation = (event) => {
      const { longitude, latitude, id } = event.detail;
      zoomToLocation(longitude, latitude, id);
    };

    const handleZoomToBbox = (event) => {
      const { bbox } = event.detail;
      zoomToBbox(bbox);
    };

    window.addEventListener('zoomToLocation', handleZoomToLocation);
    window.addEventListener('zoomToBbox', handleZoomToBbox);

    return () => {
      window.removeEventListener('zoomToLocation', handleZoomToLocation);
      window.removeEventListener('zoomToBbox', handleZoomToBbox);
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
      addLocationsToMapWithBounds();
    }
  }, [locations]);

  const addLocationsToMapWithBounds = () => {
    if (!locations.length) return;
    
    // Track bounds statistics
    let exactBoundsCount = 0;
    let fallbackBoundsCount = 0;
    
    // Create tile features with exact or fallback bounds
    const tileFeatures = locations.map(location => {
      let tileCoords;
      let hasExactBounds = false;
      
      // Check if location has exact tile bounds
      if (location.tile_bounds && location.has_exact_bounds && Array.isArray(location.tile_bounds)) {
        // Use exact bounds from the data
        tileCoords = location.tile_bounds;
        hasExactBounds = true;
        exactBoundsCount++;
      } else {
        // Use fallback bounds around centroid
        tileCoords = createFallbackTilePolygon(
          location.longitude, 
          location.latitude, 
          FALLBACK_TILE_SIZE_METERS
        );
        fallbackBoundsCount++;
      }

      return {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [tileCoords]
        },
        properties: { 
          ...location,
          hasExactBounds,
          boundsType: hasExactBounds ? 'exact' : 'fallback'
        }
      };
    });

    // Add tile source
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

    // Add layers if they don't exist
    if (!map.current.getLayer('tiles-fill')) {
      addMapLayers();
      addMapEventListeners();
    }

    // Update bounds statistics
    setBoundsStats({ exact: exactBoundsCount, fallback: fallbackBoundsCount });
  };

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

  // Update styles when selection or city filter changes
  const updateTileStyles = useCallback(() => {
    if (map.current && map.current.getLayer('tiles-fill')) {
      // Fill color styling
      map.current.setPaintProperty('tiles-fill', 'fill-color', [
        'case',
        ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
        '#ff6b6b', // Selected locations - red
        ['in', ['get', 'id'], ['literal', Array.from(cityFilteredLocations)]],
        '#4a90e2', // City filtered locations - blue
        'transparent'
      ]);

      map.current.setPaintProperty('tiles-fill', 'fill-opacity', [
        'case',
        ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
        0.6, // Selected locations
        ['in', ['get', 'id'], ['literal', Array.from(cityFilteredLocations)]],
        0.4, // City filtered locations
        0
      ]);

      // Unified border styling - white with transparency for all tiles
      map.current.setPaintProperty('tiles-border', 'line-color', [
        'case',
        ['==', ['get', 'id'], hoveredLocationId.current || -1],
        '#4ecdc4', // Hovered - light blue
        ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
        '#ff6b6b', // Selected - red
        ['in', ['get', 'id'], ['literal', Array.from(cityFilteredLocations)]],
        '#4a90e2', // City filtered - blue
        'rgba(255, 255, 255, 0.6)' // Default - white with transparency
      ]);

      map.current.setPaintProperty('tiles-border', 'line-width', [
        'case',
        ['==', ['get', 'id'], hoveredLocationId.current || -1],
        3, // Hovered
        ['in', ['get', 'id'], ['literal', Array.from(selectedLocations)]],
        2, // Selected
        ['in', ['get', 'id'], ['literal', Array.from(cityFilteredLocations)]],
        2, // City filtered
        1.5 // Default
      ]);

      // Unified dash pattern - solid lines for all tiles
      map.current.setPaintProperty('tiles-border', 'line-dasharray', [
        'literal', [] // Solid line for all tiles
      ]);
    }
  }, [selectedLocations, cityFilteredLocations]);

  useEffect(() => {
    updateTileStyles();
  }, [selectedLocations, cityFilteredLocations, updateTileStyles]);

  const addMapLayers = () => {
    // Add tile fill layer
    map.current.addLayer({
      id: 'tiles-fill',
      type: 'fill',
      source: 'tiles',
      paint: {
        'fill-color': 'transparent',
        'fill-opacity': 0
      }
    });

    map.current.addLayer({
      id: 'tiles-border',
      type: 'line',
      source: 'tiles',
      paint: {
        'line-color': 'rgba(255, 255, 255, 0.6)',
        'line-width': 1.5,
        'line-dasharray': []
      }
    });
  };

  const addMapEventListeners = () => {
    // Tile interactions - Fixed hover detection
    map.current.on('mouseenter', 'tiles-fill', (e) => {
      map.current.getCanvas().style.cursor = 'pointer';
      hoveredLocationId.current = e.features[0].properties.id;
      updateTileStyles();
    });

    map.current.on('mousemove', 'tiles-fill', (e) => {
      // Update hovered ID as mouse moves between tiles
      const newHoveredId = e.features[0].properties.id;
      if (hoveredLocationId.current !== newHoveredId) {
        hoveredLocationId.current = newHoveredId;
        updateTileStyles();
      }
    });

    map.current.on('mouseleave', 'tiles-fill', () => {
      map.current.getCanvas().style.cursor = '';
      hoveredLocationId.current = null;
      updateTileStyles();
    });

    map.current.on('click', 'tiles-fill', (e) => {
      const locationId = e.features[0].properties.id;
      onLocationSelect(locationId);
      showEnhancedLocationPopup(
        [e.features[0].properties.longitude, e.features[0].properties.latitude],
        e.features[0].properties
      );
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
        showEnhancedLocationPopup([longitude, latitude], location);
      }
    }, 1200);
  };

  const zoomToBbox = (bbox) => {
    map.current.fitBounds([
      [bbox.minLon, bbox.minLat],
      [bbox.maxLon, bbox.maxLat]
    ], {
      padding: 50,
      duration: 1000
    });
  };

  const showEnhancedLocationPopup = (coordinates, properties) => {
    // Close existing popup
    if (currentPopup.current) {
      currentPopup.current.remove();
      currentPopup.current = null;
    }

    const isSelected = selectedLocations.has(properties.id);
    const isCityFiltered = cityFilteredLocations.has(properties.id);

    let statusIcon = '';
    if (isSelected) {
      statusIcon = '✓';
    } else if (isCityFiltered) {
      statusIcon = '🏙️';
    }

    const popupHtml = `
      <div style="font-size: 10px; line-height: 1.2; max-width: 160px;">
        <div style="font-weight: 600; color: #1e293b; margin-bottom: 2px;">
          ${properties.city} ${statusIcon}
        </div>
        <div style="color: #64748b; margin-bottom: 3px;">
          ${properties.country}
        </div>
      </div>
    `;

    // Create popup
    currentPopup.current = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false,
      focusAfterOpen: false,
      maxWidth: '180px',
      className: 'enhanced-popup'
    })
      .setLngLat(coordinates)
      .setHTML(popupHtml)
      .addTo(map.current);

    // Auto-close popup after 4 seconds
    setTimeout(() => {
      if (currentPopup.current) {
        currentPopup.current.remove();
        currentPopup.current = null;
      }
    }, 4000);

    currentPopup.current.on('close', () => {
      currentPopup.current = null;
    });
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={mapContainer} className="map-view" />
      
      {/* Simplified Statistics Overlay */}
      {boundsStats.exact + boundsStats.fallback > 0 && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          background: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '6px 10px',
          borderRadius: '4px',
          fontSize: '10px',
          fontFamily: 'monospace',
          backdropFilter: 'blur(4px)',
          zIndex: 1000
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: '#fff' }}>{(boundsStats.exact + boundsStats.fallback).toLocaleString()} tiles</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MapView;