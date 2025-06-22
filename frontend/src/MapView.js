import React, { useRef, useEffect, useCallback } from 'react';
import mapboxgl from 'mapbox-gl';

const TILE_SIZE_METERS = 1120;

const MapView = ({ locations, selectedLocations, onLocationSelect, mapboxToken }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const hoveredLocationId = useRef(null);

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
        addLocationsToMap();
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
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [mapboxToken]);

  // Add locations to map when both map and locations are ready
  useEffect(() => {
    if (map.current && map.current.isStyleLoaded() && locations.length > 0) {
      addLocationsToMap();
    }
  }, [locations]);

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
    if (!map.current || !locations.length) return;

    // Create tile polygons
    const tileFeatures = locations.map(location => ({
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [createTilePolygon(location.longitude, location.latitude, TILE_SIZE_METERS)]
      },
      properties: { ...location }
    }));

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
      zoom: 15,
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

  return <div ref={mapContainer} className="map-view" />;
};

export default MapView;