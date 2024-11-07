import React, { useState, useEffect, useMemo, useCallback } from 'react';
import Map, { Source, Layer, NavigationControl, FullscreenControl, ScaleControl } from 'react-map-gl';
import bbox from '@turf/bbox';
import bboxPolygon from '@turf/bbox-polygon';
import buffer from '@turf/buffer';
import { point } from '@turf/helpers';
import 'mapbox-gl/dist/mapbox-gl.css';

const MapboxMapComponent = ({ selectedPoint, onMapClick, allPoints, selectedCity }) => {
  const MAPBOX_TOKEN = process.env.REACT_APP_MAPBOX_TOKEN;
  const SELECTED_POINT_ZOOM = 12; // Constant zoom level for selected points
  
  const [hoverPoint, setHoverPoint] = useState(null);
  const [urbanAreasGeojson, setUrbanAreasGeojson] = useState(null);
  
  const [viewState, setViewState] = useState({
    longitude: 0,
    latitude: 20,
    zoom: 1.5,
    transitionDuration: 0 // Start with no transition
  });

  useEffect(() => {
    const fetchUrbanAreas = async () => {
      try {
        const response = await fetch('http://localhost:8000/urban_areas');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setUrbanAreasGeojson(data);
      } catch (error) {
        console.error('Error fetching urban areas:', error);
      }
    };
    fetchUrbanAreas();
  }, []);

  // Update view when a point is selected
  useEffect(() => {
    if (selectedPoint) {
      setViewState({
        longitude: selectedPoint.longitude,
        latitude: selectedPoint.latitude,
        zoom: SELECTED_POINT_ZOOM,
        transitionDuration: 1000 // Smooth transition when point selected
      });
    }
  }, [selectedPoint]);

  // Update view when a city is selected
  useEffect(() => {
    if (selectedCity && urbanAreasGeojson) {
      const cityFeature = urbanAreasGeojson.features.find(
        feature => feature.properties.city.toLowerCase() === selectedCity.toLowerCase()
      );
      if (cityFeature) {
        const [minLng, minLat, maxLng, maxLat] = bbox(cityFeature);
        setViewState({
          longitude: (minLng + maxLng) / 2,
          latitude: (minLat + maxLat) / 2,
          zoom: SELECTED_POINT_ZOOM,
          transitionDuration: 1000
        });
      }
    }
  }, [selectedCity, urbanAreasGeojson]);

  const bboxPolygonData = useMemo(() => {
    if (!selectedPoint) return null;
    const center = point([selectedPoint.longitude, selectedPoint.latitude]);
    const boxSize = 2.24;
    const options = { units: 'kilometers' };
    const bboxExtent = bbox(buffer(center, boxSize / 2, options));
    return bboxPolygon(bboxExtent);
  }, [selectedPoint]);

  const hoverBboxPolygonData = useMemo(() => {
    if (!hoverPoint) return null;
    const center = point([hoverPoint.longitude, hoverPoint.latitude]);
    const boxSize = 2.24;
    const options = { units: 'kilometers' };
    const bboxExtent = bbox(buffer(center, boxSize / 2, options));
    return bboxPolygon(bboxExtent);
  }, [hoverPoint]);

  const layerStyle = {
    id: 'bbox-layer',
    type: 'line',
    paint: {
      'line-color': '#007cbf',
      'line-width': 2
    }
  };

  const hoverLayerStyle = {
    id: 'hover-bbox-layer',
    type: 'line',
    paint: {
      'line-color': '#ff7f50',
      'line-width': 2,
      'line-dasharray': [2, 2]
    }
  };

  const pointLayerStyle = {
    id: 'point-layer',
    type: 'circle',
    paint: {
      'circle-radius': 5,
      'circle-color': '#007cbf',
      'circle-stroke-width': 1,
      'circle-stroke-color': '#fff'
    }
  };

  const urbanAreaLayerStyle = {
    id: 'urban-area-layer',
    type: 'line',
    paint: {
      'line-color': '#FF0000',
      'line-width': 2
    }
  };

  const handleClick = useCallback((event) => {
    const { lngLat } = event;
    if (!allPoints?.length) return;
    
    // Find the nearest point
    let nearestPoint = null;
    let minDistance = Infinity;

    allPoints.forEach(point => {
      const distance = Math.sqrt(
        Math.pow(point.longitude - lngLat.lng, 2) + 
        Math.pow(point.latitude - lngLat.lat, 2)
      );
      if (distance < minDistance) {
        minDistance = distance;
        nearestPoint = point;
      }
    });

    if (nearestPoint && minDistance < 2) {
      onMapClick(nearestPoint);
    }
  }, [allPoints, onMapClick]);

  const handleHover = useCallback((event) => {
    const { lngLat } = event;
    if (!allPoints?.length) return;
    
    let nearestPoint = null;
    let minDistance = Infinity;

    allPoints.forEach(point => {
      const distance = Math.sqrt(
        Math.pow(point.longitude - lngLat.lng, 2) + 
        Math.pow(point.latitude - lngLat.lat, 2)
      );
      if (distance < minDistance) {
        minDistance = distance;
        nearestPoint = point;
      }
    });

    if (nearestPoint && minDistance < 2) {
      setHoverPoint(nearestPoint);
    } else {
      setHoverPoint(null);
    }
  }, [allPoints]);

  const handleMouseLeave = useCallback(() => {
    setHoverPoint(null);
  }, []);

  return (
    <div style={{ position: 'relative', height: '100%', width: '100%' }}>
      <Map
        {...viewState}
        onMove={evt => setViewState(evt.viewState)}
        onClick={handleClick}
        onMouseMove={handleHover}
        onMouseLeave={handleMouseLeave}
        style={{width: '100%', height: '100%'}}
        mapStyle="mapbox://styles/mapbox/satellite-v9"
        mapboxAccessToken={MAPBOX_TOKEN}
      >
        <NavigationControl position="top-left" />
        <FullscreenControl position="top-right" />
        <ScaleControl position="bottom-right" />
        
        {bboxPolygonData && (
          <Source type="geojson" data={bboxPolygonData}>
            <Layer {...layerStyle} />
          </Source>
        )}
        
        {hoverBboxPolygonData && (
          <Source type="geojson" data={hoverBboxPolygonData}>
            <Layer {...hoverLayerStyle} />
          </Source>
        )}
        
        {selectedPoint && (
          <Source type="geojson" data={{
            type: 'Feature',
            geometry: {
              type: 'Point',
              coordinates: [selectedPoint.longitude, selectedPoint.latitude]
            }
          }}>
            <Layer {...pointLayerStyle} />
          </Source>
        )}

        {urbanAreasGeojson && (
          <Source type="geojson" data={urbanAreasGeojson}>
            <Layer {...urbanAreaLayerStyle} />
          </Source>
        )}
      </Map>
      
      {selectedPoint && (
        <div className="absolute bottom-10 left-2 bg-white/90 p-2 rounded-lg shadow-sm text-xs space-y-1">
          <p className="font-medium">Selected Point:</p>
          <p>Lat: {selectedPoint.latitude.toFixed(6)}</p>
          <p>Lon: {selectedPoint.longitude.toFixed(6)}</p>
          <p>Country: {selectedPoint.country}</p>
          <p>City: {selectedPoint.city}</p>
        </div>
      )}
    </div>
  );
};

export default MapboxMapComponent;