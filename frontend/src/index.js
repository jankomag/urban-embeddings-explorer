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

// Autocomplete Component
function AutocompleteInput({ 
  value, 
  onChange, 
  onSelect, 
  options, 
  placeholder, 
  disabled = false 
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [filteredOptions, setFilteredOptions] = useState([]);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const inputRef = useRef(null);
  const dropdownRef = useRef(null);

  // Filter options based on input value
  useEffect(() => {
    if (!value || value.length < 1) {
      setFilteredOptions([]);
      setIsOpen(false);
      return;
    }

    const filtered = options.filter(option =>
      option.toLowerCase().includes(value.toLowerCase())
    ).slice(0, 20); // Limit to 20 results for performance

    setFilteredOptions(filtered);
    setIsOpen(filtered.length > 0);
    setHighlightedIndex(-1);
  }, [value, options]);

  // Handle input change
  const handleInputChange = (e) => {
    onChange(e.target.value);
  };

  // Handle option selection
  const handleOptionSelect = (option) => {
    onSelect(option);
    setIsOpen(false);
    setHighlightedIndex(-1);
    inputRef.current?.blur();
  };

  // Handle keyboard navigation
  const handleKeyDown = (e) => {
    if (!isOpen) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setHighlightedIndex(prev => 
          prev < filteredOptions.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setHighlightedIndex(prev => 
          prev > 0 ? prev - 1 : filteredOptions.length - 1
        );
        break;
      case 'Enter':
        e.preventDefault();
        if (highlightedIndex >= 0 && filteredOptions[highlightedIndex]) {
          handleOptionSelect(filteredOptions[highlightedIndex]);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        setHighlightedIndex(-1);
        inputRef.current?.blur();
        break;
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
        setHighlightedIndex(-1);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="autocomplete-container" ref={dropdownRef}>
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onFocus={() => {
          if (filteredOptions.length > 0) setIsOpen(true);
        }}
        placeholder={placeholder}
        className="autocomplete-input"
        disabled={disabled}
        autoComplete="off"
      />
      
      {isOpen && filteredOptions.length > 0 && (
        <div className="autocomplete-dropdown">
          {filteredOptions.map((option, index) => (
            <div
              key={option}
              className={`autocomplete-item ${index === highlightedIndex ? 'highlighted' : ''}`}
              onClick={() => handleOptionSelect(option)}
              onMouseEnter={() => setHighlightedIndex(index)}
            >
              {option}
            </div>
          ))}
        </div>
      )}
      
      {isOpen && filteredOptions.length === 0 && value.length > 0 && (
        <div className="autocomplete-dropdown">
          <div className="autocomplete-no-results">
            No matches found
          </div>
        </div>
      )}
    </div>
  );
}

// Enhanced Similarity Panel Component with Method Selection
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
  const [paginationInfo, setPaginationInfo] = useState(null);
  const [loadingMoreResults, setLoadingMoreResults] = useState(false);
  const [similarityMethod, setSimilarityMethod] = useState('regular');
  // Updated to only include the two methods that exist in Qdrant
  const [availableMethods, setAvailableMethods] = useState({
    'regular': { 
      name: "Regular Embeddings", 
      description: "Standard similarity using mean-aggregated patch embeddings" 
    },
    'global_contrastive': { 
      name: "Global Contrastive", 
      description: "Dataset mean subtracted to highlight city-level differences" 
    }
  });
  const [loadingMethods, setLoadingMethods] = useState(false);
  
  // Request cancellation
  const currentRequestRef = useRef(null);
  const loadMoreRequestRef = useRef(null);

  // Load available similarity methods on component mount - now simplified
  useEffect(() => {
    loadAvailableMethods();
  }, []);

  const loadAvailableMethods = async () => {
    setLoadingMethods(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/methods`);
      if (response.ok) {
        const data = await response.json();
        
        // Filter to only include the methods we actually have in Qdrant
        const filteredMethods = {};
        const availableKeys = ['regular', 'global_contrastive'];
        
        for (const key of availableKeys) {
          if (data.available_methods[key]) {
            filteredMethods[key] = data.available_methods[key];
          }
        }
        
        // Only update if we got valid methods, otherwise keep defaults
        if (Object.keys(filteredMethods).length > 0) {
          setAvailableMethods(filteredMethods);
          console.log('📊 Loaded similarity methods:', Object.keys(filteredMethods));
        } else {
          console.log('📊 Using default similarity methods (API returned no valid methods)');
        }
      } else {
        console.warn('⚠️ Failed to load methods from API, using defaults');
      }
    } catch (error) {
      console.error('Error loading similarity methods:', error);
      console.log('📊 Using default similarity methods due to error');
      // Keep the default methods set in state initialization
    } finally {
      setLoadingMethods(false);
    }
  };

  // Find similar locations with method parameter
  const findSimilarLocations = async (locationId, offset = 0, limit = 6, method = similarityMethod) => {
    console.log('Finding similar locations for:', locationId, 'method:', method, 'offset:', offset);
    
    // Validate method against available methods
    if (!availableMethods[method]) {
      return;
    }
    
    // Cancel any existing request for initial loads (not for pagination)
    if (offset === 0 && currentRequestRef.current) {
      currentRequestRef.current.abort();
      currentRequestRef.current = null;
    }
    
    // Cancel any existing load more request for pagination
    if (offset > 0 && loadMoreRequestRef.current) {
      console.log('🚫 Cancelling previous load more request');
      loadMoreRequestRef.current.abort();
      loadMoreRequestRef.current = null;
    }
    
    // Create new AbortController for this request
    const abortController = new AbortController();
    
    if (offset === 0) {
      currentRequestRef.current = abortController;
      setFindingSimilar(true);
      setSimilarResults([]); // Clear previous results for new search
      setPaginationInfo(null);
    } else {
      loadMoreRequestRef.current = abortController;
      setLoadingMoreResults(true);
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/similarity/${locationId}?offset=${offset}&limit=${limit}&method=${method}`,
        { signal: abortController.signal }
      );
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to find similar locations');
      }
      
      const data = await response.json();
      console.log('✅ Found similar locations:', data.similar_locations.length, 'method:', data.method_used, 'offset:', offset);
      
      // Only process results if this request wasn't cancelled
      if (!abortController.signal.aborted) {
        if (offset === 0) {
          // Initial load - replace results
          setSimilarResults(data.similar_locations);
        } else {
          // Load more - append results
          setSimilarResults(prev => [...prev, ...data.similar_locations]);
        }
        
        setPaginationInfo(data.pagination);
      }
    } catch (error) {
      // Don't show error for cancelled requests
      if (error.name === 'AbortError') {
        console.log('🔄 Request was cancelled');
        return;
      }
      
      console.error('Error finding similar locations:', error);
      
      // Only update state if request wasn't cancelled
      if (!abortController.signal.aborted) {
        // Show error state but don't clear existing results if loading more
        if (offset === 0) {
          setSimilarResults([]);
          setPaginationInfo(null);
        }
      }
    } finally {
      // Only update loading state if request wasn't cancelled
      if (!abortController.signal.aborted) {
        if (offset === 0) {
          setFindingSimilar(false);
          currentRequestRef.current = null;
        } else {
          setLoadingMoreResults(false);
          loadMoreRequestRef.current = null;
        }
      }
    }
  };

  // Load more similar locations with current method
  const loadMoreSimilarLocations = () => {
    if (paginationInfo && paginationInfo.has_more && primarySelectionId && !loadingMoreResults) {
      findSimilarLocations(primarySelectionId, paginationInfo.next_offset, 6, similarityMethod);
    }
  };

  // Handle similarity method change
  const handleMethodChange = (newMethod) => {
    console.log('🔄 Similarity method changed to:', newMethod);
    
    // Validate method
    if (!availableMethods[newMethod]) {
      console.error(`❌ Method '${newMethod}' not available`);
      return;
    }
    
    setSimilarityMethod(newMethod);
    
    // If we have a primary selection, re-run the search with new method
    if (primarySelectionId) {
      // Cancel any existing requests
      if (currentRequestRef.current) {
        currentRequestRef.current.abort();
        currentRequestRef.current = null;
      }
      if (loadMoreRequestRef.current) {
        loadMoreRequestRef.current.abort();
        loadMoreRequestRef.current = null;
      }
      
      // Reset states and search with new method
      setFindingSimilar(false);
      setLoadingMoreResults(false);
      setSimilarResults([]);
      setPaginationInfo(null);
      
      // Start new search
      findSimilarLocations(primarySelectionId, 0, 6, newMethod);
    }
  };

  const getStaticMapImage = (longitude, latitude, width = 140, height = 140, zoom = 11.3) => {
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
    if (primarySelectionId) {
      console.log('🎯 Primary selection changed to:', primarySelectionId);
      
      // Cancel any existing requests when primary selection changes
      if (currentRequestRef.current) {
        console.log('🚫 Cancelling existing request due to selection change');
        currentRequestRef.current.abort();
        currentRequestRef.current = null;
      }
      
      if (loadMoreRequestRef.current) {
        console.log('🚫 Cancelling existing load more request due to selection change');
        loadMoreRequestRef.current.abort();
        loadMoreRequestRef.current = null;
      }
      
      // Reset states
      setFindingSimilar(false);
      setLoadingMoreResults(false);
      
      findSimilarLocations(primarySelectionId, 0, 6, similarityMethod);
    }
  }, [primarySelectionId, similarityMethod]); // Include similarityMethod in dependencies
  
  // Cleanup effect to cancel requests on unmount
  useEffect(() => {
    return () => {
      if (currentRequestRef.current) {
        console.log('🧹 Cleaning up similarity request on unmount');
        currentRequestRef.current.abort();
      }
      if (loadMoreRequestRef.current) {
        console.log('🧹 Cleaning up load more request on unmount');
        loadMoreRequestRef.current.abort();
      }
    };
  }, []);

  if (!selectedLocation && !primarySelectionId) {
    return (
      <div className="empty-state">
        <p>Click any location to explore similarities</p>
        <div className="empty-info">
          <h5>AI-Powered Similarity Analysis</h5>
          <p>Using TerraMind satellite embeddings with two similarity methods to find visually similar urban areas.</p>
          <div style={{ marginTop: '8px', fontSize: '9px', color: '#666' }}>
            <strong>Available Methods:</strong><br/>
            • Regular: Standard visual similarity<br/>
            • Global Contrastive: City-level differences
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Selected Location Card */}
      {selectedLocation && (
        <div className="selected-card">
          <button 
            className="zoom-to-location-btn"
            onClick={() => {
              // Zoom map to this location
              window.dispatchEvent(new CustomEvent('zoomToLocation', {
                detail: { 
                  longitude: selectedLocation.longitude, 
                  latitude: selectedLocation.latitude, 
                  id: selectedLocation.id 
                }
              }));
              // Zoom UMAP to this point
              window.dispatchEvent(new CustomEvent('zoomToUmapPoint', {
                detail: { locationId: selectedLocation.id }
              }));
            }}
            title="Zoom to this location on map and UMAP"
          >
            🎯Back
          </button>
          <h4>{selectedLocation.city}, {selectedLocation.country}</h4>
          {mapboxToken && selectedLocation && (
            <img
              src={getStaticMapImage(selectedLocation.longitude, selectedLocation.latitude, 140, 140, 11.5)}
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
          
          {/* Similarity Method Selector - Updated with better descriptions */}
          <div className="method-selector">
            <label className="method-label">Similarity Method:</label>
            <select 
              value={similarityMethod} 
              onChange={(e) => handleMethodChange(e.target.value)}
              className="method-dropdown"
              disabled={loadingMethods || findingSimilar}
            >
              {Object.entries(availableMethods).map(([key, method]) => (
                <option key={key} value={key}>
                  {method.name}
                </option>
              ))}
            </select>
          </div>
          
          {/* Method Description - Enhanced with more detailed info */}
          {availableMethods[similarityMethod] && (
            <div className="method-info-enhanced">
              <div className="method-badge-enhanced">
                {availableMethods[similarityMethod].name}
              </div>
              <p className="method-description-enhanced">
                {availableMethods[similarityMethod].description}
              </p>
              {similarityMethod === 'regular' && (
                <div style={{ fontSize: '7px', color: '#6b7280', marginTop: '2px' }}>
                  Best for: General visual similarity
                </div>
              )}
              {similarityMethod === 'global_contrastive' && (
                <div style={{ fontSize: '7px', color: '#6b7280', marginTop: '2px' }}>
                  Best for: Finding unique city characteristics
                </div>
              )}
            </div>
          )}
          
          {/* Scrollable Similarity Content */}
          <div className="similarity-content">
            {findingSimilar ? (
              <div className="loading-state">
                <div className="spinner"></div>
                <div className="loading-text">
                  Finding similar locations using {availableMethods[similarityMethod]?.name || similarityMethod}...
                </div>
              </div>
            ) : (
              <>
                <div className="similarity-grid">
                  {similarResults.map((location, index) => {
                    const similarity = (location.similarity_score * 100).toFixed(1);
                    const imageUrl = getStaticMapImage(location.longitude, location.latitude, 140, 140, 11.3);
                    
                    return (
                      <div 
                        key={`${location.id}-${index}`}
                        className="similarity-tile"
                        onClick={() => onNavigationClick(location)}
                        title={`${location.city}, ${location.country} - ${similarity}% similar using ${availableMethods[similarityMethod]?.name || similarityMethod}`}
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
                    Showing all {similarResults.length} results using {availableMethods[similarityMethod]?.name || similarityMethod}
                  </div>
                )}
              </>
            )}
          </div>
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
  
  // Enhanced City/Country selection state with autocomplete
  const [cityFilteredLocations, setCityFilteredLocations] = useState(new Set());
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedCity, setSelectedCity] = useState('');
  const [countryInput, setCountryInput] = useState('');
  const [cityInput, setCityInput] = useState('');
  const [allCountries, setAllCountries] = useState([]);
  const [availableCities, setAvailableCities] = useState([]); // Cities filtered by selected country

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
      
      // Extract unique countries and cities for autocomplete
      const uniqueCountries = [...new Set(locationData.map(loc => loc.country))].sort();
      const allCitiesWithCountry = locationData.map(loc => ({
        city: loc.city,
        country: loc.country,
        cityCountry: `${loc.city}, ${loc.country}`
      }));
      
      setAllCountries(uniqueCountries);
      
      console.log(`Loaded ${locationData.length} locations`);
      console.log(`Found ${uniqueCountries.length} unique countries`);
    } catch (error) {
      console.error('Error loading locations:', error);
      setError('Failed to load locations');
    }
  };

  // Update available cities when country selection changes
  useEffect(() => {
    if (selectedCountry) {
      // Filter cities by selected country
      const citiesInCountry = locations
        .filter(loc => loc.country === selectedCountry)
        .map(loc => loc.city);
      const uniqueCitiesInCountry = [...new Set(citiesInCountry)].sort();
      setAvailableCities(uniqueCitiesInCountry);
    } else {
      // Show all cities when no country is selected
      const allCities = [...new Set(locations.map(loc => `${loc.city}, ${loc.country}`))].sort();
      setAvailableCities(allCities);
    }
  }, [selectedCountry, locations]);

  // Trigger UMAP highlighting immediately when city/country filter changes
  useEffect(() => {
    if (cityFilteredLocations.size > 0) {
      // Immediately center UMAP on filtered locations without waiting for hover
      window.dispatchEvent(new CustomEvent('centerOnCityTiles', {
        detail: { locationIds: Array.from(cityFilteredLocations) }
      }));
    }
  }, [cityFilteredLocations]);

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
    // Don't clear city/country filters - they remain independent
  };

  const clearCitySelection = () => {
    setCityFilteredLocations(new Set());
    setSelectedCity('');
    setSelectedCountry('');
    setCityInput('');
    setCountryInput('');
  };

  // Handle country selection with autocomplete
  const handleCountrySelect = (country) => {
    console.log('Country selected:', country);
    setSelectedCountry(country);
    setCountryInput(country);
    setSelectedCity(''); // Clear city selection when country changes
    setCityInput(''); // Clear city input when country changes
    
    // Find all locations for this country
    const countryLocations = locations.filter(loc => loc.country === country);
    const countryLocationIds = new Set(countryLocations.map(loc => loc.id));
    
    setCityFilteredLocations(countryLocationIds);
    
    // Calculate bounding box for all tiles in this country
    if (countryLocations.length > 0) {
      const lons = countryLocations.map(loc => loc.longitude);
      const lats = countryLocations.map(loc => loc.latitude);
      
      const bbox = {
        minLon: Math.min(...lons),
        maxLon: Math.max(...lons),
        minLat: Math.min(...lats),
        maxLat: Math.max(...lats)
      };
      
      // Add padding to bbox
      const lonPadding = (bbox.maxLon - bbox.minLon) * 0.1 || 0.01;
      const latPadding = (bbox.maxLat - bbox.minLat) * 0.1 || 0.01;
      
      // Zoom map to bbox
      window.dispatchEvent(new CustomEvent('zoomToBbox', {
        detail: { 
          bbox: {
            minLon: bbox.minLon - lonPadding,
            maxLon: bbox.maxLon + lonPadding,
            minLat: bbox.minLat - latPadding,
            maxLat: bbox.maxLat + latPadding
          }
        }
      }));
    }
  };

  // Handle city selection with autocomplete
  const handleCitySelect = (citySelection) => {
    console.log('City selected:', citySelection);
    
    let city, country;
    
    if (selectedCountry && !citySelection.includes(',')) {
      // If a country is already selected and city doesn't contain comma, it's just a city name
      city = citySelection;
      country = selectedCountry;
    } else {
      // Parse "City, Country" format
      const parts = citySelection.split(', ');
      if (parts.length >= 2) {
        city = parts[0];
        country = parts.slice(1).join(', '); // Handle countries with commas
      } else {
        console.warn('Invalid city selection format:', citySelection);
        return;
      }
    }
    
    setSelectedCity(city);
    setCityInput(citySelection);
    
    // If country wasn't already selected, select it now
    if (!selectedCountry) {
      setSelectedCountry(country);
      setCountryInput(country);
    }
    
    // Find all locations for this specific city
    const cityLocations = locations.filter(loc => 
      loc.city === city && loc.country === country
    );
    const cityLocationIds = new Set(cityLocations.map(loc => loc.id));
    
    setCityFilteredLocations(cityLocationIds);
    
    // Calculate bounding box for all tiles in this city
    if (cityLocations.length > 0) {
      const lons = cityLocations.map(loc => loc.longitude);
      const lats = cityLocations.map(loc => loc.latitude);
      
      const bbox = {
        minLon: Math.min(...lons),
        maxLon: Math.max(...lons),
        minLat: Math.min(...lats),
        maxLat: Math.max(...lats)
      };
      
      // Add padding to bbox
      const lonPadding = (bbox.maxLon - bbox.minLon) * 0.1 || 0.01;
      const latPadding = (bbox.maxLat - bbox.minLat) * 0.1 || 0.01;
      
      // Zoom map to bbox
      window.dispatchEvent(new CustomEvent('zoomToBbox', {
        detail: { 
          bbox: {
            minLon: bbox.minLon - lonPadding,
            maxLon: bbox.maxLon + lonPadding,
            minLat: bbox.minLat - latPadding,
            maxLat: bbox.maxLat + latPadding
          }
        }
      }));
    }
  };

  // Primary selection - triggers similarity search and UMAP centering
  // This overrides city/country selection when user clicks individual tiles
  const handlePrimarySelection = (locationId) => {
    console.log('Primary selection:', locationId);
    
    // Clear city/country selection when user selects individual tile
    clearCitySelection();
    
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

  // Navigation only - just moves views without changing primary selection or selected card
  const handleNavigationClick = (location) => {
    console.log('Navigation to:', location.city, location.country);
    
    // DON'T update the selected location or visual selection
    // DON'T change primarySelectionId or clear similarResults
    // ONLY move the views to show this location
    
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
      {/* Enhanced Header with Autocomplete City/Country Selector */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <h1>🛰️ Satellite Embeddings Explorer</h1>
          </div>
          
          <div className="header-center">
            {/* Enhanced City/Country Selector with Autocomplete */}
            <div className="location-selector">
              <div className="selector-group">
                <label>Country:</label>
                <AutocompleteInput
                  value={countryInput}
                  onChange={setCountryInput}
                  onSelect={handleCountrySelect}
                  options={allCountries}
                  placeholder="Type country..."
                />
              </div>
              
              <div className="selector-group">
                <label>City:</label>
                <AutocompleteInput
                  value={cityInput}
                  onChange={setCityInput}
                  onSelect={handleCitySelect}
                  options={availableCities}
                  placeholder={selectedCountry ? "Type city..." : "Type city, country..."}
                  disabled={allCountries.length === 0}
                />
              </div>
              
              {(selectedCity || selectedCountry) && (
                <button 
                  className="clear-filter-btn" 
                  onClick={clearCitySelection}
                  title="Clear location filter"
                >
                  ✕
                </button>
              )}
            </div>
            
            {/* Selection Info */}
            {(selectedCity || selectedCountry) && (
              <div className="filter-info">
                <span className="filter-label">
                  {selectedCity ? `City: ${selectedCity}, ${selectedCountry}` : `Country: ${selectedCountry}`}
                </span>
                <span className="filter-count">
                  {cityFilteredLocations.size} tiles
                </span>
              </div>
            )}
          </div>
          
          <div className="header-right">
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
              cityFilteredLocations={cityFilteredLocations}
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
              cityFilteredLocations={cityFilteredLocations}
              onLocationSelect={handlePrimarySelection}
            />
          </div>
        </div>

        {/* Enhanced Analysis Panel */}
        <div className="analysis-panel">
          <div className="analysis-header">
            <h3>Similarity Analysis</h3>
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