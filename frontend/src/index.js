import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import './ModernApp.css';
import MapView from './MapView';
import HighPerformanceUMapView from './HighPerformanceUMapView';
import HelpPanel, { WelcomePanel } from './HelpPanel';
import polyline from '@mapbox/polyline';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? process.env.REACT_APP_API_URL || 'https://your-railway-url.up.railway.app'
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

  useEffect(() => {
    if (!value || value.length < 1) {
      setFilteredOptions([]);
      setIsOpen(false);
      return;
    }

    const filtered = options.filter(option =>
      option.toLowerCase().includes(value.toLowerCase())
    ).slice(0, 20);

    setFilteredOptions(filtered);
    setIsOpen(filtered.length > 0);
    setHighlightedIndex(-1);
  }, [value, options]);

  const handleInputChange = (e) => {
    onChange(e.target.value);
  };

  const handleOptionSelect = (option) => {
    onSelect(option);
    setIsOpen(false);
    setHighlightedIndex(-1);
    inputRef.current?.blur();
  };

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

// Enhanced Similarity Panel Component with Adaptive Mixed Aggregation
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
  const [includeSameCity, setIncludeSameCity] = useState(true);
  const [allSimilarResults, setAllSimilarResults] = useState([]);
  const [availableMethods, setAvailableMethods] = useState({
    'regular': { 
      name: "Regular Embeddings", 
      description: "Standard similarity using mean-aggregated patch embeddings" 
    },
    'global_contrastive': { 
      name: "Global Contrastive", 
      description: "Dataset mean subtracted to highlight city-level differences" 
    },
    'adaptive_mixed': {
      name: "Adaptive Mixed",
      description: "Intelligent switching between uniform and weighted aggregation based on patch diversity"
    }
  });
  const [loadingMethods, setLoadingMethods] = useState(false);
  
  const currentRequestRef = useRef(null);
  const loadMoreRequestRef = useRef(null);

  useEffect(() => {
    loadAvailableMethods();
  }, []);

  const loadAvailableMethods = async () => {
    setLoadingMethods(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/methods`);
      if (response.ok) {
        const data = await response.json();
        
        const filteredMethods = {};
        const availableKeys = ['regular', 'global_contrastive', 'adaptive_mixed'];
        
        for (const key of availableKeys) {
          if (data.available_methods[key]) {
            filteredMethods[key] = data.available_methods[key];
          }
        }
        
        if (Object.keys(filteredMethods).length > 0) {
          setAvailableMethods(filteredMethods);
        }
      }
    } catch (error) {
      console.error('Error loading similarity methods:', error);
    } finally {
      setLoadingMethods(false);
    }
  };

  // Filter results based on same-city toggle
  const filterResultsBySameCity = useCallback((results) => {
    if (!selectedLocation) return results;
    
    if (includeSameCity) {
      return results; // Return all results
    } else {
      // Filter out results from the same city
      return results.filter(location => 
        location.city !== selectedLocation.city || 
        location.country !== selectedLocation.country
      );
    }
  }, [includeSameCity, selectedLocation]);

  const findSimilarLocations = async (locationId, offset = 0, limit = 12, method = similarityMethod) => {
    if (!availableMethods[method]) {
      return;
    }
    
    if (offset === 0 && currentRequestRef.current) {
      currentRequestRef.current.abort();
      currentRequestRef.current = null;
    }
    
    if (offset > 0 && loadMoreRequestRef.current) {
      loadMoreRequestRef.current.abort();
      loadMoreRequestRef.current = null;
    }
    
    const abortController = new AbortController();
    
    if (offset === 0) {
      currentRequestRef.current = abortController;
      setFindingSimilar(true);
      setSimilarResults([]);
      setAllSimilarResults([]);
      setPaginationInfo(null);
    } else {
      loadMoreRequestRef.current = abortController;
      setLoadingMoreResults(true);
    }

    try {
      // Request more results to account for filtering
      const adjustedLimit = includeSameCity ? limit : limit * 2;
      
      const response = await fetch(
        `${API_BASE_URL}/api/similarity/${locationId}?offset=${offset}&limit=${adjustedLimit}&method=${method}`,
        { signal: abortController.signal }
      );
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to find similar locations');
      }
      
      const data = await response.json();
      
      if (!abortController.signal.aborted) {
        if (offset === 0) {
          setAllSimilarResults(data.similar_locations);
          const filtered = filterResultsBySameCity(data.similar_locations);
          setSimilarResults(filtered.slice(0, 6));
        } else {
          setAllSimilarResults(prev => [...prev, ...data.similar_locations]);
          const allResults = [...allSimilarResults, ...data.similar_locations];
          const filtered = filterResultsBySameCity(allResults);
          setSimilarResults(filtered.slice(0, similarResults.length + 6));
        }
        
        setPaginationInfo(data.pagination);
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        return;
      }
      
      console.error('Error finding similar locations:', error);
      
      if (!abortController.signal.aborted) {
        if (offset === 0) {
          setSimilarResults([]);
          setAllSimilarResults([]);
          setPaginationInfo(null);
        }
      }
    } finally {
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

  const loadMoreSimilarLocations = () => {
    if (paginationInfo && paginationInfo.has_more && primarySelectionId && !loadingMoreResults) {
      findSimilarLocations(primarySelectionId, paginationInfo.next_offset, 6, similarityMethod);
    }
  };

  const handleMethodChange = (newMethod) => {
    if (!availableMethods[newMethod]) {
      return;
    }
    
    setSimilarityMethod(newMethod);
    
    if (primarySelectionId) {
      if (currentRequestRef.current) {
        currentRequestRef.current.abort();
        currentRequestRef.current = null;
      }
      if (loadMoreRequestRef.current) {
        loadMoreRequestRef.current.abort();
        loadMoreRequestRef.current = null;
      }
      
      setFindingSimilar(false);
      setLoadingMoreResults(false);
      setSimilarResults([]);
      setAllSimilarResults([]);
      setPaginationInfo(null);
      
      findSimilarLocations(primarySelectionId, 0, 6, newMethod);
    }
  };

  // Handle same-city toggle change
  const handleSameCityToggle = () => {
    const newIncludeSameCity = !includeSameCity;
    setIncludeSameCity(newIncludeSameCity);
    
    // Re-filter existing results
    const filtered = filterResultsBySameCity(allSimilarResults);
    setSimilarResults(filtered.slice(0, similarResults.length || 6));
  };

  // Re-filter when toggle changes
  useEffect(() => {
    if (allSimilarResults.length > 0) {
      const filtered = filterResultsBySameCity(allSimilarResults);
      setSimilarResults(filtered.slice(0, similarResults.length || 6));
    }
  }, [includeSameCity, allSimilarResults, filterResultsBySameCity]);

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

  useEffect(() => {
    if (primarySelectionId) {
      if (currentRequestRef.current) {
        currentRequestRef.current.abort();
        currentRequestRef.current = null;
      }
      
      if (loadMoreRequestRef.current) {
        loadMoreRequestRef.current.abort();
        loadMoreRequestRef.current = null;
      }
      
      setFindingSimilar(false);
      setLoadingMoreResults(false);
      
      findSimilarLocations(primarySelectionId, 0, 6, similarityMethod);
    }
  }, [primarySelectionId, similarityMethod]);
  
  useEffect(() => {
    return () => {
      if (currentRequestRef.current) {
        currentRequestRef.current.abort();
      }
      if (loadMoreRequestRef.current) {
        loadMoreRequestRef.current.abort();
      }
    };
  }, []);

  return (
    <>
      {selectedLocation && (
        <div className="selected-card">
          <button 
            className="zoom-to-location-btn"
            onClick={() => {
              window.dispatchEvent(new CustomEvent('zoomToLocation', {
                detail: { 
                  longitude: selectedLocation.longitude, 
                  latitude: selectedLocation.latitude, 
                  id: selectedLocation.id 
                }
              }));
              window.dispatchEvent(new CustomEvent('zoomToUmapPoint', {
                detail: { locationId: selectedLocation.id }
              }));
            }}
            title="Zoom to this location on map and UMAP"
          >
            🎯 Back
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

      {primarySelectionId && (
        <div className="similar-section">
          <div className="similar-header">
            <h4>
              Similar to {locations.find(l => l.id === primarySelectionId)?.city || 'Selected Location'}
            </h4>
            {paginationInfo && (
              <div className="similar-count">
                {similarResults.length} shown
              </div>
            )}
          </div>
          
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
          
          {/* Same City Toggle */}
          <div className="city-filter-toggle">
            <label className="toggle-label">
              🏙️ Include same city
            </label>
            <div 
              className={`toggle-switch ${includeSameCity ? 'active' : ''}`}
              onClick={handleSameCityToggle}
            >
              <div className="toggle-slider"></div>
            </div>
          </div>
          {!includeSameCity && (
            <div className="toggle-info">
              Showing results from other cities only
            </div>
          )}
          
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
              {similarityMethod === 'adaptive_mixed' && (
                <div style={{ fontSize: '7px', color: '#6b7280', marginTop: '2px' }}>
                  Best for: Optimal aggregation for diverse urban patterns
                </div>
              )}
            </div>
          )}
          
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
                    const isSameCity = selectedLocation && 
                      location.city === selectedLocation.city && 
                      location.country === selectedLocation.country;
                    
                    return (
                      <div 
                        key={`${location.id}-${index}`}
                        className={`similarity-tile ${isSameCity ? 'same-city' : ''}`}
                        onClick={() => onNavigationClick(location)}
                        title={`${location.city}, ${location.country} - ${similarity}% similar ${isSameCity ? '(same city)' : ''}`}
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
                      <>Show more</>
                    )}
                  </button>
                )}
                
                {paginationInfo && !paginationInfo.has_more && similarResults.length > 6 && (
                  <div className="results-summary">
                    Showing {similarResults.length} results
                    {!includeSameCity && " (excluding same city)"}
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
  const [showHelp, setShowHelp] = useState(false);
  const [hasFirstSelection, setHasFirstSelection] = useState(false);
  
  // City/Country selection state
  const [cityFilteredLocations, setCityFilteredLocations] = useState(new Set());
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedCity, setSelectedCity] = useState('');
  const [countryInput, setCountryInput] = useState('');
  const [cityInput, setCityInput] = useState('');
  const [allCountries, setAllCountries] = useState([]);
  const [availableCities, setAvailableCities] = useState([]);

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

  // Add keyboard shortcut for help
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
        setShowHelp(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
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
      
      const uniqueCountries = [...new Set(locationData.map(loc => loc.country))].sort();
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
      const citiesInCountry = locations
        .filter(loc => loc.country === selectedCountry)
        .map(loc => loc.city);
      const uniqueCitiesInCountry = [...new Set(citiesInCountry)].sort();
      setAvailableCities(uniqueCitiesInCountry);
    } else {
      const allCities = [...new Set(locations.map(loc => `${loc.city}, ${loc.country}`))].sort();
      setAvailableCities(allCities);
    }
  }, [selectedCountry, locations]);

  // Trigger UMAP highlighting when city/country filter changes
  useEffect(() => {
    if (cityFilteredLocations.size > 0) {
      window.dispatchEvent(new CustomEvent('centerOnCityTiles', {
        detail: { locationIds: Array.from(cityFilteredLocations) }
      }));
    }
  }, [cityFilteredLocations]);

  const clearSelection = () => {
    setSelectedLocations(new Set());
    setCurrentSelectedLocation(null);
    setPrimarySelectionId(null);
    setSimilarResults([]);
    setHasFirstSelection(false);
  };

  const clearCitySelection = () => {
    setCityFilteredLocations(new Set());
    setSelectedCity('');
    setSelectedCountry('');
    setCityInput('');
    setCountryInput('');
  };

  const handleCountrySelect = (country) => {
    setSelectedCountry(country);
    setCountryInput(country);
    setSelectedCity('');
    setCityInput('');
    
    const countryLocations = locations.filter(loc => loc.country === country);
    const countryLocationIds = new Set(countryLocations.map(loc => loc.id));
    
    setCityFilteredLocations(countryLocationIds);
    
    if (countryLocations.length > 0) {
      const lons = countryLocations.map(loc => loc.longitude);
      const lats = countryLocations.map(loc => loc.latitude);
      
      const bbox = {
        minLon: Math.min(...lons),
        maxLon: Math.max(...lons),
        minLat: Math.min(...lats),
        maxLat: Math.max(...lats)
      };
      
      const lonPadding = (bbox.maxLon - bbox.minLon) * 0.1 || 0.01;
      const latPadding = (bbox.maxLat - bbox.minLat) * 0.1 || 0.01;
      
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

  const handleCitySelect = (citySelection) => {
    let city, country;
    
    if (selectedCountry && !citySelection.includes(',')) {
      city = citySelection;
      country = selectedCountry;
    } else {
      const parts = citySelection.split(', ');
      if (parts.length >= 2) {
        city = parts[0];
        country = parts.slice(1).join(', ');
      } else {
        console.warn('Invalid city selection format:', citySelection);
        return;
      }
    }
    
    setSelectedCity(city);
    setCityInput(citySelection);
    
    if (!selectedCountry) {
      setSelectedCountry(country);
      setCountryInput(country);
    }
    
    const cityLocations = locations.filter(loc => 
      loc.city === city && loc.country === country
    );
    const cityLocationIds = new Set(cityLocations.map(loc => loc.id));
    
    setCityFilteredLocations(cityLocationIds);
    
    if (cityLocations.length > 0) {
      const lons = cityLocations.map(loc => loc.longitude);
      const lats = cityLocations.map(loc => loc.latitude);
      
      const bbox = {
        minLon: Math.min(...lons),
        maxLon: Math.max(...lons),
        minLat: Math.min(...lats),
        maxLat: Math.max(...lats)
      };
      
      const lonPadding = (bbox.maxLon - bbox.minLon) * 0.1 || 0.01;
      const latPadding = (bbox.maxLat - bbox.minLat) * 0.1 || 0.01;
      
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

  const handlePrimarySelection = (locationId) => {
    console.log('Primary selection:', locationId);
    
    // Mark that user has made their first selection
    setHasFirstSelection(true);
    
    // Clear city/country selection when user selects individual tile
    clearCitySelection();
    
    const newSelected = new Set();
    let newCurrentLocation = null;

    newSelected.add(locationId);
    newCurrentLocation = locations.find(loc => loc.id === locationId);
    
    setSelectedLocations(newSelected);
    setCurrentSelectedLocation(newCurrentLocation);
    setPrimarySelectionId(locationId);
    
    if (newCurrentLocation) {
      setSimilarResults([]);
      
      window.dispatchEvent(new CustomEvent('zoomToUmapPoint', {
        detail: { locationId: locationId }
      }));
    } else {
      setSimilarResults([]);
    }
  };

  const handleNavigationClick = (location) => {
    console.log('Navigation to:', location.city, location.country);
    
    window.dispatchEvent(new CustomEvent('zoomToLocation', {
      detail: { longitude: location.longitude, latitude: location.latitude, id: location.id }
    }));
    window.dispatchEvent(new CustomEvent('zoomToUmapPoint', {
      detail: { locationId: location.id }
    }));
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
      return '';
    }
    
    try {
      const tileBoundaryCoords = createTilePolygon(longitude, latitude, TILE_SIZE_METERS);
      const coordsForPolyline = tileBoundaryCoords.map(coord => [coord[1], coord[0]]);
      const encoded = polyline.encode(coordsForPolyline);
      const urlEncodedPolyline = encodeURIComponent(encoded);
      const tilePath = `path-3+ffffff-0.90(${urlEncodedPolyline})`;
      
      const imageUrl = `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/${tilePath}/${longitude},${latitude},${zoom}/${width}x${height}@2x?access_token=${mapboxToken}`;
      return imageUrl;
    } catch (error) {
      console.error('Error generating static map image:', error);
      return '';
    }
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
      {/* Help Panel */}
      <HelpPanel isOpen={showHelp} onClose={() => setShowHelp(false)} />

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <h1>
              🛰️ Satellite Embeddings Explorer
              <button 
                className="help-button" 
                onClick={() => setShowHelp(true)}
                title="Show help (press ? key)"
              >
                ?
              </button>
            </h1>
            <div className="header-stats">
              {totalLocations} locations • {stats?.countries_count || 0} countries
            </div>
          </div>
          
          <div className="header-center">
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

      {/* Main Layout */}
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

        {/* Analysis Panel - Only show after first selection */}
        <div className="analysis-panel">
          <div className="analysis-header">
            <h3>Similarity Analysis</h3>
          </div>
          <div className="analysis-content">
            {!hasFirstSelection ? (
              <WelcomePanel />
            ) : (
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