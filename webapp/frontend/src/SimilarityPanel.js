import React, { useState, useEffect, useRef, useCallback } from 'react';
import polyline from '@mapbox/polyline';

const TILE_SIZE_METERS = 2240;

// ***** ZOOM ADJUSTMENT PARAMETERS - ADJUST THESE *****
const ZOOM_CONFIG = {
  BASE_ZOOM: 11.5,        // Base zoom level for equatorial regions
  MIN_ZOOM: 11.2,         // Minimum zoom level (for high latitudes)
  MAX_ZOOM: 12.1,         // Maximum zoom level (for equatorial regions)
  LATITUDE_THRESHOLD: 60  // Latitude above which we start reducing zoom more aggressively
};

// ***** NEW PAGINATION CONFIGURATION *****
const PAGINATION_CONFIG = {
  INITIAL_DISPLAY: 6,     // Show 9 tiles initially (3x3 grid)
  BATCH_SIZE: 12,         // Load 18 tiles per batch (2x initial display)
  MAX_TOTAL_RESULTS: 100, // Maximum total results to prevent infinite loading
  AUTO_REFETCH_THRESHOLD: 2, // If less than 3 tiles after filtering, auto-refetch more
  SMART_FETCH_MULTIPLIER: 2  // Fetch 3x more when we expect filtering
};

function SimilarityPanel({ 
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
  // Enhanced state management
  const [paginationInfo, setPaginationInfo] = useState(null);
  const [loadingMoreResults, setLoadingMoreResults] = useState(false);
  const [similarityMethod, setSimilarityMethod] = useState('mean');
  const [includeSameCity, setIncludeSameCity] = useState(true);
  const [allSimilarResults, setAllSimilarResults] = useState([]);
  const [filteredResults, setFilteredResults] = useState([]);
  const [displayedResults, setDisplayedResults] = useState([]);
  const [totalFetched, setTotalFetched] = useState(0);
  const [canLoadMore, setCanLoadMore] = useState(false);
  const [autoRefetchInProgress, setAutoRefetchInProgress] = useState(false);
  const [availableMethods, setAvailableMethods] = useState({
    'mean': { 
      name: "Mean", 
      description: "Standard aggregation" 
    },
    // 'median': { 
    //   name: "Median", 
    //   description: "Robust to outliers" 
    // },
    // 'min': { 
    //   name: "Min", 
    //   description: "Shared baseline features" 
    // },
    // 'max': { 
    //   name: "Max", 
    //   description: "Distinctive features" 
    // },
    'dominant_cluster': {
      name: "Dominant Cluster",
      description: "Unique characteristics"
    }
  });
  const [loadingMethods, setLoadingMethods] = useState(false);
  
  const currentRequestRef = useRef(null);
  const loadMoreRequestRef = useRef(null);
  const lastFilterToggleRef = useRef(Date.now()); // Track when filter was last toggled

  const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? process.env.REACT_APP_API_URL || 'http://localhost:8000'
    : 'http://localhost:8000';

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
        const availableKeys = ['mean', 'dominant_cluster'];
        
        for (const key of availableKeys) {
          if (data.available_methods[key]) {
            const shortNames = {
              'mean': { name: "Mean", description: "Standard aggregation" },
              // 'median': { name: "Median", description: "Robust to outliers" },
              // 'min': { name: "Min", description: "Shared baseline features" },
              // 'max': { name: "Max", description: "Distinctive features" },
              'dominant_cluster': { name: "Dominant Cluster", description: "Unique characteristics" },
              // 'global_contrastive': { name: "Global Contrastive", description: "" },
            };
            filteredMethods[key] = shortNames[key] || data.available_methods[key];
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

  // Enhanced filtering function with statistics
  const filterResultsBySameCity = useCallback((results) => {
    if (!selectedLocation || !results.length) return results;
    
    if (includeSameCity) {
      return results;
    } else {
      const filtered = results.filter(location => 
        location.city !== selectedLocation.city || 
        location.country !== selectedLocation.country
      );
      
      // Debug info for understanding filtering
      const sameCityCount = results.length - filtered.length;
      console.log(`Filtering: ${results.length} total, ${filtered.length} after filter, ${sameCityCount} same-city excluded`);
      
      return filtered;
    }
  }, [includeSameCity, selectedLocation]);

  // ***** ENHANCED SMART FETCHING FUNCTION *****
  const findSimilarLocations = async (
    locationId, 
    offset = 0, 
    limit = PAGINATION_CONFIG.BATCH_SIZE, 
    method = similarityMethod,
    isAutoRefetch = false
  ) => {
    if (!availableMethods[method]) {
      return;
    }
    
    // Cancel existing requests
    if (offset === 0 && currentRequestRef.current) {
      currentRequestRef.current.abort();
      currentRequestRef.current = null;
    }
    
    if (offset > 0 && loadMoreRequestRef.current) {
      loadMoreRequestRef.current.abort();
      loadMoreRequestRef.current = null;
    }
    
    const abortController = new AbortController();
    
    // Set loading states
    if (offset === 0 && !isAutoRefetch) {
      currentRequestRef.current = abortController;
      setFindingSimilar(true);
      setAllSimilarResults([]);
      setFilteredResults([]);
      setDisplayedResults([]);
      setTotalFetched(0);
      setPaginationInfo(null);
      setCanLoadMore(false);
    } else {
      loadMoreRequestRef.current = abortController;
      if (isAutoRefetch) {
        setAutoRefetchInProgress(true);
      } else {
        setLoadingMoreResults(true);
      }
    }

    try {
      // ***** SMART LIMIT CALCULATION *****
      // If we're not including same city, fetch more to account for filtering
      let adjustedLimit = limit;
      if (!includeSameCity || isAutoRefetch) {
        adjustedLimit = Math.min(limit * PAGINATION_CONFIG.SMART_FETCH_MULTIPLIER, 50);
      }
      
      console.log(`Fetching: offset=${offset}, limit=${adjustedLimit}, includeSameCity=${includeSameCity}, isAutoRefetch=${isAutoRefetch}`);
      
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
        let newAllResults;
        
        if (offset === 0) {
          // Fresh search
          newAllResults = data.similar_locations;
          setAllSimilarResults(newAllResults);
          setTotalFetched(newAllResults.length);
        } else {
          // Append to existing results
          newAllResults = [...allSimilarResults, ...data.similar_locations];
          setAllSimilarResults(newAllResults);
          setTotalFetched(newAllResults.length);
        }
        
        // Apply filtering
        const newFiltered = filterResultsBySameCity(newAllResults);
        setFilteredResults(newFiltered);
        
        // Update displayed results
        const newDisplayed = newFiltered.slice(0, displayedResults.length + PAGINATION_CONFIG.INITIAL_DISPLAY);
        setDisplayedResults(newDisplayed);
        
        // Update pagination info
        setPaginationInfo(data.pagination);
        
        // ***** AUTO-REFETCH LOGIC *****
        const shouldAutoRefetch = (
          !includeSameCity && // Only when filtering out same city
          offset === 0 && // Only on initial load
          newFiltered.length < PAGINATION_CONFIG.AUTO_REFETCH_THRESHOLD && // Too few results
          data.pagination?.has_more && // More results available
          totalFetched < PAGINATION_CONFIG.MAX_TOTAL_RESULTS && // Haven't hit max limit
          !isAutoRefetch // Prevent infinite recursion
        );
        
        if (shouldAutoRefetch) {
          console.log(`Auto-refetching: only ${newFiltered.length} results after filtering, fetching more...`);
          // Auto-fetch more results
          setTimeout(() => {
            findSimilarLocations(locationId, newAllResults.length, PAGINATION_CONFIG.BATCH_SIZE, method, true);
          }, 100);
        } else {
          // Determine if we can load more
          const hasMoreResults = data.pagination?.has_more && newFiltered.length < PAGINATION_CONFIG.MAX_TOTAL_RESULTS;
          const hasMoreToDisplay = newFiltered.length > newDisplayed.length;
          setCanLoadMore(hasMoreResults || hasMoreToDisplay);
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        return;
      }
      
      console.error('Error finding similar locations:', error);
      
      if (!abortController.signal.aborted) {
        if (offset === 0 && !isAutoRefetch) {
          setAllSimilarResults([]);
          setFilteredResults([]);
          setDisplayedResults([]);
          setTotalFetched(0);
          setPaginationInfo(null);
          setCanLoadMore(false);
        }
      }
    } finally {
      if (!abortController.signal.aborted) {
        if (offset === 0 && !isAutoRefetch) {
          setFindingSimilar(false);
          currentRequestRef.current = null;
        } else {
          setLoadingMoreResults(false);
          setAutoRefetchInProgress(false);
          loadMoreRequestRef.current = null;
        }
      }
    }
  };

  // ***** ENHANCED LOAD MORE FUNCTION *****
  const loadMoreSimilarLocations = () => {
    if (!primarySelectionId || loadingMoreResults || autoRefetchInProgress) return;
    
    // Check if we can display more from already filtered results
    const canDisplayMore = filteredResults.length > displayedResults.length;
    
    if (canDisplayMore) {
      // Display more from existing filtered results
      const newDisplayCount = Math.min(
        displayedResults.length + PAGINATION_CONFIG.INITIAL_DISPLAY,
        filteredResults.length
      );
      const newDisplayed = filteredResults.slice(0, newDisplayCount);
      setDisplayedResults(newDisplayed);
      
      // Update canLoadMore based on remaining results
      const stillHasMoreToDisplay = filteredResults.length > newDisplayed.length;
      const hasMoreFromAPI = paginationInfo?.has_more && totalFetched < PAGINATION_CONFIG.MAX_TOTAL_RESULTS;
      setCanLoadMore(stillHasMoreToDisplay || hasMoreFromAPI);
    } else if (paginationInfo?.has_more && totalFetched < PAGINATION_CONFIG.MAX_TOTAL_RESULTS) {
      // Fetch more from API
      findSimilarLocations(primarySelectionId, totalFetched, PAGINATION_CONFIG.BATCH_SIZE, similarityMethod);
    }
  };

  // ***** ENHANCED METHOD CHANGE HANDLER *****
  const handleMethodChange = (newMethod) => {
    if (!availableMethods[newMethod] || newMethod === similarityMethod) {
      return;
    }
    
    setSimilarityMethod(newMethod);
    
    if (primarySelectionId) {
      // Cancel any ongoing requests
      if (currentRequestRef.current) {
        currentRequestRef.current.abort();
        currentRequestRef.current = null;
      }
      if (loadMoreRequestRef.current) {
        loadMoreRequestRef.current.abort();
        loadMoreRequestRef.current = null;
      }
      
      // Reset all states
      setFindingSimilar(false);
      setLoadingMoreResults(false);
      setAutoRefetchInProgress(false);
      setAllSimilarResults([]);
      setFilteredResults([]);
      setDisplayedResults([]);
      setTotalFetched(0);
      setPaginationInfo(null);
      setCanLoadMore(false);
      
      // Start fresh search
      findSimilarLocations(primarySelectionId, 0, PAGINATION_CONFIG.BATCH_SIZE, newMethod);
    }
  };

  // ***** ENHANCED SAME CITY TOGGLE HANDLER *****
  const handleSameCityToggle = () => {
    const newIncludeSameCity = !includeSameCity;
    setIncludeSameCity(newIncludeSameCity);
    lastFilterToggleRef.current = Date.now();
    
    if (allSimilarResults.length > 0) {
      // Apply new filtering to existing results
      const newFiltered = filterResultsBySameCity(allSimilarResults);
      setFilteredResults(newFiltered);
      
      // Reset displayed results to initial count
      const newDisplayed = newFiltered.slice(0, PAGINATION_CONFIG.INITIAL_DISPLAY);
      setDisplayedResults(newDisplayed);
      
      // ***** SMART AUTO-REFETCH ON TOGGLE *****
      const shouldAutoRefetch = (
        !newIncludeSameCity && // Toggled to exclude same city
        newFiltered.length < PAGINATION_CONFIG.AUTO_REFETCH_THRESHOLD && // Too few results after filtering
        paginationInfo?.has_more && // More results available from API
        totalFetched < PAGINATION_CONFIG.MAX_TOTAL_RESULTS && // Haven't hit max limit
        primarySelectionId // Have a selection to search for
      );
      
      if (shouldAutoRefetch) {
        console.log(`Toggle auto-refetch: only ${newFiltered.length} results after excluding same city, fetching more...`);
        setTimeout(() => {
          findSimilarLocations(primarySelectionId, totalFetched, PAGINATION_CONFIG.BATCH_SIZE, similarityMethod, true);
        }, 100);
      } else {
        // Update canLoadMore status
        const hasMoreToDisplay = newFiltered.length > newDisplayed.length;
        const hasMoreFromAPI = paginationInfo?.has_more && totalFetched < PAGINATION_CONFIG.MAX_TOTAL_RESULTS;
        setCanLoadMore(hasMoreToDisplay || hasMoreFromAPI);
      }
    }
  };

  // ***** EFFECT FOR UPDATING DISPLAYED RESULTS WHEN FILTER CHANGES *****
  useEffect(() => {
    if (allSimilarResults.length > 0) {
      const newFiltered = filterResultsBySameCity(allSimilarResults);
      setFilteredResults(newFiltered);
      
      // Keep current display count, but don't exceed filtered results
      const currentDisplayCount = displayedResults.length || PAGINATION_CONFIG.INITIAL_DISPLAY;
      const newDisplayed = newFiltered.slice(0, Math.min(currentDisplayCount, newFiltered.length));
      setDisplayedResults(newDisplayed);
      
      // Update canLoadMore
      const hasMoreToDisplay = newFiltered.length > newDisplayed.length;
      const hasMoreFromAPI = paginationInfo?.has_more && totalFetched < PAGINATION_CONFIG.MAX_TOTAL_RESULTS;
      setCanLoadMore(hasMoreToDisplay || hasMoreFromAPI);
    }
  }, [allSimilarResults, filterResultsBySameCity]);

  // ***** ZOOM CALCULATION FUNCTIONS (UNCHANGED) *****
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

  const calculateZoomForLatitude = (latitude) => {
    const absLat = Math.abs(latitude);
    
    if (absLat <= 20) {
      return ZOOM_CONFIG.MAX_ZOOM;
    } else if (absLat >= ZOOM_CONFIG.LATITUDE_THRESHOLD) {
      return ZOOM_CONFIG.MIN_ZOOM;
    } else {
      const ratio = (absLat - 20) / (ZOOM_CONFIG.LATITUDE_THRESHOLD - 20);
      return ZOOM_CONFIG.MAX_ZOOM - (ZOOM_CONFIG.MAX_ZOOM - ZOOM_CONFIG.MIN_ZOOM) * ratio;
    }
  };

  const getStaticMapImage = (longitude, latitude, width = 140, height = 140, customZoom = null) => {
    if (!mapboxToken || !longitude || !latitude) {
      return '';
    }
    
    try {
      const tileBoundaryCoords = createTilePolygon(longitude, latitude, TILE_SIZE_METERS);
      const coordsForPolyline = tileBoundaryCoords.map(coord => [coord[1], coord[0]]);
      const encoded = polyline.encode(coordsForPolyline);
      const urlEncodedPolyline = encodeURIComponent(encoded);
      const tilePath = `path-3+ffffff-0.90(${urlEncodedPolyline})`;
      
      const zoom = customZoom || calculateZoomForLatitude(latitude);
      
      const imageUrl = `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/${tilePath}/${longitude},${latitude},${zoom}/${width}x${height}@2x?access_token=${mapboxToken}`;
      return imageUrl;
    } catch (error) {
      console.error('Error generating static map image:', error);
      return '';
    }
  };

  // ***** INITIAL SEARCH EFFECT *****
  useEffect(() => {
    if (primarySelectionId) {
      // Cancel any ongoing requests
      if (currentRequestRef.current) {
        currentRequestRef.current.abort();
        currentRequestRef.current = null;
      }
      
      if (loadMoreRequestRef.current) {
        loadMoreRequestRef.current.abort();
        loadMoreRequestRef.current = null;
      }
      
      // Reset states
      setFindingSimilar(false);
      setLoadingMoreResults(false);
      setAutoRefetchInProgress(false);
      
      // Start fresh search
      findSimilarLocations(primarySelectionId, 0, PAGINATION_CONFIG.BATCH_SIZE, similarityMethod);
    }
  }, [primarySelectionId, similarityMethod]);
  
  // ***** CLEANUP EFFECT *****
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
          <div className="selected-tile-container">
            {mapboxToken && selectedLocation && (
              <img
                src={getStaticMapImage(selectedLocation.longitude, selectedLocation.latitude, 120, 120)}
                alt={`${selectedLocation.city} satellite view`}
                className="selected-image"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            )}
          </div>
          
          <div className="selected-info">
            {/* UPDATED: Title with inline button */}
            <div className="selected-title-row">
              <div>
                <h4 className="selected-title">
                  {selectedLocation.city}, {selectedLocation.country}
                </h4>
                <div className="selected-meta">
                  {selectedLocation.continent}
                </div>
              </div>
              
              <button 
                className="zoom-to-location-btn-inline"
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
                Show Source
              </button>
            </div>
            
            {/* UPDATED: Controls in one row */}
            <div className="controls-row">
              <div className="method-selector-inline">
                <label className="method-selector-label">Method:</label>
                <select 
                  value={similarityMethod} 
                  onChange={(e) => handleMethodChange(e.target.value)}
                  className="method-dropdown-inline"
                  disabled={loadingMethods || findingSimilar}
                >
                  {Object.entries(availableMethods).map(([key, method]) => (
                    <option key={key} value={key}>
                      {method.name}
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="city-filter-toggle-inline">
                <span className="toggle-label-inline">Same City</span>
                <div 
                  className={`toggle-switch-inline ${includeSameCity ? 'active' : ''}`}
                  onClick={handleSameCityToggle}
                >
                  <div className="toggle-slider-inline"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {primarySelectionId && (
        <div className="similar-section">
          <div className="similar-header">
            <h4>
              Similar to {locations.find(l => l.id === primarySelectionId)?.city || 'Selected Location'}
            </h4>
          </div>
          
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
                  {displayedResults.map((location, index) => {
                    const similarity = (location.similarity_score * 100).toFixed(1);
                    const imageUrl = getStaticMapImage(location.longitude, location.latitude, 140, 140);
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
                
                {/* Enhanced Load More Button */}
                {canLoadMore && (
                  <button 
                    className="show-more"
                    onClick={loadMoreSimilarLocations}
                    disabled={loadingMoreResults || autoRefetchInProgress}
                  >
                    {loadingMoreResults ? (
                      <>
                        <div className="button-spinner"></div>
                        Loading more...
                      </>
                    ) : autoRefetchInProgress ? (
                      <>
                        <div className="button-spinner"></div>
                        Finding more results...
                      </>
                    ) : (
                      <>
                        Show more
                        {filteredResults.length > displayedResults.length ? 
                          ` (${Math.min(PAGINATION_CONFIG.INITIAL_DISPLAY, filteredResults.length - displayedResults.length)} more ready)` :
                          ''
                        }
                      </>
                    )}
                  </button>
                )}
                
                {/* Results Summary */}
                {displayedResults.length > 0 && !canLoadMore && (
                  <div className="results-summary">
                    Showing all {displayedResults.length} results
                    {!includeSameCity && " (excluding same city)"}
                    {totalFetched >= PAGINATION_CONFIG.MAX_TOTAL_RESULTS && " (search limit reached)"}
                  </div>
                )}
                
                {/* Empty State Handling */}
                {!findingSimilar && !loadingMoreResults && !autoRefetchInProgress && displayedResults.length === 0 && (
                  <div className="empty-state">
                    <p>No similar locations found</p>
                    {!includeSameCity && (
                      <div className="empty-info">
                        <h5>Try enabling "Same City"</h5>
                        <p>You might find similar areas within the same city.</p>
                      </div>
                    )}
                  </div>
                )}
                
                {/* Auto-refetch indicator */}
                {autoRefetchInProgress && (
                  <div className="loading-state" style={{ padding: '12px' }}>
                    <div className="spinner" style={{ width: '16px', height: '16px' }}></div>
                    <div className="loading-text" style={{ fontSize: '10px' }}>
                      Searching for more diverse results...
                    </div>
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

export default SimilarityPanel;