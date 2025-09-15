import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import './ModernApp.css';
import MapView from './MapView';
import HighPerformanceUMapView from './HighPerformanceUMapView';
import HelpPanel, { WelcomePanel } from './HelpPanel';
import Header from './Header';
import SimilarityPanel from './SimilarityPanel';
import { ThemeProvider } from './ThemeProvider';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? process.env.REACT_APP_API_URL || 'http://localhost:8000'
  : 'http://localhost:8000';

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

  // Initialize app - Combined useEffect
  useEffect(() => {
    const initializeApp = async () => {
      try {
        setLoading(false);
        
        const config = await loadConfig();
        setMapboxToken(config.mapbox_token);
        
        await Promise.all([
          loadStats(),
          loadLocations()
        ]);
      } catch (error) {
        console.error('Error initializing app:', error);
        setError('Failed to initialize application');
        setLoading(false);
      }
    };

    // Add keyboard shortcut for help
    const handleKeyPress = (e) => {
      if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
        setShowHelp(prev => !prev);
      }
    };

    // Initialize app and add event listener
    initializeApp();
    window.addEventListener('keydown', handleKeyPress);
    
    // Cleanup
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

  // Show error state if there's an error
  if (error) {
    return (
      <div className="app">
        <div className="loading-state" style={{ height: '100vh', justifyContent: 'center', color: 'var(--color-error)' }}>
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
      <HelpPanel 
        isOpen={showHelp} 
        onClose={() => setShowHelp(false)}
        totalLocations={totalLocations}
        stats={stats}
      />

      {/* Header */}
      <Header
        totalLocations={totalLocations}
        stats={stats}
        setShowHelp={setShowHelp}
        cityFilteredLocations={cityFilteredLocations}
        selectedCountry={selectedCountry}
        selectedCity={selectedCity}
        countryInput={countryInput}
        cityInput={cityInput}
        setCountryInput={setCountryInput}
        setCityInput={setCityInput}
        allCountries={allCountries}
        availableCities={availableCities}
        handleCountrySelect={handleCountrySelect}
        handleCitySelect={handleCitySelect}
        clearCitySelection={clearCitySelection}
        selectedLocations={selectedLocations}
        clearSelection={clearSelection}
      />

      {/* Main Layout */}
      <div className="main-container">
        {/* Map View */}
        <div className="viz-panel">
          <div className="viz-header">
            <h3>Map View</h3>
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

        {/* Analysis Panel */}
        <div className="analysis-panel">
          <div className="analysis-header">
            <h3>Similarity Analysis</h3>
          </div>
          <div className="analysis-content">
            {!hasFirstSelection ? (
              <WelcomePanel />
            ) : (
              <SimilarityPanel
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

// Render the app with ThemeProvider
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProvider>
      <ModernApp />
    </ThemeProvider>
  </React.StrictMode>
);