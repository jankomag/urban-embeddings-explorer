import React from 'react';
import AutocompleteInput from './AutocompleteInput';
import ThemeToggle from './ThemeToggle';

function Header({
  totalLocations,
  stats,
  setShowHelp,
  cityFilteredLocations,
  selectedCountry,
  selectedCity,
  countryInput,
  cityInput,
  setCountryInput,
  setCityInput,
  allCountries,
  availableCities,
  handleCountrySelect,
  handleCitySelect,
  clearCitySelection,
  selectedLocations,
  clearSelection
}) {
  return (
    <header className="header">
      <div className="header-content">
        {/* Left: App title */}
        <div className="header-left">
          <h1 className="app-title">
            Urban Embeddings Explorer
          </h1>
        </div>
        
        {/* Center: Search */}
        <div className="header-center">
          <div className="search-container">
            <AutocompleteInput
              value={countryInput}
              onChange={setCountryInput}
              onSelect={handleCountrySelect}
              options={allCountries}
              placeholder="Search country..."
            />
            
            <AutocompleteInput
              value={cityInput}
              onChange={setCityInput}
              onSelect={handleCitySelect}
              options={availableCities}
              placeholder={selectedCountry ? "Search city..." : "Search city, country..."}
              disabled={allCountries.length === 0}
            />
            
            {(selectedCity || selectedCountry) && (
              <button 
                className="clear-search-btn" 
                onClick={clearCitySelection}
                title="Clear search"
                aria-label="Clear search"
              >
                ✕
              </button>
            )}
            
            {/* Active filter indicator - now inline with search */}
            {(selectedCity || selectedCountry) && (
              <div className="active-filter-inline">
                <span className="filter-text">
                  {selectedCity ? `${selectedCity}, ${selectedCountry}` : selectedCountry}
                </span>
                <span className="tile-count">
                  {cityFilteredLocations.size.toLocaleString()}
                </span>
              </div>
            )}
          </div>
        </div>
        
        {/* Right: Theme Toggle, Help and Actions */}
        <div className="header-right">
          <ThemeToggle />
          
          <button 
            className="help-button" 
            onClick={() => setShowHelp(true)}
            title="Help (press ? key)"
            aria-label="Show help"
          >
            ?
          </button>
          
          {selectedLocations.size > 0 && (
            <div className="selection-status">
              <span className="selected-count">{selectedLocations.size} selected</span>
              <button 
                className="clear-selection-btn" 
                onClick={clearSelection}
                title="Clear selection"
              >
                Clear
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;