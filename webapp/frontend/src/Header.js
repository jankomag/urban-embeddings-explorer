import React from 'react';
import SearchInput from './SearchInput';
import ThemeToggle from './ThemeToggle';

function Header({
  totalLocations,
  stats,
  setShowHelp,
  cityFilteredLocations,
  selectedCountry,
  selectedCity,
  locations, // Add locations prop for smart search
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
        
        {/* Center: Smart Search */}
        <div className="header-center">
          <div className="search-container">
            <SearchInput
              locations={locations}
              onCountrySelect={handleCountrySelect}
              onCitySelect={handleCitySelect}
              onClear={clearCitySelection}
              selectedCountry={selectedCountry}
              selectedCity={selectedCity}
            />
            
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