import React, { useState, useEffect, useRef } from 'react';

function SearchInput({ 
  locations, 
  onCountrySelect, 
  onCitySelect, 
  onClear,
  selectedCountry,
  selectedCity 
}) {
  const [value, setValue] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [filteredOptions, setFilteredOptions] = useState([]);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const inputRef = useRef(null);
  const dropdownRef = useRef(null);

  // Update input value when external selections change
  useEffect(() => {
    if (selectedCity && selectedCountry) {
      setValue(`${selectedCity}, ${selectedCountry}`);
    } else if (selectedCountry) {
      setValue(selectedCountry);
    } else {
      setValue('');
    }
  }, [selectedCity, selectedCountry]);

  // Create hierarchical search options
  const createSearchOptions = (searchTerm) => {
    if (!searchTerm || searchTerm.length < 1) return [];

    const term = searchTerm.toLowerCase();
    const options = [];
    const seenCountries = new Set();
    const seenCities = new Set();

    // Group locations by country
    const locationsByCountry = {};
    locations.forEach(location => {
      if (!locationsByCountry[location.country]) {
        locationsByCountry[location.country] = new Set();
      }
      locationsByCountry[location.country].add(location.city);
    });

    // First, add matching countries
    Object.keys(locationsByCountry).forEach(country => {
      if (country.toLowerCase().includes(term) && !seenCountries.has(country)) {
        const cityCount = locationsByCountry[country].size;
        options.push({
          type: 'country',
          display: country,
          value: country,
          subtitle: `${cityCount} cities`,
          searchText: country.toLowerCase()
        });
        seenCountries.add(country);
      }
    });

    // Then, add matching cities grouped by country
    Object.entries(locationsByCountry).forEach(([country, cities]) => {
      const matchingCities = Array.from(cities).filter(city => 
        city.toLowerCase().includes(term) && !seenCities.has(`${city}, ${country}`)
      );

      matchingCities.forEach(city => {
        options.push({
          type: 'city',
          display: `${city}, ${country}`,
          value: city,
          country: country,
          subtitle: 'City',
          searchText: `${city} ${country}`.toLowerCase()
        });
        seenCities.add(`${city}, ${country}`);
      });
    });

    // Sort: countries first, then cities, both alphabetically
    options.sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === 'country' ? -1 : 1;
      }
      return a.display.localeCompare(b.display);
    });

    return options.slice(0, 15); // Limit results
  };

  useEffect(() => {
    const options = createSearchOptions(value);
    setFilteredOptions(options);
    setIsOpen(options.length > 0 && document.activeElement === inputRef.current);
    setHighlightedIndex(-1);
  }, [value, locations]);

  const handleInputChange = (e) => {
    setValue(e.target.value);
  };

  const handleOptionSelect = (option) => {
    if (option.type === 'country') {
      onCountrySelect(option.value);
      setValue(option.value);
    } else if (option.type === 'city') {
      onCitySelect(`${option.value}, ${option.country}`);
      setValue(`${option.value}, ${option.country}`);
    }
    
    setIsOpen(false);
    setHighlightedIndex(-1);
    inputRef.current?.blur();
  };

  const handleClear = () => {
    setValue('');
    onClear();
    setIsOpen(false);
    setHighlightedIndex(-1);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (!isOpen || filteredOptions.length === 0) return;

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

  const getIcon = (type) => {
    return type === 'country' ? '🌍' : '🏙️';
  };

  return (
    <div className="smart-search-container" ref={dropdownRef}>
      <div className="smart-search-input-wrapper">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => {
            if (filteredOptions.length > 0) setIsOpen(true);
          }}
          placeholder="Search cities or countries..."
          className="smart-search-input"
          autoComplete="off"
        />
        
        {(selectedCity || selectedCountry) && (
          <button 
            className="smart-search-clear" 
            onClick={handleClear}
            title="Clear search"
            aria-label="Clear search"
          >
            ✕
          </button>
        )}
      </div>
      
      {isOpen && filteredOptions.length > 0 && (
        <div className="smart-search-dropdown">
          {filteredOptions.map((option, index) => (
            <div
              key={`${option.type}-${option.display}`}
              className={`smart-search-item ${option.type} ${index === highlightedIndex ? 'highlighted' : ''}`}
              onClick={() => handleOptionSelect(option)}
              onMouseEnter={() => setHighlightedIndex(index)}
            >
              <div className="search-item-main">
                <span className="search-item-icon">{getIcon(option.type)}</span>
                <span className="search-item-text">{option.display}</span>
              </div>
              <span className="search-item-subtitle">{option.subtitle}</span>
            </div>
          ))}
        </div>
      )}
      
      {isOpen && filteredOptions.length === 0 && value.length > 0 && (
        <div className="smart-search-dropdown">
          <div className="smart-search-no-results">
            No cities or countries found
          </div>
        </div>
      )}
    </div>
  );
}

export default SearchInput;