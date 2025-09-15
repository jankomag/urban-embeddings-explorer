import React, { useState, useEffect, useRef } from 'react';

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

export default AutocompleteInput;