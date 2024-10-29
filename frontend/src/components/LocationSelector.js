import React, { useState, useEffect } from 'react';

const LocationSelector = ({ onSelect }) => {
  const [countries, setCountries] = useState([]);
  const [cities, setCities] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState('');
  const [selectedCity, setSelectedCity] = useState('');

  const formatCountryName = (name) => {
    return name.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');
  };

  useEffect(() => {
    fetch('http://localhost:8000/countries')
      .then(response => response.json())
      .then(data => setCountries(data.map(formatCountryName)))
      .catch(error => console.error('Error fetching countries:', error));
  }, []);

  useEffect(() => {
    if (selectedCountry) {
      const formattedCountry = selectedCountry.replace(' ', '_').toUpperCase();
      fetch(`http://localhost:8000/cities/${formattedCountry}`)
        .then(response => response.json())
        .then(data => setCities(data))
        .catch(error => console.error('Error fetching cities:', error));
    } else {
      setCities([]);
    }
  }, [selectedCountry]);

  const handleCountryChange = (event) => {
    setSelectedCountry(event.target.value);
    setSelectedCity('');
    onSelect(event.target.value, '');
  };

  const handleCityChange = (event) => {
    setSelectedCity(event.target.value);
    onSelect(selectedCountry, event.target.value);
  };

  return (
    <div className="flex gap-2">
      <select
        value={selectedCountry}
        onChange={handleCountryChange}
        className="rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
      >
        <option value="">Select a country...</option>
        {countries.map(country => (
          <option key={country} value={country}>
            {country}
          </option>
        ))}
      </select>

      <select
        value={selectedCity}
        onChange={handleCityChange}
        disabled={!selectedCountry}
        className="rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
      >
        <option value="">Select a city...</option>
        {cities.map(city => (
          <option key={city} value={city}>
            {city}
          </option>
        ))}
      </select>
    </div>
  );
};

export default LocationSelector;
