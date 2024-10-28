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
    <div style={{
      background: 'white',
      padding: '20px',
      borderRadius: '10px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      display: 'flex',
      flexDirection: 'column',
      gap: '15px'
    }}>
      <h3 style={{
        margin: '0 0 10px 0',
        color: '#333',
        fontSize: '1.2rem'
      }}>
        Location Selection
      </h3>
      
      <div style={{
        display: 'flex',
        gap: '20px',
        flexWrap: 'wrap'
      }}>
        <div style={{
          flex: '1',
          minWidth: '200px'
        }}>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            color: '#666',
            fontSize: '0.9rem'
          }}>
            Select Country
          </label>
          <select
            value={selectedCountry}
            onChange={handleCountryChange}
            style={{
              width: '100%',
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid #ddd',
              backgroundColor: '#fff',
              fontSize: '0.9rem',
              color: '#333',
              cursor: 'pointer'
            }}
          >
            <option value="">Select a country...</option>
            {countries.map(country => (
              <option key={country} value={country}>
                {country}
              </option>
            ))}
          </select>
        </div>

        <div style={{
          flex: '1',
          minWidth: '200px'
        }}>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            color: '#666',
            fontSize: '0.9rem'
          }}>
            Select City
          </label>
          <select
            value={selectedCity}
            onChange={handleCityChange}
            disabled={!selectedCountry}
            style={{
              width: '100%',
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid #ddd',
              backgroundColor: selectedCountry ? '#fff' : '#f5f5f5',
              fontSize: '0.9rem',
              color: '#333',
              cursor: selectedCountry ? 'pointer' : 'not-allowed'
            }}
          >
            <option value="">Select a city...</option>
            {cities.map(city => (
              <option key={city} value={city}>
                {city}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
};

export default LocationSelector;
