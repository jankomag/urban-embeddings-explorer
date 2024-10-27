import React, { useState } from 'react';

const CustomUMAPControls = ({ onNewUMAPData, onReset, onLoadingChange }) => {
  const [nNeighbors, setNNeighbors] = useState(15);
  const [minDist, setMinDist] = useState(0.1);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  const startComputation = async () => {
    try {
      setError(null);
      setStatus('Starting computation...');
      onLoadingChange(true);
      
      const params = new URLSearchParams({
        n_neighbors: nNeighbors.toString(),
        min_dist: minDist.toString()
      });
      
      const response = await fetch(`http://localhost:8000/compute_umap?${params}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start computation');
      }
      
      const data = await response.json();
      pollStatus(data.computation_id);
    } catch (err) {
      setError(err.message || 'Failed to start computation');
      setStatus(null);
      onLoadingChange(false);
    }
  };

  const pollStatus = async (id) => {
    try {
      const response = await fetch(`http://localhost:8000/umap_status/${id}`);
      const data = await response.json();
      
      if (data.status === 'completed') {
        setStatus('Computation complete!');
        onNewUMAPData(data.data);
        onLoadingChange(false);
      } else if (data.status === 'failed') {
        setError(data.error);
        setStatus(null);
        onLoadingChange(false);
      } else {
        setStatus('Computing...');
        setTimeout(() => pollStatus(id), 1000);
      }
    } catch (err) {
      setError('Failed to fetch computation status');
      setStatus(null);
      onLoadingChange(false);
    }
  };

  const handleReset = () => {
    setStatus(null);
    setError(null);
    onLoadingChange(false);
    onReset();
  };

  return (
    <div style={{
      border: '1px solid #e0e0e0',
      borderRadius: '8px',
      padding: '20px',
      backgroundColor: 'white',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <h2 style={{
        margin: '0 0 20px 0',
        fontSize: '1.5rem',
        fontWeight: 'bold'
      }}>
        Custom UMAP Parameters
      </h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            fontSize: '0.875rem',
            fontWeight: '500'
          }}>
            Number of Neighbors:
          </label>
          <input
            type="number"
            value={nNeighbors}
            onChange={(e) => setNNeighbors(parseInt(e.target.value))}
            style={{
              width: '100%',
              padding: '8px',
              border: '1px solid #e0e0e0',
              borderRadius: '4px',
              fontSize: '0.875rem'
            }}
            min="2"
            max="100"
            disabled={!!status}
          />
        </div>
        
        <div>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            fontSize: '0.875rem',
            fontWeight: '500'
          }}>
            Minimum Distance:
          </label>
          <input
            type="number"
            value={minDist}
            onChange={(e) => setMinDist(parseFloat(e.target.value))}
            style={{
              width: '100%',
              padding: '8px',
              border: '1px solid #e0e0e0',
              borderRadius: '4px',
              fontSize: '0.875rem'
            }}
            min="0"
            max="1"
            step="0.1"
            disabled={!!status}
          />
        </div>
      </div>

      <div style={{
        display: 'flex',
        gap: '10px',
        marginTop: '20px'
      }}>
        <button
          onClick={startComputation}
          disabled={!!status}
          style={{
            flex: 1,
            padding: '10px',
            backgroundColor: status ? '#9ca3af' : '#3b82f6',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: status ? 'not-allowed' : 'pointer',
            fontSize: '0.875rem'
          }}
        >
          Compute UMAP
        </button>
        
        <button
          onClick={handleReset}
          style={{
            flex: 1,
            padding: '10px',
            backgroundColor: '#6b7280',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '0.875rem'
          }}
        >
          Reset to Default
        </button>
      </div>
      
      {status && (
        <div style={{
          marginTop: '16px',
          padding: '10px',
          backgroundColor: '#f3f4f6',
          borderRadius: '4px',
          fontSize: '0.875rem',
          color: '#4b5563'
        }}>
          {status}
        </div>
      )}
      
      {error && (
        <div style={{
          marginTop: '16px',
          padding: '10px',
          backgroundColor: '#fee2e2',
          borderRadius: '4px',
          fontSize: '0.875rem',
          color: '#dc2626'
        }}>
          {error}
        </div>
      )}
    </div>
  );
};

export default CustomUMAPControls;