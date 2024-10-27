import React from 'react';

const DimensionSelector = ({ 
  mode, 
  setMode, 
  dimensionX, 
  setDimensionX, 
  dimensionY, 
  setDimensionY, 
  maxDimensions,
  disabled 
}) => {
  return (
    <div style={{
      border: '1px solid #e0e0e0',
      borderRadius: '8px',
      padding: '20px',
      backgroundColor: 'white',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <div style={{
        marginBottom: '16px'
      }}>
        <h3 style={{
          fontSize: '1.2rem',
          fontWeight: 'bold',
          marginBottom: '12px'
        }}>
          Visualization Mode
        </h3>
        
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '16px'
        }}>
          <div>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              disabled={disabled}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #e0e0e0',
                borderRadius: '4px',
                backgroundColor: disabled ? '#f5f5f5' : 'white'
              }}
            >
              <option value="umap">UMAP Projection</option>
              <option value="original">Original Embeddings</option>
            </select>
          </div>
          
          {mode === 'original' && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '16px'
            }}>
              <div>
                <label style={{
                  display: 'block',
                  marginBottom: '8px',
                  fontSize: '0.875rem',
                  fontWeight: '500'
                }}>
                  X Axis Dimension
                </label>
                <select
                  value={dimensionX}
                  onChange={(e) => setDimensionX(parseInt(e.target.value))}
                  disabled={disabled}
                  style={{
                    width: '100%',
                    padding: '8px',
                    border: '1px solid #e0e0e0',
                    borderRadius: '4px',
                    backgroundColor: disabled ? '#f5f5f5' : 'white'
                  }}
                >
                  {[...Array(maxDimensions)].map((_, i) => (
                    <option key={i} value={i}>Dimension {i + 1}</option>
                  ))}
                </select>
              </div>
              <div>
                <label style={{
                  display: 'block',
                  marginBottom: '8px',
                  fontSize: '0.875rem',
                  fontWeight: '500'
                }}>
                  Y Axis Dimension
                </label>
                <select
                  value={dimensionY}
                  onChange={(e) => setDimensionY(parseInt(e.target.value))}
                  disabled={disabled}
                  style={{
                    width: '100%',
                    padding: '8px',
                    border: '1px solid #e0e0e0',
                    borderRadius: '4px',
                    backgroundColor: disabled ? '#f5f5f5' : 'white'
                  }}
                >
                  {[...Array(maxDimensions)].map((_, i) => (
                    <option key={i} value={i}>Dimension {i + 1}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DimensionSelector;