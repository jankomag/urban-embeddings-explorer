import React from 'react';
import './HelpPanel.css';

export function HelpPanel({ isOpen, onClose, totalLocations, stats }) {
  if (!isOpen) return null;
  
  return (
    <div className="help-overlay" onClick={onClose}>
      <div className="help-panel" onClick={(e) => e.stopPropagation()}>
        <button className="help-close-btn" onClick={onClose}>×</button>
        
        <h2>🛰️ Urban Embeddings Explorer</h2>
        
        <div className="help-section">
          <h3>What is this?</h3>
          <p>
            This tool uses AI-powered satellite imagery analysis to discover visually similar urban areas worldwide. 
            It processes <strong>Sentinel-2 satellite data</strong> through the <a href="https://github.com/IBM/terramind" target="_blank" rel="noopener noreferrer" className="help-link">IBM TerraMind</a> foundation 
            model to create high-dimensional <strong>embeddings</strong> - mathematical representations that capture visual patterns in urban environments.
          </p>
          <p>
            Each 224×224m urban tile is analyzed as 196 patches, generating 768-dimensional vectors that encode features like vegetation, 
            infrastructure, coastal patterns, and urban morphology. Our system removes spatially-correlated dimensions to focus on visual 
            similarity rather than geographic proximity.
          </p>
          
          <p>
            <strong>Spatial Bias Filtering:</strong>
          </p>
          <p>
            We identify and remove ~8% of embedding dimensions that are most correlated with geographic location, 
            enabling discovery of visual patterns that transcend continental boundaries.
          </p>
        </div>

        <div className="help-section">
          <h3>How to explore</h3>
          <p><strong>Click on the satellite map</strong> to select any urban tile and find visually similar locations worldwide.</p>
          <p><strong>Navigate the UMAP visualization</strong> where nearby points represent visually similar urban areas.</p>
          <p><strong>Switch between aggregation methods</strong> to explore different aspects of urban similarity:</p>
          
          <div className="method-explanation">
            <h4>📊 Mean</h4>
            <p>Standard statistical average - balanced representation of all visual features</p>
          </div>
          
          <div className="method-explanation">
            <h4>🎯 Median</h4>
            <p>Robust to outliers - emphasizes consistent visual patterns</p>
          </div>
          
          <div className="method-explanation">
            <h4>🔍 Dominant Cluster</h4>
            <p>Identifies the most frequent visual pattern within each tile using clustering analysis</p>
          </div>
        </div>

        <div className="help-section">
          <h3>Technology</h3>
          <p>
            <strong>Data:</strong> Sentinel-2 L2A satellite imagery from global urban areas<br/>
            <strong>AI Model:</strong> <a href="https://github.com/IBM/terramind" target="_blank" rel="noopener noreferrer" className="help-link">IBM TerraMind v1.0</a> foundation model for Earth observation<br/>
            <strong>Vector Search:</strong> Qdrant database with UMAP dimensionality reduction<br/>
            <strong>Interface:</strong> React + Mapbox for interactive exploration
          </p>
        </div>

        <div className="help-section">
          <h3>Discoveries</h3>
          <p>
            The system reveals fascinating global patterns: residential areas with similar vegetation clustering across continents, 
            coastal cities sharing boundary characteristics, industrial zones with recognizable signatures, and infrastructure 
            like airports showing remarkable visual consistency worldwide.
          </p>
        </div>

        <div className="help-footer">
          <p>
            For technical details visit the <a href="https://github.com/jankomag/urban-embeddings-explorer" target="_blank" rel="noopener noreferrer" className="help-link">GitHub repository </a> 
            or learn more at <a href="https://jan.magnuszewski.com" target="_blank" rel="noopener noreferrer" className="help-link">jan.magnuszewski.com</a>
          </p>
          <p style={{fontSize: '10px', marginTop: '12px', color: 'var(--text-quaternary)'}}>
            Interested or have ideas? <a href="mailto:jan@magnuszewski.com" className="help-link">Reach out!</a>
          </p>
          <p style={{marginTop: '16px'}}>Press <kbd>?</kbd> anytime to show this help</p>
        </div>
      </div>
    </div>
  );
}

// SIMPLIFIED Welcome Panel
export function WelcomePanel() {
  return (
    <div className="welcome-panel">
      <div className="welcome-content">
        <h2>🛰️ Urban Embeddings Explorer</h2>

        <div className="welcome-section">
          <p className="welcome-intro">
            Discover visually similar urban areas worldwide using AI-powered satellite imagery analysis. 
            Click on the map or UMAP visualization to explore global urban patterns.
          </p>
        </div>

        <div className="welcome-actions">
          <div className="action-card">
            <div className="action-icon">🗺️</div>
            <h4>Click on Map</h4>
            <p>Select any urban tile to find visually similar locations worldwide</p>
          </div>

          <div className="action-card">
            <div className="action-icon">📊</div>
            <h4>Explore UMAP</h4>
            <p>Navigate the 2D embedding space where nearby points share visual patterns</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HelpPanel;