import React from 'react';
import './HelpPanel.css';

export function HelpPanel({ isOpen, onClose, totalLocations, stats }) {
  if (!isOpen) return null;
  
  return (
    <div className="help-overlay" onClick={onClose}>
      <div className="help-panel" onClick={(e) => e.stopPropagation()}>
        <button className="help-close-btn" onClick={onClose}>×</button>
        
        <h2>Urban Embeddings Explorer</h2>
        
        <div className="help-section">
          <h3>What is this?</h3>
          <p>
            This tool uses AI-powered satellite imagery analysis to discover visually similar urban areas worldwide. 
            It processes <strong>Sentinel-2 satellite data</strong> through the <a href="https://github.com/IBM/terramind" target="_blank" rel="noopener noreferrer" className="help-link">IBM TerraMind</a> foundation 
            model to create high-dimensional <strong>embeddings</strong> - mathematical representations that capture visual patterns in urban environments.
          </p>
          <p>
            Each 224×224m urban tile is analyzed as 196 patches, generating 768-dimensional vectors that encode features like vegetation, 
            infrastructure, coastal patterns, and urban morphology. The system keeps the least spatially-correlated dimensions to focus on visual 
            similarity rather than geographic proximity.
          </p>
        </div>

        <div className="help-section">
          <h3>How to use it</h3>
          <p><strong>Click on the map</strong> to select any urban tile and find visually similar locations worldwide.</p>
          <p><strong>Navigate the UMAP visualization</strong> where nearby points represent visually similar urban areas.</p>
          <p><strong>Switch between aggregation methods</strong> to explore different aspects of urban similarity:</p>
          
          <div className="method-explanation">
            <h4>📊 Mean</h4>
            <p>Standard statistical average - balanced representation of all visual features</p>
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
            <strong>Vector Search:</strong> Qdrant vector database for similarity search<br/>
            <strong>UMAP Component:</strong> UMAP dimensionality reduction to visualise all tiles <br/>
            <strong>Interface:</strong> React + Mapbox for interactive exploration
          </p>
        </div>

        <div className="help-section">
          <h3>Discoveries</h3>
            <p>
              The system reveals some interesting global urban patterns: residential areas with dense vegetation clustering across continents, coastal cities sharing distinctive waterfront characteristics, 
              industrial zones with recognizable spatial signatures, and transportation infrastructure like airports showing visual consistency worldwide.
            </p>
            <p>
              <strong>What to explore:</strong> Search for different urban features across the globe - for example can you find where 
              airports concentrate in the UMAP space?
              The similarities are broad given the 10-meter resolution, revealing general urban morphology rather than 
              fine-grained details, but they showcase how AI models capture fundamental urban characteristics that 
              transcend continental boundaries.
            </p>
        </div>

        <div className="help-footer">
          <p>
            For technical details visit the <a href="https://github.com/jankomag/urban-embeddings-explorer" target="_blank" rel="noopener noreferrer" className="help-link">GitHub repository </a>
          </p>
          <p style={{fontSize: '10px', marginTop: '12px', color: 'var(--text-quaternary)'}}>
            Interested or have ideas? <a href="https://jan.magnuszewski.com" target="_blank" rel="noopener noreferrer" className="help-link">Reach out!</a>
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
        <h2>Urban Embeddings Explorer</h2>

        <div className="welcome-section">
          <p className="welcome-intro">
            Discover visually similar urban areas worldwide using AI-powered satellite imagery analysis. 
            Click on the map or UMAP visualization to explore global urban patterns.
          </p>
        </div>

        <div className="welcome-actions">
          <div className="action-card">
            <h4>Click on Map</h4>
            <p>Select any urban tile to find visually similar locations worldwide</p>
          </div>

          <div className="action-card">
            <h4>Explore UMAP</h4>
            <p>Navigate the 2D embedding space where nearby points share visual patterns</p>
          </div>

          <div className="action-card">
            <h4>Search Cities</h4>
            <p>Use the search bar above to find specific cities or countries and see their tiles highlighted on both views</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HelpPanel;