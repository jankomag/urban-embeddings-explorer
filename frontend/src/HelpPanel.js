import React from 'react';
import './HelpPanel.css';

function HelpPanel({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="help-overlay" onClick={onClose}>
      <div className="help-panel" onClick={(e) => e.stopPropagation()}>
        <button className="help-close-btn" onClick={onClose}>✕</button>
        
        <h2>🛰️ Satellite Embeddings Explorer</h2>
        
        <div className="help-section">
          <h3>What is this?</h3>
          <p>
            This tool explores urban satellite imagery using AI-generated embeddings from the TerraMind model. 
            Each point represents a 224×224 meter tile of Earth captured by satellite, encoded into a 768-dimensional vector.
          </p>
        </div>

        <div className="help-section">
          <h3>How to Use</h3>
          <ul>
            <li><strong>Map View:</strong> Click any tile to see similar locations worldwide</li>
            <li><strong>UMAP View:</strong> 2D projection of similarity space - nearby points are visually similar</li>
            <li><strong>City Filter:</strong> Use the top bar to filter by country or city</li>
            <li><strong>Similarity Panel:</strong> Appears after selecting a tile, showing the most similar locations</li>
            <li><strong>Same City Toggle:</strong> Include or exclude results from the same city as your selection</li>
          </ul>
        </div>

        <div className="help-section">
          <h3>Similarity Methods</h3>
          
          <div className="method-explanation">
            <h4>🔵 Regular Embeddings</h4>
            <p>
              Standard visual similarity using mean-aggregated patch embeddings. Each satellite tile is divided into 
              196 patches (14×14 grid), producing 196 separate 768-dim vectors. These are averaged into a single 
              768-dim vector representing the entire tile. Best for finding areas with similar overall appearance.
            </p>
          </div>

          <div className="method-explanation">
            <h4>🟣 Global Contrastive</h4>
            <p>
              Embeddings with the dataset mean subtracted, highlighting what makes each location unique compared to 
              the global average. This method emphasizes distinctive city-level patterns and characteristics that 
              differ from typical urban areas. Best for finding cities with similar unique features.
            </p>
          </div>

          <div className="method-explanation">
            <h4>🔄 Adaptive Mixed</h4>
            <p>
              <strong>Intelligent aggregation</strong> that automatically chooses between uniform and weighted averaging 
              based on patch diversity within each tile. For homogeneous areas (parks, uniform residential), uses simple 
              mean aggregation. For heterogeneous areas (mixed commercial/residential, complex urban patterns), uses 
              distinctiveness-weighted aggregation that emphasizes unique patches.
            </p>
            <div className="technical-note">
              <strong>How it works:</strong> Calculates patch homogeneity using coefficient of variation and pairwise 
              cosine similarity. If homogeneity &lt; 0.2 threshold → simple mean. If diverse → weights patches by their 
              distinctiveness from the tile mean, using adaptive temperature scaling. Best for optimal similarity 
              matching across diverse urban environments.
            </div>
          </div>
        </div>

        <div className="help-section">
          <h3>When to Use Each Method</h3>
          <ul>
            <li><strong>Regular:</strong> General exploration, finding areas with similar overall visual characteristics</li>
            <li><strong>Global Contrastive:</strong> Finding cities that stand out from typical urban patterns in similar ways</li>
            <li><strong>Adaptive Mixed:</strong> Most versatile - automatically optimizes for both uniform and complex urban areas</li>
          </ul>
        </div>

        <div className="help-section">
          <h3>Technical Details</h3>
          <p className="technical-note">
            <strong>Patch Aggregation:</strong> The TerraMind model processes each 224×224m tile as 14×14 patches. 
            Each patch generates a 768-dimensional embedding vector. The three aggregation methods differ in how 
            these 196 patch vectors are combined: Regular uses simple mean pooling, Global Contrastive subtracts 
            the dataset mean, and Adaptive Mixed intelligently weights patches based on local diversity patterns 
            to preserve both global structure and local details.
          </p>
        </div>

        <div className="help-footer">
          <p>Press <kbd>?</kbd> anytime to show this help</p>
        </div>
      </div>
    </div>
  );
}

export function WelcomePanel() {
  return (
    <div className="welcome-panel">
      <div className="welcome-content">
        <h2>🛰️ Welcome to Satellite Embeddings Explorer</h2>
        
        <div className="welcome-section">
          <p className="welcome-intro">
            Explore urban satellite imagery through AI-powered similarity search. 
            Each point represents a 224×224 meter tile of Earth, encoded by the TerraMind vision model.
          </p>
        </div>

        <div className="welcome-actions">
          <div className="action-card">
            <div className="action-icon">🗺️</div>
            <h4>Click on Map</h4>
            <p>Select any tile on the satellite map to find similar locations worldwide</p>
          </div>
          
          <div className="action-card">
            <div className="action-icon">📊</div>
            <h4>Explore UMAP</h4>
            <p>Navigate the 2D embedding space where nearby points are visually similar</p>
          </div>
          
          <div className="action-card">
            <div className="action-icon">🏙️</div>
            <h4>Filter by City</h4>
            <p>Use the top bar to focus on specific countries or cities</p>
          </div>

          <div className="action-card">
            <div className="action-icon">🔄</div>
            <h4>Try Adaptive Mixed</h4>
            <p>New intelligent aggregation method that optimizes for diverse urban patterns</p>
          </div>
        </div>

        <div className="welcome-tip">
          <strong>💡 Tip:</strong> Press <kbd>?</kbd> anytime to learn more about the three similarity methods and their technical details
        </div>
      </div>
    </div>
  );
}

export default HelpPanel;