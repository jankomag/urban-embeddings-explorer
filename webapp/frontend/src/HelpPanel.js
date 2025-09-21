import React from 'react';
import './HelpPanel.css';

function HelpPanel({ isOpen, onClose, totalLocations, stats }) {
  if (!isOpen) return null;

  return (
    <div className="help-overlay" onClick={onClose}>
      <div className="help-panel" onClick={(e) => e.stopPropagation()}>
        <button className="help-close-btn" onClick={onClose}>✕</button>

        <h2>🛰️ Urban Embeddings Explorer</h2>

        {/* Dataset Statistics */}
        <div className="help-section">
          <h3>Dataset Overview</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-number">{totalLocations}</span>
              <span className="stat-label">Satellite Tiles</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">{stats?.countries_count || 0}</span>
              <span className="stat-label">Countries</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">{Math.round((parseInt(totalLocations?.replace(/,/g, '') || 0)) / 356) || 137}</span>
              <span className="stat-label">Cities</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">768</span>
              <span className="stat-label">Dimensions</span>
            </div>
          </div>
          <p>
            Each tile represents a 224×224 meter area of Earth captured by satellite, 
            encoded into a 768-dimensional vector using the TerraMind vision model.
          </p>
        </div>

        <div className="help-section">
          <h3>How to Use It</h3>
          <ul>
            <li><strong>Map View:</strong> Click any tile to see similar locations worldwide</li>
            <li><strong>UMAP View:</strong> 2D projection of similarity space</li>
            <li><strong>Search:</strong> Use the search bar to filter by country or city</li>
            <li><strong>Similarity Panel:</strong> Appears after selecting a tile, showing the most similar locations</li>
            <li><strong>Same City Toggle:</strong> Include or exclude results from the same city as your selection</li>
          </ul>
        </div>

        <div className="help-section">
          <h3>Aggregation Methods</h3>

          <div className="method-explanation">
            <h4>🔵 Mean Aggregation</h4>
            <p>
              <strong>Standard approach:</strong> Each satellite tile is divided into 196 patches (14×14 grid), 
              producing 196 separate 768-dim vectors. These are averaged into a single vector representing 
              the entire tile. Best for general similarity based on overall visual characteristics.
            </p>
          </div>

          <div className="method-explanation">
            <h4>🟡 Median Aggregation</h4>
            <p>
              <strong>Robust to outliers:</strong> Uses median instead of mean to combine patch embeddings.
              This approach is less affected by noisy or unusual patches within a tile, providing more 
              stable similarity comparisons. Best for finding areas with consistent visual patterns.
            </p>
          </div>

          <div className="method-explanation">
            <h4>🟢 Min Aggregation</h4>
            <p>
              <strong>Conservative features:</strong> Takes the element-wise minimum across all patch embeddings.
              This captures the shared minimal features present across all patches in a tile. Best for 
              finding locations that share common baseline characteristics.
            </p>
          </div>

          <div className="method-explanation">
            <h4>🔴 Max Aggregation</h4>
            <p>
              <strong>Distinctive features:</strong> Takes the element-wise maximum across all patch embeddings.
              This captures the strongest or most distinctive features present in any patch within the tile.
              Best for finding locations with similar standout visual elements.
            </p>
          </div>

          <div className="method-explanation">
            <h4>🟣 Dominant Cluster</h4>
            <p>
              <strong>Pattern-focused:</strong> Uses machine learning clustering to identify the most common 
              visual pattern within each tile's 196 patches, then averages only those dominant patches.
              Best for finding locations with similar recurring visual themes.
            </p>
          </div>
        </div>

        <div className="help-section">
          <h3>When to Use Each Method</h3>
          <ul>
            <li><strong>Mean:</strong> General exploration, baseline similarity for most use cases</li>
            <li><strong>Median:</strong> When you want similarity that's robust to noisy patches</li>
            <li><strong>Min:</strong> Finding locations that share common foundational characteristics</li>
            <li><strong>Max:</strong> Discovering areas with similar distinctive or standout features</li>
            <li><strong>Dominant Cluster:</strong> Locations with similar recurring visual patterns</li>
          </ul>
        </div>

        <div className="help-section">
          <h3>Technical Details</h3>
          <p className="technical-note">
            <strong>Patch Processing:</strong> The TerraMind model processes each 224×224m tile as 14×14 patches.
            Each patch generates a 768-dimensional embedding vector. The aggregation methods differ in how
            these 196 patch vectors are combined: Mean uses arithmetic average, Median uses the middle value,
            Min/Max use element-wise minimum/maximum, and Dominant Cluster uses K-means clustering with 
            the elbow method to find optimal cluster count, then averages the most frequent cluster.
          </p>
        </div>

        <div className="help-footer">
          <p>Press <kbd>?</kbd> anytime to show this help</p>
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
        <h2>🛰️ Satellite Embeddings Explorer</h2>

        <div className="welcome-section">
          <p className="welcome-intro">
            Explore urban satellite imagery through AI-powered similarity search. 
            Click on the map or UMAP visualization to find visually similar locations worldwide.
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
        </div>
      </div>
    </div>
  );
}

export default HelpPanel;