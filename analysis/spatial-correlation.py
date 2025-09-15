#!/usr/bin/env python3
"""
Improved Spatial Autocorrelation Analysis with Patch-Level Resolution
=====================================================================

This script identifies spatially correlated dimensions by analyzing:
1. BETWEEN tiles: Do nearby tiles have similar values?
2. WITHIN tiles: Do patches in the same tile have similar values?
3. ACROSS cities: Generalize findings across diverse geographies

Key improvements:
- Analyzes individual patches, not just tile means
- Better cutoff detection using gradient method
- More robust spatial correlation metrics

Usage:
    python spatial_correlation_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from tqdm import tqdm
import time
import gc
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDINGS_DIR = "./embeddings/urban_embeddings_224_terramind_normalised"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Memory management
MAX_TILES_PER_CITY = 100   # Tiles per city to analyze
MIN_TILES_PER_CITY = 10   # Minimum for reliable statistics
SAMPLE_PATCHES_PER_TILE = 50  # Sample patches from each tile (out of 196)

# Files to exclude
EXCLUDED_FILES = ['dallas.gpq', 'chicago.gpq', 'miami.gpq']


class ImprovedSpatialAnalyzer:
    """
    Analyzes spatial autocorrelation at both tile and patch levels.
    """
    
    def __init__(self, n_dimensions=768):
        self.n_dimensions = n_dimensions
        
        # Store correlation scores
        self.tile_correlations = defaultdict(list)  # Tile-to-tile correlations
        self.patch_coherence = defaultdict(list)    # Within-tile patch coherence
        
        # Statistics
        self.cities_processed = 0
        self.tiles_analyzed = 0
        self.patches_analyzed = 0
    
    def get_patch_locations(self, tile_center, tile_size_meters=2240):
        """
        Calculate approximate locations of patches within a tile.
        
        Assumes 14x14 grid of patches within each 2240m x 2240m tile.
        
        Args:
            tile_center: (lon, lat) of tile center
            tile_size_meters: Size of tile in meters
        
        Returns:
            Array of (lon, lat) for each patch position
        """
        lon_center, lat_center = tile_center
        
        # Convert to approximate meters
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(lat_center))
        
        # Each patch covers tile_size/14 meters
        patch_size_meters = tile_size_meters / 14
        
        # Create 14x14 grid
        patch_locations = []
        for row in range(14):
            for col in range(14):
                # Offset from center in patches
                row_offset = (row - 6.5) * patch_size_meters
                col_offset = (col - 6.5) * patch_size_meters
                
                # Convert to degrees
                lat_offset = row_offset / meters_per_degree_lat
                lon_offset = col_offset / meters_per_degree_lon
                
                patch_locations.append([
                    lon_center + lon_offset,
                    lat_center + lat_offset
                ])
        
        return np.array(patch_locations)
    
    def analyze_within_tile_coherence(self, patches):
        """
        Analyze how coherent patches are within a single tile.
        
        High coherence in a dimension means all patches in the tile
        have similar values - suggesting that dimension encodes
        broad spatial features rather than fine visual details.
        
        Args:
            patches: Array of shape (196, 768)
        
        Returns:
            Coherence score for each dimension
        """
        coherence_scores = np.zeros(self.n_dimensions)
        
        for dim in range(self.n_dimensions):
            dim_values = patches[:, dim]
            
            # Measure coherence as inverse of coefficient of variation
            # Low variance relative to mean = high coherence
            mean_val = np.mean(dim_values)
            std_val = np.std(dim_values)
            
            if abs(mean_val) > 1e-10:
                # Coefficient of variation
                cv = std_val / abs(mean_val)
                # Convert to coherence score (inverse, bounded)
                coherence_scores[dim] = 1.0 / (1.0 + cv)
            else:
                # If mean is near zero, use just variance
                coherence_scores[dim] = 1.0 / (1.0 + std_val)
        
        return coherence_scores
    
    def analyze_tile_to_tile_correlation(self, tiles_data):
        """
        Analyze correlation between tile proximity and embedding similarity.
        
        Args:
            tiles_data: List of (location, embedding) tuples
        
        Returns:
            Correlation scores for each dimension
        """
        if len(tiles_data) < 2:
            return np.zeros(self.n_dimensions)
        
        locations = np.array([loc for loc, _ in tiles_data])
        embeddings = np.array([emb for _, emb in tiles_data])
        
        # Calculate spatial distances (in degrees, good enough for small areas)
        spatial_distances = cdist(locations, locations, metric='euclidean')
        
        # Get unique pairs (upper triangle of distance matrix)
        n_tiles = len(tiles_data)
        pairs = [(i, j) for i in range(n_tiles) for j in range(i+1, n_tiles)]
        
        # Limit number of pairs for efficiency
        if len(pairs) > 300:
            pairs = pairs[:300]
        
        correlation_scores = np.zeros(self.n_dimensions)
        
        for dim in range(self.n_dimensions):
            # Get embedding distances for this dimension
            spatial_dists = []
            embedding_diffs = []
            
            for i, j in pairs:
                spatial_dists.append(spatial_distances[i, j])
                embedding_diffs.append(abs(embeddings[i, dim] - embeddings[j, dim]))
            
            # Calculate correlation
            if len(spatial_dists) > 10 and np.std(embedding_diffs) > 0:
                # Negative correlation means close tiles have similar values
                corr, _ = spearmanr(spatial_dists, embedding_diffs)
                if not np.isnan(corr):
                    # Convert to positive score where high = spatially correlated
                    correlation_scores[dim] = -corr
        
        return correlation_scores
    
    def process_city(self, file_path):
        """
        Process a single city file.
        
        Args:
            file_path: Path to city .gpq file
        
        Returns:
            Number of tiles processed
        """
        city_name = os.path.basename(file_path).replace('.gpq', '')
        
        try:
            # Load city data
            gdf = gpd.read_parquet(file_path)
            
            if len(gdf) < MIN_TILES_PER_CITY:
                logger.warning(f"{city_name}: Too few tiles ({len(gdf)})")
                return 0
            
            # Convert categorical columns
            for col in gdf.columns:
                if gdf[col].dtype.name == 'category':
                    gdf[col] = gdf[col].astype(str)
            
            # Sample tiles if needed
            if len(gdf) > MAX_TILES_PER_CITY:
                gdf = gdf.sample(n=MAX_TILES_PER_CITY, random_state=42)
            
            logger.info(f"Processing {city_name}: {len(gdf)} tiles")
            
            tiles_data = []  # (location, mean_embedding) tuples
            coherence_accumulator = np.zeros(self.n_dimensions)
            valid_tiles = 0
            
            for _, row in gdf.iterrows():
                # Get location
                lon = row.get('centroid_lon') or row.get('longitude')
                lat = row.get('centroid_lat') or row.get('latitude')
                
                if lon is None or lat is None:
                    continue
                
                # Get patches
                full_embedding = row.get('embedding_patches_full')
                
                if full_embedding is not None:
                    try:
                        if isinstance(full_embedding, np.ndarray):
                            embedding_array = full_embedding
                        else:
                            embedding_array = np.array(full_embedding)
                        
                        if embedding_array.size == 196 * 768:
                            patches = embedding_array.reshape(196, 768)
                            
                            # Analyze within-tile coherence
                            coherence = self.analyze_within_tile_coherence(patches)
                            coherence_accumulator += coherence
                            
                            # Store tile data for between-tile analysis
                            mean_embedding = np.mean(patches, axis=0)
                            tiles_data.append(([float(lon), float(lat)], mean_embedding))
                            
                            valid_tiles += 1
                            self.patches_analyzed += 196
                            
                    except Exception as e:
                        logger.debug(f"Error processing tile: {e}")
                        continue
            
            if valid_tiles == 0:
                return 0
            
            # Average coherence scores
            avg_coherence = coherence_accumulator / valid_tiles
            
            # Analyze tile-to-tile correlation
            tile_correlation = self.analyze_tile_to_tile_correlation(tiles_data)
            
            # Store results (weighted combination)
            for dim in range(self.n_dimensions):
                # Within-tile coherence is weighted more (0.6) than between-tile (0.4)
                # because it directly measures spatial smoothness
                combined_score = 0.6 * avg_coherence[dim] + 0.4 * tile_correlation[dim]
                self.tile_correlations[dim].append(combined_score)
            
            self.cities_processed += 1
            self.tiles_analyzed += valid_tiles
            
            # Cleanup
            del gdf, tiles_data
            gc.collect()
            
            return valid_tiles
            
        except Exception as e:
            logger.error(f"Error processing {city_name}: {e}")
            return 0
    
    def compute_final_scores(self):
        """
        Compute final spatial correlation scores.
        
        Returns:
            Array of scores and sorted indices
        """
        final_scores = np.zeros(self.n_dimensions)
        
        for dim in range(self.n_dimensions):
            if dim in self.tile_correlations and len(self.tile_correlations[dim]) > 0:
                scores = np.array(self.tile_correlations[dim])
                # Use robust mean (trim outliers)
                trimmed = np.percentile(scores, [10, 90])
                valid_scores = scores[(scores >= trimmed[0]) & (scores <= trimmed[1])]
                if len(valid_scores) > 0:
                    final_scores[dim] = np.mean(valid_scores)
        
        return final_scores


def find_optimal_cutoff_gradient(scores):
    """
    Find optimal cutoff using gradient-based method.
    
    The cutoff is where the rate of change in spatial correlation
    becomes small (the curve flattens).
    
    Args:
        scores: Array of spatial correlation scores (sorted descending)
    
    Returns:
        Cutoff information
    """
    n_dims = len(scores)
    
    # Calculate gradient (rate of change)
    gradients = np.abs(np.gradient(scores))
    
    # Smooth gradient to reduce noise
    window_size = 5
    if len(gradients) > window_size:
        smoothed_gradients = np.convolve(gradients, np.ones(window_size)/window_size, mode='valid')
        # Pad to maintain size
        smoothed_gradients = np.pad(smoothed_gradients, (window_size//2, window_size//2), mode='edge')
    else:
        smoothed_gradients = gradients
    
    # Find elbow: where gradient drops below threshold
    # Use adaptive threshold based on gradient distribution
    
    # Focus on first 30% of dimensions (where spatial correlation is high)
    search_range = int(n_dims * 0.3)
    gradients_subset = smoothed_gradients[:search_range]
    
    if len(gradients_subset) > 0:
        # Threshold is 20th percentile of gradients in search range
        threshold = np.percentile(gradients_subset, 20)
        
        # Find first point where gradient stays below threshold
        below_threshold = smoothed_gradients < threshold
        
        # Look for sustained flat region (at least 5 consecutive points below threshold)
        cutoff_idx = None
        for i in range(len(below_threshold) - 5):
            if all(below_threshold[i:i+5]):
                cutoff_idx = i
                break
        
        if cutoff_idx is None:
            # Fallback: use simple threshold crossing
            flat_points = np.where(below_threshold)[0]
            if len(flat_points) > 0:
                cutoff_idx = flat_points[0]
            else:
                cutoff_idx = int(n_dims * 0.15)  # Default 15%
    else:
        cutoff_idx = int(n_dims * 0.15)
    
    # Apply bounds: between 5% and 30% of dimensions
    min_cutoff = int(n_dims * 0.05)
    max_cutoff = int(n_dims * 0.30)
    cutoff_idx = max(min_cutoff, min(cutoff_idx, max_cutoff))
    
    return {
        'cutoff_index': cutoff_idx,
        'dimensions_to_exclude': cutoff_idx,
        'exclusion_percentage': (cutoff_idx / n_dims) * 100,
        'gradient_threshold': threshold if 'threshold' in locals() else 0,
        'method': 'gradient'
    }


def create_analysis_plots(scores, sorted_indices, cutoff_info, output_dir):
    """Create comprehensive visualization plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    sorted_scores = scores[sorted_indices]
    cutoff = cutoff_info['cutoff_index']
    
    # Plot 1: Spatial correlation curve with cutoff
    ax1 = axes[0, 0]
    ax1.plot(sorted_scores, 'b-', linewidth=1.5, label='Spatial correlation')
    ax1.axvline(x=cutoff, color='red', linestyle='--', linewidth=2,
                label=f'Cutoff ({cutoff} dims)')
    ax1.fill_between(range(cutoff), 0, sorted_scores[:cutoff], 
                     alpha=0.3, color='red', label='Excluded (spatial)')
    ax1.set_xlabel('Dimension (ranked)')
    ax1.set_ylabel('Spatial Correlation Score')
    ax1.set_title('Spatial Autocorrelation by Dimension')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Gradient analysis
    ax2 = axes[0, 1]
    gradients = np.abs(np.gradient(sorted_scores))
    ax2.plot(gradients, 'g-', alpha=0.7, label='Gradient')
    ax2.axvline(x=cutoff, color='red', linestyle='--', linewidth=2)
    if 'gradient_threshold' in cutoff_info:
        ax2.axhline(y=cutoff_info['gradient_threshold'], color='orange', 
                   linestyle=':', label='Threshold')
    ax2.set_xlabel('Dimension (ranked)')
    ax2.set_ylabel('|Gradient|')
    ax2.set_title('Gradient Analysis (Rate of Change)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Zoomed view of cutoff region
    ax3 = axes[0, 2]
    zoom_start = max(0, cutoff - 20)
    zoom_end = min(len(sorted_scores), cutoff + 20)
    zoom_range = range(zoom_start, zoom_end)
    ax3.plot(zoom_range, sorted_scores[zoom_start:zoom_end], 'b-', linewidth=2)
    ax3.axvline(x=cutoff, color='red', linestyle='--', linewidth=2)
    ax3.fill_between(range(zoom_start, min(cutoff, zoom_end)), 
                     0, sorted_scores[zoom_start:min(cutoff, zoom_end)], 
                     alpha=0.3, color='red')
    ax3.set_xlabel('Dimension (ranked)')
    ax3.set_ylabel('Spatial Correlation Score')
    ax3.set_title('Zoomed View of Cutoff Region')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Distribution of scores
    ax4 = axes[1, 0]
    ax4.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    threshold_value = sorted_scores[cutoff] if cutoff < len(sorted_scores) else 0
    ax4.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2,
                label=f'Cutoff value')
    ax4.set_xlabel('Spatial Correlation Score')
    ax4.set_ylabel('Number of Dimensions')
    ax4.set_title('Distribution of Spatial Correlation')
    ax4.legend()
    
    # Plot 5: Top dimensions with labels
    ax5 = axes[1, 1]
    top_n = min(30, len(sorted_indices))
    top_dims = sorted_indices[:top_n]
    top_scores = sorted_scores[:top_n]
    colors = ['red' if i < cutoff else 'blue' for i in range(top_n)]
    bars = ax5.bar(range(top_n), top_scores, color=colors, alpha=0.7)
    ax5.set_xlabel('Rank')
    ax5.set_ylabel('Spatial Correlation')
    ax5.set_title(f'Top {top_n} Spatially Correlated Dimensions')
    # Add labels for top 10
    for i in range(min(10, top_n)):
        ax5.text(i, top_scores[i], f'd{top_dims[i]}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Cumulative explained spatial variance
    ax6 = axes[1, 2]
    cumsum = np.cumsum(sorted_scores)
    total_correlation = np.sum(scores)
    cumsum_pct = (cumsum / total_correlation * 100) if total_correlation > 0 else cumsum
    ax6.plot(cumsum_pct, 'g-', linewidth=2)
    ax6.axvline(x=cutoff, color='red', linestyle='--', linewidth=2)
    ax6.axhline(y=cumsum_pct[cutoff], color='red', linestyle=':', alpha=0.5)
    ax6.set_xlabel('Number of Dimensions')
    ax6.set_ylabel('Cumulative Spatial Information (%)')
    ax6.set_title('Cumulative Spatial Correlation')
    ax6.grid(alpha=0.3)
    ax6.text(cutoff + 5, cumsum_pct[cutoff], 
            f'{cumsum_pct[cutoff]:.1f}%', fontsize=10)
    
    plt.suptitle(f'Spatial Correlation Analysis: Excluding {cutoff} dimensions ({cutoff_info["exclusion_percentage"]:.1f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    timestamp = int(time.time())
    plot_file = os.path.join(output_dir, f'spatial_correlation_analysis_{timestamp}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved: {plot_file}")
    plt.close()


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("IMPROVED SPATIAL AUTOCORRELATION ANALYSIS")
    logger.info("=" * 70)
    logger.info("Analyzing both within-tile and between-tile spatial patterns")
    
    start_time = time.time()
    
    # Find all city files
    pattern = os.path.join(EMBEDDINGS_DIR, "full_patch_embeddings", "*", "*.gpq")
    all_files = glob(pattern)
    
    # Filter excluded files
    files = [f for f in all_files if not any(excluded in os.path.basename(f) for excluded in EXCLUDED_FILES)]
    
    if not files:
        logger.error(f"No files found in {pattern}")
        return 1
    
    logger.info(f"Found {len(files)} city files")
    
    # Initialize analyzer
    analyzer = ImprovedSpatialAnalyzer()
    
    # Process cities
    for file_path in tqdm(files, desc="Analyzing cities"):
        tiles_processed = analyzer.process_city(file_path)
        
        if analyzer.cities_processed % 10 == 0:
            gc.collect()
            logger.info(f"Progress: {analyzer.cities_processed} cities, "
                       f"{analyzer.tiles_analyzed} tiles, "
                       f"{analyzer.patches_analyzed:,} patches")
    
    # Compute final scores
    logger.info("\nComputing final spatial correlation scores...")
    final_scores = analyzer.compute_final_scores()
    
    # Sort dimensions by spatial correlation
    sorted_indices = np.argsort(final_scores)[::-1]
    sorted_scores = final_scores[sorted_indices]
    
    # Find optimal cutoff using gradient method
    cutoff_info = find_optimal_cutoff_gradient(sorted_scores)
    
    # Get excluded dimensions
    excluded_dimensions = sorted_indices[:cutoff_info['dimensions_to_exclude']].tolist()
    
    # Create visualizations
    create_analysis_plots(final_scores, sorted_indices, cutoff_info, OUTPUT_DIR)
    
    # Save results
    results = {
        'analysis_type': 'spatial_autocorrelation_improved',
        'analysis_params': {
            'cities_processed': analyzer.cities_processed,
            'tiles_analyzed': analyzer.tiles_analyzed,
            'patches_analyzed': analyzer.patches_analyzed,
            'max_tiles_per_city': MAX_TILES_PER_CITY,
            'timestamp': time.time()
        },
        'cutoff_results': {
            'method': cutoff_info['method'],
            'dimensions_to_exclude': cutoff_info['dimensions_to_exclude'],
            'excluded_dimensions': [int(d) for d in excluded_dimensions],
            'exclusion_percentage': cutoff_info['exclusion_percentage'],
            'gradient_threshold': cutoff_info.get('gradient_threshold', 0)
        },
        'top_30_spatial_dimensions': [
            {
                'dimension': int(sorted_indices[i]),
                'spatial_correlation': float(sorted_scores[i])
            }
            for i in range(min(30, len(sorted_indices)))
        ],
        'interpretation': {
            'within_tile_coherence': 'How similar patches are within same tile',
            'between_tile_correlation': 'How similar nearby tiles are',
            'combined_score': '60% within-tile + 40% between-tile',
            'high_score_meaning': 'Dimension encodes spatial/geographic information',
            'low_score_meaning': 'Dimension encodes visual features independent of location'
        }
    }
    
    output_file = os.path.join(OUTPUT_DIR, 'spatial_correlation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    duration = (time.time() - start_time) / 60
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info(f"Duration: {duration:.1f} minutes")
    logger.info(f"Cities: {analyzer.cities_processed}")
    logger.info(f"Tiles: {analyzer.tiles_analyzed}")
    logger.info(f"Patches: {analyzer.patches_analyzed:,}")
    logger.info(f"\nRESULTS:")
    logger.info(f"Dimensions to exclude: {cutoff_info['dimensions_to_exclude']} ({cutoff_info['exclusion_percentage']:.1f}%)")
    logger.info(f"Top 5 spatial dims: {excluded_dimensions[:5]}")
    logger.info(f"Cutoff method: Gradient-based (curve flattening)")
    logger.info(f"\nThese dimensions encode WHERE (location), not WHAT (visual).")
    logger.info(f"Removing them focuses similarity on visual features.")
    logger.info(f"\nResults saved: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())