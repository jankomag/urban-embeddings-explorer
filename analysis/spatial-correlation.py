#!/usr/bin/env python3
"""
Spatial Autocorrelation Analysis for Dimension Filtering
========================================================

Identifies spatially correlated dimensions in satellite image embeddings
by analyzing spatial patterns at both tile and patch levels.

Usage: python spatial_correlation.py
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
EMBEDDINGS_DIR = "../../../terramind/embeddings/urban_embeddings_224_terramind_normalised"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_TILES_PER_CITY = 200
MIN_TILES_PER_CITY = 20
EXCLUDED_FILES = []

# Enhanced gradient sensitivity parameters
GRADIENT_SENSITIVITY = 0.1  # Lower = more sensitive to curve bends (0.5-1.5)
MIN_EXCLUSION_PCT = 3.0     # Minimum 3% exclusion
MAX_EXCLUSION_PCT = 10.0
ELBOW_WINDOW = 2            # Points to check for sustained low gradient


class SpatialAnalyzer:
    def __init__(self, n_dimensions=768):
        self.n_dimensions = n_dimensions
        self.tile_correlations = defaultdict(list)
        self.cities_processed = 0
        self.tiles_analyzed = 0
        self.patches_analyzed = 0
    
    def analyze_patch_coherence(self, patches):
        """Measure how coherent patches are within a tile."""
        coherence_scores = np.zeros(self.n_dimensions)
        
        for dim in range(self.n_dimensions):
            dim_values = patches[:, dim]
            mean_val = np.mean(dim_values)
            std_val = np.std(dim_values)
            
            if abs(mean_val) > 1e-10:
                cv = std_val / abs(mean_val)
                coherence_scores[dim] = 1.0 / (1.0 + cv)
            else:
                coherence_scores[dim] = 1.0 / (1.0 + std_val)
        
        return coherence_scores
    
    def analyze_tile_correlation(self, tiles_data):
        """Analyze correlation between tile proximity and embedding similarity."""
        if len(tiles_data) < 2:
            return np.zeros(self.n_dimensions)
        
        locations = np.array([loc for loc, _ in tiles_data])
        embeddings = np.array([emb for _, emb in tiles_data])
        
        spatial_distances = cdist(locations, locations, metric='euclidean')
        n_tiles = len(tiles_data)
        pairs = [(i, j) for i in range(n_tiles) for j in range(i+1, n_tiles)]
        
        if len(pairs) > 300:
            pairs = pairs[:300]
        
        correlation_scores = np.zeros(self.n_dimensions)
        
        for dim in range(self.n_dimensions):
            spatial_dists = []
            embedding_diffs = []
            
            for i, j in pairs:
                spatial_dists.append(spatial_distances[i, j])
                embedding_diffs.append(abs(embeddings[i, dim] - embeddings[j, dim]))
            
            if len(spatial_dists) > 10 and np.std(embedding_diffs) > 0:
                corr, _ = spearmanr(spatial_dists, embedding_diffs)
                if not np.isnan(corr):
                    correlation_scores[dim] = -corr
        
        return correlation_scores
    
    def process_city(self, file_path):
        """Process a single city file."""
        city_name = os.path.basename(file_path).replace('.gpq', '')
        
        try:
            gdf = gpd.read_parquet(file_path)
            
            if len(gdf) < MIN_TILES_PER_CITY:
                logger.warning(f"{city_name}: Too few tiles ({len(gdf)})")
                return 0
            
            for col in gdf.columns:
                if gdf[col].dtype.name == 'category':
                    gdf[col] = gdf[col].astype(str)
            
            if len(gdf) > MAX_TILES_PER_CITY:
                gdf = gdf.sample(n=MAX_TILES_PER_CITY, random_state=42)
            
            logger.info(f"Processing {city_name}: {len(gdf)} tiles")
            
            tiles_data = []
            coherence_accumulator = np.zeros(self.n_dimensions)
            valid_tiles = 0
            
            for _, row in gdf.iterrows():
                lon = row.get('centroid_lon') or row.get('longitude')
                lat = row.get('centroid_lat') or row.get('latitude')
                
                if lon is None or lat is None:
                    continue
                
                full_embedding = row.get('embedding_patches_full')
                
                if full_embedding is not None:
                    try:
                        if isinstance(full_embedding, np.ndarray):
                            embedding_array = full_embedding
                        else:
                            embedding_array = np.array(full_embedding)
                        
                        if embedding_array.size == 196 * 768:
                            patches = embedding_array.reshape(196, 768)
                            
                            coherence = self.analyze_patch_coherence(patches)
                            coherence_accumulator += coherence
                            
                            mean_embedding = np.mean(patches, axis=0)
                            tiles_data.append(([float(lon), float(lat)], mean_embedding))
                            
                            valid_tiles += 1
                            self.patches_analyzed += 196
                            
                    except Exception as e:
                        logger.debug(f"Error processing tile: {e}")
                        continue
            
            if valid_tiles == 0:
                return 0
            
            avg_coherence = coherence_accumulator / valid_tiles
            tile_correlation = self.analyze_tile_correlation(tiles_data)
            
            # Combine scores: 60% patch coherence + 40% tile correlation
            for dim in range(self.n_dimensions):
                combined_score = 0.6 * avg_coherence[dim] + 0.4 * tile_correlation[dim]
                self.tile_correlations[dim].append(combined_score)
            
            self.cities_processed += 1
            self.tiles_analyzed += valid_tiles
            
            del gdf, tiles_data
            gc.collect()
            
            return valid_tiles
            
        except Exception as e:
            logger.error(f"Error processing {city_name}: {e}")
            return 0
    
    def compute_final_scores(self):
        """Compute final spatial correlation scores."""
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


def find_elbow_cutoff(scores):
    """
    Find optimal cutoff using enhanced elbow detection.
    More sensitive to the actual bend in the curve.
    """
    n_dims = len(scores)
    
    # Calculate second derivative (acceleration) to find curve bend
    first_derivative = np.gradient(scores)
    second_derivative = np.gradient(first_derivative)
    
    # Smooth second derivative to reduce noise
    window_size = 5
    if len(second_derivative) > window_size:
        smoothed_second_deriv = np.convolve(second_derivative, np.ones(window_size)/window_size, mode='valid')
        smoothed_second_deriv = np.pad(smoothed_second_deriv, (window_size//2, window_size//2), mode='edge')
    else:
        smoothed_second_deriv = second_derivative
    
    # Find where curve starts to flatten (second derivative approaches zero)
    # Focus on first 25% of dimensions where spatial correlation is high
    search_range = int(n_dims * 0.25)
    
    # Enhanced elbow detection using multiple criteria
    candidates = []
    
    # Method 1: Second derivative threshold
    abs_second_deriv = np.abs(smoothed_second_deriv[:search_range])
    if len(abs_second_deriv) > 0:
        threshold = np.percentile(abs_second_deriv, 25) * GRADIENT_SENSITIVITY
        
        # Find sustained low curvature
        for i in range(len(abs_second_deriv) - ELBOW_WINDOW):
            if all(abs_second_deriv[i:i+ELBOW_WINDOW] < threshold):
                candidates.append(i)
                break
    
    # Method 2: Gradient magnitude threshold
    gradients = np.abs(np.gradient(scores[:search_range]))
    if len(gradients) > 0:
        grad_threshold = np.percentile(gradients, 15) * GRADIENT_SENSITIVITY
        
        for i in range(len(gradients) - ELBOW_WINDOW):
            if all(gradients[i:i+ELBOW_WINDOW] < grad_threshold):
                candidates.append(i)
                break
    
    # Method 3: Relative change threshold
    if len(scores) > 10:
        max_score = np.max(scores[:10])  # Use top 10 as reference
        relative_changes = []
        for i in range(1, search_range):
            if max_score > 0:
                rel_change = abs(scores[i] - scores[i-1]) / max_score
                relative_changes.append(rel_change)
            else:
                relative_changes.append(0)
        
        if len(relative_changes) > 0:
            rel_threshold = np.percentile(relative_changes, 20) * GRADIENT_SENSITIVITY
            
            for i in range(len(relative_changes) - ELBOW_WINDOW):
                if all(np.array(relative_changes[i:i+ELBOW_WINDOW]) < rel_threshold):
                    candidates.append(i + 1)
                    break
    
    # Choose the most conservative (earliest) cutoff from candidates
    if candidates:
        cutoff_idx = min(candidates)
    else:
        # Fallback to conservative percentage
        cutoff_idx = int(n_dims * 0.08)  # 8% fallback
    
    # Apply bounds
    min_cutoff = int(n_dims * MIN_EXCLUSION_PCT / 100)
    max_cutoff = int(n_dims * MAX_EXCLUSION_PCT / 100)
    cutoff_idx = max(min_cutoff, min(cutoff_idx, max_cutoff))
    
    return {
        'cutoff_index': cutoff_idx,
        'dimensions_to_exclude': cutoff_idx,
        'exclusion_percentage': (cutoff_idx / n_dims) * 100,
        'method': 'enhanced_elbow',
        'sensitivity': GRADIENT_SENSITIVITY,
        'candidates_found': len(candidates)
    }


def create_threshold_plot(scores, sorted_indices, cutoff_info, output_dir, timestamp):
    """Create clean plot showing dimension filtering threshold."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    sorted_scores = scores[sorted_indices]
    cutoff = cutoff_info['cutoff_index']
    
    x_values = np.arange(len(sorted_scores))
    
    # Plot all dimensions
    ax.plot(x_values, sorted_scores, 'b-', linewidth=2, alpha=0.8, label='Spatial Correlation Score')
    
    # Highlight regions
    ax.fill_between(x_values[:cutoff], 0, sorted_scores[:cutoff], 
                    alpha=0.4, color='red', label=f'Excluded Dimensions ({cutoff})')
    ax.fill_between(x_values[cutoff:], 0, sorted_scores[cutoff:], 
                    alpha=0.3, color='green', label=f'Kept Dimensions ({len(sorted_scores) - cutoff})')
    
    # Cutoff line
    ax.axvline(x=cutoff, color='red', linestyle='--', linewidth=3,
               label=f'Cutoff at dimension {cutoff}')
    
    # Annotation
    if cutoff < len(sorted_scores):
        threshold_value = sorted_scores[cutoff]
        ax.annotate(f'Threshold: {threshold_value:.4f}\n({cutoff_info["exclusion_percentage"]:.1f}% excluded)', 
                   xy=(cutoff, threshold_value), 
                   xytext=(cutoff + 50, threshold_value + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))
    
    # Styling
    ax.set_xlabel('Dimension Index (Ranked by Spatial Correlation)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spatial Correlation Score', fontsize=14, fontweight='bold')
    ax.set_title('Spatial Correlation Analysis: Dimension Filtering Threshold\n'
                f'Dimensions with high spatial correlation will be excluded from similarity search',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Stats box
    stats_text = (f"Total Dimensions: {len(sorted_scores)}\n"
                 f"Excluded (Spatial): {cutoff} ({cutoff_info['exclusion_percentage']:.1f}%)\n"
                 f"Kept (Visual): {len(sorted_scores) - cutoff} ({100 - cutoff_info['exclusion_percentage']:.1f}%)\n"
                 f"Method: {cutoff_info['method']}\n"
                 f"Sensitivity: {cutoff_info['sensitivity']}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                                             facecolor="lightblue", alpha=0.8))
    
    ax.set_xlim(-10, len(sorted_scores) + 10)
    ax.set_ylim(min(0, np.min(sorted_scores) * 1.1), np.max(sorted_scores) * 1.1)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f'dimension_filtering_threshold_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Threshold plot saved: {plot_file}")
    plt.close()
    
    return plot_file


def create_analysis_plots(scores, sorted_indices, cutoff_info, output_dir):
    """Create comprehensive analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    sorted_scores = scores[sorted_indices]
    cutoff = cutoff_info['cutoff_index']
    
    # Plot 1: Main correlation curve
    ax1 = axes[0, 0]
    ax1.plot(sorted_scores, 'b-', linewidth=1.5, label='Spatial correlation')
    ax1.axvline(x=cutoff, color='red', linestyle='--', linewidth=2,
                label=f'Cutoff ({cutoff} dims)')
    ax1.fill_between(range(cutoff), 0, sorted_scores[:cutoff], 
                     alpha=0.3, color='red', label='Excluded')
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
    ax2.set_xlabel('Dimension (ranked)')
    ax2.set_ylabel('|Gradient|')
    ax2.set_title('Gradient Analysis')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Cutoff region zoom
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
    ax3.set_title('Cutoff Region Detail')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Score distribution
    ax4 = axes[1, 0]
    ax4.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    threshold_value = sorted_scores[cutoff] if cutoff < len(sorted_scores) else 0
    ax4.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Spatial Correlation Score')
    ax4.set_ylabel('Number of Dimensions')
    ax4.set_title('Score Distribution')
    
    # Plot 5: Top dimensions
    ax5 = axes[1, 1]
    top_n = min(30, len(sorted_indices))
    top_scores = sorted_scores[:top_n]
    colors = ['red' if i < cutoff else 'blue' for i in range(top_n)]
    ax5.bar(range(top_n), top_scores, color=colors, alpha=0.7)
    ax5.set_xlabel('Rank')
    ax5.set_ylabel('Spatial Correlation')
    ax5.set_title(f'Top {top_n} Dimensions')
    
    # Plot 6: Cumulative variance
    ax6 = axes[1, 2]
    cumsum = np.cumsum(sorted_scores)
    total_correlation = np.sum(scores)
    cumsum_pct = (cumsum / total_correlation * 100) if total_correlation > 0 else cumsum
    ax6.plot(cumsum_pct, 'g-', linewidth=2)
    ax6.axvline(x=cutoff, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Number of Dimensions')
    ax6.set_ylabel('Cumulative Spatial Information (%)')
    ax6.set_title('Cumulative Analysis')
    ax6.grid(alpha=0.3)
    
    plt.suptitle(f'Spatial Correlation Analysis: Excluding {cutoff} dimensions ({cutoff_info["exclusion_percentage"]:.1f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    timestamp = int(time.time())
    plot_file = os.path.join(output_dir, f'spatial_correlation_analysis_{timestamp}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"Analysis plot saved: {plot_file}")
    plt.close()
    
    return timestamp


def main():
    """Main execution function."""
    logger.info("Spatial Autocorrelation Analysis")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    pattern = os.path.join(EMBEDDINGS_DIR, "full_patch_embeddings", "*", "*.gpq")
    all_files = glob(pattern)
    files = [f for f in all_files if not any(excluded in os.path.basename(f) for excluded in EXCLUDED_FILES)]
    
    if not files:
        logger.error(f"No files found in {pattern}")
        return 1
    
    logger.info(f"Found {len(files)} city files")
    logger.info(f"Gradient sensitivity: {GRADIENT_SENSITIVITY} (lower = more sensitive)")
    logger.info(f"Exclusion range: {MIN_EXCLUSION_PCT}% - {MAX_EXCLUSION_PCT}%")
    
    analyzer = SpatialAnalyzer()
    
    for file_path in tqdm(files, desc="Analyzing cities"):
        analyzer.process_city(file_path)
        
        if analyzer.cities_processed % 10 == 0:
            gc.collect()
            logger.info(f"Progress: {analyzer.cities_processed} cities, "
                       f"{analyzer.tiles_analyzed} tiles")
    
    logger.info("Computing final scores...")
    final_scores = analyzer.compute_final_scores()
    
    sorted_indices = np.argsort(final_scores)[::-1]
    sorted_scores = final_scores[sorted_indices]
    
    cutoff_info = find_elbow_cutoff(sorted_scores)
    excluded_dimensions = sorted_indices[:cutoff_info['dimensions_to_exclude']].tolist()
    
    # Create plots
    timestamp = create_analysis_plots(final_scores, sorted_indices, cutoff_info, OUTPUT_DIR)
    create_threshold_plot(final_scores, sorted_indices, cutoff_info, OUTPUT_DIR, timestamp)
    
    # Save results
    results = {
        'analysis_type': 'spatial_autocorrelation',
        'analysis_params': {
            'cities_processed': analyzer.cities_processed,
            'tiles_analyzed': analyzer.tiles_analyzed,
            'patches_analyzed': analyzer.patches_analyzed,
            'gradient_sensitivity': GRADIENT_SENSITIVITY,
            'timestamp': time.time()
        },
        'cutoff_results': {
            'method': cutoff_info['method'],
            'total_dimensions': len(final_scores),
            'dimensions_to_exclude': cutoff_info['dimensions_to_exclude'],
            'excluded_dimensions': [int(d) for d in excluded_dimensions],
            'exclusion_percentage': cutoff_info['exclusion_percentage'],
            'sensitivity': cutoff_info['sensitivity'],
            'candidates_found': cutoff_info['candidates_found']
        },
        'top_spatial_dimensions': [
            {
                'dimension': int(sorted_indices[i]),
                'spatial_correlation': float(sorted_scores[i])
            }
            for i in range(min(20, len(sorted_indices)))
        ]
    }
    
    output_file = os.path.join(OUTPUT_DIR, 'spatial_correlation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    duration = (time.time() - start_time) / 60
    logger.info("\n" + "=" * 50)
    logger.info("ANALYSIS COMPLETE!")
    logger.info(f"Duration: {duration:.1f} minutes")
    logger.info(f"Cities: {analyzer.cities_processed}")
    logger.info(f"Tiles: {analyzer.tiles_analyzed}")
    logger.info(f"Patches: {analyzer.patches_analyzed:,}")
    logger.info(f"\nRESULTS:")
    logger.info(f"Excluded dimensions: {cutoff_info['dimensions_to_exclude']} ({cutoff_info['exclusion_percentage']:.1f}%)")
    logger.info(f"Top 5 spatial dims: {excluded_dimensions[:5]}")
    logger.info(f"Sensitivity: {GRADIENT_SENSITIVITY}")
    logger.info(f"Candidates found: {cutoff_info['candidates_found']}")
    logger.info(f"\nFiles saved:")
    logger.info(f"  - Results: {output_file}")
    logger.info(f"  - Plots: outputs/spatial_correlation_*_{timestamp}.png")
    
    return 0


if __name__ == "__main__":
    exit(main())