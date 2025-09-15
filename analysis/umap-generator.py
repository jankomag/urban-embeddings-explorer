
# !/usr/bin/env python3
"""
Standalone UMAP Coordinates Generator - Local Embeddings with Dimension Filtering
================================================================================

This script generates UMAP coordinates from local embedding files with dimension filtering.
It removes city-discriminative dimensions before computing aggregations and UMAP coordinates,
following the same filtering approach as the data migration script.

Usage:
    python local_umap_generator_filtered.py

Configuration:
    Adjust UMAP parameters below to experiment with different visualizations.
    Requires dimension analysis results file.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob
import umap
import logging
from tqdm import tqdm
import time
from dotenv import load_dotenv
import gzip
import hashlib
from typing import Dict, List, Optional, Tuple

# Load environment variables
load_dotenv()

# Configuration
EMBEDDINGS_DIR = "/Users/janmagnuszewski/dev/terramind/embeddings/urban_embeddings_224_terramind_normalised"
OUTPUT_DIR = "../backend/production_data"
BACKUP_EXISTING = False  # Create backup of existing UMAP file

# Dimension analysis results file
DIMENSION_ANALYSIS_FILE = "./spatial_correlation_analysis/spatial_correlation_results.json"

# UMAP Parameters
# UMAP Parameters: {'n_components': 6, 'n_neighbors': 7, 'min_dist': 1, 'metric': 'cosine', 'random_state': 42, 'n_epochs': 200}
UMAP_PARAMS = {
    'n_components': 6,
    'n_neighbors': 7,      # Try: 15, 30, 50, 100 - Lower = more local structure, Higher = more global
    'min_dist': 1,        # Try: 0.01, 0.1, 0.3, 0.5 - Lower = tighter clusters, Higher = more spread
    'metric': 'cosine',     # Try: 'euclidean', 'cosine', 'manhattan'
    'random_state': 42,     # Keep consistent for reproducible results
    'n_epochs': 500,        # Try: 200, 500, 1000 - More epochs = better convergence but slower
    'learning_rate': 1.0,   # Try: 0.5, 1.0, 2.0 - Affects convergence speed
    'spread': 1.0,          # Try: 0.5, 1.0, 2.0 - Controls how tightly UMAP packs points
    'set_op_mix_ratio': 1.0 # Try: 0.0-1.0 - Balance between fuzzy union and intersection
}

# Memory management settings
CITY_BATCH_SIZE = 50  # Process 50 cities at a time
MAX_CITIES = None  # Set to None to process all cities, or set a number to limit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dimension_analysis_results(filepath: str) -> Optional[Dict]:
    """Load dimension analysis results from JSON file."""
    if not os.path.exists(filepath):
        logger.error(f"Dimension analysis file not found: {filepath}")
        logger.info(f"Please run: python standalone_dimension_analysis.py first")
        return None
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        excluded_dims = results['cutoff_results']['excluded_dimensions']
        
        logger.info(f"Loaded dimension analysis results:")
        logger.info(f"   Original dimensions: {results['cutoff_results']['total_dimensions']}")
        logger.info(f"   Excluded dimensions: {len(excluded_dims)}")
        logger.info(f"   Remaining dimensions: {results['cutoff_results']['total_dimensions'] - len(excluded_dims)}")
        logger.info(f"   Exclusion percentage: {results['cutoff_results']['exclusion_percentage']:.1f}%")
        logger.info(f"   Analysis method: {results['cutoff_results']['method']}")
        
        # Show some excluded dimensions
        if len(excluded_dims) > 0:
            show_dims = excluded_dims[:10] if len(excluded_dims) > 10 else excluded_dims
            logger.info(f"   Sample excluded dims: {show_dims}")
            if len(excluded_dims) > 10:
                logger.info(f"   ... and {len(excluded_dims) - 10} more")
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading dimension analysis results: {e}")
        return None

class LocalUMAPGeneratorFiltered:
    def __init__(self, excluded_dimensions: List[int]):
        """Initialize the UMAP generator with dimension filtering."""
        self.excluded_dimensions = excluded_dimensions
        self.filtered_dimension_count = 768 - len(excluded_dimensions)
        
        logger.info(f"Dimension filtering configured:")
        logger.info(f"   Excluded dimensions: {len(self.excluded_dimensions)}")
        logger.info(f"   Filtered vector size: {self.filtered_dimension_count}")

    def filter_embedding_dimensions(self, full_patches: np.ndarray) -> np.ndarray:
        """
        Remove discriminative dimensions from patch embeddings.
        
        Args:
            full_patches: Array of shape (196, 768) containing patch embeddings
            
        Returns:
            Filtered patch embeddings with discriminative dimensions removed
        """
        if len(self.excluded_dimensions) == 0:
            return full_patches
        
        # Create mask for dimensions to keep
        all_dimensions = set(range(768))
        excluded_set = set(self.excluded_dimensions)
        dimensions_to_keep = sorted(list(all_dimensions - excluded_set))
        
        # Filter dimensions
        filtered_patches = full_patches[:, dimensions_to_keep]
        
        return filtered_patches

    def extract_exact_tile_bounds(self, row: pd.Series):
        """Extract exact tile boundary coordinates from GeoParquet geometry."""
        try:
            # Method 1: Check if we have explicit bounds columns
            bounds_cols = ['bounds_min_lon', 'bounds_min_lat', 'bounds_max_lon', 'bounds_max_lat']
            if all(col in row.index for col in bounds_cols):
                min_lon = float(row['bounds_min_lon'])
                min_lat = float(row['bounds_min_lat'])
                max_lon = float(row['bounds_max_lon'])
                max_lat = float(row['bounds_max_lat'])
                
                tile_bounds = [
                    [min_lon, max_lat], [max_lon, max_lat],
                    [max_lon, min_lat], [min_lon, min_lat],
                    [min_lon, max_lat]
                ]
                return tile_bounds
            
            # Method 2: Try to extract from geometry column
            elif 'geometry' in row.index and row['geometry'] is not None:
                geom = row['geometry']
                if hasattr(geom, 'bounds'):
                    min_lon, min_lat, max_lon, max_lat = geom.bounds
                    tile_bounds = [
                        [min_lon, max_lat], [max_lon, max_lat],
                        [max_lon, min_lat], [min_lon, min_lat],
                        [min_lon, max_lat]
                    ]
                    return tile_bounds
            
            return None
        except Exception as e:
            logger.debug(f"Could not extract exact bounds: {e}")
            return None

    def process_tile_enhanced(self, row: pd.Series):
        """Enhanced tile processing with dimension filtering - same logic as migration script."""
        try:
            # Get coordinates
            longitude = row.get('centroid_lon') or row.get('longitude')
            latitude = row.get('centroid_lat') or row.get('latitude')
            
            if longitude is None or latitude is None:
                return None, None
            
            if pd.isna(longitude) or pd.isna(latitude):
                return None, None
            
            lon_val = float(longitude)
            lat_val = float(latitude)
            
            if lon_val < -180 or lon_val > 180 or lat_val < -90 or lat_val > 90:
                return None, None
            
            # Get embedding - prioritize full patches
            full_embedding = row.get('embedding_patches_full')
            
            if full_embedding is not None:
                try:
                    if isinstance(full_embedding, np.ndarray):
                        embedding_array = full_embedding
                    else:
                        embedding_array = np.array(full_embedding)
                    
                    if embedding_array.size == 196 * 768:
                        full_patches = embedding_array.reshape(196, 768)
                    else:
                        logger.warning(f"Unexpected full embedding size: {embedding_array.size}")
                        return None, None
                        
                except Exception as e:
                    logger.warning(f"Error processing full patch embedding: {e}")
                    return None, None
            else:
                logger.debug("No full patch embedding available")
                return None, None
            
            # Generate reproducible tile ID - SAME LOGIC AS MIGRATION SCRIPT
            tile_id = row.get('tile_id')
            
            if tile_id is None or pd.isna(tile_id):
                lon_str = f"{lon_val:.6f}"
                lat_str = f"{lat_val:.6f}"
                tile_id_str = f"tile_{lon_str}_{lat_str}"
            else:
                tile_id_str = str(tile_id).strip()
            
            hash_object = hashlib.sha256(tile_id_str.encode('utf-8'))
            hash_hex = hash_object.hexdigest()
            unique_id = int(hash_hex[:8], 16) % (2**31 - 1)
            
            # Get required fields
            city = row.get('city')
            country = row.get('country')
            continent = row.get('continent', 'Unknown')
            
            def is_valid_string_field(val):
                if val is None:
                    return False
                try:
                    if pd.isna(val):
                        return False
                except (ValueError, TypeError):
                    pass
                str_val = str(val).strip().lower()
                return str_val and str_val not in ['unknown', 'nan', 'none', '']
            
            if not is_valid_string_field(city) or not is_valid_string_field(country):
                return None, None
            
            if not is_valid_string_field(continent):
                continent = 'Unknown'
            
            # Get date
            date_str = None
            acq_date = row.get('acquisition_date')
            if acq_date is not None:
                try:
                    if not pd.isna(acq_date):
                        date_str = str(acq_date)
                except (ValueError, TypeError):
                    date_str = str(acq_date) if str(acq_date) != 'nan' else None
            
            # Extract exact tile bounds
            exact_tile_bounds = self.extract_exact_tile_bounds(row)
            
            # Create metadata with reproducible ID
            metadata = {
                'id': unique_id,
                'original_tile_id': tile_id_str,
                'city': str(city).strip(),
                'country': str(country).strip(),
                'continent': str(continent).strip(),
                'longitude': lon_val,
                'latitude': lat_val,
                'date': date_str,
                'tile_bounds': exact_tile_bounds,
                'has_exact_bounds': exact_tile_bounds is not None,
                'dimension_filtering_applied': True,
                'excluded_dimensions_count': len(self.excluded_dimensions),
                'filtered_dimension_count': self.filtered_dimension_count
            }
            
            # Add tile size information if we have bounds
            if exact_tile_bounds:
                lons = [coord[0] for coord in exact_tile_bounds[:-1]]
                lats = [coord[1] for coord in exact_tile_bounds[:-1]]
                
                tile_width_deg = max(lons) - min(lons)
                tile_height_deg = max(lats) - min(lats)
                
                metadata.update({
                    'tile_width_degrees': tile_width_deg,
                    'tile_height_degrees': tile_height_deg
                })
            
            return metadata, full_patches
            
        except Exception as e:
            logger.warning(f"Error processing tile: {e}")
            return None, None

    def process_city_batch(self, files_batch: list):
        """Process a batch of city files and extract filtered embeddings + metadata."""
        batch_embeddings = []
        batch_metadata = []
        
        tiles_with_exact_bounds = 0
        tiles_with_fallback_bounds = 0
        cities_in_batch = []
        
        logger.info(f"Processing batch of {len(files_batch)} cities with dimension filtering...")
        
        for file_idx, file_path in enumerate(files_batch, 1):
            try:
                file_name = os.path.basename(file_path)
                logger.info(f"[{file_idx}/{len(files_batch)}] Loading {file_name}...")
                gdf = gpd.read_parquet(file_path)
                
                if len(gdf) == 0:
                    logger.warning(f"Empty file: {file_path}")
                    continue
                
                # Get city info from first row
                first_row = gdf.iloc[0]
                city_name = first_row.get('city', '').strip()
                country_name = first_row.get('country', '').strip()
                
                if not city_name or not country_name:
                    logger.warning(f"Missing city/country info in {file_path}")
                    del gdf
                    continue
                
                city_key = f"{city_name}, {country_name}"
                cities_in_batch.append(city_key)
                
                logger.info(f"Processing {city_key} - {len(gdf)} tiles (filtered dims: {self.filtered_dimension_count})")
                
                # Convert categorical columns
                for col in gdf.columns:
                    if gdf[col].dtype.name == 'category':
                        gdf[col] = gdf[col].astype(str)
                
                city_tile_count = 0
                city_exact_bounds = 0
                
                # Process each tile in the city
                for idx, row in enumerate(gdf.iterrows(), 1):
                    _, row_data = row
                    
                    metadata, full_patches = self.process_tile_enhanced(row_data)
                    
                    if metadata and full_patches is not None:
                        # Apply dimension filtering BEFORE computing mean aggregation
                        filtered_patches = self.filter_embedding_dimensions(full_patches)
                        
                        # Compute mean aggregation on filtered patches
                        mean_embedding = np.mean(filtered_patches, axis=0)
                        
                        # Store metadata and embedding
                        batch_metadata.append(metadata)
                        batch_embeddings.append(mean_embedding)
                        
                        city_tile_count += 1
                        
                        # Track bounds statistics
                        if metadata.get('has_exact_bounds', False):
                            city_exact_bounds += 1
                            tiles_with_exact_bounds += 1
                        else:
                            tiles_with_fallback_bounds += 1
                        
                        # Clean up patches immediately
                        del full_patches, filtered_patches
                
                if city_tile_count > 0:
                    bounds_info = f"({city_exact_bounds} exact bounds)" if city_exact_bounds > 0 else "(fallback bounds)"
                    logger.info(f"✅ {city_key}: {city_tile_count} tiles processed {bounds_info} [filtered: {self.filtered_dimension_count} dims]")
                else:
                    logger.warning(f"⚠️ {city_key}: No valid tiles found")
                
                # Clean up the GeoDataFrame
                del gdf
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        # Convert embeddings to numpy array
        if batch_embeddings:
            batch_embeddings = np.array(batch_embeddings)
        else:
            batch_embeddings = np.array([]).reshape(0, self.filtered_dimension_count)
        
        # Batch summary
        if cities_in_batch:
            logger.info(f"BATCH PROCESSING COMPLETE")
            logger.info(f"   Cities processed: {len(cities_in_batch)}")
            logger.info(f"   Total tiles: {len(batch_metadata)}")
            logger.info(f"   Filtered dimensions: {self.filtered_dimension_count}/768")
            logger.info(f"   Bounds: {tiles_with_exact_bounds} exact, {tiles_with_fallback_bounds} fallback")
        
        return batch_embeddings, batch_metadata

    def load_embeddings_from_local_files(self):
        """Load embeddings from local files in batches with dimension filtering."""
        logger.info(f"Loading embeddings from local directory with dimension filtering...")
        
        # Find all parquet files
        pattern = os.path.join(EMBEDDINGS_DIR, "full_patch_embeddings", "*", "*.gpq")
        files = glob(pattern)
        
        if not files:
            raise ValueError(f"No .gpq files found in {pattern}")
        
        logger.info(f"Found {len(files)} files to process")
        
        # Limit files if MAX_CITIES is set
        if MAX_CITIES and MAX_CITIES < len(files):
            files = files[:MAX_CITIES]
            logger.info(f"Limited to first {MAX_CITIES} files")
        
        all_embeddings = []
        all_metadata = []
        
        # Process files in batches
        for i in tqdm(range(0, len(files), CITY_BATCH_SIZE), desc="Processing city batches"):
            files_batch = files[i:i + CITY_BATCH_SIZE]
            
            batch_num = i//CITY_BATCH_SIZE + 1
            total_batches = (len(files) + CITY_BATCH_SIZE - 1)//CITY_BATCH_SIZE
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Process this batch of cities
            batch_embeddings, batch_metadata = self.process_city_batch(files_batch)
            
            if len(batch_embeddings) > 0:
                all_embeddings.append(batch_embeddings)
                all_metadata.extend(batch_metadata)
        
        # Combine all embeddings
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
        else:
            combined_embeddings = np.array([]).reshape(0, self.filtered_dimension_count)
        
        logger.info(f"Loading Summary with dimension filtering:")
        logger.info(f"   Total embeddings loaded: {len(combined_embeddings)}")
        logger.info(f"   Total metadata entries: {len(all_metadata)}")
        logger.info(f"   Filtered embedding dimension: {combined_embeddings.shape[1] if len(combined_embeddings) > 0 else 0}")
        logger.info(f"   Original dimension: 768")
        logger.info(f"   Excluded dimensions: {len(self.excluded_dimensions)}")
        
        return combined_embeddings, all_metadata

    def generate_umap_coordinates(self, embeddings):
        """Generate UMAP coordinates with specified parameters on filtered embeddings."""
        logger.info("Generating UMAP coordinates from filtered embeddings...")
        logger.info(f"UMAP Parameters:")
        for param, value in UMAP_PARAMS.items():
            logger.info(f"   {param}: {value}")
        
        logger.info(f"Input embedding shape: {embeddings.shape}")
        logger.info(f"Dimension filtering applied: {len(self.excluded_dimensions)} dimensions excluded")
        
        try:
            # Initialize UMAP with specified parameters
            reducer = umap.UMAP(**UMAP_PARAMS)
            
            # Fit and transform embeddings
            logger.info("Fitting UMAP model on filtered embeddings (this may take a few minutes)...")
            umap_coords = reducer.fit_transform(embeddings)
            
            logger.info(f"UMAP coordinates generated: {umap_coords.shape}")
            return umap_coords
            
        except Exception as e:
            logger.error(f"Error generating UMAP coordinates: {e}")
            raise
    
    def create_umap_data_structure(self, umap_coords, metadata_list, analysis_results):
        """Create the UMAP data structure matching the expected format with filtering info."""
        logger.info("Creating UMAP data structure with dimension filtering info...")
        
        umap_points = []
        tiles_with_exact_bounds = 0
        tiles_with_fallback_bounds = 0
        
        for i, (coords, metadata) in enumerate(zip(umap_coords, metadata_list)):
            umap_point = {
                'location_id': metadata['id'],
                'x': float(coords[0]),
                'y': float(coords[1]),
                'city': metadata['city'],
                'country': metadata['country'],
                'continent': metadata['continent'],
                'longitude': metadata['longitude'],
                'latitude': metadata['latitude'],
                'date': metadata.get('date'),
                'bounds': metadata.get('tile_bounds')  # Preserve exact bounds if available
            }
            
            # Track bounds statistics
            if metadata.get('has_exact_bounds', False):
                tiles_with_exact_bounds += 1
            else:
                tiles_with_fallback_bounds += 1
            
            umap_points.append(umap_point)
        
        # Create the complete UMAP data structure with dimension filtering info
        umap_data = {
            'umap_points': umap_points,
            'total_points': len(umap_points),
            'bounds_statistics': {
                'tiles_with_exact_bounds': tiles_with_exact_bounds,
                'tiles_with_fallback_bounds': tiles_with_fallback_bounds,
                'exact_bounds_percentage': (tiles_with_exact_bounds / len(umap_points) * 100) if umap_points else 0
            },
            'dimension_info': {
                'original_dimensions': 768,
                'filtered_dimensions': self.filtered_dimension_count,
                'excluded_dimensions_count': len(self.excluded_dimensions),
                'excluded_dimensions': self.excluded_dimensions,
                'filtering_applied': True,
                'analysis_results': analysis_results
            },
            'generation_info': {
                'timestamp': time.time(),
                'umap_parameters': UMAP_PARAMS.copy(),
                'source': 'local_embeddings_filtered',
                'source_directory': EMBEDDINGS_DIR,
                'total_embeddings': len(umap_coords),
                'script_version': '1.0.0',
                'aggregation_method': 'mean_filtered',
                'dimension_analysis_file': DIMENSION_ANALYSIS_FILE
            }
        }
        
        logger.info(f"UMAP Data Summary with dimension filtering:")
        logger.info(f"   Total points: {len(umap_points)}")
        logger.info(f"   Original dimensions: 768")
        logger.info(f"   Filtered dimensions: {self.filtered_dimension_count}")
        logger.info(f"   Excluded dimensions: {len(self.excluded_dimensions)}")
        logger.info(f"   Exact bounds: {tiles_with_exact_bounds}")
        logger.info(f"   Fallback bounds: {tiles_with_fallback_bounds}")
        logger.info(f"   Coverage: {umap_data['bounds_statistics']['exact_bounds_percentage']:.1f}%")
        
        return umap_data
    
    def backup_existing_file(self, filepath):
        """Create backup of existing file."""
        if not os.path.exists(filepath):
            return
        
        timestamp = int(time.time())
        backup_path = f"{filepath}.backup_{timestamp}"
        
        try:
            if filepath.endswith('.gz'):
                # Copy compressed file
                with gzip.open(filepath, 'rb') as src:
                    with gzip.open(backup_path, 'wb') as dst:
                        dst.write(src.read())
            else:
                # Copy regular file
                with open(filepath, 'r') as src:
                    with open(backup_path, 'w') as dst:
                        dst.write(src.read())
            
            logger.info(f"Backup created: {backup_path}")
            
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")
    
    def save_umap_coordinates(self, umap_data):
        """Save UMAP coordinates to files."""
        logger.info("Saving filtered UMAP coordinates...")
        
        # Prepare file paths for filtered version
        json_file = os.path.join(OUTPUT_DIR, 'umap_coordinates_filtered.json')
        gzip_file = os.path.join(OUTPUT_DIR, 'umap_coordinates_filtered.json.gz')
        
        # Create backup if requested
        if BACKUP_EXISTING:
            self.backup_existing_file(json_file)
            self.backup_existing_file(gzip_file)
        
        try:
            # Save regular JSON
            with open(json_file, 'w') as f:
                json.dump(umap_data, f, indent=2)
            logger.info(f"Saved: {json_file}")
            
            # Save compressed JSON
            with gzip.open(gzip_file, 'wt') as f:
                json.dump(umap_data, f, indent=2)
            logger.info(f"Saved compressed: {gzip_file}")
            
        except Exception as e:
            logger.error(f"Error saving files: {e}")
            raise
    
    def run(self, analysis_results):
        """Main execution method with dimension filtering."""
        start_time = time.time()
        
        logger.info("Local Embeddings UMAP Coordinates Generator with Dimension Filtering")
        logger.info("=" * 80)
        logger.info(f"Embeddings directory: {EMBEDDINGS_DIR}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info(f"City batch size: {CITY_BATCH_SIZE}")
        logger.info(f"Dimension analysis file: {DIMENSION_ANALYSIS_FILE}")
        
        if MAX_CITIES:
            logger.info(f"Max cities: {MAX_CITIES}")
        else:
            logger.info(f"Processing all cities")
        
        logger.info(f"Dimension filtering:")
        logger.info(f"   Original dimensions: 768")
        logger.info(f"   Excluded dimensions: {len(self.excluded_dimensions)}")
        logger.info(f"   Remaining dimensions: {self.filtered_dimension_count}")
        logger.info(f"   Exclusion percentage: {len(self.excluded_dimensions)/768*100:.1f}%")
        
        try:
            # Step 1: Load embeddings from local files with filtering
            embeddings, metadata_list = self.load_embeddings_from_local_files()
            
            if len(embeddings) == 0:
                raise ValueError("No embeddings found!")
            
            # Step 2: Generate UMAP coordinates on filtered embeddings
            umap_coords = self.generate_umap_coordinates(embeddings)
            
            # Step 3: Create data structure with filtering info
            umap_data = self.create_umap_data_structure(umap_coords, metadata_list, analysis_results)
            
            # Step 4: Save results
            self.save_umap_coordinates(umap_data)
            
            # Summary
            duration = (time.time() - start_time) / 60
            logger.info(f"\nFiltered UMAP generation completed successfully!")
            logger.info(f"Duration: {duration:.1f} minutes")
            logger.info(f"Generated coordinates for {len(umap_coords)} points")
            logger.info(f"Used filtered embeddings with {self.filtered_dimension_count} dimensions")
            logger.info(f"Excluded {len(self.excluded_dimensions)} city-discriminative dimensions")
            logger.info(f"Files saved to: {OUTPUT_DIR}")
            logger.info(f"\nTo use new filtered coordinates:")
            logger.info(f"   1. Replace umap_coordinates.json with umap_coordinates_filtered.json in your frontend")
            logger.info(f"   2. Restart your application")
            logger.info(f"   3. The tile IDs remain consistent with Qdrant data")
            logger.info(f"   4. UMAP now reflects reduced city bias from dimension filtering")
            
            return 0
            
        except Exception as e:
            logger.error(f"Filtered UMAP generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1


def main():
    """Main function."""
    logger.info("Local Embeddings UMAP Coordinates Generator with Dimension Filtering")
    logger.info("=" * 80)
    
    # Load dimension analysis results
    analysis_results = load_dimension_analysis_results(DIMENSION_ANALYSIS_FILE)
    if analysis_results is None:
        logger.error("Cannot proceed without dimension analysis results")
        logger.info("Please run: python standalone_dimension_analysis.py first")
        return 1
    
    excluded_dimensions = analysis_results['cutoff_results']['excluded_dimensions']
    
    # Check required directories
    if not os.path.exists(EMBEDDINGS_DIR):
        logger.error(f"Embeddings directory not found: {EMBEDDINGS_DIR}")
        return 1
    
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    generator = LocalUMAPGeneratorFiltered(excluded_dimensions)
    
    try:
        return generator.run(analysis_results)
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return 1


if __name__ == "__main__":
    exit(main())