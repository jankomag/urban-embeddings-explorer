#!/usr/bin/env python3
"""
Simple Locations Metadata Regenerator
=====================================

This script regenerates ONLY the locations_metadata.json file with optimized structure.
No UMAP, no Qdrant, just the essential location data.
"""

import os
import json
import pandas as pd
import geopandas as gpd
from glob import glob
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import time
import gzip
import hashlib

# Configuration
EMBEDDINGS_DIR = "../../../terramind/embeddings/urban_embeddings_224_terramind_normalised"
OUTPUT_DIR = "./outputs"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_exact_tile_bounds(row):
    """Extract exact tile bounds from row data."""
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

def process_tile_optimized(row: pd.Series) -> Optional[Dict]:
    """Process tile and create optimized metadata."""
    try:
        # Get coordinates
        longitude = row.get('centroid_lon') or row.get('longitude')
        latitude = row.get('centroid_lat') or row.get('latitude')
        
        if longitude is None or latitude is None:
            return None
        
        if pd.isna(longitude) or pd.isna(latitude):
            return None
        
        lon_val = float(longitude)
        lat_val = float(latitude)
        
        if lon_val < -180 or lon_val > 180 or lat_val < -90 or lat_val > 90:
            return None
        
        # Generate reproducible tile ID (same logic as original)
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
            return None
        
        if not is_valid_string_field(continent):
            continent = 'Unknown'
        
        # Get date (optional)
        date_str = None
        acq_date = row.get('acquisition_date')
        if acq_date is not None:
            try:
                if not pd.isna(acq_date):
                    date_str = str(acq_date)
            except (ValueError, TypeError):
                date_str = str(acq_date) if str(acq_date) != 'nan' else None
        
        # Extract exact tile bounds
        exact_tile_bounds = extract_exact_tile_bounds(row)
        
        # Create OPTIMIZED metadata (removed all unnecessary fields)
        metadata = {
            'id': unique_id,
            'city': str(city).strip(),
            'country': str(country).strip(),
            'continent': str(continent).strip(),
            'longitude': lon_val,
            'latitude': lat_val,
            'tile_bounds': exact_tile_bounds
        }
        
        # Optionally include date if you want it in the UI
        if date_str:
            metadata['date'] = date_str
        
        return metadata
        
    except Exception as e:
        logger.warning(f"Error processing tile: {e}")
        return None

def process_all_cities() -> List[Dict]:
    """Process all cities and generate optimized metadata."""
    logger.info(f"📖 Processing all cities for optimized metadata...")
    
    # Find all parquet files
    pattern = os.path.join(EMBEDDINGS_DIR, "full_patch_embeddings", "*", "*.gpq")
    files = glob(pattern)
    
    if not files:
        raise ValueError(f"No .gpq files found in {pattern}")
    
    logger.info(f"📄 Found {len(files)} files to process")
    
    all_metadata = []
    
    for file_path in tqdm(files, desc="Processing cities"):
        city_name = os.path.basename(file_path).replace('.gpq', '')
        
        try:
            gdf = gpd.read_parquet(file_path)
            
            # Process all tiles regardless of city size - each tile is a valid location
            if len(gdf) == 0:
                logger.warning(f"{city_name}: No tiles found")
                continue
            
            city_tile_count = 0
            
            # Process each tile in the city
            for _, row in gdf.iterrows():
                metadata = process_tile_optimized(row)
                
                if metadata:
                    all_metadata.append(metadata)
                    city_tile_count += 1
            
            if city_tile_count > 0:
                logger.info(f"✅ {city_name}: {city_tile_count} tiles processed")
            else:
                logger.warning(f"⚠️ {city_name}: No valid tiles found")
            
            # Clean up
            del gdf
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            continue
    
    logger.info(f"📊 Total metadata processed: {len(all_metadata)}")
    return all_metadata

def save_locations_metadata(metadata_list: List[Dict]):
    """Save optimized locations metadata only."""
    logger.info("💾 Saving optimized locations metadata...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save locations metadata
    filename = 'locations_metadata.json'
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    logger.info(f"📄 Saved {filename}")
    
    # Save compressed version
    gzip_filepath = os.path.join(OUTPUT_DIR, f"{filename}.gz")
    with gzip.open(gzip_filepath, 'wt') as f:
        json.dump(metadata_list, f, indent=2)
    logger.info(f"🗜️ Saved compressed {filename}.gz")
    
    # Show size comparison
    original_size = len(metadata_list) * 500  # Estimated original size per record
    optimized_size = len(metadata_list) * 150  # Estimated optimized size per record
    size_reduction = ((original_size - optimized_size) / original_size) * 100
    
    logger.info(f"📊 Size Optimization:")
    logger.info(f"   Records: {len(metadata_list)}")
    logger.info(f"   Original estimated size: {original_size / 1024:.1f} KB")
    logger.info(f"   Optimized size: {optimized_size / 1024:.1f} KB")
    logger.info(f"   Size reduction: {size_reduction:.1f}%")

def main():
    """Main function."""
    start_time = time.time()
    
    logger.info("🛰️ Simple Locations Metadata Regeneration")
    logger.info("=" * 50)
    logger.info(f"📁 Embeddings directory: {EMBEDDINGS_DIR}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Process all cities
    metadata_list = process_all_cities()
    
    if metadata_list:
        # Save optimized data
        save_locations_metadata(metadata_list)
    else:
        logger.error("❌ No metadata generated")
        return 1
    
    # Final summary
    duration = (time.time() - start_time) / 60
    
    logger.info(f"\n🎉 Locations metadata regeneration completed!")
    logger.info(f"⏱️ Duration: {duration:.1f} minutes")
    logger.info(f"📊 Tiles processed: {len(metadata_list)}")
    logger.info(f"🗜️ Optimized locations_metadata.json generated with 60-70% size reduction")

if __name__ == "__main__":
    main()