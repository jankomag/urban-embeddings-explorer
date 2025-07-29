#!/usr/bin/env python3
"""
City Representatives Generator
============================

This script creates a simplified JSON file with one representative point per city,
showing city centroids for zoomed-out map views. When users zoom in, the frontend
will switch to showing all individual tiles.
"""

import os
import json
import pandas as pd
import geopandas as gpd
from glob import glob
from typing import Dict, List, Set, Tuple
import logging
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
EMBEDDINGS_DIR = "/Users/janmagnuszewski/dev/terramind/embeddings/urban_embeddings_224_terramind"
OUTPUT_DIR = "./production_data"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CityRepresentativesGenerator:
    def __init__(self):
        """Initialize the city representatives generator."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    def process_city_data(self, gdf: gpd.GeoDataFrame, city_name: str, country_name: str) -> Dict:
        """Process all tiles for a city and create a representative."""
        try:
            # Calculate city centroid from all tiles
            centroid_lon = gdf['centroid_lon'].mean() if 'centroid_lon' in gdf.columns else gdf['longitude'].mean()
            centroid_lat = gdf['centroid_lat'].mean() if 'centroid_lat' in gdf.columns else gdf['latitude'].mean()
            
            # Get city bounds (bounding box of all tiles)
            min_lon = gdf['centroid_lon'].min() if 'centroid_lon' in gdf.columns else gdf['longitude'].min()
            max_lon = gdf['centroid_lon'].max() if 'centroid_lon' in gdf.columns else gdf['longitude'].max()
            min_lat = gdf['centroid_lat'].min() if 'centroid_lat' in gdf.columns else gdf['latitude'].min()
            max_lat = gdf['centroid_lat'].max() if 'centroid_lat' in gdf.columns else gdf['latitude'].max()
            
            # Select representative tile (closest to centroid)
            distances = []
            for _, row in gdf.iterrows():
                tile_lon = row.get('centroid_lon', row.get('longitude', 0))
                tile_lat = row.get('centroid_lat', row.get('latitude', 0))
                
                # Simple distance calculation
                dist = ((tile_lon - centroid_lon) ** 2 + (tile_lat - centroid_lat) ** 2) ** 0.5
                distances.append(dist)
            
            # Find the tile closest to centroid
            representative_idx = distances.index(min(distances))
            representative_row = gdf.iloc[representative_idx]
            
            # Get all tile IDs for this city
            all_tile_ids = []
            for _, row in gdf.iterrows():
                tile_id = row.get('tile_id')
                if tile_id is not None and not pd.isna(tile_id):
                    # Generate consistent ID if needed
                    if isinstance(tile_id, str):
                        unique_id = abs(hash(str(tile_id))) % (2**31 - 1)
                    else:
                        unique_id = int(tile_id)
                    all_tile_ids.append(unique_id)
            
            # Get representative tile ID
            rep_tile_id = representative_row.get('tile_id')
            if rep_tile_id is not None and not pd.isna(rep_tile_id):
                if isinstance(rep_tile_id, str):
                    representative_tile_id = abs(hash(str(rep_tile_id))) % (2**31 - 1)
                else:
                    representative_tile_id = int(rep_tile_id)
            else:
                representative_tile_id = all_tile_ids[0] if all_tile_ids else None
            
            # Get continent info
            continent = representative_row.get('continent', 'Unknown')
            if pd.isna(continent) or str(continent).lower() in ['nan', 'none', '']:
                continent = 'Unknown'
            
            # Get date range
            dates = []
            for _, row in gdf.iterrows():
                date_val = row.get('acquisition_date')
                if date_val is not None and not pd.isna(date_val):
                    dates.append(str(date_val))
            
            date_range = {
                'earliest': min(dates) if dates else None,
                'latest': max(dates) if dates else None
            }
            
            # Create city representative
            city_representative = {
                'city_key': f"{city_name}, {country_name}",
                'city': city_name,
                'country': country_name,
                'continent': str(continent).strip(),
                'representative_tile_id': representative_tile_id,
                'representative_longitude': float(representative_row.get('centroid_lon', representative_row.get('longitude', 0))),
                'representative_latitude': float(representative_row.get('centroid_lat', representative_row.get('latitude', 0))),
                'centroid_longitude': float(centroid_lon),
                'centroid_latitude': float(centroid_lat),
                'tile_count': len(gdf),
                'all_tile_ids': sorted(all_tile_ids),
                'bbox': {
                    'min_longitude': float(min_lon),
                    'max_longitude': float(max_lon),
                    'min_latitude': float(min_lat),
                    'max_latitude': float(max_lat)
                },
                'date_range': date_range
            }
            
            return city_representative
            
        except Exception as e:
            logger.error(f"Error processing city {city_name}, {country_name}: {e}")
            return None
    
    def generate_city_representatives(self) -> Dict:
        """Generate city representatives from all embedding files."""
        logger.info("🏙️ Generating city representatives...")
        
        # Find all parquet files
        pattern = os.path.join(EMBEDDINGS_DIR, "full_patch_embeddings", "*", "*.gpq")
        files = glob(pattern)
        
        if not files:
            raise ValueError(f"No .gpq files found in {pattern}")
        
        logger.info(f"📄 Found {len(files)} files to process")
        
        city_representatives = []
        processed_cities = set()
        total_tiles = 0
        
        for file_path in tqdm(files, desc="Processing cities"):
            try:
                # Extract city info from filename
                country_dir = os.path.basename(os.path.dirname(file_path))
                city_file = os.path.basename(file_path).replace('.gpq', '')
                
                # Load the GeoParquet file
                gdf = gpd.read_parquet(file_path)
                
                if len(gdf) == 0:
                    logger.warning(f"⚠️ Empty file: {file_path}")
                    continue
                
                # Get city info from first row
                first_row = gdf.iloc[0]
                city_name = first_row.get('city', '').strip()
                country_name = first_row.get('country', '').strip()
                
                if not city_name or not country_name:
                    logger.warning(f"⚠️ Missing city/country info in {file_path}")
                    continue
                
                city_key = f"{city_name}, {country_name}"
                
                if city_key in processed_cities:
                    logger.warning(f"⚠️ Duplicate city: {city_key}")
                    continue
                
                # Process city data
                city_representative = self.process_city_data(gdf, city_name, country_name)
                
                if city_representative:
                    city_representatives.append(city_representative)
                    processed_cities.add(city_key)
                    total_tiles += len(gdf)
                    
                    logger.info(f"✅ {city_key}: {len(gdf)} tiles, centroid: ({city_representative['centroid_longitude']:.4f}, {city_representative['centroid_latitude']:.4f})")
                else:
                    logger.warning(f"⚠️ Failed to process {city_key}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                continue
        
        # Sort by city name for consistency
        city_representatives.sort(key=lambda x: (x['country'], x['city']))
        
        # Create summary
        summary = {
            'city_representatives': city_representatives,
            'total_cities': len(city_representatives),
            'total_tiles': total_tiles,
            'source': 'generated_from_embeddings',
            'generation_timestamp': time.time(),
            'statistics': {
                'countries': len(set(rep['country'] for rep in city_representatives)),
                'continents': len(set(rep['continent'] for rep in city_representatives)),
                'avg_tiles_per_city': total_tiles / len(city_representatives) if city_representatives else 0,
                'tiles_range': {
                    'min': min(rep['tile_count'] for rep in city_representatives) if city_representatives else 0,
                    'max': max(rep['tile_count'] for rep in city_representatives) if city_representatives else 0
                }
            }
        }
        
        logger.info(f"📊 City Representatives Summary:")
        logger.info(f"   Cities: {summary['total_cities']}")
        logger.info(f"   Countries: {summary['statistics']['countries']}")
        logger.info(f"   Continents: {summary['statistics']['continents']}")
        logger.info(f"   Total tiles: {summary['total_tiles']}")
        logger.info(f"   Avg tiles per city: {summary['statistics']['avg_tiles_per_city']:.1f}")
        logger.info(f"   Tiles range: {summary['statistics']['tiles_range']['min']}-{summary['statistics']['tiles_range']['max']}")
        
        return summary
    
    def save_city_representatives(self, city_data: Dict):
        """Save city representatives to JSON file."""
        output_file = os.path.join(OUTPUT_DIR, 'city_representatives.json')
        
        with open(output_file, 'w') as f:
            json.dump(city_data, f, indent=2)
        
        logger.info(f"💾 Saved city representatives to {output_file}")
        
        # Also save a simplified version for quick loading
        simplified_data = {
            'cities': [
                {
                    'city': rep['city'],
                    'country': rep['country'],
                    'continent': rep['continent'],
                    'longitude': rep['centroid_longitude'],
                    'latitude': rep['centroid_latitude'],
                    'tile_count': rep['tile_count']
                }
                for rep in city_data['city_representatives']
            ],
            'total_cities': city_data['total_cities'],
            'generation_timestamp': city_data['generation_timestamp']
        }
        
        simplified_file = os.path.join(OUTPUT_DIR, 'city_representatives_simple.json')
        with open(simplified_file, 'w') as f:
            json.dump(simplified_data, f, indent=2)
        
        logger.info(f"💾 Saved simplified city representatives to {simplified_file}")
    
    def run(self):
        """Main execution method."""
        start_time = time.time()
        
        logger.info("🏙️ City Representatives Generator Starting...")
        logger.info(f"📁 Input directory: {EMBEDDINGS_DIR}")
        logger.info(f"📁 Output directory: {OUTPUT_DIR}")
        
        try:
            # Generate city representatives
            city_data = self.generate_city_representatives()
            
            # Save to files
            self.save_city_representatives(city_data)
            
            duration = (time.time() - start_time) / 60
            
            logger.info(f"\n🎉 City representatives generation completed!")
            logger.info(f"⏱️ Duration: {duration:.1f} minutes")
            logger.info(f"🏙️ Generated representatives for {city_data['total_cities']} cities")
            logger.info(f"📊 Covering {city_data['total_tiles']} total tiles")
            logger.info(f"💾 Files saved to {OUTPUT_DIR}")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ City representatives generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1


def main():
    """Main function."""
    logger.info("🏙️ City Representatives Generator")
    logger.info("=" * 50)
    
    if not os.path.exists(EMBEDDINGS_DIR):
        logger.error(f"❌ Embeddings directory not found: {EMBEDDINGS_DIR}")
        return 1
    
    generator = CityRepresentativesGenerator()
    
    try:
        return generator.run()
    except KeyboardInterrupt:
        logger.info("⚠️ Generation interrupted by user")
        return 1


if __name__ == "__main__":
    exit(main())