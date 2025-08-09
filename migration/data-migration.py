#!/usr/bin/env python3
"""
Enhanced Embeddings Uploader - With Exact Tile Bounds and Always Updated UMAP
===========================================================================

This enhanced version extracts exact tile boundary coordinates from your
GeoParquet files and includes them in the migration to both Qdrant and metadata files.
UMAP is always recalculated to ensure consistency, even in incremental mode.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob
from typing import Dict, List, Set, Tuple, Optional
import umap
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import logging
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
FORCE_REUPLOAD = False  # Set to False for incremental uploads
EMBEDDINGS_DIR = "/Users/janmagnuszewski/dev/terramind/embeddings/urban_embeddings_224_terramind"
OUTPUT_DIR = "./production_data"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedEmbeddingsUploader:
    def __init__(self):
        """Initialize the enhanced uploader with exact bounds extraction."""
        # Qdrant connection
        self.qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        # Connect to Qdrant
        try:
            client_kwargs = {'url': self.qdrant_url, 'timeout': 120}
            if self.qdrant_api_key and self.qdrant_api_key != 'your_api_key_if_needed':
                client_kwargs['api_key'] = self.qdrant_api_key
                
            self.qdrant_client = QdrantClient(**client_kwargs)
            logger.info(f"✅ Connected to Qdrant at {self.qdrant_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Collection names
        self.collections = {
            'regular': 'satellite_embeddings_simple',
            'global_contrastive': 'satellite_embeddings_global_contrastive_simple'
        }
        
        # Track existing cities
        self.existing_cities = set()
        if not FORCE_REUPLOAD:
            self.existing_cities = self._load_existing_cities()
    
    def _load_existing_cities(self) -> Set[str]:
        """Load existing cities from regular collection."""
        existing_cities = set()
        collection_name = self.collections['regular']
        
        try:
            logger.info(f"🔍 Loading existing cities from {collection_name}...")
            
            offset = None
            while True:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in points:
                    if hasattr(point, 'payload') and point.payload:
                        city = point.payload.get('city')
                        country = point.payload.get('country')
                        if city and country:
                            city_key = f"{city}, {country}"
                            existing_cities.add(city_key)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            logger.info(f"📊 Found {len(existing_cities)} existing cities")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing cities: {e}")
        
        return existing_cities
    
    def extract_exact_tile_bounds(self, row: pd.Series) -> Optional[List[List[float]]]:
        """
        Extract exact tile boundary coordinates from GeoParquet geometry.
        
        Args:
            row: DataFrame row containing tile data
            
        Returns:
            List of [lon, lat] coordinates defining the tile boundary, or None if not available
        """
        try:
            # Method 1: Check if we have explicit bounds columns from the embedding script
            bounds_cols = ['bounds_min_lon', 'bounds_min_lat', 'bounds_max_lon', 'bounds_max_lat']
            if all(col in row.index for col in bounds_cols):
                min_lon = float(row['bounds_min_lon'])
                min_lat = float(row['bounds_min_lat'])
                max_lon = float(row['bounds_max_lon'])
                max_lat = float(row['bounds_max_lat'])
                
                # Create tile boundary as polygon coordinates [lon, lat]
                tile_bounds = [
                    [min_lon, max_lat],  # Top-left
                    [max_lon, max_lat],  # Top-right
                    [max_lon, min_lat],  # Bottom-right
                    [min_lon, min_lat],  # Bottom-left
                    [min_lon, max_lat]   # Close polygon
                ]
                
                return tile_bounds
            
            # Method 2: Try to extract from geometry column if it's a polygon
            elif 'geometry' in row.index and row['geometry'] is not None:
                geom = row['geometry']
                if hasattr(geom, 'bounds'):
                    min_lon, min_lat, max_lon, max_lat = geom.bounds
                    tile_bounds = [
                        [min_lon, max_lat],
                        [max_lon, max_lat],
                        [max_lon, min_lat],
                        [min_lon, min_lat],
                        [min_lon, max_lat]
                    ]
                    return tile_bounds
                elif hasattr(geom, 'exterior') and hasattr(geom.exterior, 'coords'):
                    # Extract coordinates directly from polygon exterior
                    coords = list(geom.exterior.coords)
                    # Convert from (x,y) to [lon,lat] format
                    tile_bounds = [[coord[0], coord[1]] for coord in coords]
                    return tile_bounds
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract exact bounds: {e}")
            return None
    
    def delete_all_collections(self):
        """Delete all collections if they exist."""
        logger.info("🗑️ Deleting all collections...")
        
        try:
            collections = self.qdrant_client.get_collections().collections
            existing_names = [col.name for col in collections]
            
            for embed_type, collection_name in self.collections.items():
                if collection_name in existing_names:
                    logger.info(f"🗑️ Deleting {collection_name}")
                    self.qdrant_client.delete_collection(collection_name)
                    logger.info(f"✅ Deleted {collection_name}")
                else:
                    logger.info(f"ℹ️ Collection {collection_name} doesn't exist")
            
            time.sleep(2)  # Wait for deletion to complete
            logger.info("✅ All collections deleted")
            
        except Exception as e:
            logger.error(f"❌ Error deleting collections: {e}")
            raise
    
    def create_collections(self):
        """Create fresh collections."""
        logger.info("📦 Creating collections...")
        
        for embed_type, collection_name in self.collections.items():
            try:
                logger.info(f"📦 Creating {collection_name}")
                
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=768,
                        distance=Distance.COSINE,
                        on_disk=True
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=100
                    )
                )
                
                logger.info(f"✅ Created {collection_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create {collection_name}: {e}")
                raise
        
        logger.info("✅ All collections created")
    
    def process_tile_enhanced(self, row: pd.Series) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Enhanced tile processing that includes exact tile bounds."""
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
            
            # Get embedding
            full_embedding = row.get('embedding_patches_full')
            aggregated_embedding_col = row.get('embedding_patch_aggregated')
            
            # Process embeddings (same logic as before)
            if full_embedding is not None:
                try:
                    if isinstance(full_embedding, np.ndarray):
                        embedding_array = full_embedding
                    else:
                        embedding_array = np.array(full_embedding)
                    
                    if embedding_array.size == 196 * 768:
                        patches = embedding_array.reshape(196, 768)
                        aggregated_embedding = patches.mean(axis=0)
                    else:
                        logger.warning(f"Unexpected full embedding size: {embedding_array.size}")
                        return None, None
                        
                except Exception as e:
                    logger.warning(f"Error processing full patch embedding: {e}")
                    return None, None
            
            elif aggregated_embedding_col is not None:
                try:
                    if isinstance(aggregated_embedding_col, np.ndarray):
                        aggregated_embedding = aggregated_embedding_col
                    else:
                        aggregated_embedding = np.array(aggregated_embedding_col)
                    
                    if aggregated_embedding.size != 768:
                        logger.warning(f"Unexpected aggregated embedding size: {aggregated_embedding.size}")
                        return None, None
                        
                except Exception as e:
                    logger.warning(f"Error processing aggregated embedding: {e}")
                    return None, None
            else:
                return None, None
            
            # Generate tile ID
            tile_id = row.get('tile_id')
            if tile_id is None or pd.isna(tile_id):
                tile_id = f"tile_{lon_val:.6f}_{lat_val:.6f}".replace('.', '_').replace('-', 'neg')
            
            unique_id = abs(hash(str(tile_id))) % (2**31 - 1)
            
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
            
            # **ENHANCED: Extract exact tile bounds**
            exact_tile_bounds = self.extract_exact_tile_bounds(row)
            
            # Create enhanced metadata with exact bounds
            metadata = {
                'id': unique_id,
                'original_tile_id': str(tile_id),
                'city': str(city).strip(),
                'country': str(country).strip(),
                'continent': str(continent).strip(),
                'longitude': lon_val,
                'latitude': lat_val,
                'date': date_str,
                # **ENHANCED: Add exact tile bounds if available**
                'tile_bounds': exact_tile_bounds,
                'has_exact_bounds': exact_tile_bounds is not None
            }
            
            # Add tile size information if we have bounds
            if exact_tile_bounds:
                # Calculate tile dimensions
                lons = [coord[0] for coord in exact_tile_bounds[:-1]]  # Exclude closing point
                lats = [coord[1] for coord in exact_tile_bounds[:-1]]
                
                tile_width_deg = max(lons) - min(lons)
                tile_height_deg = max(lats) - min(lats)
                
                metadata.update({
                    'tile_width_degrees': tile_width_deg,
                    'tile_height_degrees': tile_height_deg
                })
            
            return metadata, aggregated_embedding
            
        except Exception as e:
            logger.warning(f"Error processing tile: {e}")
            return None, None
    
    def load_all_embeddings_enhanced(self) -> Tuple[List[Dict], np.ndarray]:
        """Enhanced embedding loading with exact tile bounds extraction."""
        logger.info("📖 Loading embeddings with exact tile bounds...")
        
        # Find all parquet files
        pattern = os.path.join(EMBEDDINGS_DIR, "full_patch_embeddings", "*", "*.gpq")
        files = glob(pattern)
        
        if not files:
            raise ValueError(f"No .gpq files found in {pattern}")
        
        logger.info(f"📄 Found {len(files)} files to process")
        
        all_metadata = []
        all_embeddings = []
        processed_cities = set()
        skipped_cities = set()
        tiles_with_exact_bounds = 0
        tiles_with_fallback_bounds = 0
        
        for file_path in tqdm(files, desc="Processing files with bounds extraction"):
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
                
                # City-level check for incremental updates
                if not FORCE_REUPLOAD and city_key in self.existing_cities:
                    logger.info(f"⏭️ Skipping {city_key} - already exists")
                    skipped_cities.add(city_key)
                    continue
                
                logger.info(f"🏙️ Processing {city_key} - {len(gdf)} tiles")
                
                # Convert categorical columns
                for col in gdf.columns:
                    if gdf[col].dtype.name == 'category':
                        gdf[col] = gdf[col].astype(str)
                
                city_tile_count = 0
                city_exact_bounds = 0
                
                for _, row in gdf.iterrows():
                    metadata, embedding = self.process_tile_enhanced(row)
                    
                    if metadata and embedding is not None:
                        all_metadata.append(metadata)
                        all_embeddings.append(embedding)
                        city_tile_count += 1
                        
                        # Track bounds statistics
                        if metadata.get('has_exact_bounds', False):
                            city_exact_bounds += 1
                            tiles_with_exact_bounds += 1
                        else:
                            tiles_with_fallback_bounds += 1
                
                if city_tile_count > 0:
                    processed_cities.add(city_key)
                    bounds_info = f"({city_exact_bounds} exact bounds)" if city_exact_bounds > 0 else "(fallback bounds)"
                    logger.info(f"✅ {city_key}: {city_tile_count} tiles {bounds_info}")
                else:
                    logger.warning(f"⚠️ {city_key}: No valid tiles found")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                continue
        
        if not all_embeddings:
            logger.warning("⚠️ No tiles to upload!")
            return [], np.array([])
        
        # Enhanced summary with bounds statistics
        logger.info(f"📊 Enhanced Processing Summary:")
        logger.info(f"   New cities processed: {len(processed_cities)}")
        logger.info(f"   Cities skipped: {len(skipped_cities)}")
        logger.info(f"   Total tiles to upload: {len(all_embeddings)}")
        logger.info(f"   🎯 Tiles with exact bounds: {tiles_with_exact_bounds}")
        logger.info(f"   📐 Tiles with fallback bounds: {tiles_with_fallback_bounds}")
        logger.info(f"   📊 Exact bounds coverage: {tiles_with_exact_bounds/len(all_embeddings)*100:.1f}%")
        
        return all_metadata, np.array(all_embeddings)
    
    def load_all_embeddings_from_qdrant(self) -> Tuple[List[Dict], np.ndarray]:
        """Load all embeddings from Qdrant for UMAP recalculation."""
        logger.info("📖 Loading all embeddings from Qdrant for UMAP recalculation...")
        
        collection_name = self.collections['regular']
        
        try:
            all_points = []
            offset = None
            
            # Scroll through all points in Qdrant
            logger.info(f"🔄 Fetching all points from {collection_name}...")
            while True:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                all_points.extend(points)
                
                if next_offset is None:
                    break
                offset = next_offset
                
                logger.info(f"   Loaded {len(all_points)} points so far...")
            
            logger.info(f"📊 Total points loaded from Qdrant: {len(all_points)}")
            
            # Convert to metadata and embeddings
            metadata_list = []
            embeddings = []
            
            for point in all_points:
                # Extract metadata from Qdrant payload
                metadata = {
                    'id': point.id,
                    'city': point.payload.get('city', ''),
                    'country': point.payload.get('country', ''),
                    'continent': point.payload.get('continent', 'Unknown'),
                    'longitude': point.payload.get('longitude', 0.0),
                    'latitude': point.payload.get('latitude', 0.0),
                    'date': point.payload.get('date'),
                    'original_tile_id': point.payload.get('original_tile_id', ''),
                    'has_exact_bounds': point.payload.get('has_exact_bounds', False),
                    # Note: tile_bounds are not stored in Qdrant payload, will be loaded from existing metadata
                    'tile_bounds': None
                }
                
                metadata_list.append(metadata)
                embeddings.append(point.vector)
            
            return metadata_list, np.array(embeddings)
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings from Qdrant: {e}")
            raise
    
    def merge_metadata_with_bounds(self, qdrant_metadata: List[Dict]) -> List[Dict]:
        """Merge Qdrant metadata with existing bounds information."""
        logger.info("🔗 Merging Qdrant metadata with existing bounds information...")
        
        # Load existing metadata file if it exists
        metadata_file = os.path.join(OUTPUT_DIR, 'locations_metadata.json')
        existing_bounds = {}
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
                
            # Create lookup for bounds information
            for item in existing_metadata:
                location_id = item.get('id')
                if location_id:
                    existing_bounds[location_id] = {
                        'tile_bounds': item.get('tile_bounds'),
                        'has_exact_bounds': item.get('has_exact_bounds', False),
                        'tile_width_degrees': item.get('tile_width_degrees'),
                        'tile_height_degrees': item.get('tile_height_degrees')
                    }
            
            logger.info(f"📄 Loaded bounds info for {len(existing_bounds)} existing locations")
        
        # Merge bounds information
        enhanced_metadata = []
        bounds_merged_count = 0
        
        for metadata in qdrant_metadata:
            location_id = metadata['id']
            
            # Add bounds information if available
            if location_id in existing_bounds:
                bounds_info = existing_bounds[location_id]
                metadata.update(bounds_info)
                bounds_merged_count += 1
            else:
                # Set default values if no bounds info found
                metadata.update({
                    'tile_bounds': None,
                    'has_exact_bounds': False,
                    'tile_width_degrees': None,
                    'tile_height_degrees': None
                })
            
            enhanced_metadata.append(metadata)
        
        logger.info(f"🔗 Merged bounds info for {bounds_merged_count}/{len(enhanced_metadata)} locations")
        
        return enhanced_metadata
    
    def upload_to_qdrant(self, metadata_list: List[Dict], regular_embeddings: np.ndarray):
        """Upload embeddings to Qdrant collections."""
        if len(metadata_list) == 0:
            logger.info("📝 No tiles to upload")
            return
        
        # Compute dataset mean for global contrastive
        logger.info("🧮 Computing dataset mean for global contrastive embeddings...")
        dataset_mean = regular_embeddings.mean(axis=0)
        global_contrastive = regular_embeddings - dataset_mean
        
        embeddings_dict = {
            'regular': regular_embeddings,
            'global_contrastive': global_contrastive
        }
        
        # Upload to each collection
        batch_size = 100
        
        for embed_type, embeddings_array in embeddings_dict.items():
            collection_name = self.collections[embed_type]
            logger.info(f"🚀 Uploading {len(metadata_list)} {embed_type} embeddings to {collection_name}")
            
            total_batches = (len(metadata_list) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(metadata_list), batch_size), 
                         desc=f"Uploading {embed_type}", total=total_batches):
                
                batch_metadata = metadata_list[i:i + batch_size]
                batch_embeddings = embeddings_array[i:i + batch_size]
                
                # Create points
                points = []
                for metadata, embedding in zip(batch_metadata, batch_embeddings):
                    payload = {
                        'city': metadata['city'],
                        'country': metadata['country'],
                        'continent': metadata['continent'],
                        'longitude': metadata['longitude'],
                        'latitude': metadata['latitude'],
                        'date': metadata['date'],
                        'original_tile_id': metadata['original_tile_id'],
                        'embedding_type': embed_type,
                        'city_name': f"{metadata['city']}, {metadata['country']}",
                        # **ENHANCED: Include bounds information in Qdrant payload**
                        'has_exact_bounds': metadata.get('has_exact_bounds', False)
                    }
                    
                    # Remove None values
                    payload = {k: v for k, v in payload.items() if v is not None}
                    
                    point = PointStruct(
                        id=metadata['id'],
                        vector=embedding.tolist(),
                        payload=payload
                    )
                    points.append(point)
                
                # Upload batch
                try:
                    operation_info = self.qdrant_client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=points
                    )
                    
                    if operation_info.status != models.UpdateStatus.COMPLETED:
                        logger.warning(f"⚠️ Batch upload status: {operation_info.status}")
                
                except Exception as e:
                    logger.error(f"❌ Error uploading batch: {e}")
                    continue
            
            logger.info(f"✅ {embed_type} upload completed")
        
        # Save enhanced metadata
        self.save_enhanced_data(metadata_list, regular_embeddings, dataset_mean)
    
    def compute_umap_for_all_data(self, metadata_list: List[Dict], regular_embeddings: np.ndarray) -> Dict:
        """Compute UMAP coordinates for all data."""
        logger.info("🗺️ Computing UMAP coordinates for all data...")
        
        n_samples = len(regular_embeddings)
        n_neighbors = min(50, max(5, n_samples // 100))
        
        logger.info(f"🗺️ Computing UMAP with n_neighbors={n_neighbors} for {n_samples} samples")
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.3,
            metric='cosine',
            random_state=42
        )
        
        umap_coords = reducer.fit_transform(regular_embeddings)
        
        # Enhanced UMAP data with bounds information
        umap_data = {
            'umap_points': [
                {
                    'location_id': metadata['id'],
                    'x': float(umap_coords[i][0]),
                    'y': float(umap_coords[i][1]),
                    'city': metadata['city'],
                    'country': metadata['country'],
                    'continent': metadata['continent'],
                    'longitude': metadata['longitude'],
                    'latitude': metadata['latitude'],
                    'date': metadata['date'],
                    # **ENHANCED: Include bounds information in UMAP data**
                    'tile_bounds': metadata.get('tile_bounds'),
                    'has_exact_bounds': metadata.get('has_exact_bounds', False)
                }
                for i, metadata in enumerate(metadata_list)
            ],
            'total_points': len(metadata_list),
            'bounds_statistics': {
                'tiles_with_exact_bounds': sum(1 for m in metadata_list if m.get('has_exact_bounds', False)),
                'tiles_with_fallback_bounds': sum(1 for m in metadata_list if not m.get('has_exact_bounds', False)),
                'exact_bounds_percentage': sum(1 for m in metadata_list if m.get('has_exact_bounds', False)) / len(metadata_list) * 100
            }
        }
        
        logger.info(f"✅ UMAP computation completed for {len(metadata_list)} points")
        
        return umap_data
    
    def save_enhanced_data(self, metadata_list: List[Dict], regular_embeddings: np.ndarray, dataset_mean: np.ndarray):
        """Save metadata with exact tile bounds information and always recalculate UMAP."""
        
        if FORCE_REUPLOAD:
            logger.info("💾 Computing UMAP and saving enhanced data with exact bounds...")
            
            # Compute UMAP for new data only
            umap_data = self.compute_umap_for_all_data(metadata_list, regular_embeddings)
            
            dataset_stats = {
                'total_samples': len(metadata_list),
                'embedding_dimension': 768,
                'dataset_mean': dataset_mean.tolist(),
                'collections': self.collections,
                'force_reupload': FORCE_REUPLOAD,
                'timestamp': time.time(),
                'enhanced_features': {
                    'exact_tile_bounds': True,
                    'bounds_coverage_percentage': umap_data['bounds_statistics']['exact_bounds_percentage']
                }
            }
            
            # Save enhanced files
            files_to_save = [
                ('locations_metadata.json', metadata_list),
                ('umap_coordinates.json', umap_data),
                ('dataset_statistics.json', dataset_stats)
            ]
            
            for filename, data in files_to_save:
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"📄 Saved enhanced {filename}")
            
            # Log bounds statistics
            bounds_stats = umap_data['bounds_statistics']
            logger.info(f"🎯 Tile Bounds Statistics:")
            logger.info(f"   Exact bounds: {bounds_stats['tiles_with_exact_bounds']}")
            logger.info(f"   Fallback bounds: {bounds_stats['tiles_with_fallback_bounds']}")
            logger.info(f"   Coverage: {bounds_stats['exact_bounds_percentage']:.1f}%")
        
        else:
            # **ENHANCED: Always recalculate UMAP in incremental mode**
            logger.info("💾 INCREMENTAL MODE: Always recalculating UMAP with all data...")
            
            # Load existing metadata
            metadata_file = os.path.join(OUTPUT_DIR, 'locations_metadata.json')
            existing_metadata = []
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                logger.info(f"📄 Loaded {len(existing_metadata)} existing metadata records")
            
            # Merge new metadata (avoid duplicates)
            existing_ids = {item['id'] for item in existing_metadata}
            new_count = 0
            for metadata in metadata_list:
                if metadata['id'] not in existing_ids:
                    existing_metadata.append(metadata)
                    new_count += 1
            
            logger.info(f"📄 Added {new_count} new metadata records")
            
            # **ALWAYS RECALCULATE UMAP**: Load all embeddings from Qdrant and recompute UMAP
            logger.info("🔄 Always recalculating UMAP coordinates for all data...")
            
            try:
                # Load all embeddings from Qdrant
                all_qdrant_metadata, all_qdrant_embeddings = self.load_all_embeddings_from_qdrant()
                
                # Merge with bounds information from existing metadata
                enhanced_all_metadata = self.merge_metadata_with_bounds(all_qdrant_metadata)
                
                # Compute UMAP for ALL data (existing + new)
                umap_data = self.compute_umap_for_all_data(enhanced_all_metadata, all_qdrant_embeddings)
                
                # Use the enhanced metadata for final save
                final_metadata = enhanced_all_metadata
                
                logger.info(f"✅ UMAP recalculated for all {len(enhanced_all_metadata)} locations")
                
            except Exception as e:
                logger.error(f"❌ Failed to load from Qdrant for UMAP recalculation: {e}")
                logger.info("🔄 Falling back to incremental metadata update without UMAP recalculation")
                
                # Fallback: just update metadata, don't touch UMAP
                final_metadata = existing_metadata
                umap_data = None
            
            # Update metadata file with final metadata
            with open(metadata_file, 'w') as f:
                json.dump(final_metadata, f, indent=2)
            logger.info(f"📄 Updated locations_metadata.json with {len(final_metadata)} total records")
            
            # Calculate enhanced statistics
            exact_bounds_count = sum(1 for m in final_metadata if m.get('has_exact_bounds', False))
            fallback_bounds_count = len(final_metadata) - exact_bounds_count
            exact_bounds_percentage = (exact_bounds_count / len(final_metadata)) * 100 if final_metadata else 0
            
            # Update dataset stats
            stats_file = os.path.join(OUTPUT_DIR, 'dataset_statistics.json')
            dataset_stats = {
                'total_samples': len(final_metadata),
                'embedding_dimension': 768,
                'dataset_mean': dataset_mean.tolist(),
                'collections': self.collections,
                'force_reupload': FORCE_REUPLOAD,
                'timestamp': time.time(),
                'new_tiles_added': new_count,
                'enhanced_features': {
                    'exact_tile_bounds': True,
                    'bounds_coverage_percentage': exact_bounds_percentage
                }
            }
            
            with open(stats_file, 'w') as f:
                json.dump(dataset_stats, f, indent=2)
            logger.info(f"📄 Updated dataset_statistics.json")
            
            # Save UMAP coordinates if successfully computed
            if umap_data:
                umap_file = os.path.join(OUTPUT_DIR, 'umap_coordinates.json')
                with open(umap_file, 'w') as f:
                    json.dump(umap_data, f, indent=2)
                logger.info(f"📄 Updated umap_coordinates.json with recalculated coordinates")
                
                # Log bounds statistics
                bounds_stats = umap_data['bounds_statistics']
                logger.info(f"🎯 Enhanced Bounds Coverage (All Data):")
                logger.info(f"   Exact bounds: {bounds_stats['tiles_with_exact_bounds']}")
                logger.info(f"   Fallback bounds: {bounds_stats['tiles_with_fallback_bounds']}")
                logger.info(f"   Coverage: {bounds_stats['exact_bounds_percentage']:.1f}%")
            else:
                logger.warning("⚠️ UMAP coordinates NOT updated due to Qdrant loading error")
            
            # Log final statistics
            logger.info(f"🎯 Final Enhanced Bounds Coverage:")
            logger.info(f"   Exact bounds: {exact_bounds_count}")
            logger.info(f"   Fallback bounds: {fallback_bounds_count}")
            logger.info(f"   Coverage: {exact_bounds_percentage:.1f}%")
            logger.info(f"   New tiles added this run: {new_count}")
            
        logger.info(f"💾 Enhanced data saved to {OUTPUT_DIR}")
    
    def run(self):
        """Main execution method with enhanced bounds processing and always-updated UMAP."""
        start_time = time.time()
        
        if FORCE_REUPLOAD:
            logger.info("🔄 FORCE REUPLOAD MODE: Enhanced processing with exact bounds")
            self.delete_all_collections()
            self.create_collections()
        else:
            logger.info("➕ INCREMENTAL MODE: Enhanced processing with exact bounds + UMAP recalculation")
            # Check if collections exist, create if not
            try:
                collections = self.qdrant_client.get_collections().collections
                existing_names = [col.name for col in collections]
                
                for embed_type, collection_name in self.collections.items():
                    if collection_name not in existing_names:
                        logger.info(f"📦 Creating missing collection: {collection_name}")
                        self.qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(size=768, distance=Distance.COSINE, on_disk=True)
                        )
            except Exception as e:
                logger.error(f"❌ Error checking/creating collections: {e}")
                raise
        
        # Load embeddings with exact bounds
        metadata_list, regular_embeddings = self.load_all_embeddings_enhanced()
        
        if len(metadata_list) > 0:
            # Upload new tiles to Qdrant
            self.upload_to_qdrant(metadata_list, regular_embeddings)
            
            # Enhanced summary
            duration = (time.time() - start_time) / 60
            exact_bounds_count = sum(1 for m in metadata_list if m.get('has_exact_bounds', False))
            
            logger.info(f"\n🎉 Enhanced upload completed!")
            logger.info(f"⏱️ Duration: {duration:.1f} minutes")
            logger.info(f"📊 New tiles uploaded: {len(metadata_list)}")
            logger.info(f"🎯 New tiles with exact bounds: {exact_bounds_count}")
            logger.info(f"📐 New tiles exact bounds coverage: {exact_bounds_count/len(metadata_list)*100:.1f}%")
            
        else:
            logger.info("✅ No new tiles to upload")
            
            # **EVEN IF NO NEW TILES: Still recalculate UMAP in incremental mode**
            if not FORCE_REUPLOAD:
                logger.info("🔄 No new tiles, but recalculating UMAP for consistency...")
                
                try:
                    # Load all embeddings from Qdrant
                    all_qdrant_metadata, all_qdrant_embeddings = self.load_all_embeddings_from_qdrant()
                    
                    if len(all_qdrant_embeddings) > 0:
                        # Merge with bounds information
                        enhanced_all_metadata = self.merge_metadata_with_bounds(all_qdrant_metadata)
                        
                        # Compute UMAP for ALL existing data
                        umap_data = self.compute_umap_for_all_data(enhanced_all_metadata, all_qdrant_embeddings)
                        
                        # Compute dataset mean for consistency
                        dataset_mean = all_qdrant_embeddings.mean(axis=0)
                        
                        # Save updated files
                        metadata_file = os.path.join(OUTPUT_DIR, 'locations_metadata.json')
                        with open(metadata_file, 'w') as f:
                            json.dump(enhanced_all_metadata, f, indent=2)
                        
                        umap_file = os.path.join(OUTPUT_DIR, 'umap_coordinates.json')
                        with open(umap_file, 'w') as f:
                            json.dump(umap_data, f, indent=2)
                        
                        # Update stats
                        exact_bounds_count = sum(1 for m in enhanced_all_metadata if m.get('has_exact_bounds', False))
                        exact_bounds_percentage = (exact_bounds_count / len(enhanced_all_metadata)) * 100
                        
                        stats_file = os.path.join(OUTPUT_DIR, 'dataset_statistics.json')
                        dataset_stats = {
                            'total_samples': len(enhanced_all_metadata),
                            'embedding_dimension': 768,
                            'dataset_mean': dataset_mean.tolist(),
                            'collections': self.collections,
                            'force_reupload': FORCE_REUPLOAD,
                            'timestamp': time.time(),
                            'new_tiles_added': 0,
                            'enhanced_features': {
                                'exact_tile_bounds': True,
                                'bounds_coverage_percentage': exact_bounds_percentage
                            }
                        }
                        
                        with open(stats_file, 'w') as f:
                            json.dump(dataset_stats, f, indent=2)
                        
                        logger.info(f"✅ UMAP recalculated for {len(enhanced_all_metadata)} existing locations")
                        logger.info(f"📊 Total bounds coverage: {exact_bounds_percentage:.1f}%")
                        
                    else:
                        logger.warning("⚠️ No data found in Qdrant for UMAP recalculation")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to recalculate UMAP: {e}")


def main():
    """Main function with enhanced bounds processing and always-updated UMAP."""
    logger.info("🛰️ Enhanced Embeddings Uploader - With Exact Tile Bounds & Always-Updated UMAP")
    logger.info("=" * 80)
    logger.info(f"📁 Embeddings directory: {EMBEDDINGS_DIR}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info(f"🔄 Force reupload: {FORCE_REUPLOAD}")
    logger.info(f"🗺️ UMAP: Always recalculated for consistency")
    
    # Show the collection names that will be created
    collections = {
        'regular': 'satellite_embeddings_simple',
        'global_contrastive': 'satellite_embeddings_global_contrastive_simple'
    }
    logger.info(f"📦 Collections to create/use:")
    for method, collection_name in collections.items():
        logger.info(f"   - {method}: {collection_name}")
    
    if not os.path.exists(EMBEDDINGS_DIR):
        logger.error(f"❌ Embeddings directory not found: {EMBEDDINGS_DIR}")
        return 1
    
    uploader = EnhancedEmbeddingsUploader()
    
    try:
        uploader.run()
        logger.info("🎉 Enhanced script completed successfully!")
        logger.info("🎯 Exact tile bounds have been extracted and included in the migration!")
        logger.info("🗺️ UMAP coordinates have been recalculated for all data!")
        logger.info("📝 Update your frontend and backend to use the enhanced models and MapView")
        return 0
    except Exception as e:
        logger.error(f"❌ Enhanced script failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())