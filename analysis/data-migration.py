#!/usr/bin/env python3
"""
Simplified Memory-Efficient Embeddings Uploader - Using Pre-computed Dimension Analysis
=====================================================================================

This script reads dimension analysis results from JSON file and applies the filtering
during embeddings processing and upload. Much faster since analysis is already done.

Usage:
    1. First run: python standalone_dimension_analysis.py
    2. Then run: python simplified_migration.py

Configuration:
    - Reads excluded dimensions from: ./dimension_analysis_results.json
    - Uploads filtered embeddings with all 6 aggregation methods
    - FORCE_REUPLOAD: Controls whether to delete/recreate all collections or skip existing ones
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
import gzip
import gc
import psutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load environment variables
load_dotenv()

# Configuration
FORCE_REUPLOAD = False  # Set to True to perform full reupload with dimension filtering
EMBEDDINGS_DIR = "../../../terramind/embeddings/urban_embeddings_224_terramind_normalised"
OUTPUT_DIR = "./production_data"

# Dimension analysis results file
DIMENSION_ANALYSIS_FILE = "./outputs/spatial_correlation_results.json"

# Memory management settings
CITY_BATCH_SIZE = 10  # Process 10 cities at a time
UPLOAD_BATCH_SIZE = 200  # Upload 100 tiles at a time
MAX_MEMORY_GB = 12  # Maximum memory usage before forcing cleanup

# Clustering parameters for dominant cluster method
USE_ELBOW_METHOD = True  # Set to True to use elbow method for optimal k
MAX_CLUSTERS_TEST = 4  # Maximum number of clusters to test in elbow method
FALLBACK_CLUSTERS = 3  # Fallback if elbow method fails

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING) 
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def force_garbage_collection():
    """Force garbage collection and log memory usage."""
    gc.collect()
    memory_gb = get_memory_usage()
    logger.info(f"🧹 Memory after cleanup: {memory_gb:.2f} GB")
    return memory_gb

def load_dimension_analysis_results(filepath: str) -> Optional[Dict]:
    """Load dimension analysis results from JSON file."""
    if not os.path.exists(filepath):
        logger.error(f"❌ Dimension analysis file not found: {filepath}")
        logger.info(f"💡 Please run: python standalone_dimension_analysis.py first")
        return None
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        excluded_dims = results['cutoff_results']['excluded_dimensions']
        
        logger.info(f"📊 Loaded dimension analysis results:")
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
        logger.error(f"❌ Error loading dimension analysis results: {e}")
        return None

class SimplifiedEmbeddingsUploader:
    def __init__(self, excluded_dimensions: List[int], force_reupload: bool = False):
        """Initialize the simplified uploader with pre-computed excluded dimensions."""
        self.excluded_dimensions = set(excluded_dimensions)
        self.filtered_dimension_count = 768 - len(excluded_dimensions)  # Assuming 768 original dimensions
        self.force_reupload = force_reupload
        
        # Initialize Qdrant client
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Collection names
        self.collections = {
            'mean': 'terramind_embeddings_mean',
            'median': 'terramind_embeddings_median',
            'max': 'terramind_embeddings_max',
            'global_contrastive': 'terramind_embeddings_global_contrastive',
            'dominant_cluster': 'terramind_embeddings_dominant_cluster'
        }
        
        # For global contrastive method
        self.dataset_mean_accumulator = None
        self.total_tiles_processed = 0
        
        logger.info(f"🔧 Initialized uploader:")
        logger.info(f"   Force reupload: {self.force_reupload}")
        logger.info(f"   Filtered dimensions: {self.filtered_dimension_count}")
        logger.info(f"   Excluded dimensions: {len(self.excluded_dimensions)}")

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

    def check_existing_collections(self) -> Dict[str, bool]:
        """Check which collections already exist."""
        try:
            collections = self.qdrant_client.get_collections().collections
            existing_names = [col.name for col in collections]
            
            collection_status = {}
            for embed_type, collection_name in self.collections.items():
                exists = collection_name in existing_names
                collection_status[embed_type] = exists
                
            return collection_status
            
        except Exception as e:
            logger.error(f"❌ Error checking existing collections: {e}")
            return {key: False for key in self.collections.keys()}

    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get information about an existing collection."""
        try:
            info = self.qdrant_client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'vector_size': info.config.params.vectors.size
            }
        except Exception as e:
            logger.warning(f"⚠️ Could not get info for collection {collection_name}: {e}")
            return None

    def delete_collection_if_exists(self, collection_name: str):
        """Delete a specific collection if it exists."""
        try:
            collections = self.qdrant_client.get_collections().collections
            existing_names = [col.name for col in collections]
            
            if collection_name in existing_names:
                logger.info(f"🗑️ Deleting existing collection: {collection_name}")
                self.qdrant_client.delete_collection(collection_name)
                logger.info(f"✅ Deleted {collection_name}")
                time.sleep(1)  # Brief pause
            else:
                logger.info(f"ℹ️ Collection {collection_name} does not exist, skipping deletion")
                
        except Exception as e:
            logger.error(f"❌ Error deleting collection {collection_name}: {e}")
            raise

    def delete_all_collections(self):
        """Delete all collections if they exist."""
        logger.info("🗑️ Deleting all filtered collections...")
        
        try:
            collections = self.qdrant_client.get_collections().collections
            existing_names = [col.name for col in collections]
            
            for embed_type, collection_name in self.collections.items():
                if collection_name in existing_names:
                    logger.info(f"🗑️ Deleting {collection_name}")
                    self.qdrant_client.delete_collection(collection_name)
                    logger.info(f"✅ Deleted {collection_name}")
            
            time.sleep(2)
            logger.info("✅ All filtered collections deleted")
            
        except Exception as e:
            logger.error(f"❌ Error deleting collections: {e}")
            raise

    def create_collection(self, embed_type: str, collection_name: str):
        """Create a single collection."""
        try:
            logger.info(f"📦 Creating {collection_name} (vector size: {self.filtered_dimension_count})")
            
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.filtered_dimension_count,
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

    def create_collections(self):
        """Create fresh collections with filtered vector size."""
        logger.info("📦 Creating filtered collections...")
        
        for embed_type, collection_name in self.collections.items():
            self.create_collection(embed_type, collection_name)
        
        logger.info("✅ All filtered collections created")

    def manage_collections(self):
        """Manage collections based on force_reupload setting."""
        logger.info("🔄 PHASE 1: Collection Management")
        
        if self.force_reupload:
            logger.info("🔥 Force reupload enabled - deleting and recreating ALL collections")
            self.delete_all_collections()
            self.create_collections()
        else:
            logger.info("🔍 Checking existing collections...")
            existing_collections = self.check_existing_collections()
            
            collections_to_create = []
            collections_to_skip = []
            
            for embed_type, collection_name in self.collections.items():
                exists = existing_collections.get(embed_type, False)
                
                if exists:
                    # Get collection info
                    info = self.get_collection_info(collection_name)
                    if info:
                        logger.info(f"ℹ️ Collection {collection_name} exists with {info['points_count']} points, vector size {info['vector_size']}")
                        
                        # Check if vector size matches our filtered dimension count
                        if info['vector_size'] != self.filtered_dimension_count:
                            logger.warning(f"⚠️ Vector size mismatch for {collection_name}: {info['vector_size']} vs expected {self.filtered_dimension_count}")
                            logger.info(f"🔄 Will recreate {collection_name} with correct vector size")
                            self.delete_collection_if_exists(collection_name)
                            collections_to_create.append((embed_type, collection_name))
                        else:
                            collections_to_skip.append((embed_type, collection_name))
                    else:
                        collections_to_create.append((embed_type, collection_name))
                else:
                    collections_to_create.append((embed_type, collection_name))
            
            # Create missing collections
            if collections_to_create:
                logger.info(f"📦 Creating {len(collections_to_create)} missing collections...")
                for embed_type, collection_name in collections_to_create:
                    self.create_collection(embed_type, collection_name)
                logger.info("✅ Missing collections created")
            
            # Log skipped collections
            if collections_to_skip:
                logger.info(f"⏭️ Skipping {len(collections_to_skip)} existing collections:")
                for embed_type, collection_name in collections_to_skip:
                    logger.info(f"   - {embed_type}: {collection_name}")

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

    def find_optimal_clusters_elbow(self, patches_scaled: np.ndarray, max_k: int = 15) -> int:
        """Find optimal number of clusters using elbow method."""
        try:
            n_samples = len(patches_scaled)
            max_k = min(max_k, n_samples // 3, 15)
            
            if max_k < 2:
                return 2
            
            k_range = range(2, max_k + 1)
            inertias = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=50)
                kmeans.fit(patches_scaled)
                inertias.append(kmeans.inertia_)
            
            if len(inertias) < 3:
                return 2
            
            differences = np.diff(inertias)
            second_diff = np.diff(differences)
            
            elbow_idx = np.argmax(second_diff) + 2
            optimal_k = k_range[elbow_idx - 2]
            optimal_k = max(2, min(optimal_k, 12))
            
            return optimal_k
            
        except Exception as e:
            logger.debug(f"Error in elbow method, using default k=5: {e}")
            return 5

    def compute_dominant_cluster_embedding(self, filtered_patches: np.ndarray) -> np.ndarray:
        """Compute dominant cluster embedding using elbow method on filtered patches."""
        try:
            n_patches = len(filtered_patches)
            
            if n_patches != 196:
                logger.warning(f"Expected 196 patches, got {n_patches}. Using mean fallback.")
                return np.mean(filtered_patches, axis=0)
            
            scaler = StandardScaler()
            patches_scaled = scaler.fit_transform(filtered_patches)
            
            optimal_k = self.find_optimal_clusters_elbow(patches_scaled, max_k=MAX_CLUSTERS_TEST)
            
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=100)
            cluster_labels = kmeans.fit_predict(patches_scaled)
            
            unique_labels, cluster_counts = np.unique(cluster_labels, return_counts=True)
            dominant_cluster_label = unique_labels[np.argmax(cluster_counts)]
            dominant_cluster_size = np.max(cluster_counts)
            
            dominant_cluster_mask = cluster_labels == dominant_cluster_label
            dominant_patches = filtered_patches[dominant_cluster_mask]
            
            dominant_cluster_embedding = np.mean(dominant_patches, axis=0)
            
            logger.debug(f"Elbow k={optimal_k}, Dominant cluster: {dominant_cluster_size}/196 patches")
            
            return dominant_cluster_embedding
            
        except Exception as e:
            logger.warning(f"Error in dominant cluster computation, using mean fallback: {e}")
            return np.mean(filtered_patches, axis=0)

    def compute_aggregated_embeddings(self, full_patches: np.ndarray, tile_num: int = None, total_tiles: int = None, city_name: str = None) -> Dict[str, np.ndarray]:
        """Compute multiple aggregation methods for patch embeddings with pre-computed dimension filtering."""
        try:
            # Apply dimension filtering
            filtered_patches = self.filter_embedding_dimensions(full_patches)
            
            aggregations = {}
            
            # Progress prefix for logging
            progress_prefix = ""
            if tile_num and total_tiles and city_name:
                progress_prefix = f"   Tile {tile_num}/{total_tiles} ({city_name}): "
            
            # Standard aggregations (fast)
            aggregations['mean'] = np.mean(filtered_patches, axis=0)
            aggregations['median'] = np.median(filtered_patches, axis=0)
            aggregations['max'] = np.max(filtered_patches, axis=0)
            
            # Log after fast aggregations
            if tile_num and total_tiles and tile_num % 10 == 0:
                logger.debug(f"{progress_prefix}Computed mean/median/max (filtered: {filtered_patches.shape})")
            
            # Dominant cluster aggregation (slow)
            aggregations['dominant_cluster'] = self.compute_dominant_cluster_embedding(filtered_patches)
            
            # Log completion for every 10th tile
            if tile_num and total_tiles and tile_num % 10 == 0:
                progress_pct = (tile_num / total_tiles) * 100
                logger.info(f"   📊 {city_name}: Processed {tile_num}/{total_tiles} tiles ({progress_pct:.1f}%) - all aggregations computed (dims: {filtered_patches.shape[1]})")
            
            return aggregations
            
        except Exception as e:
            logger.warning(f"Error in aggregations, using mean fallback: {e}")
            filtered_patches = self.filter_embedding_dimensions(full_patches)
            mean_embedding = np.mean(filtered_patches, axis=0)
            return {
                'mean': mean_embedding,
                'median': mean_embedding,
                'max': mean_embedding,
                'dominant_cluster': mean_embedding
            }

    def update_dataset_mean(self, mean_embedding: np.ndarray):
        """Update dataset mean incrementally."""
        if self.dataset_mean_accumulator is None:
            self.dataset_mean_accumulator = mean_embedding.copy()
        else:
            self.dataset_mean_accumulator += (mean_embedding - self.dataset_mean_accumulator) / (self.total_tiles_processed + 1)
        
        self.total_tiles_processed += 1

    def process_tile_enhanced(self, row: pd.Series) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Enhanced tile processing."""
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
            
            # Generate reproducible tile ID
            tile_id = row.get('tile_id') or row.get('id')
            city = row.get('city') or row.get('city_name', 'Unknown')
            country = row.get('country') or row.get('country_name', 'Unknown')
            continent = row.get('continent') or row.get('continent_name', 'Unknown')
            date = row.get('date') or row.get('capture_date')
            
            if tile_id is None:
                tile_id = f"{city}_{country}_{lon_val:.6f}_{lat_val:.6f}"
            
            tile_id_str = str(tile_id)
            date_str = str(date) if date is not None else None
            
            # Create unique ID for Qdrant
            unique_id = hash(f"{tile_id_str}_{city}_{country}_{lon_val}_{lat_val}") % (2**63 - 1)
            if unique_id < 0:
                unique_id = abs(unique_id)
            
            unique_id = int(unique_id) if unique_id != 0 else 1
            
            # Generate fallback bounds if needed
            tile_size_deg = 0.01  # Default tile size
            fallback_bounds = [
                [lon_val - tile_size_deg/2, lat_val + tile_size_deg/2],
                [lon_val + tile_size_deg/2, lat_val + tile_size_deg/2],
                [lon_val + tile_size_deg/2, lat_val - tile_size_deg/2],
                [lon_val - tile_size_deg/2, lat_val - tile_size_deg/2],
                [lon_val - tile_size_deg/2, lat_val + tile_size_deg/2]
            ] if row.get('geometry') is None else None
            
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

    def upload_batch_to_qdrant(self, batch_metadata: List[Dict], batch_aggregations: Dict[str, np.ndarray]):
        """Upload a batch to Qdrant collections with detailed progress tracking."""
        if not batch_metadata:
            return
        
        # Add global contrastive to batch aggregations
        if 'mean' in batch_aggregations and len(batch_aggregations['mean']) > 0:
            if self.dataset_mean_accumulator is not None:
                batch_aggregations['global_contrastive'] = batch_aggregations['mean'] - self.dataset_mean_accumulator
            else:
                batch_aggregations['global_contrastive'] = batch_aggregations['mean']  # Fallback
        
        # Track overall upload progress
        methods_to_upload = [m for m in batch_aggregations.keys() if len(batch_aggregations.get(m, [])) > 0]
        
        if len(methods_to_upload) == 0:
            logger.warning("⚠️ No methods to upload")
            return
        
        logger.info(f"📤 Starting upload of {len(batch_metadata)} tiles to {len(methods_to_upload)} filtered collections...")
        
        # Upload to collections with progress tracking
        method_count = 0
        for method in methods_to_upload:
            method_count += 1
            collection_name = self.collections[method]
            embeddings_array = batch_aggregations[method]
            
            # Method-specific progress header
            method_emoji = {
                'mean': '📊',
                'median': '📈',
                'max': '⬆️',
                'global_contrastive': '🌍',
                'dominant_cluster': '🎯'
            }.get(method, '📦')
            
            logger.info(f"{method_emoji} Uploading to {method} collection ({method_count}/{len(methods_to_upload)})...")
            
            # Ensure we have matching metadata and embeddings
            if len(embeddings_array) != len(batch_metadata):
                logger.warning(f"⚠️ Mismatch: {len(embeddings_array)} embeddings vs {len(batch_metadata)} metadata for {method}")
                min_len = min(len(embeddings_array), len(batch_metadata))
                embeddings_array = embeddings_array[:min_len]
                current_metadata = batch_metadata[:min_len]
            else:
                current_metadata = batch_metadata
            
            # Upload in smaller sub-batches
            successful_uploads = 0
            failed_uploads = 0
            
            for i in range(0, len(current_metadata), UPLOAD_BATCH_SIZE):
                sub_batch_metadata = current_metadata[i:i + UPLOAD_BATCH_SIZE]
                sub_batch_embeddings = embeddings_array[i:i + UPLOAD_BATCH_SIZE]
                
                # Create points
                points = []
                for metadata, embedding in zip(sub_batch_metadata, sub_batch_embeddings):
                    payload = {
                        'city': metadata['city'],
                        'country': metadata['country'],
                        'continent': metadata['continent'],
                        'longitude': metadata['longitude'],
                        'latitude': metadata['latitude'],
                        'date': metadata['date'],
                        'original_tile_id': metadata['original_tile_id'],
                        'embedding_type': method,
                        'city_name': f"{metadata['city']}, {metadata['country']}",
                        'has_exact_bounds': metadata.get('has_exact_bounds', False),
                        'dimension_filtering_applied': metadata.get('dimension_filtering_applied', True),
                        'excluded_dimensions_count': metadata.get('excluded_dimensions_count', 0),
                        'filtered_dimension_count': metadata.get('filtered_dimension_count', self.filtered_dimension_count)
                    }
                    
                    # Remove None values
                    payload = {k: v for k, v in payload.items() if v is not None}
                    
                    point = PointStruct(
                        id=metadata['id'],
                        vector=embedding.tolist(),
                        payload=payload
                    )
                    points.append(point)
                
                # Upload sub-batch
                try:
                    operation_info = self.qdrant_client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=points
                    )
                    
                    if operation_info.status == models.UpdateStatus.COMPLETED:
                        successful_uploads += len(points)
                    else:
                        logger.warning(f"   ⚠️ Upload status for {collection_name}: {operation_info.status}")
                        failed_uploads += len(points)
                
                except Exception as e:
                    logger.error(f"   ❌ Error uploading to {collection_name}: {e}")
                    failed_uploads += len(points)
                    continue
                
                # Clean up points
                points.clear()
            
            # Report method completion
            if failed_uploads > 0:
                success_rate = (successful_uploads / (successful_uploads + failed_uploads)) * 100
                logger.warning(f"   ⚠️ {method}: {successful_uploads} uploaded, {failed_uploads} failed ({success_rate:.1f}% success)")
            else:
                logger.info(f"   ✅ {method}: {successful_uploads} tiles uploaded successfully")
        
        # Final summary
        logger.info(f"📊 Batch upload complete: {len(batch_metadata)} tiles × {len(methods_to_upload)} methods")

    def process_city_batch(self, files_batch: List[str]) -> Tuple[List[Dict], Dict[str, List[np.ndarray]]]:
        """Process a batch of city files with dimension filtering."""
        batch_metadata = []
        batch_aggregations = {method: [] for method in self.collections.keys()}
        
        cities_in_batch = []
        tiles_with_exact_bounds = 0
        tiles_with_fallback_bounds = 0
        
        for file_path in files_batch:
            try:
                city_name = os.path.splitext(os.path.basename(file_path))[0]
                cities_in_batch.append(city_name)
                
                logger.info(f"🏙️ Processing {city_name}...")
                
                # Load GeoParquet file
                gdf = gpd.read_parquet(file_path)
                
                if len(gdf) == 0:
                    logger.warning(f"⚠️ Empty file: {file_path}")
                    continue
                
                city_tile_count = 0
                city_exact_bounds = 0
                
                # Process each tile in the city
                for idx, row in enumerate(gdf.iterrows(), 1):
                    _, row_data = row
                    
                    metadata, full_patches = self.process_tile_enhanced(row_data)
                    
                    if metadata and full_patches is not None:
                        # Compute aggregations with dimension filtering
                        all_aggregations = self.compute_aggregated_embeddings(
                            full_patches, 
                            tile_num=idx,
                            total_tiles=len(gdf),
                            city_name=city_name
                        )
                        
                        # Store metadata and aggregations
                        batch_metadata.append(metadata)
                        for method, embedding in all_aggregations.items():
                            if method in batch_aggregations:
                                batch_aggregations[method].append(embedding)
                        
                        # Update dataset mean incrementally
                        self.update_dataset_mean(all_aggregations['mean'])
                        
                        city_tile_count += 1
                        
                        # Track bounds statistics
                        if metadata.get('has_exact_bounds', False):
                            city_exact_bounds += 1
                            tiles_with_exact_bounds += 1
                        else:
                            tiles_with_fallback_bounds += 1
                        
                        # Clean up full_patches immediately
                        del full_patches
                
                if city_tile_count > 0:
                    bounds_info = f"({city_exact_bounds} exact bounds)" if city_exact_bounds > 0 else "(fallback bounds)"
                    logger.info(f"✅ {city_name}: {city_tile_count} tiles processed {bounds_info} [filtered: {self.filtered_dimension_count} dims]")
                else:
                    logger.warning(f"⚠️ {city_name}: No valid tiles found")
                
                # Clean up the GeoDataFrame
                del gdf
                
                # Check memory usage
                memory_gb = get_memory_usage()
                if memory_gb > MAX_MEMORY_GB:
                    logger.warning(f"⚠️ High memory usage: {memory_gb:.2f} GB")
                    force_garbage_collection()
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                continue
        
        # Convert lists to numpy arrays
        for method in batch_aggregations:
            if batch_aggregations[method]:
                batch_aggregations[method] = np.array(batch_aggregations[method])
            else:
                batch_aggregations[method] = np.array([]).reshape(0, self.filtered_dimension_count)
        
        # Batch summary before upload
        if cities_in_batch:
            logger.info(f"📊 BATCH PROCESSING COMPLETE - Ready to upload")
            logger.info(f"   Cities processed: {len(cities_in_batch)}")
            logger.info(f"   Total tiles: {len(batch_metadata)}")
            logger.info(f"   Filtered dimensions: {self.filtered_dimension_count}/768")
            logger.info(f"   Bounds: {tiles_with_exact_bounds} exact, {tiles_with_fallback_bounds} fallback")
        
        return batch_metadata, batch_aggregations

    def process_all_cities_in_batches(self) -> Tuple[List[Dict], int, int]:
        """Process all cities in memory-efficient batches with dimension filtering."""
        logger.info(f"📖 Processing all cities with dimension filtering...")
        
        # Find all parquet files
        pattern = os.path.join(EMBEDDINGS_DIR, "full_patch_embeddings", "*", "*.gpq")
        files = glob(pattern)
        
        if not files:
            raise ValueError(f"No .gpq files found in {pattern}")
        
        logger.info(f"📄 Found {len(files)} files to process")
        
        all_metadata = []
        total_tiles_with_exact_bounds = 0
        total_tiles_with_fallback_bounds = 0
        
        # Process files in batches
        for i in tqdm(range(0, len(files), CITY_BATCH_SIZE), desc="Processing city batches"):
            files_batch = files[i:i + CITY_BATCH_SIZE]
            
            batch_num = i//CITY_BATCH_SIZE + 1
            total_batches = (len(files) + CITY_BATCH_SIZE - 1)//CITY_BATCH_SIZE
            logger.info(f"📄 Processing batch {batch_num}/{total_batches}")
            
            # Process this batch of cities
            batch_metadata, batch_aggregations = self.process_city_batch(files_batch)
            
            if batch_metadata:
                # Upload this batch to Qdrant
                self.upload_batch_to_qdrant(batch_metadata, batch_aggregations)
                
                # Accumulate metadata for final save
                all_metadata.extend(batch_metadata)
                
                # Update bounds statistics
                batch_exact = sum(1 for m in batch_metadata if m.get('has_exact_bounds', False))
                batch_fallback = len(batch_metadata) - batch_exact
                
                total_tiles_with_exact_bounds += batch_exact
                total_tiles_with_fallback_bounds += batch_fallback
            
            # Clean up batch data
            del batch_metadata, batch_aggregations
            
            # Force garbage collection after each batch
            force_garbage_collection()
        
        logger.info(f"📊 Processing Summary with dimension filtering:")
        logger.info(f"   Total tiles processed: {total_tiles_with_exact_bounds + total_tiles_with_fallback_bounds}")
        logger.info(f"   🎯 Tiles with exact bounds: {total_tiles_with_exact_bounds}")
        logger.info(f"   📍 Tiles with fallback bounds: {total_tiles_with_fallback_bounds}")
        if total_tiles_with_exact_bounds + total_tiles_with_fallback_bounds > 0:
            coverage = total_tiles_with_exact_bounds/(total_tiles_with_exact_bounds + total_tiles_with_fallback_bounds)*100
            logger.info(f"   📊 Exact bounds coverage: {coverage:.1f}%")
        
        return all_metadata, total_tiles_with_exact_bounds, total_tiles_with_fallback_bounds

    def compute_umap_for_all_data(self, metadata_list: List[Dict]) -> Dict:
        """Compute UMAP coordinates using filtered mean embeddings from Qdrant."""
        try:
            logger.info(f"🗺️ Computing UMAP from filtered embeddings in Qdrant...")
            
            # Get all point IDs
            point_ids = [metadata['id'] for metadata in metadata_list]
            
            # Retrieve embeddings from mean collection
            collection_name = self.collections['mean']
            
            logger.info(f"📥 Retrieving {len(point_ids)} embeddings from {collection_name}...")
            
            # Retrieve points in batches
            batch_size = 1000
            all_embeddings = []
            valid_metadata = []
            
            for i in range(0, len(point_ids), batch_size):
                batch_ids = point_ids[i:i + batch_size]
                points = self.qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=batch_ids,
                    with_vectors=True
                )
                
                for point in points:
                    if point.vector is not None:
                        all_embeddings.append(point.vector)
                        # Find corresponding metadata
                        for metadata in metadata_list:
                            if metadata['id'] == point.id:
                                valid_metadata.append(metadata)
                                break
            
            if len(all_embeddings) == 0:
                raise ValueError("No embeddings retrieved from Qdrant")
            
            embeddings_array = np.array(all_embeddings)
            logger.info(f"✅ Retrieved {len(embeddings_array)} embeddings with shape {embeddings_array.shape}")
            
            # Compute UMAP
            logger.info("🔄 Computing UMAP coordinates...")
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                metric='cosine',
                random_state=42
            )
            
            umap_coords = reducer.fit_transform(embeddings_array)
            
            # Create UMAP points with metadata
            umap_points = []
            for i, (metadata, coords) in enumerate(zip(valid_metadata, umap_coords)):
                umap_point = {
                    'location_id': metadata['id'],
                    'x': float(coords[0]),
                    'y': float(coords[1]),
                    'city': metadata['city'],
                    'country': metadata['country'],
                    'continent': metadata['continent'],
                    'longitude': metadata['longitude'],
                    'latitude': metadata['latitude'],
                    'date': metadata['date'],
                    'bounds': metadata.get('tile_bounds')
                }
                umap_points.append(umap_point)
            
            umap_data = {
                'umap_points': umap_points,
                'total_points': len(umap_points),
                'bounds_statistics': {
                    'tiles_with_exact_bounds': sum(1 for m in valid_metadata if m.get('has_exact_bounds', False)),
                    'tiles_with_fallback_bounds': sum(1 for m in valid_metadata if not m.get('has_exact_bounds', False)),
                    'exact_bounds_percentage': sum(1 for m in valid_metadata if m.get('has_exact_bounds', False)) / len(valid_metadata) * 100
                },
                'dimension_info': {
                    'original_dimensions': 768,
                    'filtered_dimensions': self.filtered_dimension_count,
                    'excluded_dimensions_count': len(self.excluded_dimensions),
                    'filtering_applied': True,
                    'excluded_dimensions': list(self.excluded_dimensions)
                }
            }
            
            logger.info(f"✅ UMAP computation completed for {len(umap_points)} points using filtered embeddings")
            return umap_data
            
        except Exception as e:
            logger.error(f"❌ Error computing UMAP: {e}")
            raise

    def save_enhanced_data(self, metadata_list: List[Dict], analysis_results: Dict):
        """Save metadata and compute UMAP with dimension analysis results."""
        logger.info("💾 Computing UMAP and saving enhanced data...")
        
        # Compute UMAP using filtered embeddings from Qdrant
        umap_data = self.compute_umap_for_all_data(metadata_list)
        
        dataset_stats = {
            'total_samples': len(metadata_list),
            'embedding_dimension_original': 768,
            'embedding_dimension_filtered': self.filtered_dimension_count,
            'dimension_filtering': {
                'applied': True,
                'excluded_dimensions_count': len(self.excluded_dimensions),
                'excluded_dimensions': list(self.excluded_dimensions),
                'exclusion_percentage': (len(self.excluded_dimensions) / 768 * 100),
                'analysis_results': analysis_results,
                'filtering_method': 'pre_computed_from_json'
            },
            'dataset_mean': self.dataset_mean_accumulator.tolist() if self.dataset_mean_accumulator is not None else None,
            'collections': self.collections,
            'force_reupload': self.force_reupload,
            'timestamp': time.time(),
            'clustering_parameters': {
                'use_elbow_method': USE_ELBOW_METHOD,
                'max_clusters_test': MAX_CLUSTERS_TEST,
                'fallback_clusters': FALLBACK_CLUSTERS,
                'clustering_algorithm': 'KMeans',
                'scaler': 'StandardScaler',
                'cluster_selection': 'elbow_method' if USE_ELBOW_METHOD else 'fixed_k'
            },
            'aggregation_methods': {
                'mean': 'Standard mean aggregation of filtered patches',
                'median': 'Median aggregation of filtered patches (robust to outliers)',
                'max': 'Element-wise maximum of filtered patches',
                'global_contrastive': 'Dataset mean subtracted from mean embeddings (filtered)',
                'dominant_cluster': f'Average of patches in most frequent cluster (Elbow method, max k={MAX_CLUSTERS_TEST}, filtered)' if USE_ELBOW_METHOD else f'Average of patches in most frequent cluster (Fixed k={FALLBACK_CLUSTERS}, filtered)'
            },
            'enhanced_features': {
                'exact_tile_bounds': True,
                'bounds_coverage_percentage': umap_data['bounds_statistics']['exact_bounds_percentage'],
                'multiple_aggregations': True,
                'aggregation_count': len(self.collections),
                'dominant_cluster_method': True,
                'dimension_filtering': True,
                'city_discriminative_dimensions_removed': True,
                'pre_computed_dimension_analysis': True
            }
        }
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save enhanced files (with gzip compression)
        files_to_save = [
            ('locations_metadata.json', metadata_list),
            ('umap_coordinates.json', umap_data),
            ('dataset_statistics.json', dataset_stats)
        ]
        
        for filename, data in files_to_save:
            # Save regular JSON
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"📄 Saved {filename}")
            
            # Save gzipped version
            gzipped_filepath = f"{filepath}.gz"
            with gzip.open(gzipped_filepath, 'wt') as f:
                json.dump(data, f, indent=2)
            logger.info(f"🗜️ Saved compressed {filename}.gz")
        
        logger.info(f"💾 Enhanced data saved to {OUTPUT_DIR}")
        logger.info(f"📍 Enhanced similarity search with reduced city bias!")
        logger.info(f"📈 UMAP visualization computed from filtered embeddings!")

    def run(self):
        """Main execution method."""
        start_time = time.time()
        
        logger.info(f"🛰️ Simplified Memory-Efficient Embeddings Uploader - Using Pre-computed Analysis")
        logger.info("=" * 80)
        logger.info(f"🧠 Memory-efficient processing with batches of {CITY_BATCH_SIZE} cities")
        logger.info(f"📦 Upload batches of {UPLOAD_BATCH_SIZE} tiles")
        logger.info(f"⚠️ Memory limit: {MAX_MEMORY_GB} GB")
        logger.info(f"📁 Embeddings directory: {EMBEDDINGS_DIR}")
        logger.info(f"📁 Output directory: {OUTPUT_DIR}")
        logger.info(f"📄 Using pre-computed dimension analysis from: {DIMENSION_ANALYSIS_FILE}")
        logger.info(f"🔄 Force reupload: {self.force_reupload}")
        
        initial_memory = get_memory_usage()
        logger.info(f"🧠 Initial memory usage: {initial_memory:.2f} GB")
        
        logger.info("📄 SIMPLIFIED MODE: Using pre-computed dimension analysis + exact bounds + aggregation methods")
        
        # Phase 1: Manage collections based on force_reupload setting
        self.manage_collections()
        
        # Phase 2: Process all cities in memory-efficient batches
        # Only process if we have collections to upload to or if force_reupload is True
        existing_collections = self.check_existing_collections()
        collections_with_data = []
        
        if not self.force_reupload:
            # Check which collections have data
            for embed_type, collection_name in self.collections.items():
                if existing_collections.get(embed_type, False):
                    info = self.get_collection_info(collection_name)
                    if info and info['points_count'] > 0:
                        collections_with_data.append((embed_type, collection_name))
        
        if self.force_reupload or len(collections_with_data) < len(self.collections):
            logger.info("🔄 PHASE 2: Data Processing and Upload with Filtering")
            
            if not self.force_reupload and collections_with_data:
                logger.info(f"ℹ️ Found {len(collections_with_data)} collections with existing data:")
                for embed_type, collection_name in collections_with_data:
                    info = self.get_collection_info(collection_name)
                    logger.info(f"   - {embed_type}: {collection_name} ({info['points_count']} points)")
                logger.info(f"📤 Will only upload to {len(self.collections) - len(collections_with_data)} empty collections")
            
            metadata_list, tiles_with_exact_bounds, tiles_with_fallback_bounds = self.process_all_cities_in_batches()
            
            if len(metadata_list) > 0:
                # Phase 3: Save enhanced data and compute UMAP
                logger.info("🔄 PHASE 3: UMAP Computation and Data Export")
                analysis_results = load_dimension_analysis_results(DIMENSION_ANALYSIS_FILE)
                self.save_enhanced_data(metadata_list, analysis_results)
        else:
            logger.info("⏭️ PHASE 2: Skipping data processing - all collections have existing data")
            logger.info("ℹ️ Use FORCE_REUPLOAD=True to reprocess and reupload all data")
            tiles_with_exact_bounds = 0
            tiles_with_fallback_bounds = 0
        
        # Final summary
        duration = (time.time() - start_time) / 60
        final_memory = get_memory_usage()
        
        logger.info(f"\n🎉 Simplified upload completed!")
        logger.info(f"⏱️ Duration: {duration:.1f} minutes")
        logger.info(f"🧠 Peak memory usage: {final_memory:.2f} GB")
        
        if tiles_with_exact_bounds + tiles_with_fallback_bounds > 0:
            logger.info(f"📊 Tiles processed: {tiles_with_exact_bounds + tiles_with_fallback_bounds}")
            logger.info(f"🎯 Tiles with exact bounds: {tiles_with_exact_bounds}")
            logger.info(f"📍 Exact bounds coverage: {tiles_with_exact_bounds/(tiles_with_exact_bounds + tiles_with_fallback_bounds)*100:.1f}%")
        
        logger.info(f"📍 Dimension filtering applied:")
        logger.info(f"   ❌ Excluded {len(self.excluded_dimensions)} city-discriminative dimensions")
        logger.info(f"   ✅ Used {self.filtered_dimension_count} dimensions for similarity")
        logger.info(f"   📊 Exclusion rate: {len(self.excluded_dimensions)/768*100:.1f}%")
        
        logger.info(f"📊 Filtered aggregation methods available: {', '.join(self.collections.keys())}")
        logger.info(f"🆕 All methods use pre-computed dimension filtering!")


def main():
    """Main function using pre-computed dimension analysis."""
    logger.info("🛰️ Simplified Memory-Efficient Embeddings Uploader - Using Pre-computed Analysis")
    logger.info("=" * 100)
    
    # Load dimension analysis results
    analysis_results = load_dimension_analysis_results(DIMENSION_ANALYSIS_FILE)
    if analysis_results is None:
        logger.error("❌ Cannot proceed without dimension analysis results")
        logger.info("💡 Please run: python standalone_dimension_analysis.py first")
        return 1
    
    excluded_dimensions = analysis_results['cutoff_results']['excluded_dimensions']
    
    logger.info(f"📍 Configuration:")
    logger.info(f"   Embeddings directory: {EMBEDDINGS_DIR}")
    logger.info(f"   Output directory: {OUTPUT_DIR}")
    logger.info(f"   Force reupload: {FORCE_REUPLOAD}")
    logger.info(f"   Memory management: {CITY_BATCH_SIZE} cities per batch, {MAX_MEMORY_GB}GB limit")
    logger.info(f"   Dimension filtering: {len(excluded_dimensions)} dimensions excluded")
    
    # Show the collection names that will be created
    collections = {
        'mean': 'terramind_embeddings_mean',
        'median': 'terramind_embeddings_median',
        'max': 'terramind_embeddings_max',
        'global_contrastive': 'terramind_embeddings_global_contrastive',
        'dominant_cluster': 'terramind_embeddings_dominant_cluster'
    }
    logger.info(f"📦 Filtered collections to manage:")
    for method, collection_name in collections.items():
        logger.info(f"   - {method}: {collection_name}")
    
    if not os.path.exists(EMBEDDINGS_DIR):
        logger.error(f"❌ Embeddings directory not found: {EMBEDDINGS_DIR}")
        return 1
    
    uploader = SimplifiedEmbeddingsUploader(excluded_dimensions, force_reupload=FORCE_REUPLOAD)
    
    try:
        uploader.run()
        logger.info("🎉 Simplified script completed successfully!")
        logger.info("🎯 Exact tile bounds have been extracted and included!")
        logger.info("📊 Aggregation methods have been computed and uploaded with dimension filtering!")
        logger.info("🆕 Dominant cluster method captures most frequent visual patterns (filtered)!")
        logger.info("📍 City-discriminative dimensions have been filtered from all embeddings!")
        logger.info("📈 Enhanced similarity search with reduced city bias!")
        logger.info("🗺️ UMAP coordinates computed from filtered Qdrant data!")
        logger.info("📍 Update your frontend and backend to use the new filtered collections!")
        logger.info("💡 All similarity searches will now have reduced city bias!")
        logger.info("📊 Pre-computed dimension analysis used for maximum efficiency!")
        return 0
    except Exception as e:
        logger.error(f"❌ Simplified script failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())