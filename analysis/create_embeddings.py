import os
import numpy as np
import pystac_client
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import datetime
from datetime import datetime
from shapely.geometry import Point
from odc.stac import stac_load
from tqdm import tqdm
import pandas as pd
from typing import Literal
import time
from dask.distributed import Client, LocalCluster
from dask.distributed import Adaptive
from shapely.geometry import Point, Polygon, box
from terratorch.registry import BACKBONE_REGISTRY
import hashlib
import json
import zarr
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import dask
import resource
import warnings
import logging

# 1. SILENCE NUMPY/DASK RUNTIME WARNINGS
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=UserWarning, module='dask')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 2. REDUCE DASK/DISTRIBUTED LOGGING VERBOSITY
logging.getLogger('distributed').setLevel(logging.ERROR)       # Only show errors
logging.getLogger('distributed.worker').setLevel(logging.ERROR)
logging.getLogger('distributed.nanny').setLevel(logging.ERROR)
logging.getLogger('distributed.scheduler').setLevel(logging.ERROR)
logging.getLogger('distributed.client').setLevel(logging.ERROR)
logging.getLogger('distributed.core').setLevel(logging.ERROR)
logging.getLogger('distributed.utils').setLevel(logging.ERROR)
logging.getLogger('tornado').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)

# 3. SILENCE SPECIFIC PROBLEMATIC LOGGERS
logging.getLogger('distributed.worker.memory').setLevel(logging.CRITICAL)  # Memory warnings
logging.getLogger('distributed.nanny.memory').setLevel(logging.CRITICAL)   # Nanny memory warnings
logging.getLogger('distributed.scheduler').setLevel(logging.CRITICAL)      # Task recomputation warnings

# Set environment variables
os.environ['GDAL_HTTP_MAX_RETRY'] = '10'  # Increased from 40
os.environ['GDAL_HTTP_RETRY_DELAY'] = '5'  # Reduced from 15
os.environ['GDAL_HTTP_TIMEOUT'] = '120'
os.environ['GDAL_HTTP_CONNECTTIMEOUT'] = '60'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['GDAL_HTTP_VERSION'] = '2'

# Configuration
TILE_SIZE = 224  # Optimal for TerraMind model (224/16 = 14x14 = 196 patches)

# Storage configuration
SAVE_TO_GEOPARQUET = True
SAVE_RAW_DATA_LOCALLY = True  # NEW FLAG: Set to True to cache raw STAC data locally
LOCAL_DATA_DIR = "cached_stac_data"  # Directory to store cached raw data in Zarr format

# NEW: Flag to control saving full patch embeddings
SAVE_FULL_PATCH_EMBEDDINGS = True  # Set to True to save full 196x768 patch embeddings

def create_data_cache_key(bbox_coords, resolution=10):
    """Create a unique cache key for the data based on bbox and resolution."""
    # Create a hash based on bbox and resolution for unique identification
    key_data = f"{bbox_coords[0]:.6f}_{bbox_coords[1]:.6f}_{bbox_coords[2]:.6f}_{bbox_coords[3]:.6f}_{resolution}"
    return hashlib.md5(key_data.encode()).hexdigest()

# Add this at the beginning of your script
def increase_file_limits():
    """Increase the number of open file descriptors allowed"""
    try:
        # Get current limits
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current file descriptor limits: soft={soft}, hard={hard}")
        
        # Set to maximum allowed
        new_soft = min(4096, hard)  # Don't exceed hard limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"Increased file descriptor limit to: {new_soft}")
        
    except Exception as e:
        print(f"Warning: Could not increase file limits: {e}")

def save_raw_data_locally(stac_data, cache_key):
    """Save raw STAC data to local cache using Zarr format for optimal performance."""
    if not SAVE_RAW_DATA_LOCALLY:
        return
        
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    cache_dir = os.path.join(LOCAL_DATA_DIR, f"{cache_key}.zarr")
    
    try:
        # Save the main data array with compression and chunking
        full_data = stac_data['full_data']
        bands, height, width = full_data.shape
        
        # Optimize chunks for tile access (224x224 tiles)
        chunk_size = min(224, height, width)
        chunks = (1, chunk_size, chunk_size)  # One band at a time, spatial chunks
        
        # Use Zarr v3 API with BloscCodec
        compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
        
        # Create array directly - Zarr v3 style
        z = zarr.create_array(
            store=cache_dir,
            shape=full_data.shape,
            chunks=chunks,
            dtype=np.float32,
            compressors=compressors,
            overwrite=True
        )
        
        # Write data
        z[:] = full_data
        
        # Save metadata as JSON attributes
        z.attrs['bbox'] = stac_data['bbox']
        z.attrs['bands'] = stac_data['bands']
        z.attrs['total_height'] = stac_data['total_height']
        z.attrs['total_width'] = stac_data['total_width']
        z.attrs['acquisition_date'] = stac_data['acquisition_date'].isoformat() if stac_data['acquisition_date'] else None
        z.attrs['quality_metrics'] = stac_data['quality_metrics']
        
        print(f"💾 Cached data to Zarr store: {cache_dir}")
        
        # Calculate compression ratio for info
        original_size = full_data.nbytes / 1024 / 1024  # MB
        compressed_size = sum(os.path.getsize(os.path.join(root_dir, f)) 
                            for root_dir, _, files in os.walk(cache_dir) 
                            for f in files) / 1024 / 1024  # MB
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
        print(f"📊 Compression: {original_size:.1f}MB → {compressed_size:.1f}MB ({compression_ratio:.1f}x)")
        
    except Exception as e:
        print(f"❌ Error saving Zarr cache: {str(e)}")

def load_raw_data_locally(cache_key):
    """Load raw STAC data from local Zarr cache."""
    if not SAVE_RAW_DATA_LOCALLY:
        return None
        
    cache_dir = os.path.join(LOCAL_DATA_DIR, f"{cache_key}.zarr")
    
    if not os.path.exists(cache_dir):
        return None
        
    try:
        # Open Zarr array directly - Zarr v3 style
        z = zarr.open_array(cache_dir, mode='r')
        
        # Load data and reconstruct stac_data dict
        stac_data = {
            'full_data': np.array(z[:]),  # Load into memory
            'bbox': list(z.attrs['bbox']),
            'bands': list(z.attrs['bands']),
            'total_height': int(z.attrs['total_height']),
            'total_width': int(z.attrs['total_width']),
            'acquisition_date': datetime.fromisoformat(z.attrs['acquisition_date']) if z.attrs['acquisition_date'] else None,
            'quality_metrics': dict(z.attrs['quality_metrics'])
        }
        
        print(f"📁 Loaded cached data from Zarr store: {cache_dir}")
        return stac_data
        
    except Exception as e:
        print(f"❌ Error loading Zarr cache: {str(e)}")
        return None

def aggregate_patch_embeddings(
    features: torch.Tensor, 
    method: Literal['mean', 'max', 'median', 'std', 'sum', 'weighted_mean'] = 'mean',
    weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Aggregate patch embeddings from shape [batch, 196, 768] to [batch, 768].
    
    Args:
        features: Tensor of shape [batch, 196, 768] containing patch embeddings
        method: Aggregation method to use
        weights: Optional weights for weighted aggregation [196] or [batch, 196, 1]
    
    Returns:
        Aggregated tensor of shape [batch, 768]
    """
    if features.dim() != 3 or features.shape[2] != 768:
        raise ValueError(f"Expected features of shape [batch, 196, 768], got {features.shape}")
    
    if method == 'mean':
        return features.mean(dim=1)  # [batch, 768]
    
    elif method == 'max':
        return features.max(dim=1)[0]  # [batch, 768]
    
    elif method == 'median':
        return features.median(dim=1)[0]  # [batch, 768]
    
    elif method == 'std':
        return features.std(dim=1)  # [batch, 768]
    
    elif method == 'sum':
        return features.sum(dim=1)  # [batch, 768]
    
    elif method == 'weighted_mean':
        if weights is None:
            # Use attention-like weights based on norm of each patch
            weights = torch.norm(features, dim=2, keepdim=True)  # [batch, 196, 1]
            weights = torch.softmax(weights, dim=1)
        else:
            if weights.dim() == 1:
                weights = weights.unsqueeze(0).unsqueeze(2)  # [1, 196, 1]
            elif weights.dim() == 2:
                weights = weights.unsqueeze(2)  # [batch, 196, 1]
        
        weighted_features = features * weights
        return weighted_features.sum(dim=1)  # [batch, 768]
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def load_stac_data_with_quality_control(bbox, resolution: int = 10, retry_attempts: int = 3, 
                                        min_valid_pixel_threshold: float = 10.0):
    """
    Memory-optimized STAC data loading with smaller chunks and better error handling.
    """
    # Parse bbox if string
    if isinstance(bbox, str):
        bbox_coords = [float(x.strip()) for x in bbox.split(',')]
    else:
        bbox_coords = list(bbox)
    
    # Create cache key for this data request
    cache_key = create_data_cache_key(bbox_coords, resolution)
    
    # Try to load from cache first
    cached_data = load_raw_data_locally(cache_key)

    if cached_data is not None:
        print(f"✅ Using cached data (quality: {cached_data['quality_metrics']['min_valid_percentage']:.1f}%)")
        return cached_data
    
    print(f"🔍 No cached data found, fetching from STAC...")
    
    # Try with progressively relaxed parameters
    search_configs = [
        {"max_items": 20, "cloud_cover": 12, "attempt": "initial"},  # Reduced from 10
        {"max_items": 25, "cloud_cover": 16, "attempt": "relaxed_items"},  # Reduced from 15
        {"max_items": 30, "cloud_cover": 38, "attempt": "very_relaxed"},  # Reduced from 25
    ]
    
    city_name = f"bbox_{bbox_coords[0]:.3f}_{bbox_coords[1]:.3f}"
    
    for config in search_configs:
        print(f"\n🔍 Attempt: {config['attempt']} - max_items={config['max_items']}, cloud_cover<{config['cloud_cover']}%")
        
        try:
            # Add retry logic for network issues
            import time
            for retry in range(retry_attempts):
                try:
                    # Search STAC with current config
                    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
                    search = catalog.search(
                        collections=["sentinel-2-l2a"],
                        bbox=bbox_coords,
                        max_items=config['max_items'],
                        datetime="2025-08-26/2025-08-26",
                        query={"eo:cloud_cover": {"lt": config['cloud_cover']}}
                    )
                    items = search.item_collection()
                    print(f"Found {len(items)} items with cloud cover < {config['cloud_cover']}%")
                    break  # Success, exit retry loop
                    
                except Exception as network_error:
                    print(f"🔄 Network retry {retry+1}/{retry_attempts}: {str(network_error)}")
                    if retry < retry_attempts - 1:
                        time.sleep(5 * (retry + 1))  # Exponential backoff
                        continue
                    else:
                        raise  # Re-raise if all retries failed
            
            # Load data with SMALLER CHUNKS to reduce memory usage
            bands = ["coastal", "blue", "green", "red", "rededge1", "rededge2", 
                     "rededge3", "nir", "nir08", "nir09", "swir16", "swir22"]
            
            ds = stac_load(
                items,
                bands=["scl"] + bands,
                bbox=bbox_coords,
                resolution=resolution,
                chunks={"time": 1, "x": 512, "y": 512},  # REDUCED from 1024 to 512
                groupby="solar_day",
                resampling="bilinear"
            )
            
            print(f"Dataset dimensions: {ds.dims}")
            
            # Cloud mask - keep clear pixels (4=vegetation, 5=not_vegetated, 6=water)
            cloud_mask = ds.scl.isin([4, 5, 6])
            
            # Apply mask and compute median WITH MEMORY OPTIMIZATION
            ds_masked = ds[bands].where(cloud_mask)
            
            # Compute in smaller batches to avoid memory issues
            median = (
                ds_masked.where(ds_masked > 0)
                .median(dim="time", skipna=True)
            )
            
            print("🔄 Computing median composite...")
            median = median.compute()
            
            # Convert to numpy array [bands, y, x]
            full_data = median.to_array().values
            
            # Check data quality BEFORE replacing NaN
            total_pixels = full_data.shape[1] * full_data.shape[2]
            valid_pixels_per_band = []
            
            for band_idx in range(full_data.shape[0]):
                band_data = full_data[band_idx]
                valid_pixels = ~np.isnan(band_data)
                valid_pixel_count = valid_pixels.sum()
                valid_percentage = (valid_pixel_count / total_pixels) * 100
                valid_pixels_per_band.append(valid_percentage)
            
            # Use the worst band as overall quality metric
            min_valid_percentage = min(valid_pixels_per_band)
            avg_valid_percentage = np.mean(valid_pixels_per_band)
            
            print(f"📊 Data quality:")
            print(f"   Min valid pixels (worst band): {min_valid_percentage:.1f}%")
            print(f"   Avg valid pixels (all bands): {avg_valid_percentage:.1f}%")
            
            # Check if quality meets threshold
            if min_valid_percentage >= min_valid_pixel_threshold:
                print(f"✅ Quality sufficient ({min_valid_percentage:.1f}% >= {min_valid_pixel_threshold}%)")
                
                # Now safe to replace NaN with 0
                full_data = np.nan_to_num(full_data, nan=0.0)
                
                # Get acquisition date
                acquisition_date = None
                if len(items) > 0:
                    acquisition_date = datetime.fromisoformat(items[len(items)//2].properties['datetime'].replace('Z', '+00:00'))
                
                stac_data = {
                    'full_data': full_data,
                    'bbox': bbox_coords,
                    'bands': bands,
                    'total_height': full_data.shape[1],
                    'total_width': full_data.shape[2],
                    'acquisition_date': acquisition_date,
                    'quality_metrics': {
                        'min_valid_percentage': min_valid_percentage,
                        'avg_valid_percentage': avg_valid_percentage,
                        'items_used': len(items),
                        'cloud_threshold': config['cloud_cover'],
                        'search_config': config['attempt']
                    }
                }
                
                # Save to cache
                save_raw_data_locally(stac_data, cache_key)
                
                # Clean up intermediate objects
                del ds, ds_masked, median, cloud_mask
                import gc
                gc.collect()
                
                return stac_data
            else:
                print(f"❌ Quality insufficient ({min_valid_percentage:.1f}% < {min_valid_pixel_threshold}%)")
                
                # Save problematic image for inspection
                save_problematic_image(full_data, bbox_coords, city_name, config, min_valid_percentage)
                
                # Clean up and try next configuration
                del ds, ds_masked, median, cloud_mask, full_data
                import gc
                gc.collect()
                continue
                
        except Exception as e:
            print(f"❌ Error with {config['attempt']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # If all attempts failed
    print(f"❌ All attempts failed for {city_name}")
    return None

def save_problematic_image(full_data, bbox_coords, city_name, config, valid_percentage):
    """Save problematic images with low data coverage for inspection."""
    
    # Create directory for problematic images
    problem_dir = "images_with_missing_data"
    os.makedirs(problem_dir, exist_ok=True)
    
    # Replace NaN for visualization
    full_data_vis = np.nan_to_num(full_data, nan=0.0)
    
    # Create a simple RGB composite (bands 3,2,1 = red,green,blue)
    if full_data_vis.shape[0] >= 4:
        rgb_composite = full_data_vis[[3,2,1], :, :]  # red, green, blue
        rgb_composite = np.transpose(rgb_composite, (1, 2, 0))
        
        # Normalize for visualization (simple percentile stretch)
        for i in range(3):
            band = rgb_composite[:, :, i]
            if band.max() > 0:
                p2, p98 = np.percentile(band[band > 0], [2, 98])
                rgb_composite[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        
        # Save the image
        filename = f"{city_name}_{config['attempt']}_valid{valid_percentage:.1f}pct.png"
        filepath = os.path.join(problem_dir, filename)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_composite)
        plt.title(f"Problematic Image: {city_name}\n{config['attempt']} - {valid_percentage:.1f}% valid pixels")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved problematic image: {filepath}")

def create_non_overlapping_tiles(full_data: np.ndarray, bbox_coords, city_geometry, tile_size: int = 224):
    """
    Create non-overlapping tiles that completely fill the bbox area, but only return tiles
    that intersect with the actual city geometry (regardless of overlap fraction).
    
    Args:
        full_data: Full data array [bands, y, x]
        bbox_coords: [min_lon, min_lat, max_lon, max_lat]
        city_geometry: Shapely geometry representing the city boundary
        tile_size: Size of square tiles
        
    Returns:
        list of (tile_data, bounds) tuples for tiles that intersect city
    """
    bands, total_height, total_width = full_data.shape

    # Calculate how many complete tiles fit
    rows = total_height // tile_size
    cols = total_width // tile_size

    print(f"Image size: {total_height} x {total_width} pixels")
    print(f"Total possible tiles: {rows} x {cols} = {rows*cols} non-overlapping {tile_size}x{tile_size} tiles")

    min_lon, min_lat, max_lon, max_lat = bbox_coords
    lon_step = (max_lon - min_lon) / cols
    lat_step = (max_lat - min_lat) / rows

    tiles = []
    tiles_filtered_out = 0

    for row in range(rows):
        for col in range(cols):
            y_start = row * tile_size
            y_end = y_start + tile_size
            x_start = col * tile_size
            x_end = x_start + tile_size

            tile_min_lon = min_lon + col * lon_step
            tile_max_lon = min_lon + (col + 1) * lon_step
            tile_max_lat = max_lat - row * lat_step
            tile_min_lat = max_lat - (row + 1) * lat_step

            bounds = (tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat)
            tile_geom = box(tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat)

            if tile_geom.intersects(city_geometry):
                tile = full_data[:, y_start:y_end, x_start:x_end]
                assert tile.shape == (bands, tile_size, tile_size), f"Tile shape mismatch: {tile.shape}"
                tiles.append((tile, bounds))
            else:
                tiles_filtered_out += 1

    print(f"Tiles intersecting city boundary: {len(tiles)}")
    print(f"Tiles filtered out (no intersection): {tiles_filtered_out}")
    print(f"City coverage: {len(tiles)}/{rows*cols} ({len(tiles)/(rows*cols)*100:.1f}%)")

    return tiles

def save_embeddings_to_geoparquet(tile_embeddings, city_info, acquisition_date, base_dir):
    """
    Save TerraMind embeddings to GeoParquet files (.gpq) with enhanced metadata including exact tile bounds.
    Now supports both aggregated and full patch embeddings based on SAVE_FULL_PATCH_EMBEDDINGS flag.
    """
    if not SAVE_TO_GEOPARQUET:
        print("⏭️  Skipping GeoParquet storage (SAVE_TO_GEOPARQUET=False)")
        return
        
    processing_date = datetime.now().date()
    
    # Prepare tile-level data for aggregated embeddings (always saved)
    tile_data_aggregated = []
    
    # NEW: Prepare tile-level data for full patch embeddings (optional)
    tile_data_full_patches = []
    
    for tile_emb in tile_embeddings:
        centroid = tile_emb['centroid']
        bounds = tile_emb['bounds']
        
        # Base record for both aggregated and full patch versions
        base_record = {
            # Core identification
            'tile_id': f"{city_info['city']}_{city_info['country']}_{tile_emb['tile_index']}",
            'city': city_info['city'],
            'country': city_info['country'],
            'continent': city_info['continent'],
            
            # Embedding metadata
            'embedding_model': 'terramind-v1-base',
            'embedding_model_version': '1.0',
            'tile_size_pixels': TILE_SIZE,
            'input_bands': 'coastal,blue,green,red,rededge1,rededge2,rededge3,nir,nir08,nir09,swir16,swir22',
            
            # Enhanced geospatial data with exact bounds
            'geometry': box(bounds[0], bounds[1], bounds[2], bounds[3]),  # Use polygon as main geometry
            'centroid_lon': centroid[0],
            'centroid_lat': centroid[1],
            'bounds_min_lon': bounds[0],
            'bounds_min_lat': bounds[1], 
            'bounds_max_lon': bounds[2],
            'bounds_max_lat': bounds[3],
            'tile_width_degrees': bounds[2] - bounds[0],
            'tile_height_degrees': bounds[3] - bounds[1],
            
            # Additional metadata
            'processing_date': processing_date.isoformat(),
            'acquisition_date': acquisition_date.isoformat() if acquisition_date else None,
        }
        
        # Add data quality metrics if available
        if 'data_quality' in city_info:
            base_record.update({
                'data_min_valid_percentage': city_info['data_quality']['min_valid_percentage'],
                'data_avg_valid_percentage': city_info['data_quality']['avg_valid_percentage'],
                'data_items_used': city_info['data_quality']['items_used'],
                'data_cloud_threshold': city_info['data_quality']['cloud_threshold'],
                'data_search_config': city_info['data_quality']['search_config']
            })
        
        # Aggregated embeddings record
        aggregated_record = base_record.copy()
        aggregated_record.update({
            'embedding_patch_aggregated': tile_emb['patch_embedding_aggregated'].tolist(),
            'embedding_dimension_patch': len(tile_emb['patch_embedding_aggregated']),
            'aggregation_method': 'mean',
            'embedding_type': 'aggregated'
        })
        tile_data_aggregated.append(aggregated_record)
        
        # NEW: Full patch embeddings record (if enabled)
        if SAVE_FULL_PATCH_EMBEDDINGS and 'patch_embeddings_full' in tile_emb:
            full_patch_record = base_record.copy()
            full_patch_embeddings = tile_emb['patch_embeddings_full']  # [196, 768]
            
            # Flatten the full patch embeddings for storage
            full_patch_record.update({
                'embedding_patches_full': full_patch_embeddings.flatten().tolist(),  # Flatten [196, 768] -> [150528]
                'embedding_dimension_total': full_patch_embeddings.size,  # 196 * 768 = 150528
                'embedding_shape_patches': full_patch_embeddings.shape[0],  # 196
                'embedding_shape_features': full_patch_embeddings.shape[1],  # 768
                'aggregation_method': 'none',
                'embedding_type': 'full_patches'
            })
            tile_data_full_patches.append(full_patch_record)
    
    # Create simple directory structure (not Hive-style partitioning!)
    country_clean = city_info['country'].replace(' ', '_').lower()
    city_clean = city_info['city'].replace(' ', '_').lower()
    
    # Save aggregated embeddings (always)
    tiles_aggregated_dir = os.path.join(base_dir, "tile_embeddings", country_clean)
    os.makedirs(tiles_aggregated_dir, exist_ok=True)
    tiles_aggregated_file = os.path.join(tiles_aggregated_dir, f"{city_clean}.gpq")
    
    # Create tile-level GeoDataFrame for aggregated embeddings
    tiles_aggregated_gdf = gpd.GeoDataFrame(tile_data_aggregated, crs="EPSG:4326")
    
    # FORCE ALL STRING COLUMNS TO BE STORED AS STRINGS, NOT CATEGORICAL
    string_columns = ['tile_id', 'city', 'country', 'continent', 'embedding_model', 
                      'embedding_model_version', 'aggregation_method', 'input_bands',
                      'processing_date', 'acquisition_date', 'data_search_config', 'embedding_type']
    for col in string_columns:
        if col in tiles_aggregated_gdf.columns:
            tiles_aggregated_gdf[col] = tiles_aggregated_gdf[col].astype(str)
    
    # Save aggregated embeddings without schema conflicts
    tiles_aggregated_gdf.to_parquet(
        tiles_aggregated_file, 
        index=False, 
        engine='pyarrow',
        compression='snappy',
        row_group_size=10000
    )
    
    print(f"✅ Saved {len(tile_embeddings)} aggregated tile embeddings to {tiles_aggregated_file}")
    
    # NEW: Save full patch embeddings (if enabled and data exists)
    if SAVE_FULL_PATCH_EMBEDDINGS and tile_data_full_patches:
        tiles_full_patch_dir = os.path.join(base_dir, "full_patch_embeddings", country_clean)
        os.makedirs(tiles_full_patch_dir, exist_ok=True)
        tiles_full_patch_file = os.path.join(tiles_full_patch_dir, f"{city_clean}.gpq")
        
        # Create tile-level GeoDataFrame for full patch embeddings
        tiles_full_patch_gdf = gpd.GeoDataFrame(tile_data_full_patches, crs="EPSG:4326")
        
        # FORCE ALL STRING COLUMNS TO BE STORED AS STRINGS, NOT CATEGORICAL
        for col in string_columns:
            if col in tiles_full_patch_gdf.columns:
                tiles_full_patch_gdf[col] = tiles_full_patch_gdf[col].astype(str)
        
        # Save full patch embeddings without schema conflicts
        tiles_full_patch_gdf.to_parquet(
            tiles_full_patch_file, 
            index=False, 
            engine='pyarrow',
            compression='snappy',
            row_group_size=1000  # Smaller row groups due to larger data size
        )
        
        print(f"✅ Saved {len(tile_data_full_patches)} full patch embeddings to {tiles_full_patch_file}")
        print(f"📊 Full patch embedding stats:")
        print(f"   Shape per tile: 196 patches × 768 features = 150,528 values")
        print(f"   Total values stored: {len(tile_data_full_patches) * 196 * 768:,}")
        
        # Calculate file sizes for comparison
        try:
            aggregated_size = os.path.getsize(tiles_aggregated_file) / 1024 / 1024  # MB
            full_patch_size = os.path.getsize(tiles_full_patch_file) / 1024 / 1024  # MB
            print(f"📁 File sizes:")
            print(f"   Aggregated embeddings: {aggregated_size:.1f} MB")
            print(f"   Full patch embeddings: {full_patch_size:.1f} MB")
            print(f"   Size ratio: {full_patch_size/aggregated_size:.1f}x larger")
        except:
            pass
    
    # Print summary statistics for aggregated embeddings
    print(f"📊 Aggregated tile bounds summary:")
    print(f"   Lon range: {tiles_aggregated_gdf['bounds_min_lon'].min():.6f} to {tiles_aggregated_gdf['bounds_max_lon'].max():.6f}")
    print(f"   Lat range: {tiles_aggregated_gdf['bounds_min_lat'].min():.6f} to {tiles_aggregated_gdf['bounds_max_lat'].max():.6f}")
    print(f"   Avg tile size: {tiles_aggregated_gdf['tile_width_degrees'].mean():.6f}° x {tiles_aggregated_gdf['tile_height_degrees'].mean():.6f}°")

def process_city(city_row, device):
    """
    Enhanced city processing pipeline with quality control, city boundary filtering, local caching,
    and optional full patch embedding storage.
    
    Args:
        city_row: Row from cities dataframe containing city info and geometry
        device: PyTorch device for model inference
    """
    print(f"\n{'='*60}")
    print(f"Processing city: {city_row['city']}, {city_row['country']}")
    print(f"🔧 Configuration:")
    print(f"   Tile size: {TILE_SIZE}×{TILE_SIZE} pixels")
    print(f"   Save aggregated embeddings: {SAVE_TO_GEOPARQUET}")
    print(f"   Save full patch embeddings: {SAVE_FULL_PATCH_EMBEDDINGS}")
    print(f"   Local data caching: {SAVE_RAW_DATA_LOCALLY}")
    
    try:
        base_dir = "embeddings/urban_embeddings_224_terramind_normalised"
        os.makedirs(base_dir, exist_ok=True)
        
        # Check if already processed (check both aggregated and optionally full patch)
        country_clean = city_row['country'].replace(' ', '_').lower()
        city_clean = city_row['city'].replace(' ', '_').lower()
        
        # Check aggregated embeddings
        tiles_aggregated_file = os.path.join(base_dir, "tile_embeddings", country_clean, f"{city_clean}.gpq")
        aggregated_exists = os.path.exists(tiles_aggregated_file)
        
        # Check full patch embeddings (if enabled)
        tiles_full_patch_file = os.path.join(base_dir, "full_patch_embeddings", country_clean, f"{city_clean}.gpq")
        full_patch_exists = os.path.exists(tiles_full_patch_file)
        
        
        
        
        city_geometry = city_row['geometry']
        
        # Create a buffer around the city's geometry for data loading
        buffer_distance = 0.01
        bbox_geom = city_geometry.buffer(buffer_distance)
        bbox = bbox_geom.bounds
        
        if isinstance(bbox, str):
            bbox_coords = [float(x.strip()) for x in bbox.split(',')]
        else:
            bbox_coords = list(bbox)
            
        cache_key = create_data_cache_key(bbox_coords, 10)
        print(f"✅ Cache key: {cache_key}")
        
        # Determine if we need to process
        skip_processing = False
        if aggregated_exists:
            if SAVE_FULL_PATCH_EMBEDDINGS:
                if full_patch_exists:
                    print(f"✅ City {city_row['city']} already processed (both aggregated and full patch embeddings exist). Skipping.")
                    skip_processing = True
                else:
                    print(f"⚠️  City {city_row['city']} has aggregated embeddings but missing full patch embeddings. Reprocessing...")
            else:
                print(f"✅ City {city_row['city']} already processed (aggregated embeddings exist). Skipping.")
                skip_processing = True
        
        if skip_processing:
            return
        
        # Get city geometry
        city_geometry = city_row['geometry']
        
        # Create a buffer around the city's geometry for data loading
        buffer_distance = 0.01
        bbox_geom = city_geometry.buffer(buffer_distance)
        bbox = bbox_geom.bounds
        
        print(f"📡 Loading STAC data with quality control for bbox: {bbox}")
        print(f"🏙️  City geometry type: {city_geometry.geom_type}")
        
        # Use enhanced data loading with quality control and caching
        stac_data = load_stac_data_with_quality_control(
            bbox, 
            min_valid_pixel_threshold=95.0  # Require at least 95% valid pixels
        )
        
        if stac_data is None:
            print(f"❌ Failed to obtain sufficient quality data for {city_row['city']}")
            return
        
        print(f"✅ Data quality: {stac_data['quality_metrics']}")
        
        # Create tiles filtered by city geometry
        tiles = create_non_overlapping_tiles(
            stac_data['full_data'], 
            stac_data['bbox'], 
            city_geometry,
            TILE_SIZE
            )
        
        print(f"✅ Created {len(tiles)} tiles intersecting city boundary (size {TILE_SIZE}×{TILE_SIZE})")
        
        # Extract embeddings with geometry information
        tile_embeddings = extract_tile_embeddings(
            tiles, device, stac_data['acquisition_date']
        )
        
        # Prepare city info with quality metrics
        city_info = {
            'city': city_row['city'],
            'country': city_row['country'],
            'continent': city_row['CONTINENT'],
            'data_quality': stac_data['quality_metrics'],
            'boundary_filtering': {
                'tiles_within_boundary': len(tiles)
            }
        }
        
        # Save to GeoParquet files with enhanced metadata
        if SAVE_TO_GEOPARQUET:
            save_embeddings_to_geoparquet(
                tile_embeddings, city_info, stac_data['acquisition_date'], base_dir
            )
        
        processing_summary = f"✅ Successfully processed {city_row['city']} with {len(tile_embeddings)} tiles within city boundary"
        if SAVE_FULL_PATCH_EMBEDDINGS:
            processing_summary += f" (saved both aggregated and full patch embeddings)"
        else:
            processing_summary += f" (saved aggregated embeddings only)"
        print(processing_summary)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error processing city {city_row['city']}: {str(e)}")
        import traceback
        print(traceback.format_exc())

def normalize_sentinel2_for_terramind(tile, band_order):
    """
    Normalize Sentinel-2 L2A data for TerraMind model input.
    
    Args:
        tile: numpy array of shape [bands, height, width] with raw reflectance values
        band_order: list of band names in the order they appear in the tile
    
    Returns:
        normalized tile tensor ready for TerraMind
    """
    # TerraMind's expected means and stds for S2L2A bands
    # Order: coastal, blue, green, red, rededge1, rededge2, rededge3, nir, nir08, nir09, swir16, swir22
    S2L2A_MEANS = torch.tensor([
        1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975, 
        2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648
    ]).reshape(-1, 1, 1)
    
    S2L2A_STDS = torch.tensor([
        2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 
        1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311
    ]).reshape(-1, 1, 1)
    
    # Expected band order for TerraMind
    expected_order = ["coastal", "blue", "green", "red", "rededge1", "rededge2", 
                      "rededge3", "nir", "nir08", "nir09", "swir16", "swir22"]
    
    # Verify band order matches
    if band_order != expected_order:
        # Reorder bands if necessary
        indices = [band_order.index(b) for b in expected_order]
        tile = tile[indices]
        print(f"⚠️ Reordered bands to match TerraMind expected order")
    
    # Convert to tensor
    tile_tensor = torch.from_numpy(tile).float()
    
    # Handle NaN/Inf values BEFORE normalization
    tile_tensor = torch.nan_to_num(tile_tensor, nan=0.0, posinf=10000.0, neginf=0.0)
    
    # Apply standardization: (x - mean) / std
    normalized_tile = (tile_tensor - S2L2A_MEANS) / S2L2A_STDS
    
    # Add batch dimension: [1, C, H, W]
    normalized_tile = normalized_tile.unsqueeze(0)
    
    return normalized_tile

def extract_tile_embeddings(tiles, device, acquisition_date=None, band_order=None):
    """
    Fixed version of extract_tile_embeddings with proper normalization.
    """
    tile_embeddings = []
    
    # Default band order if not specified
    if band_order is None:
        band_order = ["coastal", "blue", "green", "red", "rededge1", "rededge2", 
                      "rededge3", "nir", "nir08", "nir09", "swir16", "swir22"]
    
    # Initialize the model once
    from terratorch.registry import BACKBONE_REGISTRY
    model = BACKBONE_REGISTRY.build(
        'terramind_v1_base',
        pretrained=True,
        modalities=['S2L2A'],
    )
    model = model.to(device).eval()
    
    for i, (tile, bounds) in enumerate(tqdm(tiles, desc="Extracting encoder features")):
        # CRITICAL: Apply proper normalization
        tile_tensor = normalize_sentinel2_for_terramind(tile, band_order)
        tile_tensor = tile_tensor.to(device)
        
        with torch.no_grad():
            # Extract features
            features_list = model(tile_tensor)
            encoder_features = features_list[-1]  # [1, 196, 768]
        
        # Aggregate embeddings
        aggregated_embedding = aggregate_patch_embeddings(encoder_features, method='mean')
        
        tile_embedding = {
            'tile_index': i,
            'patch_embedding_aggregated': aggregated_embedding.cpu().numpy().flatten(),
            'bounds': bounds,
            'centroid': ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        }
        
        # Optionally add full patch embeddings
        if SAVE_FULL_PATCH_EMBEDDINGS:
            full_patch_embeddings = encoder_features.squeeze(0).cpu().numpy()
            tile_embedding['patch_embeddings_full'] = full_patch_embeddings
        
        tile_embeddings.append(tile_embedding)
    
    return tile_embeddings

# Main execution
if __name__ == "__main__":
    
    try:
        # Call this before processing cities
        increase_file_limits()

        # Load city data
        gdf_with_cities = gpd.read_file('data/urban_areas_TOP6.geojson')
        gdf_with_cities = gdf_with_cities.sort_values(by="population", ascending=True)
        
        gdf_with_cities = gdf_with_cities[gdf_with_cities['city'] == 'London']
        
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"{'='*80}")
        print(f"🚀 URBAN EMBEDDINGS EXTRACTION WITH TERRAMIND")
        print(f"{'='*80}")
        print(f"📊 Configuration:")
        print(f"   Cities to process: {len(gdf_with_cities)}")
        print(f"   Tile size: {TILE_SIZE}×{TILE_SIZE} pixels")
        print(f"   Device: {device}")
        print(f"   Save aggregated embeddings: {SAVE_TO_GEOPARQUET}")
        print(f"   Save full patch embeddings: {SAVE_FULL_PATCH_EMBEDDINGS}")
        print(f"   Cache raw STAC data locally: {SAVE_RAW_DATA_LOCALLY}")
        
        cluster = LocalCluster(memory_limit='4GB')
        
        adaptive = Adaptive(cluster, interval="5s", minimum=1, maximum=12)
        client = Client(cluster)
        
        print(f"🔗 Dask dashboard available at: {client.dashboard_link}")
        # print(f"Cluster configuration: {cluster}")

        # Process cities with better error handling
        for idx, (_, city_row) in enumerate(tqdm(gdf_with_cities.iterrows(), 
                              total=len(gdf_with_cities), 
                              desc="Processing cities")):
            print(f"\n🏙️  Starting city {idx+1}/{len(gdf_with_cities)}: {city_row['city']}, {city_row['country']}")
            try:
                process_city(city_row, device)
                
                # Force garbage collection after each city
                import gc
                gc.collect()
                        
            except Exception as e:
                print(f"❌ Error processing city {city_row['city']}: {str(e)}")
                continue

        print(f"\n🎉 Finished processing all cities!")
        print(f"📁 Output directories:")
        print(f"   Aggregated embeddings: embeddings/urban_embeddings_224_terramind/tile_embeddings/")
        if SAVE_FULL_PATCH_EMBEDDINGS:
            print(f"   Full patch embeddings: embeddings/urban_embeddings_224_terramind/full_patch_embeddings/")
        if SAVE_RAW_DATA_LOCALLY:
            print(f"   Cached STAC data: {LOCAL_DATA_DIR}/")

    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")
        raise