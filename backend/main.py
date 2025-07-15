from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from models import LocationData, EnhancedSimilarityResponse, SimilarityMethodsResponse, EnhancedStatsResponse, ConfigResponse, TileBoundsResponse
import numpy as np
from typing import List, Literal
import umap
import asyncio
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import pandas as pd
from glob import glob
from shapely.geometry import Point, Polygon
import math
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum

# Load environment variables
load_dotenv()

app = FastAPI(title="Simplified Embeddings Explorer", version="3.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded data
embeddings_data = None
umap_cache = None
aggregated_cache = {}  # Cache for on-demand aggregated embeddings
similarity_results_cache = {}  # Cache for complete similarity results
executor = ThreadPoolExecutor(max_workers=2)

# Similarity method definitions
class SimilarityMethod(str, Enum):
    SIMPLE_COSINE = "simple_cosine"           # Fast: Mean-aggregated on-demand
    ATTENTION_WEIGHTED = "attention_weighted"  # Medium: Better visual features
    SPATIAL_PYRAMID = "spatial_pyramid"       # Medium: Preserves spatial structure  
    PATCH_CHAMFER = "patch_chamfer"           # Slow: Best visual similarity
    PATCH_HAUSDORFF = "patch_hausdorff"       # Slow: Robust visual similarity
    SPATIAL_AWARE = "spatial_aware"           # Medium: Spatial + visual balance

# Similarity method configurations
SIMILARITY_CONFIGS = {
    SimilarityMethod.SIMPLE_COSINE: {
        "name": "Simple Cosine (Fast)",
        "description": "Mean-aggregated embeddings with cosine similarity",
        "speed": "Fast",
        "quality": "Medium",
        "requires_full_patches": False  # We compute on-demand
    },
    SimilarityMethod.ATTENTION_WEIGHTED: {
        "name": "Attention Weighted (Recommended)",
        "description": "Emphasizes visually important patches using attention mechanism",
        "speed": "Medium",
        "quality": "High",
        "requires_full_patches": True
    },
    SimilarityMethod.SPATIAL_PYRAMID: {
        "name": "Spatial Pyramid",
        "description": "Preserves spatial structure with multi-scale representation",
        "speed": "Medium", 
        "quality": "High",
        "requires_full_patches": True
    },
    SimilarityMethod.PATCH_CHAMFER: {
        "name": "Patch-Level Chamfer (Best Visual)",
        "description": "Robust patch-level comparison using Chamfer distance",
        "speed": "Slow",
        "quality": "Highest",
        "requires_full_patches": True
    },
    SimilarityMethod.PATCH_HAUSDORFF: {
        "name": "Patch-Level Hausdorff",
        "description": "Strict patch-level comparison using modified Hausdorff distance",
        "speed": "Slow", 
        "quality": "Highest",
        "requires_full_patches": True
    },
    SimilarityMethod.SPATIAL_AWARE: {
        "name": "Spatial Aware",
        "description": "Balances spatial arrangement and visual features",
        "speed": "Medium",
        "quality": "High",
        "requires_full_patches": True
    }
}

def is_valid_coordinate(lon, lat):
    """Check if coordinates are valid (not NaN, within valid ranges)"""
    if pd.isna(lon) or pd.isna(lat):
        return False
    if math.isnan(lon) or math.isnan(lat):
        return False
    if lon < -180 or lon > 180:
        return False
    if lat < -90 or lat > 90:
        return False
    return True

def get_or_compute_aggregated_embedding(location_id: int, method: str = 'mean') -> np.ndarray:
    """
    Get or compute aggregated embedding for a location using specified method.
    Results are cached for efficiency.
    """
    cache_key = f"{location_id}_{method}"
    
    # Check cache first
    if cache_key in aggregated_cache:
        return aggregated_cache[cache_key]
    
    # Get full patch embeddings
    if location_id not in embeddings_data['embeddings_dict']:
        raise ValueError(f"Location {location_id} not found")
    
    full_patches = embeddings_data['embeddings_dict'][location_id]['patch_embeddings_full']
    if full_patches is None:
        raise ValueError(f"No full patch embeddings for location {location_id}")
    
    # Reshape from flattened to [196, 768]
    patches = np.array(full_patches).reshape(196, 768)
    
    # Compute aggregation
    if method == 'mean':
        aggregated = patches.mean(axis=0)
    elif method == 'attention_weighted':
        # Use attention mechanism to weight patches
        attention_scores = np.linalg.norm(patches, axis=1, keepdims=True)  # [196, 1]
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))  # Softmax
        weighted_features = patches * attention_weights
        aggregated = weighted_features.sum(axis=0)  # [768]
    elif method == 'spatial_pyramid':
        # Create spatial pyramid representation
        spatial_features = patches.reshape(14, 14, 768)
        
        # Global average
        global_avg = spatial_features.mean(axis=(0, 1))  # [768]
        
        # Quadrant averages (4 regions)
        quad1 = spatial_features[:7, :7, :].mean(axis=(0, 1))    # Top-left
        quad2 = spatial_features[:7, 7:, :].mean(axis=(0, 1))    # Top-right  
        quad3 = spatial_features[7:, :7, :].mean(axis=(0, 1))    # Bottom-left
        quad4 = spatial_features[7:, 7:, :].mean(axis=(0, 1))    # Bottom-right
        
        # Concatenate all levels
        aggregated = np.concatenate([global_avg, quad1, quad2, quad3, quad4])  # [768*5]
    else:
        # Default to mean
        aggregated = patches.mean(axis=0)
    
    # Cache result
    aggregated_cache[cache_key] = aggregated
    
    return aggregated

def calculate_patch_level_similarity(target_patches: np.ndarray, candidate_patches: np.ndarray, method: str) -> float:
    """Calculate similarity between tiles using patch-level features."""
    
    if method == 'chamfer':
        # Chamfer distance - more robust than Hausdorff
        distances = cdist(target_patches, candidate_patches, metric='cosine')
        
        # Average of minimum distances in both directions
        chamfer_dist = (distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2
        return max(0, 1 - chamfer_dist)
    
    elif method == 'hausdorff':
        # Modified Hausdorff distance for visual similarity
        distances = cdist(target_patches, candidate_patches, metric='cosine')
        
        # Hausdorff distance (max of min distances in both directions)
        min_dist_target_to_candidate = distances.min(axis=1).max()
        min_dist_candidate_to_target = distances.min(axis=0).max()
        hausdorff_dist = max(min_dist_target_to_candidate, min_dist_candidate_to_target)
        
        # Convert to similarity (0 to 1, higher is better)
        return max(0, 1 - hausdorff_dist)
    
    elif method == 'spatial_aware':
        # Consider spatial arrangement of patches (14x14 grid)
        target_spatial = target_patches.reshape(14, 14, 768)
        candidate_spatial = candidate_patches.reshape(14, 14, 768)
        
        # Compute similarity with spatial weights
        similarities = []
        for i in range(14):
            for j in range(14):
                # Cosine similarity between corresponding spatial patches
                sim = cosine_similarity(
                    target_spatial[i, j:j+1], 
                    candidate_spatial[i, j:j+1]
                )[0, 0]
                
                # Weight by distance from center (center patches more important)
                center_weight = 1.0 / (1 + 0.3 * ((i-7)**2 + (j-7)**2)**0.5)
                similarities.append(sim * center_weight)
        
        return np.mean(similarities)
    
    else:
        raise ValueError(f"Unknown patch-level method: {method}")

def calculate_similarity_by_method(target_location_id: int, candidate_location_id: int, method: SimilarityMethod) -> float:
    """Calculate similarity using the specified method."""
    
    if method == SimilarityMethod.SIMPLE_COSINE:
        # Use mean-aggregated embeddings computed on-demand
        try:
            target_emb = get_or_compute_aggregated_embedding(target_location_id, 'mean')
            candidate_emb = get_or_compute_aggregated_embedding(candidate_location_id, 'mean')
            
            # Calculate cosine similarity
            dot_product = np.dot(target_emb, candidate_emb)
            norm_target = np.linalg.norm(target_emb)
            norm_candidate = np.linalg.norm(candidate_emb)
            
            if norm_target == 0 or norm_candidate == 0:
                return 0.0
            
            return float(dot_product / (norm_target * norm_candidate))
        except Exception as e:
            print(f"Error in simple cosine similarity: {e}")
            return 0.0
    
    elif method in [SimilarityMethod.ATTENTION_WEIGHTED, SimilarityMethod.SPATIAL_PYRAMID]:
        # Use enhanced aggregation methods
        try:
            method_name = method.value
            target_agg = get_or_compute_aggregated_embedding(target_location_id, method_name)
            candidate_agg = get_or_compute_aggregated_embedding(candidate_location_id, method_name)
            
            # Calculate cosine similarity on aggregated features
            dot_product = np.dot(target_agg, candidate_agg)
            norm_target = np.linalg.norm(target_agg)
            norm_candidate = np.linalg.norm(candidate_agg)
            
            if norm_target == 0 or norm_candidate == 0:
                return 0.0
            
            return float(dot_product / (norm_target * norm_candidate))
        except Exception as e:
            print(f"Error in {method.value} similarity: {e}")
            # Fallback to simple cosine
            return calculate_similarity_by_method(target_location_id, candidate_location_id, SimilarityMethod.SIMPLE_COSINE)
    
    elif method in [SimilarityMethod.PATCH_CHAMFER, SimilarityMethod.PATCH_HAUSDORFF, SimilarityMethod.SPATIAL_AWARE]:
        # Use patch-level comparison methods
        try:
            target_data = embeddings_data['embeddings_dict'][target_location_id]
            candidate_data = embeddings_data['embeddings_dict'][candidate_location_id]
            
            target_patches = np.array(target_data['patch_embeddings_full']).reshape(196, 768)
            candidate_patches = np.array(candidate_data['patch_embeddings_full']).reshape(196, 768)
            
            # Extract method name (remove 'patch_' prefix)
            patch_method = method.value.replace('patch_', '')
            
            return calculate_patch_level_similarity(target_patches, candidate_patches, patch_method)
        except Exception as e:
            print(f"Error in {method.value} similarity: {e}")
            # Fallback to simple cosine
            return calculate_similarity_by_method(target_location_id, candidate_location_id, SimilarityMethod.SIMPLE_COSINE)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def load_embeddings_from_files():
    """Simplified version that loads only full patch embeddings."""
    try:
        print("Loading full patch embeddings from local files...")
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..','terramind', 'embeddings', 'urban_embeddings_224_terramind'))
        
        # Load from full_patch_embeddings directory only
        full_patch_pattern = os.path.join(base_dir, "full_patch_embeddings", "*", "*.gpq")
        full_patch_files = glob(full_patch_pattern)
        
        print(f"Found {len(full_patch_files)} full patch embedding files")
        
        if not full_patch_files:
            raise Exception("No full patch embedding files found")
        
        # Load full patch embeddings
        full_patch_gdfs = []
        for file in full_patch_files:
            try:
                gdf = gpd.read_parquet(file)
                for col in gdf.columns:
                    if gdf[col].dtype.name == 'category':
                        gdf[col] = gdf[col].astype(str)
                full_patch_gdfs.append(gdf)
                print(f"Loaded {len(gdf)} full patch tiles from {os.path.basename(file)}")
            except Exception as e:
                print(f"Error loading full patch file {file}: {e}")
        
        if not full_patch_gdfs:
            raise Exception("No valid full patch embedding data could be loaded")
        
        # Combine full patch embeddings
        all_full_patches = pd.concat(full_patch_gdfs, ignore_index=True)
        print(f"Combined full patch embeddings: {len(all_full_patches)} records")
        
        # Process the data
        locations_data = []
        embeddings_dict = {}
        skipped_count = 0
        
        for idx, row in all_full_patches.iterrows():
            try:
                # Extract coordinates
                longitude = row.get('centroid_lon') or row.get('longitude')
                latitude = row.get('centroid_lat') or row.get('latitude')
                
                if longitude is None or latitude is None or not is_valid_coordinate(longitude, latitude):
                    skipped_count += 1
                    continue
                
                # Get full patch embedding
                full_embedding = row.get('embedding_patches_full')
                if full_embedding is None:
                    skipped_count += 1
                    continue
                
                # Create unique ID
                tile_id = row.get('tile_id')
                if tile_id:
                    unique_id = hash(str(tile_id)) % (10**9)
                    if unique_id < 0:
                        unique_id = -unique_id
                else:
                    unique_id = idx
                
                # Extract tile bounds if available
                tile_bounds = None
                if all(col in row for col in ['bounds_min_lon', 'bounds_min_lat', 'bounds_max_lon', 'bounds_max_lat']):
                    min_lon = row['bounds_min_lon']
                    min_lat = row['bounds_min_lat']
                    max_lon = row['bounds_max_lon']
                    max_lat = row['bounds_max_lat']
                    
                    if (not pd.isna(min_lon) and not pd.isna(min_lat) and 
                        not pd.isna(max_lon) and not pd.isna(max_lat)):
                        tile_bounds = [
                            [float(min_lon), float(max_lat)],  # top-left
                            [float(max_lon), float(max_lat)],  # top-right
                            [float(max_lon), float(min_lat)],  # bottom-right
                            [float(min_lon), float(min_lat)],  # bottom-left
                            [float(min_lon), float(max_lat)]   # close polygon
                        ]
                
                # Create location data
                location_data = {
                    'id': int(unique_id),
                    'city': str(row.get('city', 'Unknown')),
                    'country': str(row.get('country', 'Unknown')),
                    'continent': str(row.get('continent', 'Unknown')),
                    'longitude': float(longitude),
                    'latitude': float(latitude),
                    'date': str(row.get('acquisition_date')) if row.get('acquisition_date') else None,
                    'tile_bounds': tile_bounds
                }
                
                # Store embedding data (only full patches now)
                embedding_data = {
                    'patch_embeddings_full': full_embedding
                }
                
                locations_data.append(location_data)
                embeddings_dict[int(unique_id)] = embedding_data
                
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                skipped_count += 1
                continue
        
        if not locations_data:
            raise Exception("No valid data could be processed from files")
        
        print(f"Successfully processed {len(locations_data)} records")
        print(f"Skipped {skipped_count} records due to invalid data")
        print(f"All records have full patch embeddings: 196×768 = 150,528 values each")
        
        return {
            'locations': locations_data,
            'embeddings_dict': embeddings_dict
        }
    
    except Exception as e:
        print(f"Error loading data from files: {str(e)}")
        import traceback
        print("Detailed error:")
        print(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_event():
    """Load embeddings data on startup"""
    global embeddings_data
    try:
        print("Loading embeddings data from files...")
        embeddings_data = load_embeddings_from_files()
        print(f"Successfully loaded {len(embeddings_data['locations'])} locations")
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        embeddings_data = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Simplified Embeddings Explorer API v3.1", "status": "running"}

@app.get("/api/similarity-methods", response_model=SimilarityMethodsResponse)
async def get_similarity_methods():
    """Get available similarity methods with their configurations"""
    return {
        "methods": [
            {
                "id": method.value,
                "config": config
            }
            for method, config in SIMILARITY_CONFIGS.items()
        ],
        "recommended": SimilarityMethod.ATTENTION_WEIGHTED.value,
        "fastest": SimilarityMethod.SIMPLE_COSINE.value,
        "best_quality": SimilarityMethod.PATCH_CHAMFER.value
    }

@app.get("/api/similarity/{location_id}", response_model=EnhancedSimilarityResponse)
async def find_similar_locations(
    location_id: int, 
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    limit: int = Query(6, ge=1, le=20, description="Number of results to return"),
    method: SimilarityMethod = Query(SimilarityMethod.ATTENTION_WEIGHTED, description="Similarity calculation method")
):
    """Find most similar locations using specified method with pagination"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    # Find target location
    target_location = None
    
    for location in embeddings_data['locations']:
        if location['id'] == location_id:
            target_location = location
            break
    
    if target_location is None:
        raise HTTPException(status_code=404, detail="Location not found")
    
    # Check if we have cached similarities for this location and method
    cache_key = f"{location_id}_{method.value}"
    cached_similarities = similarity_results_cache.get(cache_key)
    
    if cached_similarities is None:
        print(f"Computing similarities for location {location_id} using {method.value}...")
        
        # Calculate similarities for all locations
        similarities = []
        
        for location in embeddings_data['locations']:
            if location['id'] == location_id:
                continue  # Skip target location
                
            try:
                similarity = calculate_similarity_by_method(location_id, location['id'], method)
                similarities.append((location, similarity))
            except Exception as e:
                print(f"Error calculating similarity for location {location['id']}: {e}")
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Cache the complete sorted results
        similarity_results_cache[cache_key] = similarities
        print(f"Cached {len(similarities)} similarity results for location {location_id}")
        
        cached_similarities = similarities
    else:
        print(f"Using cached similarities for location {location_id} using {method.value}")
    
    # Apply pagination
    total_results = len(cached_similarities)
    paginated_similarities = cached_similarities[offset:offset + limit]
    
    # Format response
    similar_locations = []
    for location, sim_score in paginated_similarities:
        # Validate coordinates before adding to response
        if not is_valid_coordinate(location['longitude'], location['latitude']):
            continue
            
        similar_location = {
            'id': location['id'],
            'city': location['city'],
            'country': location['country'],
            'continent': location['continent'],
            'longitude': location['longitude'],
            'latitude': location['latitude'],
            'date': location['date'],
            "similarity_score": float(sim_score)
        }
        similar_locations.append(similar_location)
    
    method_config = SIMILARITY_CONFIGS[method]
    
    # Add pagination metadata
    has_more = (offset + limit) < total_results
    next_offset = offset + limit if has_more else None
    
    return {
        "target_location_id": location_id,
        "similar_locations": similar_locations,
        "method_used": method.value,
        "method_config": method_config,
        "pagination": {
            "offset": offset,
            "limit": limit,
            "total_results": total_results,
            "has_more": has_more,
            "next_offset": next_offset,
            "returned_count": len(similar_locations)
        }
    }

@app.get("/api/locations", response_model=List[LocationData])
async def get_locations():
    """Get all locations with their basic info"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    locations = []
    for location_data in embeddings_data['locations']:
        if not is_valid_coordinate(location_data['longitude'], location_data['latitude']):
            continue
            
        location = LocationData(
            id=location_data['id'],
            city=location_data['city'],
            country=location_data['country'],
            continent=location_data['continent'],
            longitude=location_data['longitude'],
            latitude=location_data['latitude'],
            date=location_data['date']
        )
        locations.append(location)
    
    return locations

@app.get("/api/locations/{location_id}")
async def get_location(location_id: int):
    """Get detailed info for a specific location"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    # Find location by ID
    location_data = None
    for loc in embeddings_data['locations']:
        if loc['id'] == location_id:
            location_data = loc
            break
    
    if location_data is None:
        raise HTTPException(status_code=404, detail="Location not found")
    
    # Get embedding info - we now compute aggregated on-demand
    embedding_shape = (196, 768)  # Full patches shape
    
    response_data = {
        **location_data,
        "embedding_shape": embedding_shape,
        "has_full_patches": True
    }
    
    # Remove tile_bounds from response as it's not part of the model
    if 'tile_bounds' in response_data:
        del response_data['tile_bounds']
    
    return response_data

@app.get("/api/tile-bounds/{location_id}", response_model=TileBoundsResponse)
async def get_tile_bounds(location_id: int):
    """Get exact tile boundary coordinates for a location if available"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    # Find location by ID
    location_data = None
    for loc in embeddings_data['locations']:
        if loc['id'] == location_id:
            location_data = loc
            break
    
    if location_data is None:
        raise HTTPException(status_code=404, detail="Location not found")
    
    return {
        "location_id": location_id,
        "city": location_data['city'],
        "country": location_data['country'],
        "tile_bounds": location_data.get('tile_bounds'),
        "has_exact_bounds": location_data.get('tile_bounds') is not None
    }

@app.get("/api/stats", response_model=EnhancedStatsResponse)
async def get_stats():
    """Get basic statistics about the dataset"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    locations = embeddings_data['locations']
    countries = list(set(loc['country'] for loc in locations))
    continents = list(set(loc['continent'] for loc in locations))
    
    # All locations have full patches in this simplified version
    embedding_dim = 768  # Base dimension before aggregation
    
    return {
        "total_locations": len(locations),
        "countries_count": len(countries),
        "continents_count": len(continents),
        "embedding_dimension": embedding_dim,
        "locations_with_full_patches": len(locations),  # All have full patches
        "countries": sorted(countries),
        "continents": sorted(continents),
        "available_similarity_methods": len(SIMILARITY_CONFIGS)
    }

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get configuration including Mapbox token"""
    return {
        "mapbox_token": os.getenv("MAPBOX_TOKEN", "")
    }

@app.get("/api/umap")
async def get_umap_embeddings():
    """Get UMAP 2D embeddings for all locations using mean-aggregated embeddings"""
    global umap_cache
    
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    # Return cached result if available
    if umap_cache is not None:
        print("Returning cached UMAP data")
        return umap_cache
    
    try:
        print("Computing UMAP embeddings from full patch data...")
        
        # Get mean-aggregated embeddings for UMAP computation
        embeddings_list = []
        valid_locations = []
        
        for location in embeddings_data['locations']:
            try:
                # Compute mean aggregation on-demand
                aggregated_embedding = get_or_compute_aggregated_embedding(location['id'], 'mean')
                embeddings_list.append(aggregated_embedding)
                valid_locations.append(location)
            except Exception as e:
                print(f"Error processing location {location['id']} for UMAP: {e}")
                continue
        
        if not embeddings_list:
            raise HTTPException(status_code=500, detail="No valid embeddings found for UMAP")
        
        # Run UMAP computation in thread pool
        loop = asyncio.get_event_loop()
        umap_embeddings = await loop.run_in_executor(
            executor, 
            compute_umap_embeddings, 
            embeddings_list
        )
        
        # Format response data
        umap_points = []
        for i, umap_coord in enumerate(umap_embeddings):
            location_data = valid_locations[i]
            
            # Validate coordinates before adding to response
            if not is_valid_coordinate(location_data['longitude'], location_data['latitude']):
                continue
            
            umap_points.append({
                "location_id": location_data['id'],
                "x": float(umap_coord[0]),
                "y": float(umap_coord[1]),
                "city": location_data['city'],
                "country": location_data['country'],
                "continent": location_data['continent'],
                "longitude": location_data['longitude'],
                "latitude": location_data['latitude'],
                "date": location_data['date']
            })
        
        result = {
            "umap_points": umap_points,
            "total_points": len(umap_points)
        }
        
        umap_cache = result
        
        print(f"UMAP computation successful: {len(umap_points)} points")
        return result
        
    except Exception as e:
        print(f"Error computing UMAP: {str(e)}")
        import traceback
        print("Detailed error:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to compute UMAP embeddings: {str(e)}")

def compute_umap_embeddings(embeddings_array):
    """Compute UMAP embeddings in a separate thread"""
    print("Computing UMAP embeddings...")
    
    embeddings_matrix = np.array(embeddings_array)
    n_samples = len(embeddings_matrix)
    
    # Adaptive parameters based on dataset size
    if n_samples > 10000:
        n_neighbors = min(50, n_samples // 100)
        min_dist = 0.5
    elif n_samples > 5000:
        n_neighbors = min(30, n_samples // 50)
        min_dist = 0.3
    else:
        n_neighbors = min(15, max(5, n_samples // 20))
        min_dist = 0.1
    
    print(f"UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    # Create UMAP reducer
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    
    # Fit and transform the embeddings
    umap_embeddings = reducer.fit_transform(embeddings_matrix)
    
    print("UMAP computation completed")
    return umap_embeddings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)