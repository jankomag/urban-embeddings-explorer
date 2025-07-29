from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from models import LocationData, SimplifiedSimilarityResponse, SimplifiedStatsResponse, ConfigResponse, TileBoundsResponse
import numpy as np
from typing import List
import umap
import asyncio
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import pandas as pd
from glob import glob
import math

# Load environment variables
load_dotenv()

app = FastAPI(title="Simplified Embeddings Explorer", version="2.0.0")

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
aggregated_cache = {}  # Cache for mean-aggregated embeddings
similarity_results_cache = {}  # Cache for similarity results
executor = ThreadPoolExecutor(max_workers=2)

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

def get_or_compute_aggregated_embedding(location_id: int) -> np.ndarray:
    """
    Get or compute mean-aggregated embedding for a location.
    Results are cached for efficiency.
    """
    cache_key = f"{location_id}_mean"
    
    # Check cache first
    if cache_key in aggregated_cache:
        return aggregated_cache[cache_key]
    
    # Get full patch embeddings
    if location_id not in embeddings_data['embeddings_dict']:
        raise ValueError(f"Location {location_id} not found")
    
    full_patches = embeddings_data['embeddings_dict'][location_id]['patch_embeddings_full']
    if full_patches is None:
        raise ValueError(f"No full patch embeddings for location {location_id}")
    
    # Reshape from flattened to [196, 768] and compute mean
    patches = np.array(full_patches).reshape(196, 768)
    aggregated = patches.mean(axis=0)
    
    # Cache result
    aggregated_cache[cache_key] = aggregated
    
    return aggregated

def calculate_cosine_similarity(target_location_id: int, candidate_location_id: int) -> float:
    """Calculate simple cosine similarity using mean-aggregated embeddings."""
    try:
        target_emb = get_or_compute_aggregated_embedding(target_location_id)
        candidate_emb = get_or_compute_aggregated_embedding(candidate_location_id)
        
        # Calculate cosine similarity
        dot_product = np.dot(target_emb, candidate_emb)
        norm_target = np.linalg.norm(target_emb)
        norm_candidate = np.linalg.norm(candidate_emb)
        
        if norm_target == 0 or norm_candidate == 0:
            return 0.0
        
        return float(dot_product / (norm_target * norm_candidate))
    except Exception as e:
        print(f"Error in cosine similarity calculation: {e}")
        return 0.0

def load_embeddings_from_files():
    """Load only full patch embeddings from local files."""
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
                
                # Store embedding data
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
    return {"message": "Simplified Embeddings Explorer API v2.0", "status": "running", "similarity_method": "cosine"}

@app.get("/api/similarity/{location_id}", response_model=SimplifiedSimilarityResponse)
async def find_similar_locations(
    location_id: int, 
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    limit: int = Query(6, ge=1, le=20, description="Number of results to return")
):
    """Find most similar locations using cosine similarity with pagination"""
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
    
    # Check if we have cached similarities for this location
    cache_key = f"{location_id}_cosine"
    cached_similarities = similarity_results_cache.get(cache_key)
    
    if cached_similarities is None:
        print(f"Computing cosine similarities for location {location_id}...")
        
        # Calculate similarities for all locations
        similarities = []
        
        for location in embeddings_data['locations']:
            if location['id'] == location_id:
                continue  # Skip target location
                
            try:
                similarity = calculate_cosine_similarity(location_id, location['id'])
                similarities.append((location, similarity))
            except Exception as e:
                print(f"Error calculating similarity for location {location['id']}: {e}")
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Cache the complete sorted results
        similarity_results_cache[cache_key] = similarities
        print(f"Cached {len(similarities)} similarity results for location {location_id}")
        
        # Print some diagnostics
        if similarities:
            scores = [s[1] for s in similarities]
            print(f"   📊 Similarity range: {min(scores):.4f} - {max(scores):.4f}")
            print(f"   📈 Mean similarity: {np.mean(scores):.4f}")
        
        cached_similarities = similarities
    else:
        print(f"Using cached similarities for location {location_id}")
    
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
    
    # Add pagination metadata
    has_more = (offset + limit) < total_results
    next_offset = offset + limit if has_more else None
    
    return {
        "target_location_id": location_id,
        "similar_locations": similar_locations,
        "method_used": "cosine_similarity",
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
    
    # Get embedding info
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

@app.get("/api/stats", response_model=SimplifiedStatsResponse)
async def get_stats():
    """Get basic statistics about the dataset"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    locations = embeddings_data['locations']
    countries = list(set(loc['country'] for loc in locations))
    continents = list(set(loc['continent'] for loc in locations))
    
    embedding_dim = 768  # Base dimension
    
    return {
        "total_locations": len(locations),
        "countries_count": len(countries),
        "continents_count": len(continents),
        "embedding_dimension": embedding_dim,
        "locations_with_full_patches": len(locations),
        "countries": sorted(countries),
        "continents": sorted(continents),
        "similarity_method": "cosine_similarity"
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
                aggregated_embedding = get_or_compute_aggregated_embedding(location['id'])
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