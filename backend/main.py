from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from models import LocationData
import numpy as np
from typing import List
import umap
import asyncio
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import pandas as pd
from glob import glob

# Load environment variables
load_dotenv()

app = FastAPI(title="Embeddings Explorer", version="2.0.0")

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
executor = ThreadPoolExecutor(max_workers=2)

def load_all_embeddings(base_dir):
    """
    Load all embeddings from all countries into a single GeoDataFrame.
    
    Args:
        base_dir: Base directory containing the embeddings
        
    Returns:
        GeoDataFrame with all tile embeddings
    """
    
    # Find all .gpq files - updated pattern for new directory structure
    pattern = os.path.join(base_dir, "tile_embeddings", "*", "*.gpq")
    files = glob(pattern)
    
    print(f"Found {len(files)} embedding files")
    
    if not files:
        print(f"No files found in {pattern}")
        return None
    
    # Load and concatenate all files
    gdfs = []
    for file in files:
        try:
            gdf = gpd.read_parquet(file)  # Remove engine parameter
            # Convert any categorical columns to string to avoid schema conflicts
            for col in gdf.columns:
                if gdf[col].dtype.name == 'category':
                    gdf[col] = gdf[col].astype(str)
            gdfs.append(gdf)
            print(f"Loaded {len(gdf)} tiles from {os.path.basename(file)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not gdfs:
        return None
        
    # Combine all into single GeoDataFrame
    all_embeddings = pd.concat(gdfs, ignore_index=True)
    
    print(f"\nTotal: {len(all_embeddings)} tiles from {len(gdfs)} cities")
    print(f"Countries: {sorted(all_embeddings['country'].unique())}")
    print(f"Cities: {len(all_embeddings['city'].unique())}")
    
    return all_embeddings

def load_embeddings_from_files():
    """Load embeddings from local files."""
    try:
        print("Loading embeddings from local files...")
        
        # Get base directory from environment or use default path structure
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..','terramind', 'embeddings', 'urban_embeddings_224_terramind'))
        
        # Load all embeddings
        gdf = load_all_embeddings(base_dir)
        
        if gdf is None or len(gdf) == 0:
            raise Exception("No embedding data could be loaded from files")
        
        print(f"Successfully loaded {len(gdf)} records from files")
        print(f"Available columns: {list(gdf.columns)}")
        
        # Process the GeoDataFrame
        embeddings = []
        locations_data = []
        
        for idx, row in gdf.iterrows():
            try:
                # Extract geometry coordinates (use lon/lat columns if available, otherwise geometry)
                if 'lon' in row and 'lat' in row:
                    longitude = float(row['lon'])
                    latitude = float(row['lat'])
                elif hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                    longitude = float(row.geometry.x)
                    latitude = float(row.geometry.y)
                else:
                    # Handle other geometry types (centroid)
                    centroid = row.geometry.centroid
                    longitude = float(centroid.x)
                    latitude = float(centroid.y)
                
                # Process embedding - look for your TerraMind embedding
                embedding = None
                
                if 'embedding_patch_aggregated' in row and row['embedding_patch_aggregated'] is not None:
                    embedding = np.array(row['embedding_patch_aggregated'], dtype=np.float32)
                
                # Skip if no embedding available
                if embedding is None:
                    continue
                
                # Create unique ID from tile_id or generate one
                if 'tile_id' in row and row['tile_id'] is not None:
                    # Use hash of tile_id to create numeric ID
                    unique_id = hash(str(row['tile_id'])) % (10**9)
                else:
                    unique_id = idx
                
                # Create location data
                location_data = {
                    'id': unique_id,
                    'city': str(row.get('city', 'Unknown')),
                    'country': str(row.get('country', 'Unknown')),
                    'continent': str(row.get('continent', 'Unknown')),
                    'longitude': longitude,
                    'latitude': latitude,
                    'date': str(row.get('acquisition_date', row.get('date', None))) if row.get('acquisition_date', row.get('date', None)) is not None else None
                }
                
                locations_data.append(location_data)
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        if not locations_data:
            print("ERROR: No valid data could be processed from files")
            print(f"Available columns in files: {list(gdf.columns)}")
            raise Exception("No valid data could be processed from files")
        
        print(f"Successfully processed {len(locations_data)} records")
        print(f"Embedding dimension: {embeddings[0].shape[0]}")
        
        return {
            'locations': locations_data,
            'embeddings': embeddings
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
    return {"message": "Embeddings Explorer API v2.0", "status": "running"}

@app.get("/api/locations", response_model=List[LocationData])
async def get_locations():
    """Get all locations with their basic info"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    locations = []
    for location_data in embeddings_data['locations']:
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
    location_index = None
    for i, loc in enumerate(embeddings_data['locations']):
        if loc['id'] == location_id:
            location_data = loc
            location_index = i
            break
    
    if location_data is None:
        raise HTTPException(status_code=404, detail="Location not found")
    
    # Get embedding shape
    embedding_shape = embeddings_data['embeddings'][location_index].shape
    
    return {
        **location_data,
        "embedding_shape": embedding_shape
    }

@app.get("/api/stats")
async def get_stats():
    """Get basic statistics about the dataset"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    locations = embeddings_data['locations']
    countries = list(set(loc['country'] for loc in locations))
    continents = list(set(loc['continent'] for loc in locations))
    
    # Get embedding dimension
    embedding_dim = embeddings_data['embeddings'][0].shape[0]
    
    return {
        "total_locations": len(locations),
        "countries_count": len(countries),
        "continents_count": len(continents),
        "embedding_dimension": embedding_dim,
        "countries": sorted(countries),
        "continents": sorted(continents)
    }

@app.get("/api/config")
async def get_config():
    """Get configuration including Mapbox token"""
    return {
        "mapbox_token": os.getenv("MAPBOX_TOKEN", "")
    }

def calculate_similarity(target_embedding, embeddings_list):
    """Calculate cosine similarity between target and list of embeddings"""
    similarities = []
    
    for i, embedding in enumerate(embeddings_list):
        if embedding is None or target_embedding is None:
            similarities.append((i, 0.0))
            continue
            
        # Calculate cosine similarity
        dot_product = np.dot(target_embedding, embedding)
        norm_target = np.linalg.norm(target_embedding)
        norm_embedding = np.linalg.norm(embedding)
        
        if norm_target == 0 or norm_embedding == 0:
            cosine_sim = 0
        else:
            cosine_sim = dot_product / (norm_target * norm_embedding)
        
        similarities.append((i, cosine_sim))
    
    return similarities

@app.get("/api/similarity/{location_id}")
async def find_similar_locations(
    location_id: int, 
    top_k: int = Query(20, ge=1, le=50)
):
    """Find most similar locations"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    # Find target location
    target_index = None
    for i, loc in enumerate(embeddings_data['locations']):
        if loc['id'] == location_id:
            target_index = i
            break
    
    if target_index is None:
        raise HTTPException(status_code=404, detail="Location not found")
    
    target_embedding = embeddings_data['embeddings'][target_index]
    
    # Calculate similarities
    similarities = calculate_similarity(target_embedding, embeddings_data['embeddings'])
    
    # Remove target location and sort by similarity
    similarities = [(i, sim) for i, sim in similarities if i != target_index]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar = similarities[:top_k]
    
    # Format response
    similar_locations = []
    for idx, sim_score in top_similar:
        location_data = embeddings_data['locations'][idx]
        similar_locations.append({
            **location_data,
            "similarity_score": float(sim_score)
        })
    
    return {
        "target_location_id": location_id,
        "similar_locations": similar_locations
    }

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

@app.get("/api/umap")
async def get_umap_embeddings():
    """Get UMAP 2D embeddings for all locations"""
    global umap_cache
    
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    # Return cached result if available
    if umap_cache is not None:
        print("Returning cached UMAP data")
        return umap_cache
    
    try:
        print("Computing UMAP embeddings...")
        
        # Run UMAP computation in thread pool
        loop = asyncio.get_event_loop()
        umap_embeddings = await loop.run_in_executor(
            executor, 
            compute_umap_embeddings, 
            embeddings_data['embeddings']
        )
        
        # Format response data
        umap_points = []
        for i, umap_coord in enumerate(umap_embeddings):
            location_data = embeddings_data['locations'][i]
            
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)