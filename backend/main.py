from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from database import load_embeddings_from_db
from models import LocationData
import numpy as np
from typing import List
import umap
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Embeddings Explorer", version="1.0.0")

# Add CORS middleware - Updated for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://localhost:3001",  # Backup port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded data
embeddings_data = None
umap_cache = None
executor = ThreadPoolExecutor(max_workers=2)

@app.on_event("startup")
async def startup_event():
    """Load embeddings data on startup"""
    global embeddings_data
    try:
        print("Loading embeddings data...")
        embeddings, city_labels, country_labels, continent_labels, geometries, dates = load_embeddings_from_db()
        embeddings_data = {
            'embeddings': embeddings,
            'cities': city_labels,
            'countries': country_labels,
            'continents': continent_labels,
            'geometries': geometries,
            'dates': dates
        }
        print(f"Successfully loaded {len(embeddings)} locations")
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        embeddings_data = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Embeddings Explorer API", "status": "running"}

@app.get("/api/locations", response_model=List[LocationData])
async def get_locations():
    """Get all locations with their basic info"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    locations = []
    for i in range(len(embeddings_data['cities'])):
        geom = embeddings_data['geometries'][i]
        location = LocationData(
            id=i,
            city=embeddings_data['cities'][i],
            country=embeddings_data['countries'][i],
            continent=embeddings_data['continents'][i],
            longitude=geom.x,
            latitude=geom.y,
            date=str(embeddings_data['dates'][i]) if embeddings_data['dates'][i] else None
        )
        locations.append(location)
    
    return locations

@app.get("/api/locations/{location_id}")
async def get_location(location_id: int):
    """Get detailed info for a specific location"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    if location_id >= len(embeddings_data['cities']) or location_id < 0:
        raise HTTPException(status_code=404, detail="Location not found")
    
    geom = embeddings_data['geometries'][location_id]
    return {
        "id": location_id,
        "city": embeddings_data['cities'][location_id],
        "country": embeddings_data['countries'][location_id],
        "continent": embeddings_data['continents'][location_id],
        "longitude": geom.x,
        "latitude": geom.y,
        "date": str(embeddings_data['dates'][location_id]) if embeddings_data['dates'][location_id] else None,
        "embedding_shape": embeddings_data['embeddings'][location_id].shape
    }

@app.get("/api/stats")
async def get_stats():
    """Get basic statistics about the dataset"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    countries = list(set(embeddings_data['countries']))
    continents = list(set(embeddings_data['continents']))
    
    return {
        "total_locations": len(embeddings_data['cities']),
        "countries_count": len(countries),
        "continents_count": len(continents),
        "embedding_dimension": embeddings_data['embeddings'][0].shape[0] if len(embeddings_data['embeddings']) > 0 else 0,
        "countries": sorted(countries),
        "continents": sorted(continents)
    }

@app.get("/api/config")
async def get_config():
    """Get configuration including Mapbox token"""
    return {
        "mapbox_token": os.getenv("MAPBOX_TOKEN", "")
    }

@app.get("/api/similarity/{location_id}")
async def find_similar_locations(location_id: int, top_k: int = 5):
    """Find most similar locations to the given location"""
    if embeddings_data is None:
        raise HTTPException(status_code=503, detail="Embeddings data not loaded")
    
    if location_id >= len(embeddings_data['cities']) or location_id < 0:
        raise HTTPException(status_code=404, detail="Location not found")
    
    # Get the target embedding
    target_embedding = embeddings_data['embeddings'][location_id]
    
    # Calculate cosine similarities with all other embeddings
    similarities = []
    for i, embedding in enumerate(embeddings_data['embeddings']):
        if i != location_id:  # Skip the target location itself
            # Calculate cosine similarity
            dot_product = np.dot(target_embedding, embedding)
            norm_target = np.linalg.norm(target_embedding)
            norm_embedding = np.linalg.norm(embedding)
            
            if norm_target == 0 or norm_embedding == 0:
                cosine_sim = 0
            else:
                cosine_sim = dot_product / (norm_target * norm_embedding)
            
            similarities.append((i, cosine_sim))
    
    # Sort by similarity (highest first) and get top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar = similarities[:top_k]
    
    # Format response
    similar_locations = []
    for idx, sim_score in top_similar:
        geom = embeddings_data['geometries'][idx]
        similar_locations.append({
            "id": idx,
            "city": embeddings_data['cities'][idx],
            "country": embeddings_data['countries'][idx],
            "continent": embeddings_data['continents'][idx],
            "longitude": geom.x,
            "latitude": geom.y,
            "date": str(embeddings_data['dates'][idx]) if embeddings_data['dates'][idx] else None,
            "similarity_score": float(sim_score)
        })
    
    return {
        "target_location_id": location_id,
        "similar_locations": similar_locations
    }

def compute_umap_embeddings(embeddings_array):
    """Compute UMAP embeddings in a separate thread"""
    print("Computing UMAP embeddings...")
    
    # Configure UMAP for large datasets
    n_samples = len(embeddings_array)
    
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
    umap_embeddings = reducer.fit_transform(embeddings_array)
    
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
        print("Computing UMAP embeddings asynchronously...")
        
        # Convert embeddings to numpy array if needed
        embeddings_array = np.array(embeddings_data['embeddings'])
        
        # Run UMAP computation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        umap_embeddings = await loop.run_in_executor(
            executor, 
            compute_umap_embeddings, 
            embeddings_array
        )
        
        # Format response data
        umap_points = []
        for i in range(len(umap_embeddings)):
            geom = embeddings_data['geometries'][i]
            umap_points.append({
                "location_id": i,
                "x": float(umap_embeddings[i][0]),
                "y": float(umap_embeddings[i][1]),
                "city": embeddings_data['cities'][i],
                "country": embeddings_data['countries'][i],
                "continent": embeddings_data['continents'][i],
                "longitude": geom.x,
                "latitude": geom.y,
                "date": str(embeddings_data['dates'][i]) if embeddings_data['dates'][i] else None
            })
        
        umap_cache = {
            "umap_points": umap_points,
            "total_points": len(umap_points),
            "embedding_dimension": embeddings_array.shape[1]
        }
        
        print(f"UMAP computation successful: {len(umap_points)} points")
        return umap_cache
        
    except Exception as e:
        print(f"Error computing UMAP: {str(e)}")
        import traceback
        print("Detailed error:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to compute UMAP embeddings: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)