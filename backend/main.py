import os
import numpy as np
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import DBSCAN
from typing import List, Dict
import logging
import json
import uuid
from datetime import timedelta
import umap.umap_ as umap
from collections import defaultdict
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for UMAP results
umap_results = defaultdict(dict)

# Global variables to store the data
global_df = None
urban_areas_gdf = None

def format_country_name(name: str) -> str:
    """Format country name to uppercase with underscores."""
    return name.replace(' ', '_').upper()

def format_continent_name(name: str) -> str:
    """Format continent name to uppercase with underscores."""
    return name.replace(' ', '_').upper()



def load_tsne_data():
    """Load embeddings from parquet file."""
    global global_df
    if global_df is None:
        try:
            current_dir = Path(__file__).resolve().parent
            data_path = current_dir.parent / 'data' / 'global_umap_results.parquet'
            
            logger.info(f"Attempting to load data from: {data_path}")
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found at {data_path}")
            
            global_df = pd.read_parquet(data_path)
            
            # Convert centroid string to actual coordinates
            global_df[['longitude', 'latitude']] = global_df['centroid'].str.split(',', expand=True).astype(float)
            
            # Format country and continent names
            global_df['country'] = global_df['country'].apply(format_country_name)
            global_df['continent'] = global_df['continent'].apply(format_continent_name)
            
            # Load original embeddings
            embeddings_path = current_dir.parent / 'data' / 'original_embeddings.parquet'
            if embeddings_path.exists():
                embeddings_df = pd.read_parquet(embeddings_path)
                embedding_columns = [col for col in embeddings_df.columns if col.startswith('dim_')]
                global_df['original_embeddings'] = embeddings_df[embedding_columns].values.tolist()
            else:
                logger.warning("Original embeddings file not found")
                global_df['original_embeddings'] = [[0] * 128] * len(global_df)  # placeholder
            
            # Pre-compute the dict representation for faster access
            global_df['dict_rep'] = global_df.apply(lambda row: {
                'x': float(row['x']),
                'y': float(row['y']),
                'country': row['country'],
                'continent': row['continent'],
                'city': row['city'],
                'longitude': float(row['longitude']),
                'latitude': float(row['latitude']),
                'original_embeddings': row['original_embeddings']
            }, axis=1).tolist()
            
            logger.info(f"Successfully loaded {len(global_df)} data points")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")
    return global_df

@app.get("/embedding_dimensions")
async def get_embedding_dimensions():
    """Get the number of dimensions in the original embeddings."""
    try:
        df = load_tsne_data()
        if len(df['original_embeddings'].iloc[0]) > 0:
            return {"dimensions": len(df['original_embeddings'].iloc[0])}
        else:
            raise HTTPException(status_code=404, detail="No embedding dimensions found")
    except Exception as e:
        logger.error(f"Error in /embedding_dimensions: {e}")
        raise HTTPException(status_code=500, detail="Error fetching embedding dimensions")

def load_urban_areas():
    global urban_areas_gdf
    if urban_areas_gdf is None:
        try:
            # Get the absolute path to the data directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, '..', 'data', 'urban_areas.geojson')
            
            logger.info(f"Loading urban areas from: {file_path}")
            
            urban_areas_gdf = gpd.read_file(file_path)
            urban_areas_gdf['city'] = urban_areas_gdf['city'].str.lower()
            logger.info(f"Loaded {len(urban_areas_gdf)} urban areas")
        except Exception as e:
            logger.error(f"Error loading urban areas data: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading urban areas data: {str(e)}")
    return urban_areas_gdf

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup."""
    load_tsne_data()
    load_urban_areas()

async def compute_umap(computation_id: str, params: Dict):
    """Compute UMAP transformation with given parameters."""
    try:
        # Load the original data
        df = load_tsne_data()
        logger.info("Data loaded for UMAP computation")
        
        # Get coordinates for UMAP input
        coords = df[['x', 'y']].values
        logger.info(f"Input shape for UMAP: {coords.shape}")
        
        # Configure UMAP
        reducer = umap.UMAP(
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            n_components=2,
            random_state=42
        )
        
        # Compute UMAP
        logger.info(f"Starting UMAP computation with params: {params}")
        reduced_embeddings = reducer.fit_transform(coords)
        logger.info(f"UMAP computation completed, output shape: {reduced_embeddings.shape}")
        
        # Store results in memory
        result_dict = {
            'status': 'completed',
            'data': reduced_embeddings.tolist(),
            'params': params
        }
        umap_results[computation_id] = result_dict
        logger.info(f"Results stored for {computation_id}")
        
    except Exception as e:
        logger.error(f"Error in UMAP computation: {str(e)}")
        umap_results[computation_id] = {
            'status': 'failed',
            'error': str(e),
            'params': params
        }

@app.post("/compute_umap")
async def request_umap_computation(
    background_tasks: BackgroundTasks,
    n_neighbors: int = Query(..., ge=2, le=100),
    min_dist: float = Query(..., ge=0.0, le=1.0)
):
    """Initiate UMAP computation with specified parameters."""
    try:
        params = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist
        }
        
        computation_id = str(uuid.uuid4())
        umap_results[computation_id] = {'status': 'pending', 'params': params}
        
        background_tasks.add_task(
            compute_umap,
            computation_id,
            params
        )
        
        return {"computation_id": computation_id}
        
    except Exception as e:
        logger.error(f"Error initiating UMAP computation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/umap_status/{computation_id}")
async def get_umap_status(computation_id: str):
    """Get status of a UMAP computation."""
    try:
        result = umap_results.get(computation_id)
        if not result:
            raise HTTPException(status_code=404, detail="Computation not found")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching UMAP status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/urban_areas")
async def get_urban_areas():
    """Get all urban areas as GeoJSON."""
    try:
        gdf = load_urban_areas()
        if gdf is None:
            logger.error("Urban areas GeoDataFrame is None")
            raise HTTPException(status_code=500, detail="Failed to load urban areas data")
        
        logger.info(f"Converting {len(gdf)} urban areas to GeoJSON")
        geojson = gdf.to_crs(epsg=4326).to_json()
        return JSONResponse(content=json.loads(geojson))
    except Exception as e:
        logger.error(f"Error in /urban_areas: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching urban areas: {str(e)}")

@app.get("/urban_areas/{city}")
async def get_urban_area(city: str):
    """Get urban area for a specific city."""
    try:
        gdf = load_urban_areas()
        if gdf is None:
            raise HTTPException(status_code=404, detail="Urban areas data not found")
        
        city_data = gdf[gdf['city'].str.lower() == city.lower()]
        
        if city_data.empty:
            raise HTTPException(status_code=404, detail="City not found")
        
        geojson = city_data.to_crs(epsg=4326).to_json()
        return JSONResponse(content=json.loads(geojson))
    except Exception as e:
        logger.error(f"Error in /urban_area: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching urban area for {city}")

@app.get("/tsne_data")
async def get_tsne_data():
    """Get all TSNE/UMAP data points."""
    try:
        df = load_tsne_data()
        return JSONResponse(content={
            'data': df['dict_rep'].tolist(),
            'total_points': len(df)
        })
    except Exception as e:
        logger.error(f"Error in /tsne_data: {e}")
        raise HTTPException(status_code=500, detail="Error fetching t-SNE data")

@app.get("/countries")
async def get_countries() -> List[str]:
    """Get list of all countries."""
    try:
        df = load_tsne_data()
        return sorted(df['country'].unique().tolist())
    except Exception as e:
        logger.error(f"Error in /countries: {e}")
        raise HTTPException(status_code=500, detail="Error fetching country list")

@app.get("/cities/{country}")
async def get_cities(country: str) -> List[str]:
    """Get list of cities for a specific country."""
    try:
        df = load_tsne_data()
        cities = sorted(df[df['country'] == country]['city'].unique().tolist())
        if not cities:
            raise HTTPException(status_code=404, detail="Country not found")
        return cities
    except Exception as e:
        logger.error(f"Error in /cities: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching cities for {country}")

@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Embeddings Viewer API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)