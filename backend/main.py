# main.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
import json
from pathlib import Path
from sqlalchemy import create_engine, text
import sys

# Add the path to the Clay model
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(parent_dir)

from db_config import get_db_url

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
    allow_origins=["http://your-ec2-public-ip:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the data
global_df = None
urban_areas_gdf = None

def format_country_name(name: str) -> str:
    """Format country name to uppercase with underscores."""
    return name.replace(' ', '_').upper()

def format_continent_name(name: str) -> str:
    """Format continent name to uppercase with underscores."""
    return name.replace(' ', '_').upper()

def load_pca_data():
    """Load pre-computed PCA visualization data."""
    global global_df
    if global_df is None:
        try:
            current_dir = Path(__file__).resolve().parent
            data_path = current_dir.parent / 'data' / 'global_pca_results.parquet'
            
            logger.info(f"Attempting to load data from: {data_path}")
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found at {data_path}")
            
            global_df = pd.read_parquet(data_path)
            
            # Convert centroid string to actual coordinates
            global_df[['longitude', 'latitude']] = global_df['centroid'].str.split(',', expand=True).astype(float)
            
            # Format country and continent names
            global_df['country'] = global_df['country'].apply(format_country_name)
            global_df['continent'] = global_df['continent'].apply(format_continent_name)
            
            # Pre-compute the dict representation for faster access
            # Note: Using only PC1 and PC2 for visualization
            global_df['dict_rep'] = global_df.apply(lambda row: {
                'x': float(row['PC1']),  # Changed from 'x' to PC1
                'y': float(row['PC2']),  # Changed from 'y' to PC2
                'country': row['country'],
                'continent': row['continent'],
                'city': row['city'],
                'longitude': float(row['longitude']),
                'latitude': float(row['latitude']),
                # Add all PCs to the response
                'pcs': {f'PC{i}': float(row[f'PC{i}']) for i in range(1, 11)}
            }, axis=1).tolist()
            
            logger.info(f"Successfully loaded {len(global_df)} data points")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")
    return global_df

def load_urban_areas():
    """Load urban areas GeoJSON data."""
    global urban_areas_gdf
    if urban_areas_gdf is None:
        try:
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
    load_pca_data()
    load_urban_areas()

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

@app.get("/pca_data")
async def get_pca_data():
    """Get all PCA data points."""
    try:
        df = load_pca_data()
        return {"data": df['dict_rep'].tolist()}
    except Exception as e:
        logger.error(f"Error in /pca_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/countries")
async def get_countries() -> List[str]:
    """Get list of all countries."""
    try:
        df = load_pca_data()
        return sorted(df['country'].unique().tolist())
    except Exception as e:
        logger.error(f"Error in /countries: {e}")
        raise HTTPException(status_code=500, detail="Error fetching country list")

@app.get("/cities/{country}")
async def get_cities(country: str) -> List[str]:
    """Get list of cities for a specific country."""
    try:
        df = load_pca_data()
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
    return {"message": "Welcome to the PCA Urban Embeddings Viewer API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
