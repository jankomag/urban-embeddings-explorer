import os
import numpy as np
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import DBSCAN
from typing import List
import json
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
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

# Global variable to store the data
global_df = None
urban_areas_gdf = None

def format_country_name(name):
    return name.replace(' ', '_').upper()

def format_continent_name(name):
    return name.replace(' ', '_').upper()

@lru_cache(maxsize=1)
def load_tsne_data():
    global global_df
    if global_df is None:
        try:
            global_df = pd.read_parquet('../data/global_umap_results.parquet')
            # Convert centroid string to actual coordinates
            global_df[['longitude', 'latitude']] = global_df['centroid'].str.split(',', expand=True).astype(float)
            # Format country and continent names
            global_df['country'] = global_df['country'].apply(format_country_name)
            global_df['continent'] = global_df['continent'].apply(format_continent_name)
            
            # Make sure continent is not null
            if global_df['continent'].isnull().any():
                logger.warning("Some continent values are null!")
            
            # Log unique continents for debugging
            unique_continents = global_df['continent'].unique()
            logger.info(f"Unique continents in data: {unique_continents}")
            
            # Pre-compute the dict representation for faster access
            global_df['dict_rep'] = global_df.apply(lambda row: {
                'x': float(row['x']),
                'y': float(row['y']),
                'country': row['country'],
                'continent': row['continent'],
                'city': row['city'],
                'longitude': float(row['longitude']),
                'latitude': float(row['latitude'])
            }, axis=1).tolist()
            logger.info(f"Loaded {len(global_df)} data points")
            
            # Log a sample point for verification
            sample_point = global_df['dict_rep'][0]
            logger.info(f"Sample point data: {sample_point}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise HTTPException(status_code=500, detail="Error loading data")
    return global_df

@lru_cache(maxsize=1)
def load_urban_areas():
    global urban_areas_gdf
    if urban_areas_gdf is None:
        try:
            urban_areas_gdf = gpd.read_file('../data/urban_areas.geojson')
            urban_areas_gdf['city'] = urban_areas_gdf['city'].str.lower()
            logger.info(f"Loaded {len(urban_areas_gdf)} urban areas")
        except Exception as e:
            logger.error(f"Error loading urban areas data: {e}")
            raise HTTPException(status_code=500, detail="Error loading urban areas data")
    return urban_areas_gdf

@app.on_event("startup")
async def startup_event():
    load_tsne_data()
    load_urban_areas()

@app.get("/urban_areas")
async def get_urban_areas():
    try:
        gdf = load_urban_areas()
        geojson = gdf.to_crs(epsg=4326).to_json()
        return JSONResponse(content=json.loads(geojson))
    except Exception as e:
        logger.error(f"Error in /urban_areas: {e}")
        raise HTTPException(status_code=500, detail="Error fetching urban areas")

@app.get("/urban_area/{city}")
async def get_urban_area(city: str):
    try:
        gdf = load_urban_areas()
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
    try:
        df = load_tsne_data()
        total_points = len(df)
        
        # Get the response data
        response_data = df['dict_rep'].tolist()
        
        # Log some sample data for verification
        logger.info(f"Total points: {total_points}")
        if response_data:
            logger.info(f"Sample point continent: {response_data[0].get('continent', 'NO_CONTINENT')}")
        
        # Verify continent data
        continents_present = set(d.get('continent') for d in response_data)
        logger.info(f"Continents present in response: {continents_present}")
        
        return JSONResponse(content={
            'data': response_data,
            'total_points': total_points,
        })
    except Exception as e:
        logger.error(f"Error in /tsne_data: {e}")
        raise HTTPException(status_code=500, detail="Error fetching t-SNE data")

@app.get("/countries")
async def get_countries() -> List[str]:
    try:
        df = load_tsne_data()
        return sorted(df['country'].unique().tolist())
    except Exception as e:
        logger.error(f"Error in /countries: {e}")
        raise HTTPException(status_code=500, detail="Error fetching country list")

@app.get("/cities/{country}")
async def get_cities(country: str) -> List[str]:
    try:
        df = load_tsne_data()
        cities = sorted(df[df['country'] == country]['city'].unique().tolist())
        if not cities:
            raise HTTPException(status_code=404, detail="Country not found")
        return cities
    except Exception as e:
        logger.error(f"Error in /cities: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching cities for {country}")

@app.get("/clustered_map_data")
async def get_clustered_map_data(
    zoom: float = Query(..., description="Current map zoom level"),
    bbox: str = Query(..., description="Map bounding box (minLon,minLat,maxLon,maxLat)")
):
    try:
        df = load_tsne_data()
        
        # Filter by viewport
        minLon, minLat, maxLon, maxLat = map(float, bbox.split(','))
        df_filtered = df[
            (df['longitude'].between(minLon, maxLon)) & 
            (df['latitude'].between(minLat, maxLat))
        ]
        
        # Adjust epsilon based on zoom level
        epsilon = 10 / (2 ** zoom)  # Adjust this formula as needed
        
        # Perform DBSCAN clustering
        coords = np.radians(df_filtered[['latitude', 'longitude']].values)
        db = DBSCAN(eps=epsilon, min_samples=1, metric='haversine').fit(coords)
        df_filtered['cluster'] = db.labels_
        
        # Aggregate clusters
        clusters = df_filtered.groupby('cluster').agg({
            'country': lambda x: ', '.join(set(x)),
            'longitude': 'mean',
            'latitude': 'mean',
            'x': 'count'  # Use 'x' column to count points
        }).reset_index()
        
        clusters.columns = ['cluster', 'countries', 'longitude', 'latitude', 'point_count']
        
        return JSONResponse(content=clusters.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error in /clustered_map_data: {e}")
        raise HTTPException(status_code=500, detail="Error fetching clustered map data")

@app.get("/country_data/{country}")
async def get_country_data(country: str):
    try:
        df = load_tsne_data()
        country_df = df[df['country'] == country]
        
        if country_df.empty:
            raise HTTPException(status_code=404, detail="Country not found")
        
        response_data = country_df['dict_rep'].tolist()
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error in /country_data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching data for {country}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Embeddings Viewer API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)