import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine, text
from db_config import get_db_url

def load_continent_data():
    """Load and prepare continent boundaries from GeoJSON file."""
    print("Loading continent boundaries...")
    continents = gpd.read_file('data/World_Continents.geojson')
    # Ensure the CRS is set to WGS84 (EPSG:4326)
    if continents.crs is None or continents.crs.to_string() != 'EPSG:4326':
        continents = continents.to_crs('EPSG:4326')
    return continents

def update_database_with_continents():
    """Update the database with continent information for each city."""
    try:
        # Load continent boundaries
        continents_gdf = load_continent_data()
        print(f"Loaded {len(continents_gdf)} continent boundaries")
        
        # Connect to database
        print("Connecting to database...")
        engine = create_engine(get_db_url())
        
        # Get city data using SQLAlchemy
        with engine.begin() as connection:
            # First, add the continent column if it doesn't exist
            connection.execute(text("ALTER TABLE city_embeddings ADD COLUMN IF NOT EXISTS continent VARCHAR(50);"))
            
            # Fetch all cities
            print("Loading cities from database...")
            result = connection.execute(text("""
                SELECT 
                    id,
                    ST_X(geom) as longitude,
                    ST_Y(geom) as latitude
                FROM city_embeddings
            """))
            
            # Convert to DataFrame
            df = pd.DataFrame(result.fetchall(), columns=['id', 'longitude', 'latitude'])
            print(f"Loaded {len(df)} cities from database")
            
            # Create GeoDataFrame from cities
            geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
            cities_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            
            # Perform spatial join
            print("Performing spatial join with continents...")
            joined = gpd.sjoin(cities_gdf, continents_gdf, how='left', predicate='within')
            
            # Check for cities that didn't get assigned to a continent
            unmatched = joined[joined['index_right'].isna()]
            if len(unmatched) > 0:
                print(f"Warning: {len(unmatched)} cities could not be matched to a continent")
                print("Using nearest continent for unmatched points...")
                
                # For each unmatched city, find the nearest continent
                for idx in unmatched.index:
                    point = cities_gdf.loc[idx, 'geometry']
                    # Calculate distance to each continent
                    distances = continents_gdf.geometry.distance(point)
                    # Get the nearest continent's index
                    nearest_idx = distances.idxmin()
                    # Assign the continent information
                    joined.loc[idx, 'index_right'] = nearest_idx
                    for col in continents_gdf.columns:
                        if col != 'geometry':
                            joined.loc[idx, col] = continents_gdf.loc[nearest_idx, col]
            
            # Prepare and execute batch update
            print("Updating database with continent information...")
            update_data = []
            for _, row in joined.iterrows():
                update_data.append({
                    'continent': row['CONTINENT'],  # Adjust this column name based on your GeoJSON structure
                    'longitude': row['longitude'],
                    'latitude': row['latitude']
                })
            
            # Use batch update for better performance
            if update_data:
                batch_size = 1000
                for i in range(0, len(update_data), batch_size):
                    batch = update_data[i:i + batch_size]
                    connection.execute(
                        text("""
                        UPDATE city_embeddings
                        SET continent = :continent
                        WHERE ST_X(geom) = :longitude AND ST_Y(geom) = :latitude;
                        """),
                        batch
                    )
            
            # Print summary
            print("\nUpdate Summary:")
            result = connection.execute(text("SELECT continent, COUNT(*) FROM city_embeddings GROUP BY continent"))
            for row in result.fetchall():
                print(f"{row[0]}: {row[1]} cities")

    except Exception as e:
        print(f"Error updating database with continents: {str(e)}")
        import traceback
        print("Detailed error:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    update_database_with_continents()