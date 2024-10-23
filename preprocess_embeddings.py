import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import geopandas as gpd
from geoalchemy2 import Geometry
from shapely import wkb
from shapely.geometry import Point
import umap
from tqdm import tqdm
from db_config import get_db_url

def load_embeddings_from_db():
    """Load embeddings from PostgreSQL database."""
    try:
        print("Connecting to database...")
        engine = create_engine(get_db_url())
        
        print("Loading data from database...")
        # Updated query to include continent
        query = text("""
        SELECT 
            embeddings,
            city,
            country,
            continent,
            ST_X(geom) as longitude,
            ST_Y(geom) as latitude,
            date
        FROM city_embeddings
        """)
        
        with engine.connect() as connection:
            # First, get the data using SQLAlchemy
            result = connection.execute(query)
            rows = result.fetchall()
            
            if not rows:
                raise Exception("No data found in the database")
                
            print(f"Successfully loaded {len(rows)} records from database")
            
            # Process the results
            embeddings = []
            city_labels = []
            country_labels = []
            continent_labels = []  # New list for continents
            geometries = []
            dates = []
            
            print("Processing embeddings...")
            for row in rows:
                try:
                    # Process embeddings
                    embedding = np.array(row[0], dtype=np.float32)
                    
                    # Create Point geometry from coordinates
                    longitude = float(row[4])  # Updated index
                    latitude = float(row[5])   # Updated index
                    geometry = Point(longitude, latitude)
                    
                    # Append all data
                    embeddings.append(embedding)
                    city_labels.append(row[1])
                    country_labels.append(row[2])
                    continent_labels.append(row[3])  # Add continent
                    geometries.append(geometry)
                    dates.append(row[6])  # Updated index
                    
                except Exception as e:
                    print(f"Error processing row: {row}")
                    print(f"Error details: {str(e)}")
                    continue
            
            if not embeddings:
                raise Exception("No valid data could be processed")
            
            # Convert lists to numpy arrays
            embeddings = np.array(embeddings)
            city_labels = np.array(city_labels)
            country_labels = np.array(country_labels)
            continent_labels = np.array(continent_labels)  # Convert continent list to array
            dates = np.array(dates)
            
            print(f"Successfully processed {len(embeddings)} records")
            
            # Print some debug information
            print("\nData sample:")
            print(f"First embedding shape: {embeddings[0].shape}")
            print(f"First geometry: {geometries[0]}")
            print(f"First city: {city_labels[0]}")
            print(f"First country: {country_labels[0]}")
            print(f"First continent: {continent_labels[0]}")  # Add continent to debug info
            
            return (embeddings, city_labels, country_labels, continent_labels, geometries, dates)
    
    except Exception as e:
        print(f"Error loading data from database: {str(e)}")
        import traceback
        print("Detailed error:")
        print(traceback.format_exc())
        raise

def main():
    print("Loading embeddings from database...")
    embeddings, city_labels, country_labels, continent_labels, geometries, dates = load_embeddings_from_db()

    print("Performing UMAP...")
    reducer = umap.UMAP(n_neighbors=6, min_dist=1, n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Reduce precision of UMAP results
    embeddings_2d = np.round(reduced_embeddings, decimals=5)

    print("Creating DataFrame...")
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'centroid': [f"{geom.x},{geom.y}" for geom in geometries],
        'continent': continent_labels,  # Add continent to DataFrame
        'country': country_labels,
        'city': city_labels,
        'date': dates
    })

    print("Saving results...")
    import os
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/global_umap_results.parquet'
    df.to_parquet(output_path, compression='snappy', index=False)
    print(f"Results saved to {output_path}")

    # Print debug information
    print(f"\nSummary Statistics:")
    print(f"Total number of data points: {len(df)}")
    print(f"Number of unique continents: {df['continent'].nunique()}")  # Add continent stats
    print(f"Number of unique countries: {df['country'].nunique()}")
    print(f"Number of unique cities: {df['city'].nunique()}")
    print(f"\nSample of processed data:")
    print(df.head(2))
    
    print("\nShape information:")
    print(f"Original embeddings shape: {embeddings.shape}")
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

if __name__ == "__main__":
    main()