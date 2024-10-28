# preprocess_embeddings.py
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import geopandas as gpd
from geoalchemy2 import Geometry
from shapely import wkb
from shapely.geometry import Point
import umap
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
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
            continent_labels = []
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

    print("Performing PCA...")
    # Standardize the embeddings first
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply PCA with 10 components
    n_components = 10
    pca = PCA(n_components=n_components, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings_scaled)

    # Reduce precision of PCA results
    embeddings_nd = np.round(reduced_embeddings, decimals=5)

    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")

    print("Creating DataFrame...")
    # Create columns for all PCs
    pc_columns = {f'PC{i+1}': embeddings_nd[:, i] for i in range(n_components)}
    
    df = pd.DataFrame({
        **pc_columns,  # Unpack all PC columns
        'centroid': [f"{geom.x},{geom.y}" for geom in geometries],
        'continent': continent_labels,
        'country': country_labels,
        'city': city_labels,
        'date': dates
    })

    print("Saving results...")
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/global_pca_results.parquet'
    df.to_parquet(output_path, compression='snappy', index=False)
    print(f"Results saved to {output_path}")

    # Print debug information
    print(f"\nSummary Statistics:")
    print(f"Total number of data points: {len(df)}")
    print(f"Number of unique continents: {df['continent'].nunique()}")
    print(f"Number of unique countries: {df['country'].nunique()}")
    print(f"Number of unique cities: {df['city'].nunique()}")
    print(f"\nSample of processed data:")
    print(df.head(2))
    
    print("\nShape information:")
    print(f"Original embeddings shape: {embeddings.shape}")
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

if __name__ == "__main__":
    main()
