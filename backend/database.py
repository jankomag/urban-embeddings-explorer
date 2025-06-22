import numpy as np
from sqlalchemy import create_engine, text
from shapely.geometry import Point
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_url():
    """Get database URL from environment variables"""
    # Assuming you have these in your .env file
    return os.getenv("DATABASE_URL")

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
        FROM city_embeddings_new
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