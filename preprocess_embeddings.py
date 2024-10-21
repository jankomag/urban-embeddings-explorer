import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import umap

def load_embeddings(root_dir):
    all_embeddings = []
    city_labels = []
    country_labels = []
    geometries = []
    dates = []

    for country in os.listdir(root_dir):
        country_dir = os.path.join(root_dir, country)
        if os.path.isdir(country_dir):
            for file in os.listdir(country_dir):
                if file.endswith('.gpq'):
                    file_path = os.path.join(country_dir, file)
                    gdf = gpd.read_parquet(file_path)
                    
                    # Extract embeddings
                    embeddings = np.array(gdf['embeddings'].tolist())
                    all_embeddings.append(embeddings)
                    
                    # Extract geometries (centroids)
                    geometries.extend(gdf['geometry'].tolist())
                    
                    # Extract dates
                    dates.extend(gdf['date'].tolist())
                    
                    # Extract labels
                    city_name = file.split('_')[1].split('.')[0]
                    city_labels.extend([city_name] * len(embeddings))
                    country_labels.extend([country] * len(embeddings))

    return (np.vstack(all_embeddings), np.array(city_labels), np.array(country_labels), 
            geometries, dates)

print("Loading embeddings...")
embeddings, city_labels, country_labels, geometries, dates = load_embeddings("224embeddings")

print("Performing UMAP...")
reducer = umap.UMAP(n_neighbors=6, min_dist=1, n_components=2, random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)

# Reduce precision of UMAP results
embeddings_2d = np.round(reduced_embeddings, decimals=5)

df = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'centroid': [f"{geom.x},{geom.y}" for geom in geometries],
    'country': country_labels,
    'city': city_labels,
    'date': dates
})

print("Saving results...")
df.to_parquet('data/global_umap_results.parquet', compression='snappy', index=False)

print("Data processing complete. Results saved to 'data/global_umap_results.parquet'")

# Print some debug information
print(f"Total number of data points: {len(df)}")
print(f"Number of unique countries: {df['country'].nunique()}")
print(f"Number of unique cities: {df['city'].nunique()}")
print(f"Sample of data:")
print(df.head(2))