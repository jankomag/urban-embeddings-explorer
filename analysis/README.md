# 🛰️ Urban Embeddings Analysis Pipeline

This pipeline processes satellite imagery from cities worldwide to create AI-powered visual embeddings for similarity search.

## 🔄 Pipeline Overview

1. **Extract Embeddings** → 2. **Find Spatial Correlations** → 3. **Generate UMAP** → 4. **Upload to Vector DB**

---

## 📁 Files & Purpose

### 1. `create_embeddings.py` 🎯

**Purpose**: Downloads satellite imagery and extracts TerraMind vision embeddings  
**Output**: Local GeoParquet files with 768-dimensional embeddings per tile

```bash
# Install dependencies
pip install terratorch geopandas pystac-client matplotlib torch dask

# Configure and run
python create_embeddings.py
```

**What it does**:

- Fetches Sentinel-2 satellite data from STAC catalogs
- Creates 224×224m tiles for each city
- Extracts embeddings using TerraMind model (196 patches → 768 dimensions)
- Saves both aggregated embeddings and full patch data
- Handles data quality control and caching

---

### 2. `spatial-correlation.py` 🗺️

**Purpose**: Identifies dimensions that encode geographic location rather than visual features  
**Output**: `spatial_correlation_results.json` with excluded dimensions list

```bash
python spatial-correlation.py
```

**Why this matters**:

- Some embedding dimensions encode "WHERE" (lat/lon) instead of "WHAT" (visual content)
- Removing spatial dims improves similarity search quality
- Focuses on visual patterns rather than geographic proximity

**What it analyzes**:

- Within-tile patch coherence (spatial smoothness)
- Between-tile spatial correlation
- Uses gradient-based cutoff detection

---

### 3. `umap-generator.py` 📊

**Purpose**: Creates 2D visualization coordinates from filtered embeddings  
**Output**: `umap_coordinates_filtered.json` with dimension reduction

```bash
python umap-generator.py
```

**Process**:

- Loads embeddings and applies dimension filtering
- Generates UMAP coordinates using optimized parameters
- Creates interactive 2D projection for web interface

**UMAP Parameters**:

```python
UMAP_PARAMS = {
    'n_components': 6,      # 6D intermediate space
    'n_neighbors': 7,       # Local structure preservation
    'min_dist': 1,          # Cluster tightness
    'metric': 'cosine',     # Distance metric
    'n_epochs': 500         # Training iterations
}
```

### 4. `data-migration.py` 🚀

**Purpose**: Prepares filtered embeddings for vector database with multiple aggregation methods  
**Output**: 6 collections with different similarity approaches

**Creates 6 Collections**:

- `mean`: Standard patch averaging
- `median`: Robust to outliers
- `min/max`: Conservative/distinctive features
- `dominant_cluster`: Most frequent visual patterns
- `global_contrastive`: Dataset-relative encoding

**Why multiple methods**: Different aggregations capture different visual aspects, enabling diverse similarity search modes.

## 📊 Output Structure

```
embeddings/
├── urban_embeddings_224_terramind_normalised/
│   ├── tile_embeddings/           # Aggregated embeddings
│   └── full_patch_embeddings/     # Raw patch data
├── production_data/
│   ├── umap_coordinates_filtered.json
│   ├── locations_metadata.json
│   └── dataset_statistics.json
└── spatial_correlation_analysis/
    └── spatial_correlation_results.json
```

## ⚙️ Configuration

Key settings in each script:

- `TILE_SIZE = 224`: Matches TerraMind input requirements
- `SAVE_FULL_PATCH_EMBEDDINGS = True`: Enables advanced aggregation methods
- `MAX_TILES_PER_CITY = 100`: Memory management for analysis
- City filtering via `EXCLUDED_FILES` for problematic areas

## 🎯 Why This Pipeline?

1. **TerraMind Embeddings**: State-of-the-art satellite imagery understanding
2. **Dimension Filtering**: Removes geographic bias, focuses on visual similarity
3. **Multiple Aggregations**: Captures different aspects of urban visual patterns
4. **Vector Database**: Enables fast similarity search at scale
5. **UMAP Visualization**: Makes high-dimensional data explorable

The result: An AI system that finds visually similar urban areas worldwide based on satellite imagery patterns! 🌍
