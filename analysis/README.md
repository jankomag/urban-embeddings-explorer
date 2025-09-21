# Analysis Pipeline

Satellite imagery processing pipeline for extracting TerraMind embeddings and preparing data for similarity search applications.

## Overview

This pipeline processes Sentinel-2 satellite imagery to create visual similarity embeddings using IBM TerraMind foundation model. The system identifies and removes spatially-biased dimensions to focus on visual content rather than geographic location, then prepares filtered embeddings for vector database storage.

## Pipeline Components

### 1. Embedding Extraction (`create_embeddings.py`)

Extracts TerraMind embeddings from Sentinel-2 satellite imagery.

**Process:**

- Downloads Sentinel-2 L2A data from STAC catalogs
- Creates 224×224m tiles matching TerraMind input requirements
- Processes tiles through TerraMind model (196 patches → 768 dimensions per tile)
- Saves both aggregated embeddings and raw patch data

**TerraMind Model:**

- Model: `ibm-esa-geospatial/TerraMind-1.0-base`
- Repository: https://github.com/IBM/terramind
- Input: 224×224 pixels at 10m resolution
- Output: 768-dimensional embeddings per 16×16 patch

**Configuration:**

```python
TILE_SIZE = 224                    # Matches TerraMind requirements
SAVE_FULL_PATCH_EMBEDDINGS = True # Enables advanced aggregation
MAX_TILES_PER_CITY = 100          # Memory management
```

### 2. Spatial Correlation Analysis (`spatial-correlation.py`)

Identifies embedding dimensions that encode geographic location rather than visual features.

**Analysis Method:**

- Measures spatial autocorrelation within tiles (patch coherence)
- Calculates correlation between tile proximity and embedding similarity
- Uses gradient-based cutoff detection
- Generates statistical plots and dimension exclusion results, like this one:

![Spatial Correlation Analysis](./res/spatial_correlation_analysis.png)

**Output:** `spatial_correlation_results.json` with excluded dimensions list

### 3. UMAP Coordinate Generation (`umap-generator.py`)

Creates 2D visualization coordinates from filtered embeddings.

**Process:**

- Loads embeddings and applies identical dimension filtering
- Computes UMAP projection using optimized parameters
- Generates precomputed coordinates for interactive web visualization

### 4. Vector Database Migration (`data-migration.py`)

Prepares filtered embeddings for vector database with multiple aggregation strategies.

**Aggregation Methods:**

- `mean`
- `median`
- `dominant_cluster`: First clusters all 196 pacthes in a tile and extractes the mean of the most frequent cluster
- `global_contrastive`: Average of all patches in the tile minus global mean

## Execution Sequence

```bash
# 1. Extract TerraMind embeddings
python create_embeddings.py

# 2. Analyze spatial correlations
python spatial-correlation.py

# 3. Generate UMAP coordinates
python umap-generator.py

# 4. Upload to vector database
python data-migration.py
```

## Technical Requirements

**Dependencies:**

```bash
pip install terratorch geopandas qdrant-client umap-learn scikit-learn
```

**System Requirements:**

- Python 3.10+
- GDAL for geospatial processing
- 12GB+ RAM for city batch processing
- GPU recommended for TerraMind inference

## Output Structure

```
embeddings/
├── urban_embeddings_224_terramind_normalised/
│   ├── tile_embeddings/           # Aggregated tile embeddings
│   └── full_patch_embeddings/     # Raw 196×768 patch data
├── production_data/
│   ├── umap_coordinates_filtered.json
│   ├── locations_metadata.json
│   └── dataset_statistics.json
└── spatial_correlation_analysis/
    └── spatial_correlation_results.json
```

## Key Innovation

**Patch-Level Filtering:** The pipeline applies dimension filtering to raw 196×768 patch embeddings before aggregation, ensuring spatial bias removal affects the fundamental building blocks rather than just final aggregated vectors.

## Data Quality

**Spatial Bias Removal:** Excluded roughly 8% of dimensions identified as most representative of geographic location
**Consistency:** Identical filtering applied across UMAP generation and vector database preparation
**Validation:** Comprehensive logging and statistical analysis of filtering effectiveness
