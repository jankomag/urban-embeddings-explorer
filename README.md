# Urban Embeddings Explorer 🌏🏙️🔎

AI-powered satellite imagery analysis system for discovering visually similar urban areas worldwide using IBM TerraMind foundation model embeddings.
Explore the live webapp: https://urban-embeddings-explorer.vercel.app/

## Overview

This system processes satellite imagery from global cities to extract visual similarity features and enables interactive exploration through a web application. The pipeline uses TerraMind v1.0 to generate 768-dimensional embeddings, applies spatial bias filtering, and provides multiple similarity search methods through a vector database backend.

## Repository Structure

```
urban-embeddings-explorer/
├── analysis/              # Data processing pipeline
│   ├── create_embeddings.py
│   ├── spatial-correlation.py
│   ├── umap-generator.py
│   ├── data-migration.py
│   └── README.md         # Analysis pipeline documentation
├── webapp/               # Web application
│   ├── backend/          # FastAPI + Qdrant server
│   ├── frontend/         # React + Mapbox interface
│   └── README.md        # Web application documentation
└── README.md            # This file
```

## System Architecture

```
Satellite Data → TerraMind Model → Spatial Filtering → Vector Database → Web Interface
```

**Data Flow:**

1. **Analysis Pipeline**: Extracts and processes embeddings from satellite imagery
2. **Web Application**: Provides interactive similarity search and visualization

## Quick Start

### Analysis Pipeline

```bash
cd analysis/
python create_embeddings.py      # Extract TerraMind embeddings
python spatial-correlation.py    # Remove spatial bias
python umap-generator.py          # Generate 2D coordinates
python data-migration.py         # Upload to vector database
```

### Web Application

```bash
cd webapp/backend/
python main.py                   # Start FastAPI server

cd ../frontend/
npm install && npm start        # Start React interface
```

## Core Technology

**AI Model:** IBM TerraMind v1.0 foundation model for Earth observation

- Repository: https://github.com/IBM/terramind
- Generates 768-dimensional embeddings from 224×224m satellite tiles
- Processes 196 patches per tile through vision transformer architecture

**Vector Database:** Qdrant for similarity search with three distinct aggregation methods
**Visualization:** UMAP dimensionality reduction for interactive 2D exploration
**Web Stack:** FastAPI backend with React frontend and Mapbox satellite imagery

## Key Features

**Spatial Bias Filtering:** Removes geographic location encoding to focus on visual similarity
**Multiple Aggregation Methods:** Six different approaches for various similarity search modes
**Interactive Interface:** Real-time similarity search with map and UMAP visualization
**Global Coverage:** Processes urban areas from cities worldwide

## Requirements

**Analysis Pipeline:**

- Python 3.10+, TerraTorch, GDAL
- GPU recommended for TerraMind inference

**Web Application:**

- Node.js 16+, Python 3.10+, Qdrant instance
- Mapbox API token for satellite imagery

See individual README files in `analysis/` and `webapp/` directories for detailed setup instructions and technical specifications.
