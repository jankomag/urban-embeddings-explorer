# 🛰️ Urban Embeddings Explorer

**AI-powered satellite imagery analysis to find visually similar urban areas worldwide**

## 🎯 What It Does

Click any city on the map → AI finds visually similar places globally using satellite imagery patterns. Discover urban areas that look alike from space! 🌍

## 🏗️ Architecture

```
Satellite Data → TerraMind AI → Vector Search → Interactive Web App
```

- **Analysis Pipeline**: Extract embeddings, filter dimensions, create visualizations
- **Web Application**: Interactive map + similarity search with 6 different AI methods
- **Vector Database**: Fast similarity search across 100K+ urban tiles

## 🌐 Live Demo

**[View the live application here →](YOUR_DEPLOYED_URL_HERE)**

## 🧪 Research Pipeline

The analysis pipeline processes satellite imagery to create AI embeddings:

```bash
cd analysis
python create_embeddings.py      # Extract AI embeddings from satellite data
python spatial-correlation.py    # Identify & remove geographic bias
python umap-generator.py          # Create 2D visualizations
python data-migration.py         # Prepare vector database
```

## 🧠 AI Methods

- **Mean/Median**: General visual similarity
- **Min/Max**: Shared/distinctive features
- **Dominant Cluster**: Recurring patterns
- **Global Contrastive**: Unique vs typical

## 🎨 Features

- 🗺️ Interactive satellite map (Mapbox)
- 📊 2D embedding visualization (UMAP)
- 🔍 Real-time similarity search
- 🌓 Dark/light themes
- 📱 Mobile responsive

## 📁 Structure

```
urban-exp-app/
├── analysis/          # AI pipeline scripts
│   ├── create_embeddings.py
│   ├── spatial-correlation.py
│   ├── umap-generator.py
│   └── data-migration.py
└── webapp/           # Web application
    ├── backend/      # FastAPI + Qdrant
    └── frontend/     # React + Mapbox
```

## 🔧 Requirements

- Python 3.11+ (TerraMind, FastAPI)
- Node.js 16+ (React frontend)
- Qdrant vector database
- Mapbox API token

## 🌟 Perfect For

Urban planners, researchers, travelers, or anyone curious about how cities look from space and which ones share similar visual DNA!

**Example**: Click Manhattan → Find dense urban cores in Hong Kong, São Paulo, Tokyo... 🏙️✨
