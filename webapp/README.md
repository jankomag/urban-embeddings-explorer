# 🌍 Urban Embeddings Explorer Web App

Interactive web application for exploring visually similar urban areas using AI-powered satellite imagery analysis.

## 🌐 Live Application

**[Explore the deployed app here →](YOUR_DEPLOYED_URL_HERE)**

## 🏗️ Architecture

**Frontend** (React) ↔️ **Backend** (FastAPI) ↔️ **Vector DB** (Qdrant)

---

## 📁 Backend Files

### Core Application

- **`main.py`** 🚀: FastAPI server with similarity search endpoints
- **`models.py`** 📝: Pydantic data models for API requests/responses
- **`database.py`** 🗄️: Database connection utilities (PostgreSQL optional)

### Configuration

- **`requirements.txt`** 📦: Python dependencies
- **`railway.toml`** 🚂: Railway.app deployment config
- **`runtime.txt`** 🐍: Python version specification

### Data

- **`production_data/`** 📊:
  - `locations_metadata.json.gz`: City locations and metadata
  - `umap_coordinates.json.gz`: 2D visualization coordinates

## 🖥️ Frontend Files

### Core Components

- **`index.js`** 🎯: Main app component with state management
- **`MapView.js`** 🗺️: Interactive Mapbox satellite map
- **`HighPerformanceUMapView.js`** 📊: Canvas-based UMAP visualization
- **`SimilarityPanel.js`** 🔍: Search results and controls

### UI Components

- **`Header.js`** 📱: Navigation bar with search and filters
- **`HelpPanel.js`** ❓: Documentation and method explanations
- **`AutocompleteInput.js`** 🔎: Smart city/country search
- **`ThemeProvider.js`** 🎨: Dark/light mode management
- **`ThemeToggle.js`** 🌓: Theme switcher button

### Styling

- **`ModernApp.css`** 💄: Main application styles with CSS variables
- **`HelpPanel.css`** 📖: Help modal specific styles

### Configuration

- **`package.json`** 📦: Node.js dependencies and scripts
- **`public/index.html`** 🌐: HTML entry point

## 🔧 Development Setup

For local development:

```bash
# Backend
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd webapp/frontend
npm install && npm start
```

Requires environment variables: `MAPBOX_TOKEN`, `QDRANT_URL`

---

## 🔌 API Endpoints

### Core Endpoints

- `GET /api/locations` 📍: All city locations with coordinates
- `GET /api/similarity/{id}` 🔍: Find similar locations using vector search
- `GET /api/umap` 📊: 2D coordinates for visualization
- `GET /api/stats` 📈: Dataset statistics and metadata

### Configuration

- `GET /api/config` ⚙️: Mapbox token and app settings
- `GET /api/methods` 🧠: Available similarity methods
- `GET /api/health` 💚: System health check

### Utility

- `POST /api/cache/clear` 🧹: Clear similarity search cache

---

## 🎛️ Features

### Interactive Map 🗺️

- **Satellite imagery** powered by Mapbox
- **Click tiles** to select and find similar areas
- **City filtering** via search bar
- **Zoom controls** and smooth navigation

### UMAP Visualization 📊

- **2D projection** of high-dimensional embeddings
- **Continental coloring** for geographic context
- **Interactive selection** synced with map
- **Pan and zoom** with smooth animations

### Similarity Search 🔍

- **6 aggregation methods**: Mean, Median, Min, Max, Dominant Cluster, Global Contrastive
- **Same-city filtering** toggle
- **Pagination** for large result sets
- **Satellite thumbnails** for each result

### Smart UI 🧠

- **Dark/light themes** with CSS variables
- **Responsive design** for mobile/desktop
- **Auto-complete search** for cities and countries
- **Contextual help** panel with method explanations

---

## 🎨 Similarity Methods

| Method                    | Description               | Best For                |
| ------------------------- | ------------------------- | ----------------------- |
| **Mean** 📊               | Standard patch averaging  | General similarity      |
| **Median** 📈             | Robust to outliers        | Consistent patterns     |
| **Min** ⬇️                | Shared baseline features  | Common characteristics  |
| **Max** ⬆️                | Distinctive features      | Unique landmarks        |
| **Dominant Cluster** 🎯   | Most frequent patterns    | Recurring visual themes |
| **Global Contrastive** 🌍 | Dataset-relative encoding | Distinctive vs average  |

---

## 🎯 What This Does

The webapp enables users to:

1. **Explore** cities on an interactive satellite map 🛰️
2. **Click** any urban area to analyze its visual patterns 🔍
3. **Discover** visually similar places worldwide using AI 🤖
4. **Compare** different similarity methods and see how they work 🔬
5. **Visualize** the embedding space through UMAP projections 📊

Perfect for urban planners, researchers, or anyone curious about how cities look from space! 🌍✨
