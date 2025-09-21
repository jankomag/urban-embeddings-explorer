# Web Application

Interactive web interface for exploring urban visual similarity using TerraMind embeddings and vector search.

## Overview

React-based web application with FastAPI backend that enables real-time similarity search across global urban satellite imagery. Users can click any location to find visually similar urban areas using multiple AI aggregation methods.

## Architecture

```
React Frontend ←→ FastAPI Backend ←→ Qdrant Vector Database
      ↓                                      ↓
   Mapbox Maps                        TerraMind Embeddings
   UMAP Visualization                 Similarity Search
```

## Technology Stack

### Frontend (`frontend/`)

**Core Framework:**

- React 18 with functional components and hooks
- Create React App build system
- Modern ES6+ JavaScript

**Mapping & Visualization:**

- Mapbox GL JS for satellite imagery and interactive maps
- Custom UMAP scatter plot visualization using HTML5 Canvas
- Responsive design with CSS Grid and Flexbox

**State Management:**

- React Context API for global application state
- useState and useEffect hooks for component state
- Custom hooks for API interactions and data fetching

**UI Components:**

- Custom-built interface components
- Dark/light theme support
- Mobile-responsive design
- Real-time loading states and error handling

### Backend (`backend/`)

**API Framework:**

- FastAPI with automatic OpenAPI documentation
- Pydantic models for request/response validation
- CORS middleware for cross-origin requests
- Async/await for non-blocking operations

**Vector Database:**

- Qdrant for high-performance similarity search
- Three collection types for different aggregation methods
- Cosine similarity with configurable result limits
- Batch processing for multiple similarity queries

## Key Features

### Similarity Search

- **Click-based Search:** Click any map location to find similar areas
- **Multiple Methods:** Three different aggregation approaches (mean, median, dominant_cluster)
- **Real-time Results:** Sub-second similarity search across 40k+ locations
- **Configurable Parameters:** Adjustable similarity thresholds and result counts

### Interactive Visualization

- **Dual View:** Satellite map + UMAP embedding space
- **Synchronized Selection:** Selections sync between map and UMAP views
- **Zoom and Pan:** Smooth navigation across global dataset
- **Result Highlighting:** Visual highlighting of similar locations

### Data Display

- **Location Metadata:** City, country, continent information
- **Similarity Scores:** Quantitative similarity measurements
- **Coordinate Information:** Latitude/longitude display
- **Tile Boundaries:** Optional display of analysis tile boundaries

## API Endpoints

### Similarity Search

```
POST /api/similarity
{
  "location_id": "string",
  "method": "mean|median|min|max|dominant_cluster|global_contrastive",
  "limit": 50
}
```

### Location Data

```
GET /api/locations          # All location metadata
GET /api/umap-coordinates   # UMAP 2D coordinates
GET /api/stats              # Dataset statistics
```

### Health Check

```
GET /api/health            # API status
```

## Setup Instructions

### Frontend Setup

```bash
cd frontend/
npm install
npm start                  # Development server on localhost:3000
```

**Environment Configuration:**

```bash
# .env file
REACT_APP_API_URL=http://localhost:8000
REACT_APP_MAPBOX_TOKEN=your_mapbox_token
```

### Backend Setup

```bash
cd backend/
pip install fastapi uvicorn qdrant-client pandas numpy
python main.py            # Server on localhost:8000
```

**Environment Configuration:**

```bash
# .env file
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=optional_api_key
CORS_ORIGINS=http://localhost:3000
```

## Development

### Frontend Development

```bash
npm run start             # Development server with hot reload
npm run build             # Production build
npm run test              # Run test suite
```

### Backend Development

```bash
uvicorn main:app --reload # Development server with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000  # Production server
```

## Deployment

**Production Build:**

```bash
# Frontend
npm run build
# Serve static files with nginx or similar

# Backend
uvicorn main:app --host 0.0.0.0 --port 8000
# Use gunicorn for production WSGI server
```
