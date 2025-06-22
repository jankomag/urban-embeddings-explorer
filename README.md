# 🛰️ Satellite Embeddings Explorer

Interactive map for exploring city embeddings from satellite imagery. Click cities to find similar locations using AI.

## Quick Start

### Backend

```bash
cd backend
pip install fastapi uvicorn python-dotenv sqlalchemy shapely numpy
echo "MAPBOX_TOKEN=your_mapbox_token_here" > .env
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm start
```

Open http://localhost:3000

## Features

- 🌍 Interactive satellite map (Mapbox GL)
- 🔍 Click cities to view details
- 🤝 AI-powered similarity search
- 📊 Dataset statistics

## Tech Stack

- **Frontend**: React + Mapbox GL JS
- **Backend**: FastAPI + Python
- **Database**: PostgreSQL (optional, includes test data)

## API Endpoints

- `GET /api/config` - Mapbox token
- `GET /api/locations` - All locations
- `GET /api/similarity/{id}` - Find similar cities

## Environment

Create `backend/.env`:

```
MAPBOX_TOKEN=pk.eyJ...your_token_here
DATABASE_URL=postgresql://user:pass@host:5432/db  # optional
```

Get free Mapbox token at https://mapbox.com
