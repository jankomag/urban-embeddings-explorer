# Embeddings Viewer Web Application

This web application visualizes global embeddings using t-SNE and geographic mapping.

## Setup

1. Ensure you have Docker and Docker Compose installed.
2. Place your `global_tsne_results.parquet` file in the `data/` directory.
3. Run `docker-compose up --build` to start the application.
4. Access the web application at `http://localhost:3000`.

## Development

- Backend: FastAPI application in `backend/`
- Frontend: React application in `frontend/`

To run in development mode:

1. For backend: `cd backend && pip install -r requirements.txt && uvicorn main:app --reload`
2. For frontend: `cd frontend && npm install && npm start`

## API Endpoints

- `/tsne_data`: Get t-SNE plot data
- `/map_data`: Get geographic map data

## Technologies Used

- Backend: FastAPI, PyArrow, Shapely
- Frontend: React, Chart.js, Leaflet
- Data Storage: Parquet

```
webapp/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js
│   │   ├── components/
│   │   │   ├── ScatterPlot.js
│   │   │   └── MapComponent.js
│   │   └── index.js
│   ├── package.json
│   └── Dockerfile
│
├── data/
│   └── global_tsne_results.parquet
│
├── docker-compose.yml
└── README.md
```
