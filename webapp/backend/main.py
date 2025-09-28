from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import json
import time
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import logging
import gzip
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from functools import lru_cache

# Import simplified models
from models import (
    LocationData, SimilarityResponse, StatsResponse, 
    ConfigResponse, UMapResponse, SimilarLocation,
    PaginationInfo, UMapPoint, BoundsStatistics
)

@lru_cache(maxsize=1000)
def get_cached_similarity(location_id: int, method: str, limit: int):
    # Cache frequent queries in memory
    pass

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)

# Load environment variables
load_dotenv()

def load_data():
    """Load lightweight metadata from gzipped files."""
    try:
        data_dir = 'production_data'
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Load location metadata
        locations_file = os.path.join(data_dir, 'locations_metadata.json.gz')
        if os.path.exists(locations_file):
            with gzip.open(locations_file, 'rt') as f:
                locations = json.load(f)
            logger.info(f"📍 Loaded {len(locations)} locations")
        else:
            raise FileNotFoundError(f"Locations metadata not found: {locations_file}")
        
        # Load UMAP coordinates
        umap_file = os.path.join(data_dir, 'umap_coordinates.json.gz')
        if not os.path.exists(umap_file):
            umap_file = os.path.join(data_dir, 'umap_coordinates.json')
        
        if os.path.exists(umap_file):
            if umap_file.endswith('.gz'):
                with gzip.open(umap_file, 'rt') as f:
                    umap_coords = json.load(f)
            else:
                with open(umap_file, 'r') as f:
                    umap_coords = json.load(f)
            logger.info(f"🗺️ Loaded UMAP coordinates")
        else:
            logger.warning("⚠️ UMAP coordinates not found")
            umap_coords = None
        
        # Load dataset statistics
        stats_file = os.path.join(data_dir, 'dataset_statistics.json.gz')
        if not os.path.exists(stats_file):
            stats_file = os.path.join(data_dir, 'dataset_statistics.json')
        
        if os.path.exists(stats_file):
            if stats_file.endswith('.gz'):
                with gzip.open(stats_file, 'rt') as f:
                    stats = json.load(f)
            else:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            logger.info(f"📊 Loaded dataset statistics")
        else:
            logger.warning("⚠️ Dataset statistics not found")
            stats = None
        
        return locations, umap_coords, stats
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from qdrant and httpx
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING) 
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# CREATE APP ONLY ONCE
app = FastAPI(title="Satellite Embeddings Explorer", version="4.0.0")

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Check if we're in production
is_production = os.getenv("ENVIRONMENT") == "production" or os.getenv("RAILWAY_ENVIRONMENT") == "production"

# Add security middleware (conditionally for production)
if is_production:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*.railway.app", "your-domain.com"])
    app.add_middleware(HTTPSRedirectMiddleware)

# Security headers (always add these)
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Only add HTTPS headers in production
    if is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# Add CORS middleware
cors_origins = [
    "http://localhost:3000",
    "https://urban-embeddings-explorer.vercel.app"
]

# Remove localhost in production
if is_production:
    cors_origins = [origin for origin in cors_origins if not origin.startswith("http://localhost")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for data
locations_data = None
umap_data = None
dataset_stats = None
qdrant_client = None

# Qdrant collection names
COLLECTIONS = {
    'mean': 'terramind_embeddings_mean',
    'dominant_cluster': 'terramind_embeddings_dominant_cluster'
}

class SimilarityCache:
    def __init__(self, max_size: int = 1000, ttl: int = 900):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[List]:
        if key in self.cache:
            if time.time() - self.access_times[key] > self.ttl:
                del self.cache[key]
                del self.access_times[key]
                return None
            
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: List):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()

# Initialize cache
similarity_cache = SimilarityCache()

def setup_qdrant_client():
    """Initialize Qdrant client."""
    qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    try:
        client_kwargs = {'url': qdrant_url, 'timeout': 60}
        if qdrant_api_key and qdrant_api_key != 'your_api_key_if_needed':
            client_kwargs['api_key'] = qdrant_api_key
        
        qdrant_client = QdrantClient(
            **client_kwargs,
            prefer_grpc=True,
            grpc_port=6334
        )
        
        # Test connection
        collections = qdrant_client.get_collections()
        available_collection_names = [c.name for c in collections.collections]
        
        logger.info(f"✅ Connected to Qdrant")
        logger.info(f"📦 Found {len(available_collection_names)} collections")
        
        return qdrant_client
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        raise

async def query_qdrant_similarity(
    target_location_id: int, 
    collection_name: str,
    limit: int = 100,
    offset: int = 0
) -> tuple[List[tuple], int]:
    """Query Qdrant for similar vectors."""
    try:
        # Get the target vector
        target_points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[target_location_id],
            with_vectors=True
        )
        
        if not target_points:
            raise ValueError(f"Target location {target_location_id} not found in {collection_name}")
        
        target_vector = target_points[0].vector
        
        # Search for results
        search_limit = max(500, (offset + limit) * 2)
    
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=target_vector,
            limit=search_limit,
            with_payload=True,
            score_threshold=0.0
        )
        
        # Deduplication
        seen_location_ids: Set[int] = set()
        seen_coordinates: Set[tuple] = set()
        seen_city_country: Set[tuple] = set()
        unique_results = []
        
        for result in search_results:
            location_id = int(result.id)
            
            # Skip the target location itself
            if location_id == target_location_id:
                continue
            
            # Skip if already seen
            if location_id in seen_location_ids:
                continue
            
            # Extract data
            try:
                longitude = float(result.payload['longitude'])
                latitude = float(result.payload['latitude'])
                city = str(result.payload['city'])
                country = str(result.payload['country'])
                continent = str(result.payload.get('continent', 'Unknown'))
            except (KeyError, ValueError, TypeError):
                continue
            
            # Skip duplicate coordinates
            coord_key = (round(latitude, 6), round(longitude, 6))
            if coord_key in seen_coordinates:
                continue
            
            # Limit results per city
            city_country_key = (city.lower().strip(), country.lower().strip())
            city_country_count = sum(1 for cc in seen_city_country if cc == city_country_key)
            
            MAX_PER_CITY = 3
            if city_country_count >= MAX_PER_CITY:
                continue
            
            # Record this location
            seen_location_ids.add(location_id)
            seen_coordinates.add(coord_key)
            seen_city_country.add(city_country_key)
            
            location_data = {
                'id': location_id,
                'city': city,
                'country': country,
                'continent': continent,
                'longitude': longitude,
                'latitude': latitude,
                'date': result.payload.get('date')
            }
            similarity_score = float(result.score)
            unique_results.append((location_data, similarity_score))
        
        total_available = len(unique_results)
        paginated_results = unique_results[offset:offset + limit]
        
        return paginated_results, total_available
        
    except Exception as e:
        logger.error(f"❌ Error querying Qdrant similarity: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    global locations_data, umap_data, dataset_stats, qdrant_client
    
    try:
        logger.info("🚀 Starting Satellite Embeddings Explorer")
        
        # Setup Qdrant client
        qdrant_client = setup_qdrant_client()
        
        # Load data
        locations_data, umap_data, dataset_stats = load_data()
        
        logger.info("✅ Application ready")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        locations_data = []
        umap_data = None
        dataset_stats = None

@app.get("/")
async def root():
    """Root endpoint with system status."""
    qdrant_status = "connected" if qdrant_client else "disconnected"
    
    collection_info = {}
    if qdrant_client:
        try:
            collections = qdrant_client.get_collections()
            for method, collection_name in COLLECTIONS.items():
                exists = any(c.name == collection_name for c in collections.collections)
                collection_info[method] = {
                    "collection_name": collection_name,
                    "exists": exists
                }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
    
    return {
        "message": "Satellite Embeddings Explorer", 
        "version": "4.0.0",
        "status": "running",
        "qdrant_status": qdrant_status,
        "similarity_methods": list(COLLECTIONS.keys()),
        "collections": collection_info,
        "locations_loaded": len(locations_data) if locations_data else 0
    }

@limiter.limit("3/minute")
@app.get("/api/similarity/{location_id}", response_model=SimilarityResponse)
async def find_similar_locations(
    request: Request,
    location_id: int, 
    offset: int = Query(0, ge=0),
    limit: int = Query(6, ge=1, le=20),
    method: str = Query("mean")
):
    # Add usage tracking per IP
    client_ip = request.client.host
    
    """Find similar locations using Qdrant vector similarity."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    
    if not locations_data:
        raise HTTPException(status_code=503, detail="Location data not loaded")
    
    # Validate method
    if method not in COLLECTIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid method '{method}'. Available: {list(COLLECTIONS.keys())}"
        )
    
    # Find target location
    target_location = None
    for location in locations_data:
        if location['id'] == location_id:
            target_location = location
            break
    
    if target_location is None:
        raise HTTPException(status_code=404, detail="Location not found")
    
    # Check cache
    cache_key = f"similarity_{location_id}_{method}_{offset}_{limit}"
    cached_result = similarity_cache.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        collection_name = COLLECTIONS[method]
        
        # Query Qdrant
        similar_results, total_results = await query_qdrant_similarity(
            target_location_id=location_id,
            collection_name=collection_name,
            limit=limit,
            offset=offset
        )
        
        # Format response
        similar_locations = []
        for location_data, sim_score in similar_results:
            if location_data['id'] == location_id:
                continue
                
            similar_location = SimilarLocation(
                id=location_data['id'],
                city=location_data['city'],
                country=location_data['country'],
                continent=location_data['continent'],
                longitude=location_data['longitude'],
                latitude=location_data['latitude'],
                date=location_data['date'],
                similarity_score=sim_score
            )
            similar_locations.append(similar_location)
        
        # Create pagination info
        has_more = (offset + len(similar_locations)) < total_results
        next_offset = offset + limit if has_more else None
        
        pagination = PaginationInfo(
            offset=offset,
            limit=limit,
            total_results=total_results,
            has_more=has_more,
            next_offset=next_offset,
            returned_count=len(similar_locations)
        )
        
        response = SimilarityResponse(
            target_location_id=location_id,
            similar_locations=similar_locations,
            method_used=f"qdrant_{method}",
            pagination=pagination
        )
        
        # Cache the result
        similarity_cache.set(cache_key, response)
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@limiter.limit("10/minute")
@app.get("/api/locations", response_model=List[LocationData])
async def get_locations(request: Request):
    """Get all locations."""
    if not locations_data:
        raise HTTPException(status_code=503, detail="Location data not loaded")
    
    locations = []
    for location_data in locations_data:
        # Extract exact tile bounds from the data
        tile_bounds = location_data.get('tile_bounds')
        
        location = LocationData(
            id=location_data['id'],
            city=location_data['city'],
            country=location_data['country'],
            continent=location_data['continent'],
            longitude=location_data['longitude'],
            latitude=location_data['latitude'],
            date=location_data.get('date'),
            bounds=tile_bounds  # Use exact bounds from embeddings data
        )
        locations.append(location)
    
    return locations

@app.get("/api/umap", response_model=UMapResponse)
async def get_umap_embeddings():
    """Get UMAP 2D coordinates."""
    if not umap_data:
        raise HTTPException(status_code=503, detail="UMAP data not available")
    
    umap_points = []
    for point_data in umap_data['umap_points']:
        # Extract exact tile bounds from UMAP data
        tile_bounds = point_data.get('tile_bounds')
        
        umap_point = UMapPoint(
            location_id=point_data['location_id'],
            x=point_data['x'],
            y=point_data['y'],
            city=point_data['city'],
            country=point_data['country'],
            continent=point_data['continent'],
            longitude=point_data['longitude'],
            latitude=point_data['latitude'],
            date=point_data.get('date'),
            bounds=tile_bounds  # Use exact bounds from embeddings data
        )
        umap_points.append(umap_point)
    
    return UMapResponse(
        umap_points=umap_points,
        total_points=len(umap_points)
    )

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get dataset statistics."""
    if not locations_data:
        raise HTTPException(status_code=503, detail="Location data not loaded")
    
    countries = list(set(loc['country'] for loc in locations_data))
    continents = list(set(loc['continent'] for loc in locations_data))
    
    stats = StatsResponse(
        total_locations=len(locations_data),
        countries_count=len(countries),
        continents_count=len(continents),
        embedding_dimension=768,
        countries=sorted(countries),
        continents=sorted(continents),
        similarity_method="qdrant_vector_search"
    )
    
    return stats

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get configuration including Mapbox token."""
    return ConfigResponse(
        mapbox_token=os.getenv("MAPBOX_TOKEN", "")
    )

@app.get("/api/methods")
async def get_similarity_methods():
    """Get available similarity methods."""
    methods = {
        "mean": {
            "name": "Mean",
            "description": "Standard aggregation",
            "collection": COLLECTIONS["mean"]
        },
        # "median": {
        #     "name": "Median", 
        #     "description": "Robust to outliers",
        #     "collection": COLLECTIONS["median"]
        # },
        # "min": {
        #     "name": "Min",
        #     "description": "Shared baseline features", 
        #     "collection": COLLECTIONS["min"]
        # },
        # "max": {
        #     "name": "Max",
        #     "description": "Distinctive features",
        #     "collection": COLLECTIONS["max"]
        # },
        "dominant_cluster": {
            "name": "Dominant Cluster",
            "description": "Most frequent pattern",
            "collection": COLLECTIONS["dominant_cluster"]
        },
        "global_contrastive": {
            "name": "Global Contrastive",
            "description": "Unique minus mean",
            "collection": COLLECTIONS["global_contrastive"]
        }
    }
    
    # Add availability information
    for method_key, method_info in methods.items():
        if qdrant_client:
            try:
                collections = qdrant_client.get_collections()
                collection_exists = any(c.name == method_info["collection"] for c in collections.collections)
                method_info["available"] = collection_exists
            except:
                method_info["available"] = False
        else:
            method_info["available"] = False
    
    return {
        "available_methods": methods,
        "default_method": "mean"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }
    
    # Check Qdrant
    try:
        if qdrant_client:
            collections = qdrant_client.get_collections()
            collection_status = {}
            for method, collection_name in COLLECTIONS.items():
                exists = any(c.name == collection_name for c in collections.collections)
                collection_status[method] = {
                    "collection_name": collection_name,
                    "exists": exists
                }
            
            health_status["components"]["qdrant"] = {
                "status": "healthy",
                "collections": collection_status
            }
        else:
            health_status["components"]["qdrant"] = {"status": "disconnected"}
    except Exception as e:
        health_status["components"]["qdrant"] = {"status": "error", "error": str(e)}
    
    health_status["components"]["data"] = {
        "locations_loaded": len(locations_data) if locations_data else 0,
        "umap_available": umap_data is not None,
        "stats_available": dataset_stats is not None
    }
    
    return health_status

@app.post("/api/cache/clear")
async def clear_cache():
    """Clear the similarity cache."""
    similarity_cache.clear()
    logger.info("🧹 Similarity cache cleared")
    return {"message": "Cache cleared", "status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)