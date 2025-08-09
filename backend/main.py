from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import json
import asyncio
import time
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import logging
import gzip

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/locations")
@limiter.limit("30/minute")  # Max 30 requests per minute per IP
async def get_locations(request: Request):
    # Your existing code
    pass

# Import your enhanced models
from models import (
    LocationData, SimplifiedSimilarityResponse, SimplifiedStatsResponse, 
    ConfigResponse, TileBoundsResponse, UMapResponse, SimilarLocation,
    PaginationInfo, UMapPoint, CityRepresentativesResponse, BoundsStatistics,
    TileBoundsBatchResponse
)

# Load environment variables
load_dotenv()

def load_lightweight_data():
    """Load enhanced lightweight metadata with exact bounds from gzipped files."""
    try:
        # Updated paths for Railway deployment
        possible_paths = [
            './production_data',  # Current structure
            '../production_data', 
            '/app/production_data',  # Railway deployment path
            './backend/production_data'  # If running from root
        ]
        
        data_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                logger.info(f"📁 Found data directory: {data_dir}")
                break
        
        if not data_dir:
            raise FileNotFoundError(f"Data directory not found. Tried: {possible_paths}")
        
        # Load location metadata with bounds - try both .gz and .json
        locations_file = os.path.join(data_dir, 'locations_metadata.json.gz')
        if not os.path.exists(locations_file):
            locations_file = os.path.join(data_dir, 'locations_metadata.json')
        
        if os.path.exists(locations_file):
            if locations_file.endswith('.gz'):
                with gzip.open(locations_file, 'rt') as f:
                    locations = json.load(f)
            else:
                with open(locations_file, 'r') as f:
                    locations = json.load(f)
                    
            # Count bounds statistics
            exact_bounds_count = sum(1 for loc in locations if loc.get('has_exact_bounds', False))
            total_count = len(locations)
            
            logger.info(f"📍 Loaded {total_count} location records")
            logger.info(f"🎯 Exact bounds: {exact_bounds_count}/{total_count} ({exact_bounds_count/total_count*100:.1f}%)")
        else:
            raise FileNotFoundError(f"Locations metadata not found: {locations_file}")
        
        # Load city representatives - try both .gz and .json
        city_file = os.path.join(data_dir, 'city_representatives.json.gz')
        if not os.path.exists(city_file):
            city_file = os.path.join(data_dir, 'city_representatives.json')
        
        if os.path.exists(city_file):
            if city_file.endswith('.gz'):
                with gzip.open(city_file, 'rt') as f:
                    city_data = json.load(f)
            else:
                with open(city_file, 'r') as f:
                    city_data = json.load(f)
            logger.info(f"🏙️ Loaded {city_data['total_cities']} city representatives")
        else:
            logger.warning("⚠️ City representatives not found, will generate from locations")
            city_data = None
        
        # Load UMAP coordinates - try both .gz and .json
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
            logger.info(f"🗺️ Loaded UMAP coordinates for {umap_coords['total_points']} points")
            
            # Log bounds statistics if available
            if 'bounds_statistics' in umap_coords:
                bounds_stats = umap_coords['bounds_statistics']
                logger.info(f"📊 UMAP bounds coverage: {bounds_stats.get('exact_bounds_percentage', 0):.1f}%")
        else:
            logger.warning("⚠️ UMAP coordinates not found, will compute on-demand")
            umap_coords = None
        
        # Load dataset statistics - try both .gz and .json
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
            
            # Log enhanced features if available
            if 'enhanced_features' in stats:
                enhanced = stats['enhanced_features']
                if enhanced.get('exact_tile_bounds'):
                    logger.info(f"✅ Enhanced features: exact bounds coverage {enhanced.get('bounds_coverage_percentage', 0):.1f}%")
                if enhanced.get('adaptive_mixed_aggregation'):
                    logger.info(f"🔄 Adaptive mixed aggregation available")
                    if 'adaptive_coverage_percentage' in enhanced:
                        logger.info(f"🔄 Adaptive coverage: {enhanced['adaptive_coverage_percentage']:.1f}%")
        else:
            logger.warning("⚠️ Dataset statistics not found")
            stats = None
        
        return locations, city_data, umap_coords, stats
        
    except Exception as e:
        logger.error(f"❌ Error loading enhanced data: {e}")
        raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Satellite Embeddings Explorer - With Adaptive Mixed Aggregation", version="3.2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://urban-embeddings-explorer.vercel.app/",
        "https://urban-embeddings-explorer-git-main-jankomags-projects.vercel.app/",
        "https://urban-embeddings-explorer-kpvnbn384-jankomags-projects.vercel.app/",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "https://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for lightweight data
locations_data = None
umap_data = None
dataset_stats = None
city_representatives_data = None
qdrant_client = None

# Updated Qdrant collection names with adaptive mixed
COLLECTIONS = {
    'regular': 'satellite_embeddings_simple',
    'global_contrastive': 'satellite_embeddings_global_contrastive_simple',
    'adaptive_mixed': 'satellite_embeddings_adaptive_mixed_simple'
}

# Simple in-memory cache
similarity_cache = {}
cache_ttl = 900  # 15 minutes
max_cache_size = 1000

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
similarity_cache = SimilarityCache(max_cache_size, cache_ttl)

def setup_qdrant_client():
    """Initialize Qdrant client with environment configuration."""
    qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    try:
        client_kwargs = {'url': qdrant_url, 'timeout': 60}
        if qdrant_api_key and qdrant_api_key != 'your_api_key_if_needed':
            client_kwargs['api_key'] = qdrant_api_key
        
        client = QdrantClient(**client_kwargs)
        
        # Test connection
        collections = client.get_collections()
        available_collection_names = [c.name for c in collections.collections]
        
        logger.info(f"✅ Connected to Qdrant at {qdrant_url}")
        logger.info(f"📦 Available collections: {available_collection_names}")
        
        for method, collection_name in COLLECTIONS.items():
            if collection_name in available_collection_names:
                logger.info(f"✅ Found collection for {method}: {collection_name}")
            else:
                logger.warning(f"❌ Missing collection for {method}: {collection_name}")
        
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        raise

# def load_lightweight_data():
#     """Load enhanced lightweight metadata with exact bounds."""
#     try:
#         possible_paths = [
#             './production_data',
#             '../production_data', 
#             './migration/production_data',
#             '../migration/production_data',
#             '/Users/janmagnuszewski/dev/terramind/embeddings/production_data'
#         ]
        
#         data_dir = None
#         for path in possible_paths:
#             if os.path.exists(path):
#                 data_dir = path
#                 logger.info(f"📁 Found data directory: {data_dir}")
#                 break
        
#         if not data_dir:
#             raise FileNotFoundError(f"Data directory not found. Tried: {possible_paths}")
        
#         # Load location metadata with bounds
#         locations_file = os.path.join(data_dir, 'locations_metadata.json')
#         if os.path.exists(locations_file):
#             with open(locations_file, 'r') as f:
#                 locations = json.load(f)
                
#                 # Count bounds statistics
#                 exact_bounds_count = sum(1 for loc in locations if loc.get('has_exact_bounds', False))
#                 total_count = len(locations)
                
#                 logger.info(f"📍 Loaded {total_count} location records")
#                 logger.info(f"🎯 Exact bounds: {exact_bounds_count}/{total_count} ({exact_bounds_count/total_count*100:.1f}%)")
#         else:
#             raise FileNotFoundError(f"Locations metadata not found: {locations_file}")
        
#         # Load city representatives
#         city_file = os.path.join(data_dir, 'city_representatives.json')
#         if os.path.exists(city_file):
#             with open(city_file, 'r') as f:
#                 city_data = json.load(f)
#                 logger.info(f"🏙️ Loaded {city_data['total_cities']} city representatives")
#         else:
#             logger.warning("⚠️ City representatives not found, will generate from locations")
#             city_data = None
        
#         # Load UMAP coordinates with bounds
#         umap_file = os.path.join(data_dir, 'umap_coordinates.json')
#         if os.path.exists(umap_file):
#             with open(umap_file, 'r') as f:
#                 umap_coords = json.load(f)
#                 logger.info(f"🗺️ Loaded UMAP coordinates for {umap_coords['total_points']} points")
                
#                 # Log bounds statistics if available
#                 if 'bounds_statistics' in umap_coords:
#                     bounds_stats = umap_coords['bounds_statistics']
#                     logger.info(f"📊 UMAP bounds coverage: {bounds_stats.get('exact_bounds_percentage', 0):.1f}%")
#         else:
#             logger.warning("⚠️ UMAP coordinates not found, will compute on-demand")
#             umap_coords = None
        
#         # Load dataset statistics
#         stats_file = os.path.join(data_dir, 'dataset_statistics.json')
#         if os.path.exists(stats_file):
#             with open(stats_file, 'r') as f:
#                 stats = json.load(f)
#                 logger.info(f"📊 Loaded dataset statistics")
                
#                 # Log enhanced features if available
#                 if 'enhanced_features' in stats:
#                     enhanced = stats['enhanced_features']
#                     if enhanced.get('exact_tile_bounds'):
#                         logger.info(f"✅ Enhanced features: exact bounds coverage {enhanced.get('bounds_coverage_percentage', 0):.1f}%")
#                     if enhanced.get('adaptive_mixed_aggregation'):
#                         logger.info(f"🔄 Adaptive mixed aggregation available")
#                         if 'adaptive_coverage_percentage' in enhanced:
#                             logger.info(f"🔄 Adaptive coverage: {enhanced['adaptive_coverage_percentage']:.1f}%")
#         else:
#             logger.warning("⚠️ Dataset statistics not found")
#             stats = None
        
#         return locations, city_data, umap_coords, stats
        
#     except Exception as e:
#         logger.error(f"❌ Error loading enhanced data: {e}")
#         raise

async def query_qdrant_similarity(
    target_location_id: int, 
    collection_name: str,
    limit: int = 100,
    offset: int = 0
) -> tuple[List[tuple], int]:
    """Query Qdrant for similar vectors with comprehensive deduplication."""
    try:
        logger.info(f"🔍 Searching for target location {target_location_id} in collection {collection_name}")
        
        # Get the target vector
        target_points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[target_location_id],
            with_vectors=True
        )
        
        if not target_points:
            logger.warning(f"❌ Target location {target_location_id} not found directly. Searching...")
            
            sample_points = qdrant_client.scroll(
                collection_name=collection_name,
                limit=10,
                with_payload=True,
                with_vectors=False
            )
            
            sample_ids = [p.id for p in sample_points[0]]
            logger.info(f"📋 Sample IDs in collection: {sample_ids[:5]}...")
            
            raise ValueError(f"Target location {target_location_id} not found in {collection_name}")
        
        target_vector = target_points[0].vector
        logger.info(f"✅ Found target vector for location {target_location_id}")
        
        # Search for results
        search_limit = max(500, (offset + limit) * 5)
        
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=target_vector,
            limit=search_limit,
            with_payload=True,
            score_threshold=0.0
        )
        
        logger.info(f"📊 Raw search results: {len(search_results)} points")
        
        # Multi-level deduplication
        seen_location_ids: Set[int] = set()
        seen_coordinates: Set[tuple] = set()
        seen_city_country: Set[tuple] = set()
        unique_results = []
        
        for result in search_results:
            location_id = int(result.id)
            
            # Skip the target location itself
            if location_id == target_location_id:
                continue
            
            # Skip if already seen this location ID
            if location_id in seen_location_ids:
                continue
            
            # Extract coordinates and location info
            try:
                longitude = float(result.payload['longitude'])
                latitude = float(result.payload['latitude'])
                city = str(result.payload['city'])
                country = str(result.payload['country'])
                continent = str(result.payload.get('continent', 'Unknown'))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"⚠️ Skipping result with invalid payload: {e}")
                continue
            
            # Skip if we've seen these exact coordinates
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
        
        logger.info(f"✅ Similarity query results:")
        logger.info(f"   - Raw results: {len(search_results)}")
        logger.info(f"   - After deduplication: {total_available}")
        logger.info(f"   - Returned: {len(paginated_results)}")
        
        return paginated_results, total_available
        
    except Exception as e:
        logger.error(f"❌ Error querying Qdrant similarity: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced application."""
    global locations_data, umap_data, dataset_stats, city_representatives_data, qdrant_client
    
    try:
        logger.info("🚀 Starting Enhanced Satellite Embeddings Explorer - With Adaptive Mixed Aggregation")
        
        # Setup Qdrant client
        qdrant_client = setup_qdrant_client()
        
        # Load enhanced data
        locations_data, city_representatives_data, umap_data, dataset_stats = load_lightweight_data()
        
        logger.info("✅ Enhanced application startup completed successfully")
        logger.info(f"💾 Memory usage: Enhanced metadata with exact bounds and adaptive mixed aggregation")
        logger.info(f"🎯 Ready to serve similarity queries via Qdrant with three aggregation methods")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Set minimal fallback data
        locations_data = []
        city_representatives_data = None
        umap_data = None
        dataset_stats = None

@app.get("/")
async def root():
    """Root endpoint with enhanced system status."""
    qdrant_status = "connected" if qdrant_client else "disconnected"
    
    # Get collection info
    collection_info = {}
    bounds_stats = {"exact": 0, "fallback": 0, "percentage": 0}
    adaptive_stats = {"available": False, "coverage": 0}
    
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
    
    # Calculate bounds statistics
    if locations_data:
        bounds_stats["exact"] = sum(1 for loc in locations_data if loc.get('has_exact_bounds', False))
        bounds_stats["fallback"] = len(locations_data) - bounds_stats["exact"]
        bounds_stats["percentage"] = (bounds_stats["exact"] / len(locations_data)) * 100 if locations_data else 0
    
    # Check adaptive mixed availability
    if dataset_stats and dataset_stats.get('enhanced_features', {}).get('adaptive_mixed_aggregation'):
        adaptive_stats["available"] = True
        adaptive_stats["coverage"] = dataset_stats.get('enhanced_features', {}).get('adaptive_coverage_percentage', 0)
    
    return {
        "message": "Enhanced Satellite Embeddings Explorer - With Adaptive Mixed Aggregation", 
        "version": "3.2.0",
        "status": "running",
        "qdrant_status": qdrant_status,
        "similarity_methods": list(COLLECTIONS.keys()),
        "collections": collection_info,
        "locations_loaded": len(locations_data) if locations_data else 0,
        "cities_loaded": len(city_representatives_data.get('city_representatives', [])) if city_representatives_data else 0,
        "bounds_coverage": bounds_stats,
        "adaptive_mixed": adaptive_stats,
        "architecture": "enhanced_qdrant_with_exact_bounds_and_adaptive_mixed"
    }

@limiter.limit("10/minute")  # Max 10 requests per minute per IP
@app.get("/api/similarity/{location_id}", response_model=SimplifiedSimilarityResponse)
async def find_similar_locations(
    request: Request,  # Add this parameter
    location_id: int, 
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    limit: int = Query(6, ge=1, le=50, description="Number of results to return"),
    method: str = Query("regular", description="Similarity method: regular, global_contrastive, adaptive_mixed")
):
    """Find most similar locations using Qdrant vector similarity with three aggregation methods."""
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
        raise HTTPException(status_code=404, detail="Location not found in metadata")
    
    # Check cache
    cache_key = f"similarity_{location_id}_{method}_{offset}_{limit}"
    cached_result = similarity_cache.get(cache_key)
    if cached_result:
        logger.info(f"🎯 Cache hit for similarity query: {cache_key}")
        return cached_result
    
    try:
        collection_name = COLLECTIONS[method]
        logger.info(f"🔍 Querying {collection_name} for location {location_id} using {method} aggregation")
        
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
        
        response = SimplifiedSimilarityResponse(
            target_location_id=location_id,
            similar_locations=similar_locations,
            method_used=f"qdrant_{method}",
            pagination=pagination
        )
        
        # Cache the result
        similarity_cache.set(cache_key, response)
        
        logger.info(f"✅ Similarity query completed using {method}: {len(similar_locations)} unique results")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@app.get("/api/methods")
async def get_similarity_methods():
    """Get available similarity methods and descriptions with adaptive mixed."""
    methods = {
        "regular": {
            "name": "Regular Embeddings",
            "description": "Standard similarity using mean-aggregated patch embeddings",
            "collection": COLLECTIONS["regular"],
            "use_case": "General similarity based on overall visual appearance",
            "aggregation_type": "uniform_mean"
        },
        "global_contrastive": {
            "name": "Global Contrastive",
            "description": "Dataset mean subtracted to highlight city-level differences",
            "collection": COLLECTIONS["global_contrastive"],
            "use_case": "Find cities that differ from the global average in similar ways",
            "aggregation_type": "contrastive_mean"
        },
        "adaptive_mixed": {
            "name": "Adaptive Mixed",
            "description": "Intelligent switching between uniform and weighted aggregation based on patch diversity",
            "collection": COLLECTIONS["adaptive_mixed"],
            "use_case": "Optimal aggregation for both homogeneous and heterogeneous urban areas",
            "aggregation_type": "adaptive_weighted",
            "technical_details": {
                "homogeneous_threshold": 0.2,
                "aggregation_strategy": "Auto-selects between simple mean (homogeneous tiles) and distinctiveness-weighted mean (heterogeneous tiles)",
                "benefits": "Better handling of diverse urban patterns while maintaining efficiency for uniform areas"
            }
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
    
    # Add adaptive mixed statistics if available
    if dataset_stats and dataset_stats.get('enhanced_features', {}).get('adaptive_mixed_aggregation'):
        adaptive_stats = dataset_stats.get('enhanced_features', {})
        methods["adaptive_mixed"]["statistics"] = {
            "coverage_percentage": adaptive_stats.get('adaptive_coverage_percentage', 0),
            "homogeneous_tiles": adaptive_stats.get('homogeneous_tiles', 0),
            "heterogeneous_tiles": adaptive_stats.get('heterogeneous_tiles', 0)
        }
    
    return {
        "available_methods": methods,
        "default_method": "regular",
        "total_methods": len(methods),
        "adaptive_mixed_available": methods["adaptive_mixed"]["available"]
    }

# Include all other existing endpoints unchanged...
@app.get("/api/locations", response_model=List[LocationData])
async def get_enhanced_locations(
    zoom: Optional[float] = Query(None, description="Map zoom level for LOD switching"),
    detail_level: Optional[str] = Query("auto", description="Detail level: 'city', 'tiles', or 'auto'")
):
    """Get locations with level-of-detail based on zoom level."""
    if not locations_data:
        raise HTTPException(status_code=503, detail="Location data not loaded")
    
    # Determine detail level based on zoom
    zoom_threshold = 6.0  # Switch to individual tiles above zoom level 6
    
    if detail_level == "auto" and zoom is not None:
        use_city_representatives = zoom <= zoom_threshold
    elif detail_level == "city":
        use_city_representatives = True
    elif detail_level == "tiles":
        use_city_representatives = False
    else:
        # Default behavior - show all tiles
        use_city_representatives = False
    
    logger.info(f"📍 Location request: zoom={zoom}, detail_level={detail_level}, using_city_reps={use_city_representatives}")
    
    if use_city_representatives and city_representatives_data:
        # Return city representatives for zoomed out view
        city_locations = []
        for city_rep in city_representatives_data['city_representatives']:
            city_location = LocationData(
                id=city_rep['representative_tile_id'],
                city=city_rep['city'],
                country=city_rep['country'],
                continent=city_rep['continent'],
                longitude=city_rep['centroid_longitude'],
                latitude=city_rep['centroid_latitude'],
                date=None,  # City representatives don't have specific dates
                tile_bounds=None,  # City representatives use computed bounds
                has_exact_bounds=False,
                tile_width_degrees=None,
                tile_height_degrees=None
            )
            city_locations.append(city_location)
        
        logger.info(f"📊 City representatives response: {len(city_locations)} cities")
        return city_locations
    
    else:
        # Return all individual tiles for zoomed in view
        enhanced_locations = []
        bounds_stats = {"exact": 0, "fallback": 0}
        
        for location_data in locations_data:
            # Check if we have tile bounds data
            tile_bounds = location_data.get('tile_bounds')
            has_exact_bounds = location_data.get('has_exact_bounds', False)
            
            if has_exact_bounds and tile_bounds:
                bounds_stats["exact"] += 1
            else:
                bounds_stats["fallback"] += 1
            
            enhanced_location = LocationData(
                id=location_data['id'],
                city=location_data['city'],
                country=location_data['country'],
                continent=location_data['continent'],
                longitude=location_data['longitude'],
                latitude=location_data['latitude'],
                date=location_data.get('date'),
                # Enhanced: Include tile bounds data
                tile_bounds=tile_bounds,
                has_exact_bounds=has_exact_bounds,
                tile_width_degrees=location_data.get('tile_width_degrees'),
                tile_height_degrees=location_data.get('tile_height_degrees')
            )
            enhanced_locations.append(enhanced_location)
        
        logger.info(f"📊 Enhanced tiles response: {len(enhanced_locations)} tiles, {bounds_stats['exact']} exact bounds, {bounds_stats['fallback']} fallback")
        return enhanced_locations

@app.get("/api/umap", response_model=UMapResponse)
async def get_enhanced_umap_embeddings():
    """Get enhanced UMAP 2D coordinates with tile bounds information."""
    if not umap_data:
        raise HTTPException(
            status_code=503, 
            detail="UMAP data not available. Run enhanced migration script."
        )
    
    logger.info("🗺️ Returning enhanced UMAP coordinates with bounds info")
    
    # Convert to response model with bounds
    enhanced_umap_points = []
    bounds_included_count = 0
    
    for point_data in umap_data['umap_points']:
        has_bounds = point_data.get('has_exact_bounds', False)
        if has_bounds:
            bounds_included_count += 1
        
        enhanced_umap_point = UMapPoint(
            location_id=point_data['location_id'],
            x=point_data['x'],
            y=point_data['y'],
            city=point_data['city'],
            country=point_data['country'],
            continent=point_data['continent'],
            longitude=point_data['longitude'],
            latitude=point_data['latitude'],
            date=point_data.get('date'),
            # Enhanced: Include bounds info
            tile_bounds=point_data.get('tile_bounds'),
            has_exact_bounds=has_bounds
        )
        enhanced_umap_points.append(enhanced_umap_point)
    
    logger.info(f"📊 UMAP points with bounds: {bounds_included_count}/{len(enhanced_umap_points)}")
    
    # Create bounds statistics
    bounds_stats = None
    if 'bounds_statistics' in umap_data:
        bounds_stats = BoundsStatistics(**umap_data['bounds_statistics'])
    
    return UMapResponse(
        umap_points=enhanced_umap_points,
        total_points=len(enhanced_umap_points),
        bounds_statistics=bounds_stats
    )

@app.get("/api/stats", response_model=SimplifiedStatsResponse)
async def get_enhanced_stats():
    """Get enhanced dataset statistics including bounds coverage and adaptive mixed info."""
    if not locations_data:
        raise HTTPException(status_code=503, detail="Location data not loaded")
    
    # Calculate bounds statistics
    exact_bounds_count = sum(1 for loc in locations_data if loc.get('has_exact_bounds', False))
    fallback_bounds_count = len(locations_data) - exact_bounds_count
    exact_bounds_percentage = (exact_bounds_count / len(locations_data)) * 100 if locations_data else 0
    
    # Use dataset stats if available
    if dataset_stats:
        countries = list(set(loc['country'] for loc in locations_data))
        continents = list(set(loc['continent'] for loc in locations_data))
        
        enhanced_stats = SimplifiedStatsResponse(
            total_locations=dataset_stats['total_samples'],
            countries_count=len(countries),
            continents_count=len(continents),
            embedding_dimension=dataset_stats['embedding_dimension'],
            locations_with_full_patches=dataset_stats['total_samples'],
            countries=sorted(countries),
            continents=sorted(continents),
            similarity_method="qdrant_vector_search_with_adaptive_mixed",
            # Enhanced: Add bounds coverage
            bounds_coverage_percentage=exact_bounds_percentage,
            tiles_with_exact_bounds=exact_bounds_count,
            tiles_with_fallback_bounds=fallback_bounds_count
        )
        
    else:
        # Fallback computation
        countries = list(set(loc['country'] for loc in locations_data))
        continents = list(set(loc['continent'] for loc in locations_data))
        
        enhanced_stats = SimplifiedStatsResponse(
            total_locations=len(locations_data),
            countries_count=len(countries),
            continents_count=len(continents),
            embedding_dimension=768,
            locations_with_full_patches=len(locations_data),
            countries=sorted(countries),
            continents=sorted(continents),
            similarity_method="qdrant_vector_search_with_adaptive_mixed",
            # Enhanced: Add bounds coverage
            bounds_coverage_percentage=exact_bounds_percentage,
            tiles_with_exact_bounds=exact_bounds_count,
            tiles_with_fallback_bounds=fallback_bounds_count
        )
    
    logger.info(f"📊 Enhanced bounds coverage: {exact_bounds_count}/{len(locations_data)} ({exact_bounds_percentage:.1f}%) exact")
    
    return enhanced_stats

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get configuration including Mapbox token."""
    return ConfigResponse(
        mapbox_token=os.getenv("MAPBOX_TOKEN", "")
    )

@app.get("/api/health")
async def enhanced_health_check():
    """Enhanced health check with bounds coverage and adaptive mixed info."""
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
                "total_collections": len(collections.collections),
                "our_collections": collection_status
            }
        else:
            health_status["components"]["qdrant"] = {"status": "disconnected"}
    except Exception as e:
        health_status["components"]["qdrant"] = {"status": "error", "error": str(e)}
    
    # Enhanced data check with bounds and adaptive mixed
    bounds_stats = {"exact": 0, "fallback": 0, "percentage": 0}
    adaptive_stats = {"available": False, "coverage": 0}
    
    if locations_data:
        bounds_stats["exact"] = sum(1 for loc in locations_data if loc.get('has_exact_bounds', False))
        bounds_stats["fallback"] = len(locations_data) - bounds_stats["exact"]
        bounds_stats["percentage"] = (bounds_stats["exact"] / len(locations_data)) * 100
    
    if dataset_stats and dataset_stats.get('enhanced_features', {}).get('adaptive_mixed_aggregation'):
        adaptive_stats["available"] = True
        adaptive_stats["coverage"] = dataset_stats.get('enhanced_features', {}).get('adaptive_coverage_percentage', 0)
    
    health_status["components"]["data"] = {
        "locations_loaded": len(locations_data) if locations_data else 0,
        "cities_loaded": len(city_representatives_data.get('city_representatives', [])) if city_representatives_data else 0,
        "umap_available": umap_data is not None,
        "stats_available": dataset_stats is not None,
        "bounds_coverage": bounds_stats,
        "adaptive_mixed": adaptive_stats
    }
    
    # Cache status
    health_status["components"]["cache"] = {
        "size": len(similarity_cache.cache),
        "max_size": similarity_cache.max_size
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