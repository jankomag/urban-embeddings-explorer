from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class LocationData(BaseModel):
    id: int
    city: str
    country: str
    continent: str
    longitude: float
    latitude: float
    date: Optional[str] = None

class LocationDetail(BaseModel):
    id: int
    city: str
    country: str
    continent: str
    longitude: float
    latitude: float
    date: Optional[str] = None
    embedding_shape: Optional[tuple] = None

class SimilarLocation(BaseModel):
    id: int
    city: str
    country: str
    continent: str
    longitude: float
    latitude: float
    date: Optional[str] = None
    similarity_score: float

class SimilarityMethodConfig(BaseModel):
    name: str
    description: str
    speed: str
    quality: str
    requires_full_patches: bool

class SimilarityMethod(BaseModel):
    id: str
    config: SimilarityMethodConfig

class SimilarityMethodsResponse(BaseModel):
    methods: List[SimilarityMethod]
    recommended: str
    fastest: str
    best_quality: str

class EnhancedSimilarityResponse(BaseModel):
    target_location_id: int
    similar_locations: List[SimilarLocation]
    method_used: str
    method_config: SimilarityMethodConfig

class UMapPoint(BaseModel):
    location_id: int
    x: float
    y: float
    city: str
    country: str
    continent: str
    longitude: float
    latitude: float
    date: Optional[str] = None

class UMapResponse(BaseModel):
    umap_points: List[UMapPoint]
    total_points: int

class EnhancedStatsResponse(BaseModel):
    total_locations: int
    countries_count: int
    continents_count: int
    embedding_dimension: int
    locations_with_full_patches: int
    countries: List[str]
    continents: List[str]
    available_similarity_methods: int

class ConfigResponse(BaseModel):
    mapbox_token: str

class TileBoundsResponse(BaseModel):
    location_id: int
    city: str
    country: str
    tile_bounds: Optional[List[List[float]]]
    has_exact_bounds: bool