from pydantic import BaseModel
from typing import List, Optional

class LocationData(BaseModel):
    """Simplified location data model."""
    id: int
    city: str
    country: str
    continent: str
    longitude: float
    latitude: float
    date: Optional[str] = None
    bounds: Optional[List[List[float]]] = None  # Simplified from tile_bounds

class SimilarLocation(BaseModel):
    """Similar location model."""
    id: int
    city: str
    country: str
    continent: str
    longitude: float
    latitude: float
    date: Optional[str] = None
    similarity_score: float

class PaginationInfo(BaseModel):
    """Pagination information for API responses."""
    offset: int
    limit: int
    total_results: int
    has_more: bool
    next_offset: Optional[int] = None
    returned_count: int

class SimilarityResponse(BaseModel):
    """Similarity response model."""
    target_location_id: int
    similar_locations: List[SimilarLocation]
    method_used: str
    pagination: PaginationInfo

class BoundsStatistics(BaseModel):
    """Statistics about tile bounds coverage."""
    tiles_with_bounds: int
    total_tiles: int
    coverage_percentage: float

class StatsResponse(BaseModel):
    """Statistics response."""
    total_locations: int
    countries_count: int
    continents_count: int
    embedding_dimension: int
    countries: List[str]
    continents: List[str]
    similarity_method: str

class ConfigResponse(BaseModel):
    """Configuration response."""
    mapbox_token: str

class UMapPoint(BaseModel):
    """UMAP point with bounds information."""
    location_id: int
    x: float
    y: float
    city: str
    country: str
    continent: str
    longitude: float
    latitude: float
    date: Optional[str] = None
    bounds: Optional[List[List[float]]] = None  # Simplified from tile_bounds

class UMapResponse(BaseModel):
    """UMAP response."""
    umap_points: List[UMapPoint]
    total_points: int