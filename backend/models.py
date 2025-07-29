from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class LocationData(BaseModel):
    """Enhanced location data model with LOD support."""
    id: int
    city: str
    country: str
    continent: str
    longitude: float
    latitude: float
    date: Optional[str] = None
    # Enhanced bounds information
    tile_bounds: Optional[List[List[float]]] = None
    has_exact_bounds: Optional[bool] = False
    tile_width_degrees: Optional[float] = None
    tile_height_degrees: Optional[float] = None
    # LOD support
    is_city_representative: Optional[bool] = False
    tile_count: Optional[int] = 1

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

class SimplifiedSimilarityResponse(BaseModel):
    """Simplified similarity response model."""
    target_location_id: int
    similar_locations: List[SimilarLocation]
    method_used: str
    pagination: PaginationInfo

class BoundsStatistics(BaseModel):
    """Statistics about tile bounds coverage."""
    tiles_with_exact_bounds: int
    tiles_with_fallback_bounds: int
    exact_bounds_percentage: float

class SimplifiedStatsResponse(BaseModel):
    """Enhanced simplified statistics response with bounds coverage."""
    total_locations: int
    countries_count: int
    continents_count: int
    embedding_dimension: int
    locations_with_full_patches: int
    countries: List[str]
    continents: List[str]
    similarity_method: str
    # Enhanced bounds coverage
    bounds_coverage_percentage: Optional[float] = 0.0
    tiles_with_exact_bounds: Optional[int] = 0
    tiles_with_fallback_bounds: Optional[int] = 0

class ConfigResponse(BaseModel):
    """Enhanced configuration response with LOD settings."""
    mapbox_token: str
    zoom_threshold: Optional[float] = 6.0
    lod_enabled: Optional[bool] = False

class TileBoundsResponse(BaseModel):
    """Response model for tile bounds endpoint."""
    location_id: int
    city: str
    country: str
    tile_bounds: Optional[List[List[float]]] = None
    has_exact_bounds: bool = False
    tile_width_degrees: Optional[float] = None
    tile_height_degrees: Optional[float] = None

class TileBoundsBatchResponse(BaseModel):
    """Response model for batch tile bounds requests."""
    tile_bounds: Dict[int, TileBoundsResponse]
    total_requested: int
    total_found: int

class UMapPoint(BaseModel):
    """Enhanced UMAP point with bounds information."""
    location_id: int
    x: float
    y: float
    city: str
    country: str
    continent: str
    longitude: float
    latitude: float
    date: Optional[str] = None
    # Enhanced bounds information
    tile_bounds: Optional[List[List[float]]] = None
    has_exact_bounds: Optional[bool] = False

class UMapResponse(BaseModel):
    """Enhanced UMAP response with bounds statistics."""
    umap_points: List[UMapPoint]
    total_points: int
    bounds_statistics: Optional[BoundsStatistics] = None

class CityRepresentative(BaseModel):
    """City representative model for LOD."""
    city_key: str
    city: str
    country: str
    continent: str
    representative_tile_id: int
    representative_longitude: float
    representative_latitude: float
    centroid_longitude: float
    centroid_latitude: float
    tile_count: int
    all_tile_ids: List[int]
    bbox: Optional[Dict[str, float]] = None
    date_range: Optional[Dict[str, Optional[str]]] = None

class CityRepresentativesResponse(BaseModel):
    """Response model for city representatives."""
    city_representatives: List[CityRepresentative]
    total_cities: int
    total_tiles: int
    source: str = "city_metadata_file"
    generation_timestamp: Optional[float] = None
    statistics: Optional[Dict[str, Any]] = None

class LODConfigResponse(BaseModel):
    """Level-of-detail configuration response."""
    zoom_threshold: float
    lod_enabled: bool
    city_representatives_available: int
    total_tiles: int
    description: str