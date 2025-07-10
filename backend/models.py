from pydantic import BaseModel
from typing import Optional, List

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

class SimilarityResponse(BaseModel):
    target_location_id: int
    similar_locations: List[SimilarLocation]

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

class StatsResponse(BaseModel):
    total_locations: int
    countries_count: int
    continents_count: int
    embedding_dimension: int
    countries: List[str]
    continents: List[str]

class ConfigResponse(BaseModel):
    mapbox_token: str