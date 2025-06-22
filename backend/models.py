from pydantic import BaseModel
from typing import Optional

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
    embedding_shape: tuple