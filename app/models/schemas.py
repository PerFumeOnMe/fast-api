# app/schemas.py

from typing import List
from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    ambience: str
    style: str
    gender: str
    season: str
    personality: str

class FragranceRecommendation(BaseModel):
    brand: str
    name: str
    topNote: str
    middleNote: str
    baseNote: str
    description: str
    relatedKeywords: List[str]
    imageUrl: str

class RecommendationResponse(BaseModel):
    scenario: str
    recommendations: List[FragranceRecommendation]
