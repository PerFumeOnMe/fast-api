# app/schemas.py

from typing import List, Dict, Any
from pydantic import BaseModel

# 기존 이미지 키워드 추천 스키마
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

# PBTI 전용 스키마들
class PbtiRequest(BaseModel):
    qOne: str
    qTwo: str
    qThree: str
    qFour: str
    qFive: str
    qSix: str
    qSeven: str
    qEight: str

class PbtiPerfumeResponse(BaseModel):
    name: str
    brand: str
    description: str
    perfumeImageUrl: str

class PbtiKeyword(BaseModel):
    keyword: str
    keywordDescription: str

class PbtiNote(BaseModel):
    category: str
    categoryDescription: str

class PbtiPerfumeStyle(BaseModel):
    description: str
    notes: List[PbtiNote]

class PbtiScentPoint(BaseModel):
    category: str
    point: int

class PbtiFullResponse(BaseModel):
    recommendation: str
    keywords: List[PbtiKeyword]
    perfumeStyle: PbtiPerfumeStyle
    scentPoint: List[PbtiScentPoint]
    summary: str
    perfumeRecommend: List[PbtiPerfumeResponse]
