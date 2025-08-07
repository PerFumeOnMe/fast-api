# app/routers/recommendations.py

from fastapi import APIRouter, HTTPException
from app.models.schemas import RecommendationRequest, RecommendationResponse
from app.services.recommend_full import recommend_full

router = APIRouter(
    prefix="/recommend",
    tags=["recommendations"],
    responses={404: {"description": "Not found"}},
)

@router.post("/full", response_model=RecommendationResponse)
async def recommend_with_scenario(req: RecommendationRequest):
    """
    감성 시나리오 + 하이브리드 향수 추천 통합 API (병렬 처리)

    - 사용자의 5가지 키워드를 기반으로 감성적인 시나리오를 작성합니다.
    - TF-IDF + SBERT 기반 Hybrid 로직으로 향수를 추천합니다.
    - GPT 시나리오 생성과 ML 추천을 병렬로 처리하여 응답 속도를 개선합니다.
    """
    try:
        return await recommend_full(
            ambience=req.ambience,
            style=req.style,
            gender=req.gender,
            season=req.season,
            personality=req.personality
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 실패: {str(e)}")