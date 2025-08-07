# app/routers/pbti.py

from fastapi import APIRouter, HTTPException
from app.models.schemas import PbtiRequest
from app.services.pbti.pbti_service import get_full_pbti_result, get_pbti_status
from typing import Dict, Any

router = APIRouter(
    prefix="/pbti",
    tags=["pbti"],
    responses={404: {"description": "Not found"}},
)

@router.post("/full-result")
async def full_result(request: PbtiRequest) -> Dict[str, Any]:
    """
    PBTI 전체 결과 API
    
    - 8개의 질문에 대한 답변을 받아서 MBTI 성향 분석
    - GPT를 통한 5개 카테고리 병렬 분석 (recommendation, keywords, perfumeStyle, scentPoint, summary)
    - SBERT 기반 향수 추천 (상위 3개)
    - 모든 결과를 통합해서 반환
    
    Args:
        request (PbtiRequest): 8개 질문에 대한 답변
    
    Returns:
        Dict[str, Any]: 통합 PBTI 결과
    """
    try:
        result = await get_full_pbti_result(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PBTI 분석 실패: {str(e)}")

@router.get("/status")
async def pbti_status() -> Dict[str, Any]:
    """
    PBTI 서비스 상태 확인 API (디버깅 및 헬스체크용)
    
    Returns:
        Dict[str, Any]: 서비스 상태 정보
    """
    try:
        status = get_pbti_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 확인 실패: {str(e)}")