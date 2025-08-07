# app/services/pbti/pbti_service.py

import asyncio
from typing import Dict, Any
from app.models.schemas import PbtiRequest
from app.services.pbti.gpt_service import (
    prompt_recommendation,
    prompt_keywords,
    prompt_perfume_style,
    prompt_scent_point,
    prompt_summary,
    call_gpt_async
)
from app.services.pbti.pbti_recommender import get_perfume_recommendations

async def get_full_pbti_result(request: PbtiRequest) -> Dict[str, Any]:
    """
    PBTI 전체 결과 생성 함수
    - GPT 병렬 호출 (5개 프롬프트)
    - 향수 추천 (SBERT 기반)
    - 결과 통합 및 반환
    
    pbti.py의 485~504줄과 동일한 로직
    """
    
    # 5개의 GPT 프롬프트 생성
    prompts = [
        prompt_recommendation(request),
        prompt_keywords(request),
        prompt_perfume_style(request),
        prompt_scent_point(request),
        prompt_summary(request),
    ]
    
    # GPT 병렬 호출 실행
    gpt_outputs = await asyncio.gather(*[call_gpt_async(p) for p in prompts])
    
    # GPT 결과들을 하나의 딕셔너리로 통합
    result = {}
    for out in gpt_outputs:
        if isinstance(out, dict):
            result.update(out)
    
    # 향수 추천 결과 추가
    result["perfumeRecommend"] = get_perfume_recommendations(request)
    
    return result

def get_pbti_status() -> Dict[str, Any]:
    """PBTI 서비스 상태 확인 (디버깅용)"""
    try:
        from app.services.pbti.pbti_recommender import get_model_info
        model_info = get_model_info()
        
        return {
            "status": "healthy",
            "model_info": model_info,
            "services": {
                "gpt_service": "available",
                "mbti_analyzer": "available", 
                "pbti_recommender": "available"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }