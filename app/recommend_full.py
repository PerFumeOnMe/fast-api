# app/recommend_full.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.generator import generate_scenario_sync
from app.recommender_tf_idf import PerfumeRecommender
from app.recommender_sbert import SBERTPerfumeRecommender
from app.recommender_hybrid import HybridPerfumeRecommender
from app.config import EXCEL_PATH

# 글로벌 객체 초기화 (한 번만 로딩)
tfidf = PerfumeRecommender(EXCEL_PATH)
sbert = SBERTPerfumeRecommender(EXCEL_PATH)
hybrid = HybridPerfumeRecommender(tfidf, sbert)

async def recommend_full(
    ambience: str, 
    style: str, 
    gender: str, 
    season: str, 
    personality: str
) -> dict:
    """
    감성 시나리오 + 향수 추천 통합 추천 결과 반환 (병렬 처리)

    :param ambience: 분위기 키워드
    :param style: 스타일 키워드
    :param gender: 성별 키워드
    :param season: 계절 키워드
    :param personality: 성격 키워드
    :return: {
        "scenario": str,
        "recommendations": list[dict]
    }
    """
    keywords = [ambience, style, gender, season, personality]

    # 병렬 처리: GPT 시나리오 생성과 하이브리드 추천을 동시에 실행
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 1. GPT 시나리오 생성 (I/O 바운드)
        scenario_future = loop.run_in_executor(
            executor, generate_scenario_sync, keywords
        )
        
        # 2. 하이브리드 추천 (CPU 바운드)
        recommend_future = loop.run_in_executor(
            executor, 
            lambda: hybrid.recommend(ambience, style, gender, season, personality)
        )
        
        # 두 작업이 모두 완료될 때까지 대기
        scenario, hybrid_result = await asyncio.gather(
            scenario_future, 
            recommend_future
        )

    return {
        "scenario": scenario,
        "recommendations": hybrid_result["results"]
    }
