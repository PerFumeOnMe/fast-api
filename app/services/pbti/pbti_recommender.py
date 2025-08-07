# app/services/pbti/pbti_recommender.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import S3_BUCKET, S3_KEY, PBTI_SBERT_MODEL_NAME
from app.core.utils import load_excel_from_s3, safe_str
from app.models.schemas import PbtiRequest
from app.services.pbti.mbti_analyzer import determine_mbti_type, build_user_description
from typing import List, Dict, Any
import pandas as pd

class PBTIPerfumeRecommender:
    """PBTI 전용 향수 추천기 (기존 SBERT 추천기와 동일한 패턴)"""
    _cache = None  # 클래스 캐싱
    
    def __init__(self):
        # S3에서 로드 (캐싱)
        if PBTIPerfumeRecommender._cache is None:
            PBTIPerfumeRecommender._cache = load_excel_from_s3(S3_BUCKET, S3_KEY)
            print(f"PBTI: S3에서 향수 데이터 로드 완료: {len(PBTIPerfumeRecommender._cache)}개 향수")
        
        self.df = PBTIPerfumeRecommender._cache.copy()
        self.df = self.df.dropna(subset=["향수이름", "향수 키워드"])
        self.model = SentenceTransformer(PBTI_SBERT_MODEL_NAME)
        
        # 향수 임베딩 데이터 준비
        self._prepare_perfume_embeddings()
        
    def _prepare_perfume_embeddings(self):
        """향수 임베딩 데이터 준비"""
        # 향수 임베딩 문장 생성
        self.df["임베딩문장"] = self.df.apply(self._build_perfume_sentence, axis=1)
        # 임베딩 벡터 생성
        self.df["임베딩벡터"] = self.df["임베딩문장"].apply(lambda x: self.model.encode(x))
        print("PBTI: 향수 데이터 임베딩 처리 완료")
        
    def _build_perfume_sentence(self, row) -> str:
        """향수 임베딩 문장 생성 함수"""
        gender = safe_str(row.get('성별', ''))
        place = safe_str(row.get('장소', ''))
        season = safe_str(row.get('계절', ''))
        keywords = safe_str(row.get('향수 키워드', ''))
        return f"Suitable for {gender} at {place} in {season}, with keywords like {keywords}"
    
    def recommend(self, request: PbtiRequest) -> List[Dict[str, Any]]:
        """PBTI 기반 향수 추천"""
        # MBTI 분석 및 사용자 벡터 생성
        mbti = determine_mbti_type(request)
        user_sentence = build_user_description(mbti)
        user_vector = self.model.encode(user_sentence)
        
        # 코사인 유사도 계산
        similarities = []
        for embedding in self.df["임베딩벡터"]:
            similarity = cosine_similarity([user_vector], [embedding])[0][0]
            similarities.append(similarity)
        
        self.df["유사도"] = similarities
        
        # 상위 3개 향수 선택
        top_matches = self.df.sort_values(by="유사도", ascending=False).head(3)
        
        # 응답 형식에 맞게 변환
        result = []
        for _, row in top_matches.iterrows():
            result.append({
                "name": safe_str(row.get("향수이름", "")),
                "brand": safe_str(row.get("브랜드", "")),
                "description": safe_str(row.get("한줄소개", "")),
                "perfumeImageUrl": safe_str(row.get("향수 이미지", ""))
            })
        
        return result

# 전역 추천기 인스턴스 (기존 패턴과 동일)
_pbti_recommender = None

def get_perfume_recommendations(request: PbtiRequest) -> List[Dict[str, Any]]:
    """전역 추천기를 사용한 향수 추천 (기존 패턴 호환)"""
    global _pbti_recommender
    
    if _pbti_recommender is None:
        _pbti_recommender = PBTIPerfumeRecommender()
    
    return _pbti_recommender.recommend(request)

def get_model_info() -> Dict[str, Any]:
    """모델 상태 정보 반환 (디버깅용)"""
    global _pbti_recommender
    return {
        "recommender_loaded": _pbti_recommender is not None,
        "data_loaded": PBTIPerfumeRecommender._cache is not None,
        "data_count": len(PBTIPerfumeRecommender._cache) if PBTIPerfumeRecommender._cache is not None else 0,
        "model_name": PBTI_SBERT_MODEL_NAME
    }