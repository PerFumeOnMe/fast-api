# recommender_sbert.py
import pandas as pd
import numpy as np
from app.core.utils import safe_str, load_excel_from_s3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import (
    SBERT_MODEL_NAME, 
    DEFAULT_TOP_N, 
    S3_BUCKET, 
    S3_KEY
)


class SBERTPerfumeRecommender:
    _cache = None  # 클래스 캐싱

    def __init__(self, excel_path: str = None):
        # S3에서 로드 (캐싱)
        if SBERTPerfumeRecommender._cache is None:
            SBERTPerfumeRecommender._cache = load_excel_from_s3(S3_BUCKET, S3_KEY)
        self.df = SBERTPerfumeRecommender._cache
        self.df = self.df.dropna(subset=["향수이름", "향수 키워드"])
        self.model = SentenceTransformer(SBERT_MODEL_NAME)

        self._prepare_texts()
        self._embed_texts()

    def _prepare_texts(self):
        """
        다층 벡터 접근법을 위한 텍스트 준비
        - 속성 그룹별로 분리하여 더 정밀한 의미 분석
        """
        # 1. 핵심 속성 그룹 (Core Features)
        self.df["core_text"] = (
            self.df["향수 키워드"].fillna('').astype(str) + '. ' +
            self.df["한줄소개"].fillna('').astype(str)
        )
        
        # 2. 노트 속성 그룹 (Note Features)
        self.df["note_text"] = (
            self.df["탑 노트 설명"].fillna('').astype(str) + '. ' +
            self.df["미들 노트 설명"].fillna('').astype(str) + '. ' +
            self.df["베이스 노트 설명"].fillna('').astype(str) + '. ' +
            self.df["탑 노트 키워드"].fillna('').astype(str) + '. ' +
            self.df["미들 노트 키워드"].fillna('').astype(str) + '. ' +
            self.df["베이스 노트 키워드"].fillna('').astype(str)
        )
        
        # 3. 컨텍스트 속성 그룹 (Context Features)
        self.df["context_text"] = (
            self.df["성별"].fillna('').astype(str) + '. ' +
            self.df["계절"].fillna('').astype(str) + '. ' +
            self.df["장소"].fillna('').astype(str) + '. ' +
            self.df["브랜드"].fillna('').astype(str)
        )
        
        # 4. 전체 텍스트 (기존 방식 유지)
        self.df["full_text"] = (
            self.df["core_text"] + '. ' +
            self.df["note_text"] + '. ' +
            self.df["context_text"]
        )

    def _embed_texts(self):
        """다층 벡터 임베딩 생성"""
        # 전체 텍스트 임베딩
        self.embeddings = self.model.encode(
            self.df["full_text"].tolist(), 
            convert_to_tensor=True
        ).cpu().numpy()
        
        # 그룹별 임베딩 (더 정밀한 분석용)
        self.core_embeddings = self.model.encode(
            self.df["core_text"].tolist(),
            convert_to_tensor=True
        ).cpu().numpy()
        
        self.note_embeddings = self.model.encode(
            self.df["note_text"].tolist(),
            convert_to_tensor=True
        ).cpu().numpy()
        
        self.context_embeddings = self.model.encode(
            self.df["context_text"].tolist(),
            convert_to_tensor=True
        ).cpu().numpy()

    def _get_top_related_keywords(self, keywords: list[str], perfume_text: str, topn: int = 3) -> list[str]:
        perfume_vec = self.model.encode(perfume_text, convert_to_tensor=True).cpu().numpy()
        scores = {}

        for kw in keywords:
            try:
                kw_vec = self.model.encode(kw, convert_to_tensor=True).cpu().numpy()
                sim = cosine_similarity([kw_vec], [perfume_vec])[0][0]
                scores[kw] = sim
            except Exception:
                continue

        sorted_kws = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kws[:topn]]

    def _create_weighted_query_embedding(self, ambience: str, style: str, gender: str, season: str, personality: str) -> np.ndarray:
        """가중치가 적용된 쿼리 임베딩 생성"""
        
        # 1. 핵심 키워드 기반 쿼리
        core_query = f"이 향수는 {ambience}, {style}, {personality} 느낌의 향수입니다."
        core_embedding = self.model.encode(core_query, convert_to_tensor=True).cpu().numpy()
        
        # 2. 컨텍스트 기반 쿼리 (성별, 계절)
        context_parts = [gender, season]
        context_query = f"이 향수는 {', '.join(context_parts)}에 어울리는 향수입니다."
        context_embedding = self.model.encode(context_query, convert_to_tensor=True).cpu().numpy()
        
        # 3. 가중 평균으로 결합
        weighted_embedding = (
            core_embedding * 0.7 +    # 핵심 특성 비중 70%
            context_embedding * 0.3   # 컨텍스트 비중 30%
        )
        
        return weighted_embedding
    
    def _calculate_multi_layer_similarity(self, query_embedding: np.ndarray, ambience: str, style: str, gender: str, season: str, personality: str) -> np.ndarray:
        """다층 벡터 기반 유사도 계산"""
        
        # 1. 전체 유사도 (70% 비중)
        full_similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 2. 핵심 속성 유사도 (20% 비중)
        core_query = f"{ambience} {style} {personality}"
        core_query_emb = self.model.encode(core_query, convert_to_tensor=True).cpu().numpy()
        core_similarities = cosine_similarity([core_query_emb], self.core_embeddings)[0]
        
        # 3. 컨텍스트 속성 유사도 (10% 비중)
        context_parts = [gender, season]
        context_query = " ".join(context_parts)
        context_query_emb = self.model.encode(context_query, convert_to_tensor=True).cpu().numpy()
        context_similarities = cosine_similarity([context_query_emb], self.context_embeddings)[0]
        
        # 가중 결합
        final_similarities = (
            full_similarities * 0.7 +
            core_similarities * 0.2 +
            context_similarities * 0.1
        )
        
        return final_similarities

    def recommend(
        self, 
        ambience: str, 
        style: str, 
        gender: str, 
        season: str, 
        personality: str,
        top_n: int = DEFAULT_TOP_N
    ) -> dict:
        """
        개선된 SBERT 추천 시스템
        - 다층 벡터 접근법
        - 데이터셋 장소 속성 내부 활용  
        - 가중치 기반 유사도 계산
        """
        keywords = [ambience, style, gender, season, personality]
        
        # 가중치 적용된 쿼리 임베딩 생성
        query_embedding = self._create_weighted_query_embedding(
            ambience, style, gender, season, personality
        )
        
        # 다층 벡터 기반 유사도 계산
        cos_scores = self._calculate_multi_layer_similarity(
            query_embedding, ambience, style, gender, season, personality
        )
        
        # 상위 결과 선별
        top_indices = cos_scores.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            full_text = row["full_text"]
            related_keywords = self._get_top_related_keywords(keywords, full_text)
            similarity_score = float(round(cos_scores[idx], 4))

            results.append({
                "similarity": similarity_score,
                "brand": safe_str(row.get("브랜드", "")),
                "name": safe_str(row.get("향수이름", "")),
                "topNote": safe_str(row.get("탑 노트 키워드", "")),
                "middleNote": safe_str(row.get("미들 노트 키워드", "")),
                "baseNote": safe_str(row.get("베이스 노트 키워드", "")),
                "description": safe_str(row.get("한줄소개", row.get("향수 키워드", ""))),
                "relatedKeywords": related_keywords,
                "imageUrl": safe_str(row.get("향수 이미지", "")),
                "removebgImageUrl": safe_str(row.get("rmbg_s3_url", ""))
            })

        avg_score = float(np.mean([cos_scores[i] for i in top_indices]))

        return {
            "average_similarity": round(avg_score, 4),
            "results": results
        }
