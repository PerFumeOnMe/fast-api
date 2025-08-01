# recommender_sbert.py
import pandas as pd
import numpy as np
from app.utils import safe_str, load_excel_from_s3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import SBERT_MODEL_NAME, DEFAULT_TOP_N, S3_BUCKET, S3_KEY


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
        self.df["full_text"] = (
            self.df["향수 키워드"].fillna('').astype(str) + ', ' +
            self.df["탑 노트 설명"].fillna('').astype(str) + ', ' +
            self.df["미들 노트 설명"].fillna('').astype(str) + ', ' +
            self.df["베이스 노트 설명"].fillna('').astype(str) + ', ' +
            self.df["한줄소개"].fillna('').astype(str)
        )

    def _embed_texts(self):
        self.embeddings = self.model.encode(
            self.df["full_text"].tolist(), 
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

    def recommend(
        self, 
        ambience: str, 
        style: str, 
        gender: str, 
        season: str, 
        personality: str, 
        top_n: int = DEFAULT_TOP_N
    ) -> dict:
        keywords = [ambience, style, gender, season, personality]
        sentence = "이 사람은 " + ", ".join(keywords) + " 느낌을 가진 사람입니다."

        query_embedding = self.model.encode(sentence, convert_to_tensor=True).cpu().numpy()
        cos_scores = cosine_similarity([query_embedding], self.embeddings)[0]
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
                "imageUrl": safe_str(row.get("향수 이미지", ""))
                })

        avg_score = float(np.mean([cos_scores[i] for i in top_indices]))

        return {
            "average_similarity": round(avg_score, 4),
            "results": results
        }
