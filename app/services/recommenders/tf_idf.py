# recommender_tf_idf.py
import pandas as pd
from app.core.utils import safe_str
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.utils import load_excel_from_s3
from app.core.config import S3_BUCKET, S3_KEY
from app.core.config import (
    TFIDF_NGRAM_RANGE,
    TFIDF_MAX_FEATURES,
)


class PerfumeRecommender:
    _cache = None  # 캐싱: 한 번만 로드

    def __init__(self, excel_path: str = None):
        if PerfumeRecommender._cache is None:
            # S3에서 로드
            PerfumeRecommender._cache = load_excel_from_s3(S3_BUCKET, S3_KEY)
        self.df = PerfumeRecommender._cache
        self.df = self.df.dropna(subset=["향수이름", "향수 키워드"])
        self._prepare_documents()

    def _prepare_documents(self):
        self.df["full_text"] = (
            self.df["향수 키워드"].fillna('').astype(str) + ',' +
            self.df["탑 노트 키워드"].fillna('').astype(str) + ',' +
            self.df["미들 노트 키워드"].fillna('').astype(str) + ',' +
            self.df["베이스 노트 키워드"].fillna('').astype(str) + ',' +
            self.df["탑 노트 설명"].fillna('').astype(str) + ',' +
            self.df["미들 노트 설명"].fillna('').astype(str) + ',' +
            self.df["베이스 노트 설명"].fillna('').astype(str) + ',' +
            self.df["한줄소개"].fillna('').astype(str) + ',' +
            self.df["성별"].fillna('').astype(str) + ',' +
            self.df["계절"].fillna('').astype(str)
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["full_text"])
        self.feature_names = self.vectorizer.get_feature_names_out()

    def _match_score(self, user_value: str, perfume_value: str) -> float:
        if pd.isna(perfume_value) or not user_value:
            return 0
        return 0.1 if user_value.strip() in str(perfume_value).strip() else 0

    def _top_influential_keywords(self, user_keywords: list[str], perfume_vector) -> list[str]:
        scores = {}
        perfume_vector_array = perfume_vector.toarray().flatten()
        for kw in user_keywords:
            if kw in self.feature_names:
                idx = list(self.feature_names).index(kw)
                scores[kw] = perfume_vector_array[idx]
        sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_keywords[:3]]

    def recommend(self, ambience: str, style: str, gender: str, season: str, personality: str, top_n: int = 3) -> dict:
        user_keywords = [ambience, style, gender, season, personality]
        
        query = " ".join(user_keywords)
        query_vec = self.vectorizer.transform([query])

        if query_vec.nnz == 0:
            print("❌ query 벡터가 0입니다. 유사도 계산 불가")
            return {
                "average_similarity": 0,
                "results": []
            }

        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        scores = []
        for i, row in self.df.iterrows():
            gender_score = self._match_score(gender, row.get("성별", ""))
            season_score = self._match_score(season, row.get("계절", ""))
            final_score = cosine_sim[i] + gender_score + season_score
            scores.append((i, final_score))

        sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for i, score in sorted_results:
            row = self.df.iloc[i]
            perfume_vector = self.tfidf_matrix[i]
            top_keywords = self._top_influential_keywords(user_keywords, perfume_vector)

            results.append({
                "brand": safe_str(row.get("브랜드", "")),
                "name": safe_str(row.get("향수이름", "")),
                "topNote": safe_str(row.get("탑 노트 키워드", "")),
                "middleNote": safe_str(row.get("미들 노트 키워드", "")),
                "baseNote": safe_str(row.get("베이스 노트 키워드", "")),
                "description": safe_str(row.get("한줄소개", row.get("향수 키워드", ""))),
                "relatedKeywords": top_keywords,
                "imageUrl": safe_str(row.get("향수 이미지", ""))
                })

        similarity_scores = [cosine_sim[i] for i, _ in sorted_results]
        avg_sim = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

        return {
            "average_similarity": round(avg_sim, 4),
            "results": results
        }
