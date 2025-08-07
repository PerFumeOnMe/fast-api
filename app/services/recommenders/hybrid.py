# recommender_hybrid.py
from app.core.config import DEFAULT_TOP_N, DEFAULT_ALPHA
from app.core.utils import safe_str


class HybridPerfumeRecommender:
    def __init__(self, tfidf_recommender, sbert_recommender):
        self.tfidf = tfidf_recommender
        self.sbert = sbert_recommender

    def recommend(
        self, 
        ambience: str, 
        style: str, 
        gender: str, 
        season: str, 
        personality: str, 
        top_n: int = DEFAULT_TOP_N, 
        alpha: float = DEFAULT_ALPHA
    ) -> dict:
        """
        Hybrid 방식 추천 시스템
        - TF-IDF 추천 점수와 SBERT 추천 점수를 가중 평균하여 결과 결정
        - TF-IDF 평균 유사도가 0이면 SBERT 결과 단독 사용

        :param alpha: TF-IDF 점수 비중 (0.0 ~ 1.0), SBERT는 (1 - alpha)
        """

        # 추천 결과 얻기
        tfidf_result = self.tfidf.recommend(ambience, style, gender, season, personality, top_n=top_n)
        sbert_result = self.sbert.recommend(ambience, style, gender, season, personality, top_n=top_n)

        if tfidf_result["average_similarity"] == 0:
            return sbert_result

        tfidf_scores = {f"{r['brand']}|{r['name']}": r.get("similarity", 0) for r in tfidf_result["results"]}
        sbert_scores = {f"{r['brand']}|{r['name']}": r.get("similarity", 0) for r in sbert_result["results"]}
        all_items = list({*tfidf_scores.keys(), *sbert_scores.keys()})
        combined_scores = []
        for item in all_items:
            tfidf_score = tfidf_scores.get(item, 0)
            sbert_score = sbert_scores.get(item, 0)
            final_score = alpha * tfidf_score + (1 - alpha) * sbert_score
            combined_scores.append((item, final_score))

        top_items = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for item_key, score in top_items:
            brand, name = item_key.split("|")
            row = self.tfidf.df[
                (self.tfidf.df["브랜드"] == brand) & (self.tfidf.df["향수이름"] == name)
            ].iloc[0]

            idx = row.name
            perfume_vector = self.tfidf.tfidf_matrix[idx]
            user_keywords = [ambience, style, gender, season, personality]
            top_keywords = self.tfidf._top_influential_keywords(user_keywords, perfume_vector)

            results.append({
                "similarity": round(score, 4),
                "brand": safe_str(brand),
                "name": safe_str(name),
                "topNote": safe_str(row.get("탑 노트 키워드", "")),
                "middleNote": safe_str(row.get("미들 노트 키워드", "")),
                "baseNote": safe_str(row.get("베이스 노트 키워드", "")),
                "description": safe_str(row.get("한줄소개", row.get("향수 키워드", ""))),
                "relatedKeywords": top_keywords,
                "imageUrl": safe_str(row.get("향수 이미지", ""))
                })

        average_score = sum([score for _, score in top_items]) / top_n

        return {
            "average_similarity": round(average_score, 4),
            "results": results
        }
