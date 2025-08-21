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
    WEIGHT_CORE_KEYWORDS,
    WEIGHT_DESCRIPTION,
    WEIGHT_NOTES,
    WEIGHT_BRAND,
    WEIGHT_CONTEXT,
    BRAND_DIVERSITY_RATIO,
    NOTE_DIVERSITY_RATIO
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
        """
        속성별 가중치를 적용하여 문서 준비
        - 핵심 속성에 더 높은 가중치 부여
        - 장소 속성 추가 활용
        """
        # 속성별 가중치 적용된 텍스트 생성
        core_keywords = (self.df["향수 키워드"].fillna('').astype(str) + ' ') * int(WEIGHT_CORE_KEYWORDS)
        description = (self.df["한줄소개"].fillna('').astype(str) + ' ') * int(WEIGHT_DESCRIPTION)
        
        note_texts = (
            (self.df["탑 노트 설명"].fillna('').astype(str) + ' ' +
             self.df["미들 노트 설명"].fillna('').astype(str) + ' ' +
             self.df["베이스 노트 설명"].fillna('').astype(str) + ' ') * int(WEIGHT_NOTES)
        )
        
        brand_text = (self.df["브랜드"].fillna('').astype(str) + ' ') * int(WEIGHT_BRAND)
        
        context_text = (
            (self.df["성별"].fillna('').astype(str) + ' ' +
             self.df["계절"].fillna('').astype(str) + ' ' +
             self.df["장소"].fillna('').astype(str) + ' ') * int(WEIGHT_CONTEXT)
        )
        
        # 모든 속성 결합
        self.df["full_text"] = (
            core_keywords +
            description +
            note_texts +
            brand_text +
            context_text +
            self.df["탑 노트 키워드"].fillna('').astype(str) + ' ' +
            self.df["미들 노트 키워드"].fillna('').astype(str) + ' ' +
            self.df["베이스 노트 키워드"].fillna('').astype(str)
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["full_text"])
        self.feature_names = self.vectorizer.get_feature_names_out()

    def _match_score(self, user_value: str, perfume_value: str, weight: float = 0.1) -> float:
        """개선된 매칭 스코어 계산"""
        if pd.isna(perfume_value) or not user_value:
            return 0
        return weight if user_value.strip().lower() in str(perfume_value).strip().lower() else 0
    
    def _calculate_diversity_penalty(self, results: list, new_item: dict) -> float:
        """다양성 페널티 계산 (브랜드/노트 중복 방지)"""
        penalty = 0
        
        # 브랜드 다양성 검사
        existing_brands = [r.get('brand', '') for r in results]
        if new_item.get('brand', '') in existing_brands:
            penalty += 0.2
        
        # 노트 계열 다양성 검사
        new_notes = (new_item.get('topNote', '') + ' ' + 
                    new_item.get('middleNote', '') + ' ' + 
                    new_item.get('baseNote', '')).lower()
        
        for result in results:
            existing_notes = (result.get('topNote', '') + ' ' + 
                            result.get('middleNote', '') + ' ' + 
                            result.get('baseNote', '')).lower()
            
            # 노트 유사성 계산 (단순 문자열 매칭)
            if existing_notes and new_notes:
                common_words = set(existing_notes.split()) & set(new_notes.split())
                if len(common_words) > 2:  # 공통 단어 3개 이상
                    penalty += 0.15
        
        return penalty

    def _calculate_keyword_rarity_weights(self, user_keywords: list[str]) -> dict:
        """
        키워드 희소성 기반 동적 가중치 계산
        - 희소한 키워드일수록 높은 가중치 부여
        """
        keyword_weights = {}
        total_documents = len(self.df)
        
        for keyword in user_keywords:
            if not keyword:
                continue
                
            # 해당 키워드를 포함하는 문서 수 계산
            matching_count = self.df['full_text'].str.contains(
                keyword.lower(), case=False, na=False
            ).sum()
            
            if matching_count == 0:
                # 전혀 매칭되지 않는 키워드는 최고 가중치
                keyword_weights[keyword] = 2.0
            else:
                # 희소성 계산: (1 - 발생빈도) * 2 + 0.5
                frequency = matching_count / total_documents
                rarity_score = max(0.5, (1 - frequency) * 2 + 0.5)
                keyword_weights[keyword] = min(2.0, rarity_score)
        
        return keyword_weights
    
    def _apply_keyword_weights(self, query: str, user_keywords: list[str]) -> str:
        """
        키워드 가중치를 적용한 쿼리 문자열 생성
        """
        keyword_weights = self._calculate_keyword_rarity_weights(user_keywords)
        
        weighted_query_parts = []
        for keyword in user_keywords:
            weight = keyword_weights.get(keyword, 1.0)
            # 가중치만큼 키워드 반복
            repeat_count = int(weight)
            weighted_query_parts.extend([keyword] * repeat_count)
        
        return " ".join(weighted_query_parts)
    
    def _top_influential_keywords(self, user_keywords: list[str], perfume_vector) -> list[str]:
        """
        키워드 가중치를 고려한 영향력 있는 키워드 추출
        """
        scores = {}
        perfume_vector_array = perfume_vector.toarray().flatten()
        keyword_weights = self._calculate_keyword_rarity_weights(user_keywords)
        
        for kw in user_keywords:
            if kw in self.feature_names:
                idx = list(self.feature_names).index(kw)
                base_score = perfume_vector_array[idx]
                # 키워드 희소성 가중치 적용
                weighted_score = base_score * keyword_weights.get(kw, 1.0)
                scores[kw] = weighted_score
        
        sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_keywords[:3]]

    def recommend(self, ambience: str, style: str, gender: str, season: str, personality: str, top_n: int = 3) -> dict:
        """
        개선된 TF-IDF 추천 시스템
        - 데이터셋 장소 속성 내부 활용
        - 다양성 고려 알고리즘
        - 개선된 매칭 스코어
        """
        user_keywords = [ambience, style, gender, season, personality]
        
        # 키워드 희소성 기반 동적 가중치 적용
        weighted_query = self._apply_keyword_weights(" ".join(user_keywords), user_keywords)
        print(f"🎯 키워드 가중치 적용 - 원본: {user_keywords}")
        print(f"🎯 가중치 적용 후: {weighted_query}")
        
        query_vec = self.vectorizer.transform([weighted_query])

        if query_vec.nnz == 0:
            print("❌ query 벡터가 0입니다. 유사도 계산 불가")
            return {
                "average_similarity": 0,
                "results": []
            }

        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        scores = []
        for i, row in self.df.iterrows():
            # 기본 코사인 유사도
            base_score = cosine_sim[i]
            
            # 컨텍스트 매칭 스코어 (가중치 적용)
            gender_score = self._match_score(gender, row.get("성별", ""), 0.15)
            season_score = self._match_score(season, row.get("계절", ""), 0.15)
            # 데이터셋의 장소 속성은 TF-IDF 벡터화에서 이미 반영됨
            
            final_score = base_score + gender_score + season_score
            scores.append((i, final_score))

        # 상위 후보군 선별 (top_n * 2 배를 선별하여 다양성 고려)
        candidate_size = min(top_n * 3, len(scores))
        sorted_candidates = sorted(scores, key=lambda x: x[1], reverse=True)[:candidate_size]

        # 다양성 고려 선별
        final_results = []
        for i, score in sorted_candidates:
            if len(final_results) >= top_n:
                break
                
            row = self.df.iloc[i]
            perfume_vector = self.tfidf_matrix[i]
            top_keywords = self._top_influential_keywords(user_keywords, perfume_vector)

            candidate_item = {
                "similarity": round(score, 4),
                "brand": safe_str(row.get("브랜드", "")),
                "name": safe_str(row.get("향수이름", "")),
                "topNote": safe_str(row.get("탑 노트 키워드", "")),
                "middleNote": safe_str(row.get("미들 노트 키워드", "")),
                "baseNote": safe_str(row.get("베이스 노트 키워드", "")),
                "description": safe_str(row.get("한줄소개", row.get("향수 키워드", ""))),
                "relatedKeywords": top_keywords,
                "imageUrl": safe_str(row.get("향수 이미지", "")),
                "removebgImageUrl": safe_str(row.get("rmbg_s3_url", ""))
            }
            
            # 다양성 페널티 계산
            diversity_penalty = self._calculate_diversity_penalty(final_results, candidate_item)
            adjusted_score = score - diversity_penalty
            
            # 다양성을 고려한 선별 (나쁜 전체 점수인 경우 제외)
            if adjusted_score > 0.05 or len(final_results) < 2:  # 최소 2개는 보장
                candidate_item["similarity"] = round(adjusted_score, 4)
                final_results.append(candidate_item)

        # 결과가 부족한 경우 추가 채우기
        if len(final_results) < top_n:
            remaining_candidates = sorted_candidates[len(final_results):]
            for i, score in remaining_candidates[:top_n - len(final_results)]:
                row = self.df.iloc[i]
                perfume_vector = self.tfidf_matrix[i]
                top_keywords = self._top_influential_keywords(user_keywords, perfume_vector)
                
                final_results.append({
                    "similarity": round(score, 4),
                    "brand": safe_str(row.get("브랜드", "")),
                    "name": safe_str(row.get("향수이름", "")),
                    "topNote": safe_str(row.get("탑 노트 키워드", "")),
                    "middleNote": safe_str(row.get("미들 노트 키워드", "")),
                    "baseNote": safe_str(row.get("베이스 노트 키워드", "")),
                    "description": safe_str(row.get("한줄소개", row.get("향수 키워드", ""))),
                    "relatedKeywords": top_keywords,
                    "imageUrl": safe_str(row.get("향수 이미지", "")),
                    "removebgImageUrl": safe_str(row.get("rmbg_s3_url", ""))
                })

        similarity_scores = [r["similarity"] for r in final_results]
        avg_sim = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

        return {
            "average_similarity": round(avg_sim, 4),
            "results": final_results
        }
