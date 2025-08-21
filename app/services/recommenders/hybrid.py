# recommender_hybrid.py
import numpy as np
import random
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import (
    DEFAULT_TOP_N, 
    DEFAULT_ALPHA,
    DYNAMIC_ALPHA_HIGH,
    DYNAMIC_ALPHA_LOW,
    TFIDF_VALIDITY_THRESHOLD,
    DIVERSITY_WEIGHT,
    BRAND_DIVERSITY_RATIO,
    NOTE_DIVERSITY_RATIO,
    FRAGRANCE_FAMILIES,
    PRICE_TIERS,
    DIVERSITY_SEED_RANGE,
    ENABLE_SEED_RANDOMIZATION
)
from app.core.utils import safe_str


class HybridPerfumeRecommender:
    def __init__(self, tfidf_recommender, sbert_recommender):
        self.tfidf = tfidf_recommender
        self.sbert = sbert_recommender
    
    def _calculate_dynamic_alpha(self, tfidf_avg_similarity: float) -> float:
        """
        TF-IDF 유효성에 따른 동적 가중치 계산
        - 유효성이 높으면 TF-IDF 비중 증가
        - 유효성이 낮으면 SBERT 비중 증가
        """
        if tfidf_avg_similarity > TFIDF_VALIDITY_THRESHOLD * 2:
            return DYNAMIC_ALPHA_HIGH  # 0.5
        elif tfidf_avg_similarity > TFIDF_VALIDITY_THRESHOLD:
            return DEFAULT_ALPHA  # 0.3
        else:
            return DYNAMIC_ALPHA_LOW  # 0.2
    
    def _calculate_mmr_score(self, similarity_score: float, item_features: dict, selected_items: list, user_keywords: list = None) -> float:
        """
        고도화된 MMR(Maximal Marginal Relevance) 스코어 계산
        - 브랜드, 노트, 스타일, 키워드, 인기도 다양성 종합 고려
        - 적응적 페널티 시스템
        """
        if not selected_items:
            return similarity_score
        
        # 다양성 페널티 계산
        diversity_penalty = 0
        selection_count = len(selected_items)
        
        # 1. 브랜드 다양성 (가장 중요)
        selected_brands = [item.get('brand', '') for item in selected_items]
        current_brand = item_features.get('brand', '')
        if current_brand in selected_brands:
            brand_count = selected_brands.count(current_brand)
            # 브랜드 중복 개수에 따라 페널티 증가 (지수적, 더 강화)
            brand_penalty = (brand_count ** 2.0) * 0.4  # 지수 1.5->2.0, 계수 0.25->0.4로 강화
            diversity_penalty += brand_penalty
        
        # 2. 노트 계열 다양성 (정밀한 계산)
        current_notes = self._extract_note_features(item_features)
        
        note_similarity_sum = 0
        for selected_item in selected_items:
            selected_notes = self._extract_note_features(selected_item)
            note_similarity = self._calculate_note_similarity(current_notes, selected_notes)
            note_similarity_sum += note_similarity
        
        avg_note_similarity = note_similarity_sum / selection_count
        if avg_note_similarity > 0.3:  # 기준 0.4에서 0.3으로 더욱 엄격화
            note_penalty = (avg_note_similarity - 0.3) * 0.4  # 0.3->0.4로 페널티 강화
            diversity_penalty += note_penalty
        
        # 3. 향수 스타일 다양성 (한줄소개 기반)
        current_style = item_features.get('description', '').lower()
        if current_style:
            style_similarity_sum = 0
            for selected_item in selected_items:
                selected_style = selected_item.get('description', '').lower()
                if selected_style:
                    # 간단한 단어 겹침 비율 계산
                    current_words = set(current_style.split())
                    selected_words = set(selected_style.split())
                    if current_words and selected_words:
                        overlap_ratio = len(current_words & selected_words) / len(current_words | selected_words)
                        style_similarity_sum += overlap_ratio
            
            avg_style_similarity = style_similarity_sum / selection_count
            if avg_style_similarity > 0.25:  # 기준 0.3에서 0.25로 더욱 엄격화
                style_penalty = (avg_style_similarity - 0.25) * 0.3  # 0.2->0.3으로 페널티 강화
                diversity_penalty += style_penalty
        
        # 4. 키워드 매칭 다양성 (새로 추가)
        if user_keywords:
            keyword_penalty = self._calculate_keyword_diversity_penalty(item_features, selected_items, user_keywords)
            diversity_penalty += keyword_penalty
        
        # 5. 인기도 다양성 (새로 추가)
        popularity_penalty = self._calculate_popularity_diversity_penalty(item_features, selected_items)
        diversity_penalty += popularity_penalty
        
        # 6. 향수 계열 다양성 (새로 추가)
        fragrance_family_penalty = self._calculate_fragrance_family_diversity_penalty(item_features, selected_items)
        diversity_penalty += fragrance_family_penalty
        
        # 7. 가격대 다양성 (새로 추가)  
        price_tier_penalty = self._calculate_price_tier_diversity_penalty(item_features, selected_items)
        diversity_penalty += price_tier_penalty
        
        # 적응적 다양성 가중치 (선택된 아이템이 많을수록 다양성 중시, 더 강화)
        adaptive_diversity_weight = DIVERSITY_WEIGHT + (selection_count * 0.12)  # 0.08에서 0.12로 증가
        
        # MMR 스코어 계산
        mmr_score = similarity_score - adaptive_diversity_weight * diversity_penalty
        
        # 최소 점수 보장 (원래 점수의 15%로 더욱 엄격화)
        min_score = similarity_score * 0.15
        return max(mmr_score, min_score)
    
    def _extract_note_features(self, item: dict) -> dict:
        """향수 노트 특징 추출"""
        return {
            'top': set(item.get('topNote', '').lower().split()),
            'middle': set(item.get('middleNote', '').lower().split()),
            'base': set(item.get('baseNote', '').lower().split())
        }
    
    def _calculate_note_similarity(self, notes1: dict, notes2: dict) -> float:
        """노트 간 유사도 계산 (가중치 적용)"""
        similarities = []
        weights = {'top': 0.4, 'middle': 0.4, 'base': 0.2}  # 탑/미들 노트에 더 큰 가중치
        
        for note_type in ['top', 'middle', 'base']:
            set1 = notes1.get(note_type, set())
            set2 = notes2.get(note_type, set())
            
            if set1 and set2:
                similarity = len(set1 & set2) / len(set1 | set2)
                similarities.append(similarity * weights[note_type])
            else:
                similarities.append(0)
        
        return sum(similarities)
    
    def _get_matched_keywords(self, item: dict, user_keywords: list) -> list:
        """향수가 매칭하는 사용자 키워드 추출"""
        matched_keywords = []
        
        # 향수 정보를 통합한 텍스트 생성
        full_perfume_text = (
            item.get('description', '') + ' ' +
            item.get('topNote', '') + ' ' +
            item.get('middleNote', '') + ' ' +
            item.get('baseNote', '') + ' ' +
            item.get('brand', '')
        ).lower()
        
        # 각 사용자 키워드가 향수 정보에 포함되어 있는지 확인
        for keyword in user_keywords:
            if keyword.lower() in full_perfume_text:
                matched_keywords.append(keyword)
        
        return matched_keywords
    
    def _calculate_keyword_diversity_penalty(self, candidate: dict, selected_items: list, user_keywords: list) -> float:
        """
        키워드 매칭 패턴의 다양성 평가
        - 동일한 키워드 패턴으로 매칭되는 경우 페널티
        """
        if not selected_items or not user_keywords:
            return 0
        
        penalty = 0
        candidate_matched = set(self._get_matched_keywords(candidate, user_keywords))
        
        for selected in selected_items:
            selected_matched = set(self._get_matched_keywords(selected, user_keywords))
            
            if candidate_matched and selected_matched:
                # 공통 매칭 키워드 비율 계산
                intersection = len(candidate_matched & selected_matched)
                union = len(candidate_matched | selected_matched)
                overlap_ratio = intersection / union if union > 0 else 0
                
                # 겹침이 많을수록 페널티 증가 (더 엄격하게)
                if overlap_ratio > 0.5:  # 50% 이상 겹치면 (기존 60%에서 강화)
                    penalty += overlap_ratio * 0.4  # 0.3->0.4로 페널티 강화
                elif intersection >= 2:  # 2개 이상 공통 키워드 (기존 3개에서 강화)
                    penalty += 0.25  # 0.2->0.25로 페널티 강화
        
        return penalty
    
    def _calculate_popularity_diversity_penalty(self, candidate: dict, selected_items: list) -> float:
        """
        인기도 기반 다양성 평가
        - 유명한 브랜드의 유명한 향수들이 과도하게 중복되는 것을 방지
        """
        # 인기 브랜드 리스트 (데이터 기반 사전 정의)
        premium_brands = [
            '샤넥', 'CHANEL', '디올', 'DIOR', '에르메스', 'HERMES',
            '톰포드', 'TOM FORD', '조르지오 아르마니', 'GIORGIO ARMANI',
            '란콤', 'LANCOME', '이브 생로랑', 'YSL'
        ]
        
        # 인기 향수 패턴 (브랜드명이 향수명에 포함)
        popular_patterns = ['No.5', '미스 디올', '샤넵', '코코', '정원']
        
        penalty = 0
        candidate_brand = candidate.get('brand', '')
        candidate_name = candidate.get('name', '')
        
        # 프리미엄 브랜드 중복 방지
        if candidate_brand in premium_brands:
            premium_count = sum(1 for item in selected_items if item.get('brand') in premium_brands)
            if premium_count >= 2:  # 이미 2개 이상의 프리미엄 브랜드
                penalty += 0.15
        
        # 인기 향수 패턴 중복 방지
        for pattern in popular_patterns:
            if pattern in candidate_name:
                pattern_count = sum(1 for item in selected_items if pattern in item.get('name', ''))
                if pattern_count >= 1:  # 이미 유사한 패턴 존재
                    penalty += 0.1
                    break
        
        return penalty
    
    def _classify_fragrance_family(self, perfume_info: dict) -> str:
        """
        향수 계열 분류 (노트와 키워드 기반)
        """
        perfume_text = (
            perfume_info.get('topNote', '') + ' ' +
            perfume_info.get('middleNote', '') + ' ' +
            perfume_info.get('baseNote', '') + ' ' +
            perfume_info.get('description', '')
        ).lower()
        
        family_scores = {}
        for family, keywords in FRAGRANCE_FAMILIES.items():
            score = sum(1 for keyword in keywords if keyword in perfume_text)
            if score > 0:
                family_scores[family] = score
        
        if family_scores:
            return max(family_scores, key=family_scores.get)
        return '미분류'
    
    def _classify_price_tier(self, brand: str) -> str:
        """
        브랜드 기반 가격대 분류
        """
        brand_lower = brand.lower()
        for tier, brands in PRICE_TIERS.items():
            for tier_brand in brands:
                if tier_brand.lower() in brand_lower or brand_lower in tier_brand.lower():
                    return tier
        return 'mid_range'  # 중간 가격대로 기본 설정
    
    def _calculate_fragrance_family_diversity_penalty(self, candidate: dict, selected_items: list) -> float:
        """
        향수 계열 다양성 페널티 계산
        """
        if not selected_items:
            return 0
        
        candidate_family = self._classify_fragrance_family(candidate)
        selected_families = [self._classify_fragrance_family(item) for item in selected_items]
        
        family_count = selected_families.count(candidate_family)
        if family_count > 0:
            # 동일 계열이 많을수록 페널티 증가
            return family_count * 0.2
        return 0
    
    def _calculate_price_tier_diversity_penalty(self, candidate: dict, selected_items: list) -> float:
        """
        가격대 다양성 페널티 계산
        """
        if not selected_items:
            return 0
        
        candidate_tier = self._classify_price_tier(candidate.get('brand', ''))
        selected_tiers = [self._classify_price_tier(item.get('brand', '')) for item in selected_items]
        
        tier_count = selected_tiers.count(candidate_tier)
        if tier_count > 0:
            # 동일 가격대가 많을수록 페널티 증가
            return tier_count * 0.15
        return 0
    
    def _generate_diversity_seed(self, user_keywords: list) -> int:
        """
        사용자 키워드 기반 다양성 시드 생성
        """
        if not ENABLE_SEED_RANDOMIZATION:
            return 42  # 고정 시드
        
        # 진정한 랜덤성을 위해 현재 시간의 마이크로초 사용
        import time
        import os
        
        # 1. 현재 시간 (마이크로초까지)
        current_time = int(time.time() * 1000000)
        
        # 2. 프로세스 ID (추가 랜덤성)
        process_factor = os.getpid()
        
        # 3. 키워드 해시 (일관성)
        keywords_str = ''.join(sorted(user_keywords))
        hash_obj = hashlib.md5(keywords_str.encode())
        keyword_factor = int(hash_obj.hexdigest()[:8], 16)
        
        # 세 요소를 결합하여 시드 생성
        final_seed = (current_time + process_factor + keyword_factor) % DIVERSITY_SEED_RANGE
        
        return final_seed
    
    def _ensure_diversity(self, candidates: list, top_n: int, user_keywords: list = None) -> list:
        """
        다양성을 보장하는 최종 선별 알고리즘
        - MMR 등들라리 선별을 통한 다양성 최적화
        - 키워드 기반 다양성 고려 추가
        """
        if len(candidates) <= top_n:
            return candidates
        
        # 다양성 시드 적용
        diversity_seed = self._generate_diversity_seed(user_keywords or [])
        random.seed(diversity_seed)
        print(f"🎲 다양성 시드 적용: {diversity_seed}")
        
        selected = []
        remaining = candidates.copy()
        
        # 첫 번째 선택에 약간의 랜덤성 추가 (상위 3개 중에서 선택)
        if ENABLE_SEED_RANDOMIZATION and len(remaining) >= 3:
            top_3_candidates = remaining[:3]
            first_choice = random.choice(top_3_candidates)
            remaining.remove(first_choice)
            selected.append(first_choice)
        else:
            # 기존 방식: 가장 높은 점수
            selected.append(remaining.pop(0))
        
        # 나머지는 MMR 기반으로 선별 (랜덤성 강화)
        while len(selected) < top_n and remaining:
            mmr_scores = []
            
            # 모든 후보의 MMR 스코어 계산
            for i, candidate in enumerate(remaining):
                mmr_score = self._calculate_mmr_score(
                    candidate['similarity'],
                    candidate,
                    selected,
                    user_keywords  # 키워드 전달
                )
                mmr_scores.append((i, mmr_score))
            
            # MMR 스코어 기준 정렬
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            
            if ENABLE_SEED_RANDOMIZATION and len(mmr_scores) >= 2:
                # 상위 후보 중에서 확률적 선택 (가중 랜덤)
                top_candidates = mmr_scores[:min(3, len(mmr_scores))]
                weights = [score for _, score in top_candidates]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    # 가중치 기반 랜덤 선택
                    rand_val = random.uniform(0, total_weight)
                    cumulative = 0
                    selected_idx = top_candidates[0][0]  # 기본값
                    
                    for idx, weight in top_candidates:
                        cumulative += weight
                        if rand_val <= cumulative:
                            selected_idx = idx
                            break
                else:
                    selected_idx = mmr_scores[0][0]
            else:
                # 기존 방식: 최고 MMR 스코어
                selected_idx = mmr_scores[0][0]
            
            # 선택된 항목 처리
            selected_item = remaining.pop(selected_idx)
            best_mmr_score = mmr_scores[selected_idx][1] if selected_idx < len(mmr_scores) else mmr_scores[0][1]
            selected_item['similarity'] = round(best_mmr_score, 4)  # MMR 스코어로 업데이트
            selected.append(selected_item)
        
        return selected
    
    def _apply_brand_diversity_filter(self, candidates: list, top_n: int) -> list:
        """
        브랜드 다양성을 보장하는 사전 필터링
        - 브랜드별 최대 할당량 설정
        - 다양한 브랜드에서 고루 선별
        """
        if len(candidates) <= top_n:
            return candidates
        
        # 브랜드별 최대 할당량 계산
        max_per_brand = max(1, top_n // 2)  # 브랜드별 최대 50%
        
        # 브랜드별 그룹화
        brand_groups = {}
        for candidate in candidates:
            brand = candidate.get('brand', '브랜드없음')
            if brand not in brand_groups:
                brand_groups[brand] = []
            brand_groups[brand].append(candidate)
        
        # 다양성 선별 전략
        selected_candidates = []
        brand_quotas = {brand: 0 for brand in brand_groups.keys()}
        
        # 라운드 로빈 방식으로 브랜드에서 번갈아가며 선별
        round_count = 0
        while len(selected_candidates) < len(candidates) and round_count < top_n:
            added_in_round = False
            
            # 각 브랜드에서 1개씩 선별 시도
            for brand, brand_candidates in brand_groups.items():
                if (brand_quotas[brand] < max_per_brand and 
                    brand_quotas[brand] < len(brand_candidates) and
                    len(selected_candidates) < len(candidates)):
                    
                    # 점수순으로 정렬된 브랜드 후보 중 다음 선별
                    candidate = brand_candidates[brand_quotas[brand]]
                    selected_candidates.append(candidate)
                    brand_quotas[brand] += 1
                    added_in_round = True
            
            if not added_in_round:
                break
            round_count += 1
        
        # 남은 자리를 점수 순으로 채우기
        remaining_candidates = [
            c for c in candidates 
            if c not in selected_candidates
        ]
        
        final_candidates = selected_candidates + remaining_candidates
        
        print(f"🎨 브랜드 다양성 필터링 - 전체: {len(candidates)}개, 선별: {len(final_candidates[:top_n*3])}개")
        print(f"🎨 브랜드 분포: {dict(brand_quotas)}")
        
        return final_candidates[:top_n*3]  # MMR용 후보군 반환
    
    def _validate_recommendation_diversity(self, results: list, threshold: float = 0.6) -> dict:
        """
        추천 결과의 다양성 검증
        - 브랜드, 노트, 스타일 다양성 종합 평가
        """
        if len(results) < 2:
            return {
                "is_diverse": True,
                "diversity_score": 1.0,
                "brand_diversity": 1.0,
                "note_diversity": 1.0,
                "style_diversity": 1.0
            }
        
        # 1. 브랜드 다양성
        brands = [r.get('brand', '') for r in results if r.get('brand')]
        brand_diversity = len(set(brands)) / len(brands) if brands else 0
        
        # 2. 노트 다양성
        all_notes = []
        for r in results:
            notes_text = f"{r.get('topNote', '')} {r.get('middleNote', '')} {r.get('baseNote', '')}"
            notes_words = [word.strip() for word in notes_text.split() if word.strip()]
            all_notes.extend(notes_words)
        
        note_diversity = len(set(all_notes)) / len(all_notes) if all_notes else 0
        
        # 3. 스타일 다양성 (한줄소개 기반)
        descriptions = [r.get('description', '').lower() for r in results if r.get('description')]
        all_desc_words = []
        for desc in descriptions:
            all_desc_words.extend(desc.split())
        
        style_diversity = len(set(all_desc_words)) / len(all_desc_words) if all_desc_words else 0
        
        # 4. 향수 계열 다양성
        fragrance_families = [self._classify_fragrance_family(r) for r in results]
        family_diversity = len(set(fragrance_families)) / len(fragrance_families) if fragrance_families else 0
        
        # 5. 가격대 다양성
        price_tiers = [self._classify_price_tier(r.get('brand', '')) for r in results]
        tier_diversity = len(set(price_tiers)) / len(price_tiers) if price_tiers else 0
        
        # 종합 다양성 점수 (가중치 재조정)
        diversity_score = (
            brand_diversity * 0.3 + 
            note_diversity * 0.25 + 
            style_diversity * 0.2 +
            family_diversity * 0.15 +
            tier_diversity * 0.1
        )
        
        return {
            "is_diverse": diversity_score >= threshold,
            "diversity_score": round(diversity_score, 3),
            "brand_diversity": round(brand_diversity, 3),
            "note_diversity": round(note_diversity, 3),
            "style_diversity": round(style_diversity, 3),
            "family_diversity": round(family_diversity, 3),
            "tier_diversity": round(tier_diversity, 3)
        }
    
    def _get_alternative_recommendations(self, original_results: list, user_keywords: list, top_n: int) -> list:
        """
        다양성 부족 시 대체 추천 생성
        - 기존 결과와 다른 브랜드/스타일 위주로 선별
        """
        try:
            # 기존 결과에서 사용된 브랜드 리스트
            used_brands = set(r.get('brand', '') for r in original_results)
            
            # 대체 후보군 생성 (다른 브랜드 위주)
            alternative_candidates = []
            
            # SBERT 결과를 사용하여 다른 관점의 추천 생성
            sbert_result = self.sbert.recommend(
                *user_keywords[:5], top_n=top_n*3  # 더 많은 후보 요청
            )
            
            for candidate in sbert_result.get('results', []):
                candidate_brand = candidate.get('brand', '')
                # 기존에 사용되지 않은 브랜드 우선 선별
                if candidate_brand not in used_brands:
                    alternative_candidates.append(candidate)
                    
                if len(alternative_candidates) >= top_n:
                    break
            
            # 추가 후보가 부족한 경우 기존 결과로 채우기
            if len(alternative_candidates) < top_n:
                remaining_count = top_n - len(alternative_candidates)
                alternative_candidates.extend(original_results[:remaining_count])
            
            return alternative_candidates[:top_n]
            
        except Exception as e:
            print(f"❌ 대체 추천 생성 실패: {e}")
            return original_results

    def recommend(
        self, 
        ambience: str, 
        style: str, 
        gender: str, 
        season: str, 
        personality: str,
        top_n: int = DEFAULT_TOP_N, 
        alpha: float = None
    ) -> dict:
        """
        고도화된 하이브리드 추천 시스템
        - 동적 가중치 자동 조정
        - MMR 기반 다양성 보장
        - 데이터셋 장소 속성 내부 활용
        - 개선된 유사도 계산
        """

        # 후보군 확장: 더 많은 후보를 확보하여 다양성 개선
        expansion_factor = 5  # 기존 2배에서 5배로 확장
        expanded_top_n = max(top_n * expansion_factor, 15)  # 최소 15개 보장
        
        print(f"📈 후보군 확장 - 요청: {top_n}개, 후보군: {expanded_top_n}개")
        
        # 추천 결과 얻기 (확장된 후보군)
        tfidf_result = self.tfidf.recommend(
            ambience, style, gender, season, personality, top_n=expanded_top_n
        )
        sbert_result = self.sbert.recommend(
            ambience, style, gender, season, personality, top_n=expanded_top_n
        )

        # 동적 가중치 계산
        if alpha is None:
            alpha = self._calculate_dynamic_alpha(tfidf_result["average_similarity"])
        
        print(f"🔧 하이브리드 가중치 최적화 - TF-IDF: {alpha:.2f}, SBERT: {1-alpha:.2f}")

        # TF-IDF 결과가 없는 경우 SBERT 단독 사용
        if tfidf_result["average_similarity"] == 0:
            print("⚠️ TF-IDF 결과 없음 - SBERT 단독 모드로 전환")
            final_results = sbert_result["results"][:top_n]
            return {
                "average_similarity": sbert_result["average_similarity"],
                "results": final_results
            }
        
        # SBERT 결과가 없는 경우 TF-IDF 단독 사용 (추가 안전장치)
        if not sbert_result["results"]:
            print("⚠️ SBERT 결과 없음 - TF-IDF 단독 모드로 전환")
            final_results = tfidf_result["results"][:top_n]
            return {
                "average_similarity": tfidf_result["average_similarity"],
                "results": final_results
            }

        # 스코어 매핑 및 결합 (안전한 처리)
        try:
            tfidf_scores = {f"{r['brand']}|{r['name']}": r.get("similarity", 0) for r in tfidf_result["results"] if r.get('brand') and r.get('name')}
            sbert_scores = {f"{r['brand']}|{r['name']}": r.get("similarity", 0) for r in sbert_result["results"] if r.get('brand') and r.get('name')}
        except Exception as e:
            print(f"❌ 스코어 매핑 중 오류: {e}")
            # Fallback: 가장 좋은 단일 결과 반환
            best_result = tfidf_result if tfidf_result["average_similarity"] > sbert_result["average_similarity"] else sbert_result
            return {
                "average_similarity": best_result["average_similarity"],
                "results": best_result["results"][:top_n]
            }
        
        # 안전한 항목 결합
        all_items = list({*tfidf_scores.keys(), *sbert_scores.keys()})
        combined_candidates = []
        
        if not all_items:
            print("❌ 결합할 추천 결과가 없음 - 기본 결과 반환")
            return {
                "average_similarity": 0.1,
                "results": []
            }
        
        for item in all_items:
            try:
                tfidf_score = tfidf_scores.get(item, 0)
                sbert_score = sbert_scores.get(item, 0)
                
                # 가중 평균 점수 계산
                final_score = alpha * tfidf_score + (1 - alpha) * sbert_score
                
                # 안전한 파싱
                if "|" not in item:
                    print(f"⚠️ 잘못된 아이템 형식: {item}")
                    continue
                    
                brand, name = item.split("|", 1)  # 최대 1번만 분할
                
                # TF-IDF 또는 SBERT 결과에서 상세 정보 가져오기
                item_info = None
                for result in tfidf_result["results"] + sbert_result["results"]:
                    if result.get("brand") == brand and result.get("name") == name:
                        item_info = result.copy()
                        break
                
                if item_info:
                    item_info["similarity"] = round(final_score, 4)
                    combined_candidates.append(item_info)
                else:
                    print(f"⚠️ 아이템 정보 찾을 수 없음: {brand} | {name}")
                    
            except Exception as e:
                print(f"❌ 아이템 처리 중 오류: {item}, 에러: {e}")
                continue
        
        # 점수순 정렬 (안전한 처리)
        try:
            combined_candidates.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        except Exception as e:
            print(f"❌ 정렬 중 오류: {e}")
            # 정렬 실패 시 그대로 사용
        
        # 브랜드 다양성 사전 필터링
        try:
            diverse_candidates = self._apply_brand_diversity_filter(combined_candidates, top_n)
        except Exception as e:
            print(f"❌ 브랜드 다양성 필터링 중 오류: {e}")
            diverse_candidates = combined_candidates
        
        # 키워드 준비
        user_keywords = [ambience, style, gender, season, personality]
        
        # MMR 기반 다양성 보장 선별 (키워드 전달)
        try:
            final_results = self._ensure_diversity(diverse_candidates, top_n, user_keywords)
        except Exception as e:
            print(f"❌ 다양성 선별 중 오류: {e}")
            # Fallback: 브랜드 다양성 필터링된 결과에서 상위 N개 선택
            final_results = diverse_candidates[:top_n]
            
        # 안전한 키워드 업데이트
        for i, result in enumerate(final_results):
            try:
                brand = result.get("brand", "")
                name = result.get("name", "")
                
                if not brand or not name:
                    print(f"⚠️ 브랜드 또는 향수명이 누락됨: {result}")
                    result["relatedKeywords"] = user_keywords[:3]
                    continue
                
                # DataFrame 검색
                matching_rows = self.tfidf.df[
                    (self.tfidf.df["브랜드"] == brand) & (self.tfidf.df["향수이름"] == name)
                ]
                
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    idx = row.name
                    
                    if idx < len(self.tfidf.tfidf_matrix.toarray()):
                        perfume_vector = self.tfidf.tfidf_matrix[idx]
                        top_keywords = self.tfidf._top_influential_keywords(user_keywords, perfume_vector)
                        result["relatedKeywords"] = top_keywords if top_keywords else user_keywords[:3]
                    else:
                        result["relatedKeywords"] = user_keywords[:3]
                else:
                    print(f"⚠️ DB에서 향수를 찾을 수 없음: {brand} - {name}")
                    result["relatedKeywords"] = user_keywords[:3]
                    
            except Exception as e:
                print(f"❌ 키워드 업데이트 실패 (인덱스 {i}): {e}")
                result["relatedKeywords"] = user_keywords[:3]

        # 다양성 검증 및 재추천 시스템
        diversity_check = self._validate_recommendation_diversity(final_results)
        print(f"📋 다양성 검증 - 점수: {diversity_check['diversity_score']}, 브랜드: {diversity_check['brand_diversity']}")
        
        # 다양성이 부족한 경우 대체 추천 생성
        if not diversity_check["is_diverse"] and len(final_results) >= 2:
            print("⚠️ 다양성 부족 감지 - 대체 추천 시도")
            alternative_results = self._get_alternative_recommendations(final_results, user_keywords, top_n)
            
            # 대체 추천의 다양성 재검증
            alt_diversity = self._validate_recommendation_diversity(alternative_results)
            if alt_diversity["diversity_score"] > diversity_check["diversity_score"]:
                print(f"✅ 대체 추천 적용 - 개선된 다양성: {alt_diversity['diversity_score']}")
                final_results = alternative_results
                diversity_check = alt_diversity
        
        # 안전한 평균 점수 계산
        try:
            similarity_scores = [r.get("similarity", 0) for r in final_results if r.get("similarity") is not None]
            avg_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        except Exception as e:
            print(f"❌ 평균 점수 계산 실패: {e}")
            avg_score = 0.1  # 기본값
        
        # 최종 검증
        if not final_results:
            print("❌ 최종 결과가 비어있음 - 빈 결과 반환")
            return {
                "average_similarity": 0,
                "results": []
            }
        
        print(f"✅ 추천 완료 - {len(final_results)}개 결과, 평균 유사도: {avg_score:.4f}")
        print(f"✅ 최종 다양성 - 브랜드: {diversity_check['brand_diversity']}, 노트: {diversity_check['note_diversity']}")
        print(f"✅ 확장 다양성 - 향수계열: {diversity_check['family_diversity']}, 가격대: {diversity_check['tier_diversity']}")

        return {
            "average_similarity": round(avg_score, 4),
            "results": final_results,
            "diversity_info": diversity_check  # 다양성 정보 추가
        }
