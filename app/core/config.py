# app/config.py
import os
from dotenv import load_dotenv

# .env 로드 (로컬환경에서만 사용)
load_dotenv()
class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

settings = Settings()   


S3_BUCKET = os.getenv("S3_BUCKET", "umc-perfume-bucket")
S3_KEY = os.getenv("S3_KEY", "data/perfume.xlsx")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-1")

# 추천 기본값
DEFAULT_TOP_N = 3
DEFAULT_ALPHA = 0.3  # 하이브리드 가중치: TF-IDF 비율 (최적화됨)

# 동적 가중치 설정
DYNAMIC_ALPHA_HIGH = 0.5  # TF-IDF 유효성이 높을 때
DYNAMIC_ALPHA_LOW = 0.2   # TF-IDF 유효성이 낮을 때
TFIDF_VALIDITY_THRESHOLD = 0.1  # TF-IDF 유효성 판단 임계값

# 다양성 알고리즘 설정 (강화)
DIVERSITY_WEIGHT = 0.4    # MMR에서 다양성 가중치 (0.3->0.4로 증가)
BRAND_DIVERSITY_RATIO = 0.6  # 브랜드 다양성 비율
NOTE_DIVERSITY_RATIO = 0.7   # 노트 계열 다양성 비율

# 속성별 가중치 설정
WEIGHT_CORE_KEYWORDS = 3.0      # 향수 키워드
WEIGHT_DESCRIPTION = 2.0        # 한줄소개
WEIGHT_NOTES = 1.5              # 노트 설명
WEIGHT_BRAND = 1.2              # 브랜드명
WEIGHT_CONTEXT = 1.0            # 성별, 계절, 장소

# SBERT 모델
SBERT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# 📊 TF-IDF 설정
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 3000

# 🧠 PBTI 전용 설정
PBTI_SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
PBTI_GPT_MODEL = "gpt-4o"
PBTI_GPT_TEMPERATURE = 0.7

# 향수 계열 분류 (다양성 향상을 위한 새로운 시스템)
FRAGRANCE_FAMILIES = {
    '플로럴': ['꽃', '꽃향기', '플로럴', '로즈', '자스민', '피오니', '라일락'],
    '우디': ['나무', '우드', '우디', '샌달우드', '시더', '베티버', '오크'],
    '프레쉬': ['신선', '시트러스', '레몬', '오렌지', '자몽', '민트', '아쿠아틱'],
    '오리엔탈': ['스파이시', '바닐라', '머스크', '엠버', '인센스', '향신료'],
    '푸제르': ['라벤더', '허브', '아로마틱', '세이지', '로즈마리', '바질']
}

# 가격대 분류 (브랜드 기반 추정 - 다양성 향상)
PRICE_TIERS = {
    'luxury': ['샤넬', 'CHANEL', '디올', 'DIOR', '에르메스', 'HERMES', '톰포드', 'TOM FORD', '크리드', 'CREED'],
    'premium': ['조르지오 아르마니', 'GIORGIO ARMANI', '란콤', 'LANCOME', '이브 생로랑', 'YSL', '버버리', 'BURBERRY'],
    'accessible': ['더바디샵', 'THE BODY SHOP', '미샤', 'MISSHA', '이니스프리', 'innisfree', '에뛰드하우스', 'ETUDE HOUSE']
}

# 다양성 시드 설정 (동일 요청에 대해 다른 결과 제공)
DIVERSITY_SEED_RANGE = 1000  # 시드 랜덤화 범위 (0-999)
ENABLE_SEED_RANDOMIZATION = True  # 시드 랜덤화 활성화