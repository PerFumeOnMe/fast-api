# app/config.py

# 파일 경로
EXCEL_PATH = "data/perfume.xlsx"

# 추천 기본값
DEFAULT_TOP_N = 3
DEFAULT_ALPHA = 0.1  # 하이브리드 가중치: TF-IDF 비율

# SBERT 모델
SBERT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# 📊 TF-IDF 설정
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 3000