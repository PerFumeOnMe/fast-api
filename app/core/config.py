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
DEFAULT_ALPHA = 0.1  # 하이브리드 가중치: TF-IDF 비율

# SBERT 모델
SBERT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# 📊 TF-IDF 설정
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 3000