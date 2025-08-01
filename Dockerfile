# PerfumeOnMe 추천시스템 Dockerfile (최적화 버전)
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (최소화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# 의존성 파일 먼저 복사 (Docker 레이어 캐싱 최적화)
COPY requirements.txt .

# Python 패키지 설치 (캐시 최적화)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# 애플리케이션 파일 복사 (마지막에 복사하여 코드 변경 시 캐시 활용)
COPY . .

# 포트 8000 노출 (FastAPI 기본 포트)
EXPOSE 8000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# 애플리케이션 실행
# uvicorn 서버로 FastAPI 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]