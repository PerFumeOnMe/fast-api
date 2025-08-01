# ===========================================
# 1단계: 빌드 환경 (의존성 설치용)
# ===========================================
FROM python:3.9-slim as builder

# 빌드 도구 설치 (최소한으로)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 및 wheel 생성 최적화
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 의존성 파일만 먼저 복사 (Docker 레이어 캐싱 최적화)  
COPY requirements.txt /tmp/
WORKDIR /tmp

# 휠 파일 생성 (빌드 시간 단축)
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /tmp/wheels -r requirements.txt

# ===========================================  
# 2단계: 런타임 환경 (실제 실행용)
# ===========================================
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 런타임에 필요한 최소 패키지만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 1단계에서 빌드된 휠 파일들 복사
COPY --from=builder /tmp/wheels /wheels
COPY --from=builder /tmp/requirements.txt .

# 미리 빌드된 휠에서 패키지 설치 (훨씬 빠름)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

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