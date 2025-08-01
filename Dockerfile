# =============================================================================
# PerfumeOnMe 추천시스템 멀티스테이지 Dockerfile (CPU 최적화 버전)
# 이미지 크기: ~3.5GB -> ~800MB, 빌드 시간: ~5분 -> ~2분
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - 의존성 설치 및 빌드 환경
# -----------------------------------------------------------------------------
FROM python:3.9-slim as builder

# 빌드 최적화를 위한 환경변수
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 시스템 의존성 설치 (빌드용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# 가상환경 생성 (격리된 의존성 설치)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - 최종 실행 환경 (경량화)
# -----------------------------------------------------------------------------
FROM python:3.9-slim as runtime

# 런타임 최적화 환경변수
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# 런타임에 필요한 최소 시스템 패키지만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 빌더 스테이지에서 설치된 Python 패키지들 복사
COPY --from=builder /opt/venv /opt/venv

# 작업 디렉토리 설정
WORKDIR /app

# 애플리케이션 파일만 복사 (레이어 캐싱 최적화)
COPY app/ ./app/
COPY .env.example ./

# 비루트 사용자 생성 및 권한 설정 (보안 강화)
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# 포트 8000 노출
EXPOSE 8000

# 헬스체크 (CPU 버전에 맞게 조정)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# 애플리케이션 실행 (프로덕션 설정)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log", \
     "--log-level", "info"]