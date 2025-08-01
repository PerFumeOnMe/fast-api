#!/bin/bash

# PerfumeOnMe FastAPI 추천시스템 배포 스크립트
# AWS EC2에서 Spring Boot와 함께 실행되는 독립 컨테이너 배포

set -e  # 에러 발생 시 스크립트 중단

echo "🚀 PerfumeOnMe FastAPI 추천시스템 배포 시작..."
echo "====================================="

# 현재 시간 로그
echo "⏰ 배포 시작 시간: $(date)"

# 환경변수 파일 확인 (선택사항 - GitHub Actions에서는 직접 환경변수 전달)
if [ -f ".env" ]; then
    echo "📋 환경변수 파일 발견됨"
    ENV_FILE_OPTION="--env-file .env"
else
    echo "⚠️  환경변수 파일 없음 - GitHub Actions 환경변수 사용"
    ENV_FILE_OPTION=""
fi

# Docker Hub에서 최신 이미지 가져오기 (로컬 빌드 대신)
echo "📥 Docker Hub에서 최신 이미지 가져오기..."
docker pull chanee29/perfume-recommender:latest || {
    echo "❌ Docker 이미지 풀 실패 - 로컬 빌드 시도..."
    docker build -t chanee29/perfume-recommender:latest .
}

# 기존 컨테이너 확인 및 중지
echo "🔄 기존 FastAPI 컨테이너 확인 및 중지..."
if [ "$(docker ps -q -f name=perfume-recommender-container)" ]; then
    echo "   기존 컨테이너 중지 중..."
    docker stop perfume-recommender-container
fi

# 기존 컨테이너 삭제
if [ "$(docker ps -aq -f name=perfume-recommender-container)" ]; then
    echo "   기존 컨테이너 삭제 중..."
    docker rm perfume-recommender-container
fi

# Spring Boot 컨테이너와 동일한 네트워크에서 FastAPI 컨테이너 실행
echo "🚀 FastAPI 컨테이너 실행 중..."
echo "   - 컨테이너명: perfume-recommender-container"
echo "   - 포트: 8000 (내부 통신용)"
echo "   - 네트워크: bridge (Spring Boot와 통신)"

docker run -d \
  --name perfume-recommender-container \
  --network bridge \
  -p 8000:8000 \
  ${ENV_FILE_OPTION} \
  --restart unless-stopped \
  chanee29/perfume-recommender:latest

# 컨테이너 상태 확인
echo "🔍 컨테이너 상태 확인..."
sleep 10  # 컨테이너 시작 대기

if [ "$(docker ps -q -f name=perfume-recommender-container)" ]; then
    echo "✅ 컨테이너가 성공적으로 실행되었습니다!"
    
    # 헬스체크
    echo "🩺 API 헬스체크 중..."
    sleep 5  # API 시작 대기
    
    # FastAPI 헬스체크 (Docker 컨테이너 내부에서 확인)
    if docker exec perfume-recommender-container curl -f http://localhost:8000/docs > /dev/null 2>&1; then
        echo "✅ FastAPI가 정상적으로 응답하고 있습니다!"
        echo "📋 API 문서: http://localhost:8000/docs (EC2 내부)"
        echo "🔗 Spring Boot 연동: http://perfume-recommender-container:8000"
    else
        echo "⚠️  FastAPI 응답 확인 실패. 로그를 확인하세요."
        echo "📋 로그 확인: docker logs perfume-recommender-container"
    fi
    
    # 컨테이너 정보 출력
    echo "📊 컨테이너 정보:"
    docker ps --filter name=perfume-recommender-container --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
else
    echo "❌ 컨테이너 실행 실패!"
    echo "📋 로그 확인: docker logs perfume-recommender-container"
    exit 1
fi

# 사용하지 않는 Docker 이미지 정리
echo "🧹 사용하지 않는 Docker 이미지 정리..."
docker image prune -f

echo "====================================="
echo "✅ FastAPI 배포 완료! ($(date))"
echo ""
echo "🌐 서비스 정보:"
echo "   - FastAPI 포트: 8000"
echo "   - Spring Boot 연동 URL: http://perfume-recommender-container:8000"
echo "   - 컨테이너명: perfume-recommender-container"
echo ""
echo "📋 관리 명령어:"
echo "   - 로그 확인: docker logs -f perfume-recommender-container"
echo "   - 컨테이너 재시작: docker restart perfume-recommender-container"
echo "   - 컨테이너 중지: docker stop perfume-recommender-container"
echo ""
echo "🔗 Spring Boot 설정에서 다음 URL 사용:"
echo "   EXTERNAL_FASTAPI_RECOMMEND_URL=http://perfume-recommender-container:8000"