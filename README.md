# PerfumeOnMe 추천시스템 API

AI 기반 향수 추천 및 감성 시나리오 생성 FastAPI 서버

## 🎯 주요 기능

- **하이브리드 추천 알고리즘**: TF-IDF + SBERT 조합으로 정확한 향수 추천
- **감성 시나리오 생성**: OpenAI GPT를 활용한 개인화된 감성 스토리
- **RESTful API**: FastAPI 기반 고성능 API 서버

## 🏗️ 아키텍처

```
클라이언트 → Spring Boot Backend → FastAPI Recommender → OpenAI API
                                        ↓
                               ML 모델 (TF-IDF + SBERT)
```

## 🚀 빠른 시작

### 로컬 개발 환경

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/perfume-recommender-api.git
cd perfume-recommender-api

# 2. 가상환경 생성 및 활성화
python -m venv perfumeenv
source perfumeenv/bin/activate  # Windows: perfumeenv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정

# 5. 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker 사용

```bash
# 1. Docker 이미지 빌드
docker build -t perfume-recommender .

# 2. 컨테이너 실행
docker run -d \
  --name perfume-recommender-container \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-api-key \
  perfume-recommender

# 또는 docker-compose 사용
docker-compose up -d
```

## 📋 API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 주요 엔드포인트

#### POST /recommend/full
5가지 키워드를 기반으로 감성 시나리오와 향수 추천을 제공합니다.

**요청 예시:**
```json
{
  "ambience": "세련된",
  "style": "유니크한",
  "gender": "여성스러운",
  "season": "겨울",
  "personality": "조용한"
}
```

**응답 예시:**
```json
{
  "scenario": "잔잔한 눈이 내리고 따뜻한 햇살이 얼굴을 감싸오는...",
  "recommendations": [
    {
      "brand": "메종마르지엘라",
      "name": "바이 더 파이어플레이스",
      "topNote": "핑크 페퍼, 오렌지 블로섬",
      "middleNote": "체스트넛, 통카빈",
      "baseNote": "바닐라, 캐시미어 우드",
      "description": "따뜻하고 조용한 불향. 겨울을 닮은 분위기",
      "relatedKeywords": ["세련된", "조용한", "유니크한"]
    }
  ]
}
```

## 🛠️ 기술 스택

- **Python 3.9+**
- **FastAPI**: 고성능 웹 프레임워크
- **scikit-learn**: TF-IDF 벡터화 및 유사도 계산
- **sentence-transformers**: SBERT 기반 의미론적 유사도
- **gensim**: Word2Vec 모델링
- **konlpy**: 한국어 자연어 처리
- **OpenAI API**: GPT 기반 감성 시나리오 생성
- **pandas**: 데이터 처리
- **uvicorn**: ASGI 서버

## 📁 프로젝트 구조

```
perfume-recommender/
├── app/
│   ├── main.py                    # FastAPI 애플리케이션 진입점
│   ├── config.py                  # 설정 파일
│   ├── schemas.py                 # Pydantic 모델
│   ├── recommend_full.py          # 통합 추천 로직
│   ├── recommender_hybrid.py      # 하이브리드 추천 알고리즘
│   ├── recommender_tf_idf.py      # TF-IDF 기반 추천
│   ├── recommender_sbert.py       # SBERT 기반 추천
│   ├── generator.py               # 감성 시나리오 생성
│   └── utils.py                   # 유틸리티 함수
├── data/
│   └── perfume.xlsx               # 향수 데이터셋
├── models/
│   └── ko_word2vec.model          # 한국어 Word2Vec 모델
├── scripts/
│   ├── convert_vec_to_model.py    # 모델 변환 스크립트
│   └── evaluate_hybrid_alpha.py   # 성능 평가 스크립트
├── Dockerfile                     # Docker 이미지 설정
├── docker-compose.yml             # Docker Compose 설정
├── deploy.sh                      # 배포 스크립트
├── requirements.txt               # Python 의존성
├── .env.example                   # 환경변수 템플릿
└── README.md                      # 프로젝트 문서
```

## ⚙️ 환경 변수

| 변수명 | 설명 | 필수 여부 | 기본값 |
|--------|------|-----------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 | 필수 | - |
| `PORT` | 서버 포트 | 선택 | 8000 |
| `HOST` | 서버 호스트 | 선택 | 0.0.0.0 |
| `DEFAULT_ALPHA` | 하이브리드 가중치 | 선택 | 0.1 |
| `DEFAULT_TOP_N` | 추천 향수 개수 | 선택 | 3 |
| `LOG_LEVEL` | 로그 레벨 | 선택 | INFO |

## 🚀 AWS EC2 배포

### 배포 방식 선택

#### 1. GitHub Actions 자동 배포 (권장 ⭐)

**특징:**
- Spring Boot와 동일한 CI/CD 패턴
- `main` 또는 `develop` 브랜치 푸시 시 자동 배포
- Docker Hub 기반 이미지 관리
- 완전 자동화된 배포 프로세스

**설정 방법:**
1. **GitHub Secrets 설정** (저장소 설정 → Secrets and variables → Actions)
   ```
   DOCKERHUB_USERNAME=your-dockerhub-username
   DOCKERHUB_TOKEN=your-dockerhub-token
   AWS_ACCESS_KEY_ID=your-aws-access-key
   AWS_ACCESS_KEY_PASSWORD=your-aws-secret-key
   AWS_SG_ID=your-security-group-id
   EC2_HOST=your-ec2-public-ip
   EC2_SSH_PRIVATE_KEY=your-ec2-private-key
   EC2_SSH_PORT=22
   OPENAI_API_KEY=your-openai-api-key
   ```

2. **브랜치에 푸시**
   ```bash
   git add .
   git commit -m "[Feature] FastAPI 기능 추가"
   git push origin main  # 자동 배포 트리거
   ```

3. **배포 확인**
   - GitHub Actions 탭에서 배포 진행 상황 확인
   - EC2에서 컨테이너 실행 상태 확인: `docker ps`

#### 2. 수동 배포

**전제 조건:**
- AWS EC2 인스턴스 (Ubuntu 22.04 LTS)
- Docker 설치
- Git 설치
- 포트 8000 열림

**배포 단계:**

```bash
# 1. EC2 인스턴스 접속
ssh -i your-key.pem ubuntu@your-ec2-ip

# 2. 저장소 클론
git clone https://github.com/your-username/perfume-recommender-api.git
cd perfume-recommender-api

# 3. 환경변수 설정
cp .env.example .env
nano .env  # OPENAI_API_KEY 설정

# 4. 배포 스크립트 실행
chmod +x deploy.sh
./deploy.sh
```

### Spring Boot와 연동 배포

FastAPI를 Spring Boot와 함께 동일한 EC2에서 실행하는 경우:

**아키텍처:**
```
EC2 Instance
├── perfumeonme (Spring Boot Container) :8080
├── perfume-recommender-container (FastAPI) :8000
└── Docker Network로 내부 통신
```

**Spring Boot 설정 확인:**
```yaml
# application-dev.yml
external:
  fastapi:
    recommend-url: http://perfume-recommender-container:8000
```

**컨테이너 간 통신 테스트:**
```bash
# Spring Boot 컨테이너에서 FastAPI 호출 테스트
docker exec perfumeonme curl http://perfume-recommender-container:8000/docs
```

### 배포 후 확인

```bash
# 전체 컨테이너 상태 확인
docker ps

# FastAPI 컨테이너 로그 확인
docker logs -f perfume-recommender-container

# Spring Boot 컨테이너 로그 확인
docker logs -f perfumeonme

# FastAPI API 직접 테스트 (EC2 내부)
curl http://localhost:8000/docs

# Spring Boot를 통한 연동 테스트
curl -X POST "http://localhost:8080/image-keyword/preview" \
  -H "Content-Type: application/json" \
  -d '{...키워드 데이터...}'
```

### 배포 플로우

#### FastAPI 독립 배포
```
코드 변경 → GitHub 푸시 → Actions 트리거 → 
Docker 빌드 → Docker Hub 푸시 → EC2 배포
```

#### Spring Boot와 연동
```
Spring Boot (:8080) ←→ FastAPI (:8000)
        ↓                    ↓
   클라이언트 요청        OpenAI API 호출
```

## 🧪 테스트

### 기본 테스트
```bash
# API 서버 상태 확인
curl http://localhost:8000/docs

# 추천 API 테스트
curl -X POST "http://localhost:8000/recommend/full" \
  -H "Content-Type: application/json" \
  -d '{
    "ambience": "세련된",
    "style": "유니크한",
    "gender": "여성스러운",
    "season": "겨울",
    "personality": "조용한"
  }'
```

### 성능 테스트
```bash
# 응답 시간 측정
time curl -X POST "http://localhost:8000/recommend/full" \
  -H "Content-Type: application/json" \
  -d '{"ambience":"세련된","style":"유니크한","gender":"여성스러운","season":"겨울","personality":"조용한"}'
```

## 📊 모니터링

### 로그 모니터링
```bash
# 실시간 로그 확인
docker logs -f perfume-recommender-container

# 최근 100줄 로그 확인
docker logs --tail 100 perfume-recommender-container
```

### 시스템 리소스 모니터링
```bash
# 컨테이너 리소스 사용량
docker stats perfume-recommender-container

# 시스템 전체 리소스
htop
```

## 🔧 개발 가이드

### 로컬 개발 환경 설정
```bash
# 의존성 설치 (개발용)
pip install -r requirements.txt

# 개발 서버 실행 (자동 재로드)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 코드 포맷팅
```bash
# Black 포맷터 (권장)
pip install black
black app/

# isort import 정렬
pip install isort
isort app/
```

### 새로운 추천 알고리즘 추가
1. `app/recommender_new.py` 파일 생성
2. 기존 패턴에 따라 클래스 구현
3. `app/recommender_hybrid.py`에서 새 알고리즘 통합
4. 테스트 및 성능 평가

## ⚠️ 주의사항

1. **OpenAI API 키 보안**: `.env` 파일을 Git에 커밋하지 마세요
2. **메모리 사용량**: ML 모델 로딩으로 인해 최소 4GB RAM 권장
3. **네트워크**: OpenAI API 호출을 위해 외부 네트워크 접근 필요
4. **데이터**: `data/perfume.xlsx` 파일은 저작권 보호 대상일 수 있습니다

## 🐛 문제 해결

### 일반적인 문제들

**Q: 컨테이너가 시작되지 않아요**
```bash
# 로그 확인
docker logs perfume-recommender-container

# 일반적인 원인: OpenAI API 키 미설정
```

**Q: 메모리 부족 오류가 발생해요**
```bash
# EC2 인스턴스 타입을 t3.medium 이상으로 변경
# 또는 swap 메모리 추가
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Q: 한국어 처리가 안 돼요**
```bash
# KoNLPy 관련 의존성 설치 확인
pip install konlpy
```

## 🤝 기여 방법

1. Fork 저장소
2. 기능 브랜치 생성 (`git checkout -b feature/new-feature`)
3. 변경사항 커밋 (`git commit -m 'Add new feature'`)
4. 브랜치에 Push (`git push origin feature/new-feature`)
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 👥 개발팀

- **PerfumeOnMe 백엔드팀**
- **문의**: 프로젝트 리포지토리 이슈

---

**마지막 업데이트**: 2025년 1월