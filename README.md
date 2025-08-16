<div align="center">

# 🌸 퍼퓨온미 (Perfume On Me)

**향수를 더 쉽고 즐겁게, 경험하다**  
향기와 경험을 담아내는 새로운 방식의 **향수 추천·경험 플랫폼**

<img src="images/cover.png" alt="퍼퓨온미 커버 이미지" width="600" />

</div>

---

## 📌 프로젝트 소개

퍼퓨온미는 사용자가 자신에게 어울리는 향수를 쉽고 재미있게 찾을 수 있도록 돕는 향수 추천·경험 플랫폼입니다.
GPT 기반 분석, 키워드 검색, 설문 등 다양한 방법을 통해 사용자의 취향을 파악하고,
성격·기분·스타일에 맞춘 개인 맞춤형 향수 추천을 제공합니다.
이를 통해 단순한 제품 구매를 넘어, 향수를 통해 추억과 감정을 담아내는 새로운 경험을 제안합니다.

---

## 🌱 프로젝트 배경

수천 가지 향수가 존재하지만, 대부분의 사람들은 어떤 향이 자신에게 어울릴지 몰라 선택에 어려움을 겪습니다.
또한 향에 대한 취향은 언어로 설명하기 어려워 기존의 검색·추천 방식에는 한계가 있습니다.
퍼퓨온미는 이러한 문제를 해결하고자, 다양한 접근 방식과 개인화 추천을 결합한 플랫폼을 만들었습니다.
향수를 비싸고 어려운 액세서리가 아닌, 누구나 즐길 수 있는 일상의 취미로 바꾸는 것이 우리의 목표입니다.

---

## 🔗 배포 주소

> [🌐 퍼퓨온미 바로가기](https://perfumeonme.vercel.app)

---

## ✨ 주요 기능

- 💡 **취향 맞춤 추천** : 취향 기반 개인 맞춤 향수 추천
- 📚 **향수 아카이브** :  성별, 상황, 계절, 가격, 노트별 등 검색 및 필터
- 🧾 **시향 기록** : 향에 대한 개인 다이어리 기록
- 📱 **추천 컨텐츠** : 이미지 기반, 온라인 공방, PBTI 등 다양한 경로의 추천

---

## 🎥 데모 & 미리보기

| 메인 화면                                                                                                                               | 향수 상세                                                                                                                               | 추천 화면                                                                                                                              |
|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| <img width="393" height="1254" alt="Image" src="https://github.com/user-attachments/assets/6f1d611f-8237-47f0-b0b4-8b8a7d1f961c" /> | <img width="393" height="1729" alt="Image" src="https://github.com/user-attachments/assets/df9b5539-e860-46c2-9a54-f7ad1633e48b" /> | <img width="393" height="824" alt="Image" src="https://github.com/user-attachments/assets/8dae23fe-a7df-4326-be2a-c179803ca08d" /> |

![기능 시연](images/demo.gif)

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
                               ML 모델 
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


## 📅 Roadmap

- [ ] 향수 추천 모델 고도화
- [ ] 응답속도 개선

--- 

## 👥 팀원 정보

| 이름  | 역할      | GitHub                                               |
|-----|---------|------------------------------------------------------|
| 김찬우 | AI | [@chanudevelop](https://github.com/chanudevelop)     |
| 이병웅 | AI | [@bulee5328](https://github.com/bulee5328)           |

--- 

## 📬 연락처

인스타그램: perfu_on_me

--- 










**마지막 업데이트**: 2025년 1월
