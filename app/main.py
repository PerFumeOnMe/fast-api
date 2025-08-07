# app/main.py

from fastapi import FastAPI
from app.routers import recommendations

app = FastAPI(
    title="PerfumeOnMe FAST API",
    description="향수 추천 및 감성 시나리오 생성, PBTI API 통합",
    version="1.0.0"
)

# 라우터 등록
app.include_router(recommendations.router)

@app.get("/")
async def root():
    """
    API 상태 확인
    """
    return {"message": "PerfumeOnMe FastAPI가 정상적으로 실행 중입니다."}