import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# 향수 데이터 로딩
perfume_df = pd.read_excel("perfumeOnMe_data.xlsx")

# 임베딩 모델 로딩
model = SentenceTransformer("all-MiniLM-L6-v2")

# 사용자 요청 DTO
class PbtiRequest(BaseModel):
    qOne: str
    qTwo: str
    qThree: str
    qFour: str
    qFive: str
    qSix: str
    qSeven: str
    qEight: str

# 추천 결과 DTO
class PerfumeResponse(BaseModel):
    name: str
    brand: str
    description: str
    perfumeImageUrl: str

class PbtiRecommendResponse(BaseModel):
    perfumeRecommend: List[PerfumeResponse]

# MBTI 성향별 설명 딕셔너리
mbti_trait_descriptions  = {
    # E / I
    "E": "They are outgoing, energized by social interactions, and tend to enjoy dynamic environments.",
    "I": "They are introspective, enjoy solitude, and recharge through personal reflection.",
    "EI": "They show a blend of extroversion and introversion — able to engage socially but also value time alone.",

    # S / N
    "S": "They are grounded in reality, detail-focused, and prefer concrete facts and practical experiences.",
    "N": "They are imaginative, future-oriented, and enjoy exploring abstract ideas and possibilities.",
    "SN": "They balance practicality with imagination — relying on both details and big-picture thinking.",

    # T / F
    "T": "They value logic and objective reasoning, often making decisions based on facts.",
    "F": "They prioritize emotions and empathy, and often consider values when making choices.",
    "TF": "They consider both logic and emotion in decision-making, weighing reason and compassion together.",

    # J / P
    "J": "They prefer structure, planning, and clear expectations, feeling comfortable with schedules.",
    "P": "They enjoy flexibility, spontaneity, and are comfortable adapting to changing situations.",
    "JP": "They embody both structure and adaptability — capable of planning while staying flexible."
}

# 사용자 MBTI 판별 (키워드 기반)
def determine_mbti_type(data: PbtiRequest) -> str:
    eScore = iScore = sScore = nScore = tScore = fScore = jScore = pScore = 0

    if "칫솔을" in data.qOne or "바로" in data.qOne:
        eScore += 1
    if "수건으로" in data.qOne or "닦고" in data.qOne:
        iScore += 1
    if "버튼" in data.qTwo or "분사해" in data.qTwo:
        eScore += 1
    if "중간에" in data.qTwo or "내리는" in data.qTwo:
        iScore += 1

    if "알림처럼" in data.qThree or "미리" in data.qThree:
        sScore += 1
    if "버스가" in data.qThree or "가방에서 꺼내" in data.qThree:
        nScore += 1
    if "신호가 바뀌기 직전" in data.qFour or "분사해" in data.qFour:
        sScore += 1
    if "기다리다" in data.qFour or "향이 옅어지면" in data.qFour:
        nScore += 1

    if "공간" in data.qFive or "퍼뜨린다" in data.qFive:
        tScore += 1
    if "기분" in data.qFive or "헹굴 때마다" in data.qFive:
        fScore += 1
    if "목줄" in data.qSix or "가볍게" in data.qSix:
        tScore += 1
    if "손목에" in data.qSix or "레이어링한다" in data.qSix:
        fScore += 1

    if "리모컨을" in data.qSeven or "채널을 돌리며" in data.qSeven:
        jScore += 1
    if "광고가" in data.qSeven or "확실히" in data.qSeven:
        pScore += 1
    if "이불 위에서" in data.qEight or "잔향을" in data.qEight:
        jScore += 1
    if "중앙에서" in data.qEight or "톡톡" in data.qEight:
        pScore += 1

    mbti = ""
    mbti += "EI" if eScore == iScore else ("E" if eScore > iScore else "I")
    mbti += "SN" if sScore == nScore else ("S" if sScore > nScore else "N")
    mbti += "TF" if tScore == fScore else ("T" if tScore > fScore else "F")
    mbti += "JP" if jScore == pScore else ("J" if jScore > pScore else "P")
    return mbti

# 사용자 설명 문장 생성
def build_user_description(mbti: str) -> str:
    desc = []

    # E/I
    if "E" in mbti and "I" in mbti:
        desc.append(mbti_trait_descriptions["EI"])
    elif "E" in mbti:
        desc.append(mbti_trait_descriptions["E"])
    elif "I" in mbti:
        desc.append(mbti_trait_descriptions["I"])

    # S/N
    if "S" in mbti and "N" in mbti:
        desc.append(mbti_trait_descriptions["SN"])
    elif "S" in mbti:
        desc.append(mbti_trait_descriptions["S"])
    elif "N" in mbti:
        desc.append(mbti_trait_descriptions["N"])

    # T/F
    if "T" in mbti and "F" in mbti:
        desc.append(mbti_trait_descriptions["TF"])
    elif "T" in mbti:
        desc.append(mbti_trait_descriptions["T"])
    elif "F" in mbti:
        desc.append(mbti_trait_descriptions["F"])

    # J/P
    if "J" in mbti and "P" in mbti:
        desc.append(mbti_trait_descriptions["JP"])
    elif "J" in mbti:
        desc.append(mbti_trait_descriptions["J"])
    elif "P" in mbti:
        desc.append(mbti_trait_descriptions["P"])

    return "This person shows the following traits: " + " ".join(desc)

# 향수 임베딩 문장 생성 함수
def build_perfume_sentence(row):
    gender = row['성별']
    place = row['장소']
    season = row['계절']
    keywords = row['향수 키워드']
    return f"Suitable for {gender} at {place} in {season}, with keywords like {keywords}"


# 향수 데이터 전처리: 문장 생성 및 벡터 임베딩
perfume_df["임베딩문장"] = perfume_df.apply(build_perfume_sentence, axis=1)
perfume_df["임베딩벡터"] = perfume_df["임베딩문장"].apply(lambda x: model.encode(x))

# 추천 API
@app.post("/recommend/pbti", response_model=PbtiRecommendResponse)
def recommend(data: PbtiRequest):
    # 사용자 MBTI 계산
    mbti = determine_mbti_type(data)

    # 사용자 성격 문장 생성
    user_sentence = build_user_description(mbti)
    user_vector = model.encode(user_sentence)

    # 향수 벡터와 코사인 유사도 계산
    perfume_df["유사도"] = perfume_df["임베딩벡터"].apply(
        lambda vec: cosine_similarity([user_vector], [vec])[0][0]
    )

    # 상위 향수 3개 반환
    top_matches = perfume_df.sort_values(by="유사도", ascending=False).head(3)

    result = [
        {
            "name": row["향수이름"],
            "brand": row["브랜드"],
            "description": row["한줄소개"].replace('\n', ' '),
            "perfumeImageUrl": row["향수 이미지"]
        }
        for _, row in top_matches.iterrows()
    ]
    return PbtiRecommendResponse(perfumeRecommend=result)
