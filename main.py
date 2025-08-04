import asyncio
import json
import re
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

app = FastAPI()

# 환경변수에서 OpenAI API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")

client = OpenAI(api_key=api_key)

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

# 사용자 응답 기반 키워드 결정 함수
def calculate_keywords_by_text(answers: list[str]) -> list[str]:
    q1, q2, q3, q4, q5, q6, q7, q8 = answers

    e_score = i_score = s_score = n_score = t_score = f_score = j_score = p_score = 0

    # Q1~Q2 → E/I
    if any(word in q1 for word in ["칫솔을", "바로"]):
        e_score += 1
    if any(word in q1 for word in ["수건으로", "닦고"]):
        i_score += 1

    if any(word in q2 for word in ["버튼", "분사해"]):
        e_score += 1
    if any(word in q2 for word in ["중간에", "내리는"]):
        i_score += 1

    # Q3~Q4 → S/N
    if any(word in q3 for word in ["알림처럼", "미리"]):
        s_score += 1
    if any(word in q3 for word in ["버스가", "가방에서 꺼내"]):
        n_score += 1

    if any(word in q4 for word in ["신호가 바뀌기 직전", "분사해"]):
        s_score += 1
    if any(word in q4 for word in ["기다리다", "향이 옅어지면"]):
        n_score += 1

    # Q5~Q6 → T/F
    if any(word in q5 for word in ["공간", "퍼뜨린다"]):
        t_score += 1
    if any(word in q5 for word in ["기분", "헹굴 때마다"]):
        f_score += 1

    if any(word in q6 for word in ["목줄", "가볍게"]):
        t_score += 1
    if any(word in q6 for word in ["손목에", "레이어링한다"]):
        f_score += 1

    # Q7~Q8 → J/P
    if any(word in q7 for word in ["리모컨을", "채널을 돌리며"]):
        j_score += 1
    if any(word in q7 for word in ["광고가", "확실히"]):
        p_score += 1

    if any(word in q8 for word in ["이불 위에서", "잔향을"]):
        j_score += 1
    if any(word in q8 for word in ["중앙에서", "톡톡"]):
        p_score += 1

    # 키워드 결정
    keyword1 = (
        "긍정적 임팩트를 가진 당신" if e_score > i_score else
        "은은한 집중형인 당신" if e_score < i_score else
        "외향과 내향의 균형을 지닌 당신"
    )
    keyword2 = (
        "촉각에 민감한 당신" if s_score > n_score else
        "직관으로 이끄는 당신" if s_score < n_score else
        "감각과 직관을 오가는 당신"
    )
    keyword3 = (
        "세부까지 놓치지 않는 당신" if t_score > f_score else
        "감성을 우선하는 당신" if t_score < f_score else
        "사고와 감정을 조화시키는 당신"
    )
    keyword4 = (
        "미리 움직이는 당신" if j_score > p_score else
        "순간을 즐기는 당신" if j_score < p_score else
        "계획과 즉흥이 공존하는 당신"
    )

    return [keyword1, keyword2, keyword3, keyword4]


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

def get_perfume_recommendations(request: PbtiRequest):
    mbti = determine_mbti_type(request)
    user_sentence = build_user_description(mbti)
    user_vector = model.encode(user_sentence)

    perfume_df["유사도"] = perfume_df["임베딩벡터"].apply(
        lambda vec: cosine_similarity([user_vector], [vec])[0][0]
    )
    top_matches = perfume_df.sort_values(by="유사도", ascending=False).head(3)

    return [
        {
            "name": row["향수이름"],
            "brand": row["브랜드"],
            "description": row["한줄소개"].replace('\n', ' '),
            "perfumeImageUrl": row["향수 이미지"]
        }
        for _, row in top_matches.iterrows()
    ]


# ----------------------
# GPT 병렬 호출용 프롬프트 함수들
# ----------------------

def prompt_recommendation(request: PbtiRequest) -> str:
    answers = [
        request.qOne, request.qTwo, request.qThree, request.qFour,
        request.qFive, request.qSix, request.qSeven, request.qEight
    ]
    keywords = calculate_keywords_by_text(answers)
    
    return f"""
아래는 사용자가 향수 성향 테스트에 응답한 결과입니다.:
Q1: {request.qOne}
Q2: {request.qTwo}
Q3: {request.qThree}
Q4: {request.qFour}
Q5: {request.qFive}
Q6: {request.qSix}
Q7: {request.qSeven}
Q8: {request.qEight}

이 사용자는 다음과 같은 향수 성향 키워드를 가지고 있습니다:
keyword1: {keywords[0]}
keyword2: {keywords[1]}
keyword3: {keywords[2]}
keyword4: {keywords[3]}

각 질문에 대한 사용자의 답변과 키워드를 분석하여 'recommendation' 필드만 JSON으로 출력하세요. JSON 데이터만 정확하게 출력해주세요.
추가로 "recommendation"은 '당신은' 으로 시작되도록 하세요.
예시:
{{"recommendation": "당신은 ...입니다."}}
"""

def prompt_keywords(request: PbtiRequest) -> str:
    answers = [
        request.qOne, request.qTwo, request.qThree, request.qFour,
        request.qFive, request.qSix, request.qSeven, request.qEight
    ]
    keywords = calculate_keywords_by_text(answers)
    
    return f"""
아래는 사용자가 향수 성향 테스트에 응답한 결과입니다.:
Q1: {request.qOne}
Q2: {request.qTwo}
Q3: {request.qThree}
Q4: {request.qFour}
Q5: {request.qFive}
Q6: {request.qSix}
Q7: {request.qSeven}
Q8: {request.qEight}

이 사용자는 다음과 같은 향수 성향 키워드를 가지고 있습니다:
keyword1: {keywords[0]}
keyword2: {keywords[1]}
keyword3: {keywords[2]}
keyword4: {keywords[3]}

이 키워드 각각에 대해 향수 성향 기반 설명(keywordDescription)을 작성하세요. 각 키워드는 다음 JSON 형식의 "keywords" 필드에 배열로 포함되어야 합니다.
또한 각 keywords 배열의 "keyword"는 위의 향수 성향 키워드에서 그대로 사용하세요. GPT가 임의로 바꾸지 마세요.

아래 JSON 예시처럼, "keywords" 배열에 4개 항목을 포함하세요.  JSON 데이터만 정확하게 출력해주세요.
예시:
{{
  "keywords": [
    {{"keyword": "...", "keywordDescription": "..."}},
    {{"keyword": "...", "keywordDescription": "..."}},
    {{"keyword": "...", "keywordDescription": "..."}},
    {{"keyword": "...", "keywordDescription": "..."}}
  ]
}}
"""

def prompt_perfume_style(request: PbtiRequest) -> str:
    answers = [
        request.qOne, request.qTwo, request.qThree, request.qFour,
        request.qFive, request.qSix, request.qSeven, request.qEight
    ]
    keywords = calculate_keywords_by_text(answers)
    
    return f"""
아래는 사용자가 향수 성향 테스트에 응답한 결과입니다.:
Q1: {request.qOne}
Q2: {request.qTwo}
Q3: {request.qThree}
Q4: {request.qFour}
Q5: {request.qFive}
Q6: {request.qSix}
Q7: {request.qSeven}
Q8: {request.qEight}

이 사용자는 다음과 같은 향수 성향 키워드를 가지고 있습니다:
keyword1: {keywords[0]}
keyword2: {keywords[1]}
keyword3: {keywords[2]}
keyword4: {keywords[3]}

각 질문에 대한 사용자의 답변과 키워드를 분석하여 'perfumeStyle' 객체만 JSON으로 출력하세요. JSON 데이터만 정확하게 출력해주세요.
추가로 "perfumeStyle" 내 "notes" 배열에는 5개 항목을 포함하세요.
그리고 category의 예시는 '시트러스', '우디' 등의 실제 존재하는 노트를 예시로 들어주고 한국어로 출력하세요.
예시:
{{
  "perfumeStyle": {{
    "description": "...",
    "notes": [
      {{"category": "...", "categoryDescription": "..."}},
      {{"category": "...", "categoryDescription": "..."}},
      {{"category": "...", "categoryDescription": "..."}},
      {{"category": "...", "categoryDescription": "..."}},
      {{"category": "...", "categoryDescription": "..."}}
    ]
  }}
}}
"""

def prompt_scent_point(request: PbtiRequest) -> str:
    answers = [
        request.qOne, request.qTwo, request.qThree, request.qFour,
        request.qFive, request.qSix, request.qSeven, request.qEight
    ]
    keywords = calculate_keywords_by_text(answers)
    
    return f"""
아래는 사용자가 향수 성향 테스트에 응답한 결과입니다.:
Q1: {request.qOne}
Q2: {request.qTwo}
Q3: {request.qThree}
Q4: {request.qFour}
Q5: {request.qFive}
Q6: {request.qSix}
Q7: {request.qSeven}
Q8: {request.qEight}

이 사용자는 다음과 같은 향수 성향 키워드를 가지고 있습니다:
keyword1: {keywords[0]}
keyword2: {keywords[1]}
keyword3: {keywords[2]}
keyword4: {keywords[3]}

각 질문에 대한 사용자의 답변과 키워드를 분석하여 'scentPoint' 배열만 JSON으로 출력하세요. JSON 데이터만 정확하게 출력해주세요.
그리고 category의 예시는 '시트러스', '우디' 등의 실제 존재하는 노트를 예시로 들어주고 한국어로 출력하세요.
"scentPoint" 배열 내의 "point"는 사용자와 잘 어울릴 수록 숫자를 크게 부여하고 숫자가 큰 순서대로 출력해주세요. "scentPoint" 배열에는 5개 항목을 포함하세요.
예시:
{{
  "scentPoint": [
    {{"category": "...", "point": 5}},
    {{"category": "...", "point": 4}},
    {{"category": "...", "point": 3}},
    {{"category": "...", "point": 2}},
    {{"category": "...", "point": 1}}
  ]
}}
"""

def prompt_summary(request: PbtiRequest) -> str:
    answers = [
        request.qOne, request.qTwo, request.qThree, request.qFour,
        request.qFive, request.qSix, request.qSeven, request.qEight
    ]
    keywords = calculate_keywords_by_text(answers)
    
    return f"""
아래는 사용자가 향수 성향 테스트에 응답한 결과입니다.:
Q1: {request.qOne}
Q2: {request.qTwo}
Q3: {request.qThree}
Q4: {request.qFour}
Q5: {request.qFive}
Q6: {request.qSix}
Q7: {request.qSeven}
Q8: {request.qEight}

이 사용자는 다음과 같은 향수 성향 키워드를 가지고 있습니다:
keyword1: {keywords[0]}
keyword2: {keywords[1]}
keyword3: {keywords[2]}
keyword4: {keywords[3]}

각 질문에 대한 사용자의 답변과 키워드를 분석하여 'summary' 필드만 JSON 문자열로 출력하세요. JSON 데이터만 정확하게 출력해주세요.
추가로 "summary"는 사용자의 성격이 반영되는 단어가 들어가도록 간단하게 요약해주세요. ex) “사람들과의 에너지 흐름을 잘 이끌어내는 ~한 사람”
예시:
{{"summary": "..." }}
"""

def extract_json(text: str) -> str:
    # ```json ... ``` 형태 제거
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    # 코드블록 없으면 원본 반환
    return text

async def call_gpt_async(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "정확한 JSON만 출력하는 향수 분석가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    text = response.choices[0].message.content.strip()

    json_str = extract_json(text)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}
# ----------------------
# 병렬 GPT 호출 및 통합 응답 API
# ----------------------

@app.post("/pbti/full-result")
async def full_result(request: PbtiRequest):
    prompts = [
        prompt_recommendation(request),
        prompt_keywords(request),
        prompt_perfume_style(request),
        prompt_scent_point(request),
        prompt_summary(request),
    ]

    gpt_outputs = await asyncio.gather(*[call_gpt_async(p) for p in prompts])

    result = {}
    for out in gpt_outputs:
        if isinstance(out, dict):
            result.update(out)

    # 향수 추천 결과 포함
    result["perfumeRecommend"] = get_perfume_recommendations(request)
    return result