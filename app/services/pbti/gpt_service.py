# app/services/pbti/gpt_service.py

import json
import re
from openai import OpenAI
from app.models.schemas import PbtiRequest
from app.core.config import settings, PBTI_GPT_MODEL, PBTI_GPT_TEMPERATURE
from app.services.pbti.mbti_analyzer import calculate_keywords_by_text

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# GPT 병렬 호출용 프롬프트 함수들 (pbti.py 274~454줄)

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
추가로 "recommendation"은 '당신은' 으로 시작되도록 하세요. 예시 정도의 분량으로 출력하세요.
예시:
{{"recommendation": "당신은 어디서든 존재감을 뽐내는 리더 타입!
밝고 또렷한 향이 당신의 에너지를 더 빛나게 해줄 거예요."}}
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

아래 JSON 예시처럼, "keywords" 배열에 4개 항목을 포함하세요. JSON 데이터만 정확하게 출력해주세요.
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
"description"은 ~향기로 끝나도록 해주세요. 마지막으로 예시 정도의 분량으로 출력하세요.
예시:
{{
  "perfumeStyle": {{
    "description": "선명하고 또렷한 인상을 남기는 향기",
    "notes": [
      {{"category": "시트러스", "categoryDescription": "상쾌하고 활기찬 에너지"}},
      {{"category": "앰버", "categoryDescription": "깊이 있는 고급스러운 마무리"}},
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
추가로 "summary"는 사용자의 성격이 반영되는 단어가 들어가도록 간단하게 요약해주세요. 예시 정도의 분량으로 출력하세요.
예시:
{{"summary": "사람들과의 에너지 흐름을 잘 이끌어내는 계획형 외향인" }}
"""

# JSON 추출 함수 (pbti.py 456~463줄)
def extract_json(text: str) -> str:
    # ```json ... ``` 형태 제거
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    # 코드블록 없으면 원본 반환
    return text

# GPT 비동기 호출 함수 (pbti.py 465~480줄)
async def call_gpt_async(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=PBTI_GPT_MODEL,
        messages=[
            {"role": "system", "content": "정확한 JSON만 출력하는 향수 분석가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=PBTI_GPT_TEMPERATURE
    )
    text = response.choices[0].message.content.strip()

    json_str = extract_json(text)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}