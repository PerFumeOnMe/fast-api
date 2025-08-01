# app/generator.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def _create_prompt(keywords: list[str]) -> str:
    """프롬프트 생성 함수 (중복 제거)"""
    return (
        f"당신은 향수를 뿌린 사용자의 분위기와 감정을 감성적으로 묘사하는 시나리오 작가입니다.\n"
        f"다음 키워드들을 반영하여, 향수를 뿌린 '당신'의 감정과 모습을 중심으로 감성적인 시나리오 한 문단(3~4줄)을 작성해주세요.\n"
        f"성별을 나타내는 표현(그녀, 그 등)은 사용하지 말고, 반드시 '당신'을 주어로 표현해주세요.\n"
        f"표현은 시각적, 후각적, 감정적인 분위기를 담아주세요.\n\n"
        f"키워드: {', '.join(keywords)}\n\n"
        f"예시:\n"
        f"창밖엔 눈이 내리고, 따뜻한 머플러에 얼굴을 묻은 채 조용한 거리를 걷고 있어요. "
        f"바스락거리는 낙엽, 코끝에 닿는 은은한 바닐라와 우디 향, 그 속에서 세련된 당신의 분위기가 더욱 빛나요. "
        f"사람들 속에 섞여 있지만, 뚜렷한 개성과 고요한 존재감이 느껴져요."
    )

def generate_scenario_sync(keywords: list[str]) -> str:
    """동기 버전 (ThreadPoolExecutor에서 사용)"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = _create_prompt(keywords)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 감성적인 향기 시나리오를 쓰는 향수 작가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.85,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"[OpenAI Error] 시나리오 생성 중 문제 발생: {e}")
        return "감성 시나리오 생성에 실패했어요. 다시 시도해주세요."
