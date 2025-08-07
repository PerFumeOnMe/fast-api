# app/services/pbti/mbti_analyzer.py

from app.models.schemas import PbtiRequest
from typing import List

# MBTI 성향별 설명 딕셔너리
mbti_trait_descriptions = {
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

# 사용자 MBTI 판별 (키워드 기반) - pbti.py 77~121줄
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
def calculate_keywords_by_text(answers: List[str]) -> List[str]:
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