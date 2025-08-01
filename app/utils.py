# app/utils.py

import pandas as pd

def safe_str(value) -> str:
    """
    추천 결과 필드 값이 문자열이 아닌 경우 (예: -1, NaN, None 등) → 빈 문자열로 안전 변환
    문자열인 경우에도 \n 줄바꿈 제거 및 strip 처리 포함
    """
    try:
        # 이미 문자열이면 바로 처리
        if isinstance(value, str):
            return value.replace("\n", " ").strip()
        
        # NaN 또는 -1 또는 None → 빈 문자열
        if pd.isna(value) or value == -1 or value is None:
            return ""

        # 그 외 값도 문자열로 변환 후 \n 제거 및 strip 처리
        return str(value).replace("\n", " ").strip()
    except Exception:
        return ""
