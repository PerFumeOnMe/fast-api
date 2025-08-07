# app/utils.py
import boto3
from io import BytesIO
import os
import pandas as pd

def load_excel_from_s3(bucket: str, key: str):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-northeast-1")
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_excel(BytesIO(obj['Body'].read()))

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
