import streamlit as st
import pandas as pd
import boto3
from io import StringIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="아이템베이 전략 분석실", layout="wide")

# 1. S3에서 데이터 불러오기 함수
@st.cache_data
def load_data_from_s3():
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
        aws_secret_access_key=st.secrets["AWS_SECRET_KEY"]
    )
    obj = s3.get_object(Bucket=st.secrets["S3_BUCKET_NAME"], Key="cloud_upload_data.csv")
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

try:
    df = load_data_from_s3()
    st.success("✅ 클라우드 데이터 연결 성공!")

    # 2. AI 설정 (Gemini 무료 API 활용)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["GEMINI_API_KEY"])
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

    # 3. 채팅 인터페이스
    st.title("🤖 무엇이든 물어보세요 (김부장 전용)")
    query = st.chat_input("예: 3월 리니지2 이탈 예상 유저는 몇 명이야?")
    
    if query:
        with st.chat_message("user"): st.write(query)
        with st.chat_message("assistant"):
            response = agent.run(query)
            st.write(response)

except Exception as e:
    st.error(f"연결 중 오류 발생: {e}")