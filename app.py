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
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
            aws_secret_access_key=st.secrets["AWS_SECRET_KEY"]
        )
        obj = s3.get_object(Bucket=st.secrets["S3_BUCKET_NAME"], Key="cloud_upload_data.csv")
        return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    except Exception as e:
        st.error(f"S3 연결 실패: {e}")
        return None

df = load_data_from_s3()

if df is not None:
    st.success("✅ 클라우드 데이터 연결 성공!")

    # 2. AI 설정 (최신 모델명 반영)
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GEMINI_API_KEY"])
        # 에이전트 생성
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        
        # 3. 채팅 인터페이스
        st.title("🤖 무엇이든 물어보세요 (김부장 전용)")
        query = st.chat_input("예: 우리 데이터의 전체 거래 건수와 총 금액은?")
        
        if query:
            with st.chat_message("user"): st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("데이터 분석 중..."):
                    response = agent.run(query)
                    st.write(response)
    except Exception as e:
        st.error(f"AI 엔진 초기화 실패: {e}")
else:
    st.warning("데이터를 불러오지 못했습니다. S3 설정과 Secrets를 확인해주세요.")
