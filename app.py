import streamlit as st
import pandas as pd
import boto3
from io import StringIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. 페이지 설정
st.set_page_config(page_title="아이템베이 전략 분석실", layout="wide")

# 2. S3에서 데이터 불러오기 함수
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
        st.error(f"❌ S3 연결 실패: {e}")
        return None

df = load_data_from_s3()

if df is not None:
    st.success(f"✅ 클라우드 데이터 연결 성공! (총 {len(df):,} 건)")

    # 3. AI 설정 (404 에러를 피하기 위한 가장 표준적인 설정)
    try:
        # 모델명을 'models/gemini-1.5-flash'로 고정합니다. 
        # 1.5-flash-latest가 안 될 경우 가장 기본형인 이 이름이 최선입니다.
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash", 
            google_api_key=st.secrets["GEMINI_API_KEY"],
            convert_system_message_to_human=True,
            temperature=0
        )
        
        # 에이전트 생성
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        
        # 4. 채팅 인터페이스
        st.title("🤖 아이템베이 데이터 전략 어시스턴트")
        query = st.chat_input("데이터에 대해 궁금한 점을 입력하세요.")
        
        if query:
            with st.chat_message("user"): st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("데이터 분석 중..."):
                    response = agent.run(query)
                    st.write(response)
                    
    except Exception as e:
        st.error(f"🚨 AI 엔진 초기화 실패: {e}")
else:
    st.warning("데이터를 불러오지 못했습니다. S3 설정과 Secrets를 확인해주세요.")
