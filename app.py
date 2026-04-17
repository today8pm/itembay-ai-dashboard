import streamlit as st
import pandas as pd
import boto3
from io import StringIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 페이지 설정: 부장님이 선호하시는 와이드 레이아웃
st.set_page_config(page_title="아이템베이 전략 분석실", layout="wide")

# 1. S3에서 데이터 불러오기 함수
@st.cache_data
def load_data_from_s3():
    try:
        # Streamlit Secrets에 등록된 키를 사용하여 S3 연결
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
            aws_secret_access_key=st.secrets["AWS_SECRET_KEY"]
        )
        # 버킷명과 파일명 확인
        obj = s3.get_object(Bucket=st.secrets["S3_BUCKET_NAME"], Key="cloud_upload_data.csv")
        # 데이터 로드 (UTF-8 인코딩)
        return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    except Exception as e:
        st.error(f"❌ S3 연결 실패: {e}")
        return None

# 데이터 로드 실행
df = load_data_from_s3()

if df is not None:
    st.success(f"✅ 클라우드 데이터 연결 성공! (총 {len(df):,} 건)")

    # 2. AI 설정 (부장님 화면에서 확인된 Gemini 3 모델 반영)
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash",  # 최신 모델명으로 교체
            google_api_key=st.secrets["GEMINI_API_KEY"],
            convert_system_message_to_human=True,
            temperature=0  # 분석의 정확도를 위해 0으로 설정
        )
        
        # 데이터 분석용 에이전트 생성
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True,
            handle_parsing_errors=True  # 404 외의 파싱 에러 방지용
        )
        
        # 3. 채팅 인터페이스 (부장님 전용 UI)
        st.title("🤖 아이템베이 데이터 전략 어시스턴트")
        st.info("데이터 기반의 의사결정을 돕는 김부장 전용 분석 엔진입니다.")
        
        query = st.chat_input("예: 2026년 3월 거래액 합계는 얼마야?")
        
        if query:
            with st.chat_message("user"): 
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("Gemini 3 엔진이 데이터를 분석 중입니다..."):
                    # 에이전트 실행 및 답변 출력
                    response = agent.run(query)
                    st.write(response)
                    
    except Exception as e:
        st.error(f"🚨 AI 엔진 초기화 실패: {e}")
        st.info("💡 Tip: API 키가 유효한지 또는 모델명(gemini-3-flash)이 정확한지 확인해 주세요.")

else:
    st.warning("데이터를 불러오지 못했습니다. AWS S3 권한 설정과 Streamlit Secrets를 다시 확인해 주세요.")
