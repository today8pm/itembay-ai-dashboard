import streamlit as st
import pandas as pd
import boto3
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. 페이지 설정
st.set_page_config(page_title="아이템베이 데이터 전략 분석실", layout="wide")

# 2. S3 폴더 내 모든 Parquet 파일을 읽어서 합치는 함수
@st.cache_data
def load_combined_data_from_s3():
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
            aws_secret_access_key=st.secrets["AWS_SECRET_KEY"]
        )

        bucket_name = st.secrets["S3_BUCKET_NAME"]
        folder_prefix = "transdb/"

        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)

        all_dfs = []
        if 'Contents' in response:
            for obj in response['Contents']:
                file_key = obj['Key']
                if file_key.endswith('.parquet'):
                    file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
                    temp_df = pd.read_parquet(BytesIO(file_obj['Body'].read()))
                    all_dfs.append(temp_df)

        if not all_dfs:
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df

    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        return None


# 3. 데이터 로드 실행
df = load_combined_data_from_s3()

if df is not None:
    st.success(f"✅ 'transdb' 폴더 내 {len(df):,}건의 데이터를 성공적으로 통합했습니다!")

    # ✅ 진단 코드 (데이터 구조 확인용 - 문제 해결 후 삭제 가능)
    with st.expander("🔍 데이터 진단 정보 (확인 후 삭제 가능)"):
        st.write("**📋 컬럼 목록 및 타입:**")
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "컬럼명", 0: "타입"}))

        st.write("**👀 상위 5개 행:**")
        st.dataframe(df.head(5))

        st.write("**📅 날짜/텍스트 컬럼 샘플값:**")
        date_cols = df.select_dtypes(include=['object', 'datetime64[ns]', 'datetime64[us]']).columns.tolist()
        if date_cols:
            for col in date_cols:
                st.write(f"- `{col}` : `{df[col].iloc[0]}`  (타입: `{df[col].dtype}`)")
        else:
            st.write("날짜/텍스트 컬럼 없음")

        st.write("**🔢 전체 데이터 shape:**")
        st.write(f"행: {df.shape[0]:,}개 / 열: {df.shape[1]}개")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=st.secrets["GEMINI_API_KEY"],
            convert_system_message_to_human=True,
            temperature=0,
            streaming=False
        )

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )

        st.title("🤖 아이템베이 실시간 데이터 전략 어시스턴트")
        st.info("transdb 폴더의 모든 데이터를 실시간으로 합산하여 분석합니다.")

        query = st.chat_input("궁금한 분석 내용을 입력하세요 (예: 25년 전체 거래액 알려줘)")

        if query:
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("통합 데이터를 분석 중입니다..."):
                    response = agent.run(query)
                    st.write(response)

    except Exception as e:
        st.error(f"🚨 AI 엔진 초기화 실패: {e}")

else:
    st.warning("transdb 폴더 내에 데이터 파일이 없거나 S3 설정을 확인해야 합니다.")
