import streamlit as st
import pandas as pd
import boto3
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# ========================================
# 1. 페이지 설정
# ========================================
st.set_page_config(
    page_title="아이템베이 데이터 전략 분석실",
    page_icon="🎮",
    layout="wide"
)

# ========================================
# 2. 데이터 로드
# ========================================
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

        # 완료일시 변환
        combined_df['완료일시'] = combined_df['완료일시'].str.replace('오전', 'AM').str.replace('오후', 'PM')
        combined_df['완료일시'] = pd.to_datetime(combined_df['완료일시'], format='%Y-%m-%d %p %I:%M:%S', errors='coerce')

        # 거래금액, 수수료 숫자 변환
        for col in ['거래금액', '수수료']:
            combined_df[col] = combined_df[col].astype(str).str.replace(r'[\\,\s]', '', regex=True)
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

        # 수량 변환
        combined_df['수량'] = pd.to_numeric(combined_df['수량'], errors='coerce').fillna(0).astype(int)

        # 날짜 컬럼 추가
        combined_df['날짜'] = combined_df['완료일시'].dt.date
        combined_df['연월'] = combined_df['완료일시'].dt.to_period('M')

        return combined_df

    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        return None


# ========================================
# 3. AI 응답 추출 함수
# ========================================
def extract_output(response):
    raw = ""
    if isinstance(response, dict):
        raw = response.get("output", "")
        if not str(raw).strip():
            steps = response.get("intermediate_steps", [])
            if steps:
                last = steps[-1]
                if isinstance(last, (list, tuple)) and len(last) > 1:
                    raw = last[1]
    else:
        raw = response

    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()

    return str(raw).strip()


# ========================================
# 4. 숫자 포맷 함수
# ========================================
def fmt(number):
    try:
        return f"{int(number):,}"
    except:
        return "0"


# ========================================
# 5. 메인 앱
# ========================================
df = load_combined_data_from_s3()

if df is not None:

    # 날짜 기준 설정
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    current_month = today.replace(day=1)

    # ✅ NaT 제거 및 날짜 타입 정리
    df_clean = df[df['날짜'].notna()].copy()
    df_clean = df_clean[df_clean['날짜'].apply(lambda x: isinstance(x, type(today)))]

    # 어제 / 이번달 데이터 필터
    df_yesterday = df_clean[df_clean['날짜'] == yesterday]
    df_this_month = df_clean[df_clean['날짜'] >= current_month]

    # 데이터 없으면 가장 최근 날짜로 대체
    if df_yesterday.empty:
        latest = df_clean['날짜'].max()
        if pd.notna(latest):
            yesterday = latest
            df_yesterday = df_clean[df_clean['날짜'] == yesterday]

    st.success(f"✅ 총 {fmt(len(df))}건 데이터 로드 완료 | 기준일: {yesterday}")

    # ========================================
    # 탭 구성
    # ========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 어제 현황",
        "🏆 랭킹",
        "🎮 게임별 현황",
        "📈 이번달 추이",
        "🤖 AI 분석"
    ])

    # ======================================================
    # TAB 1 : 어제 현황
    # ======================================================
    with tab1:
        st.header(f"📊 어제 현황 ({yesterday})")

        if df_yesterday.empty:
            st.warning("해당 날짜의 데이터가 없습니다.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 총 거래금액", f"{fmt(df_yesterday['거래금액'].sum())}원")
            with col2:
                st.metric("📦 총 거래건수", f"{fmt(len(df_yesterday))}건")
            with col3:
                st.metric("💸 총 수수료", f"{fmt(df_yesterday['수수료'].sum())}원")
            with col4:
                avg = df_yesterday['거래금액'].mean()
                st.metric("📊 평균 거래금액", f"{fmt(avg)}원")

            st.divider()

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🗂️ 물품종류별 거래건수")
                type_count = df_yesterday.groupby('물품종류').size().reset_index(name='거래건수')
                fig1 = px.pie(
                    type_count,
                    names='물품종류',
                    values='거래건수',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)

            with col_right:
                st.subheader("💰 물품종류별 거래금액")
                type_amount = df_yesterday.groupby('물품종류')['거래금액'].sum().reset_index()
                type_amount.columns = ['물품종류', '거래금액']
                type_amount = type_amount.sort_values('거래금액', ascending=False)
                fig2 = px.bar(
                    type_amount,
                    x='물품종류',
                    y='거래금액',
                    color='물품종류',
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    text_auto=True
                )
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            st.divider()

            st.subheader("📋 물품종류별 상세 요약")
            summary = df_yesterday.groupby('물품종류').agg(
                거래건수=('거래금액', 'count'),
                거래금액합계=('거래금액', 'sum'),
                수수료합계=('수수료', 'sum'),
                평균거래금액=('거래금액', 'mean')
            ).reset_index()
            summary['거래금액합계'] = summary['거래금액합계'].apply(lambda x: f"{int(x):,}원")
            summary['수수료합계'] = summary['수수료합계'].apply(lambda x: f"{int(x):,}원")
            summary['평균거래금액'] = summary['평균거래금액'].apply(lambda x: f"{int(x):,}원")
            st.dataframe(summary, use_container_width=True, hide_index=True)

    # ======================================================
    # TAB 2 : 랭킹
    # ======================================================
    with tab2:
        st.header(f"🏆 랭킹 ({yesterday})")

        period = st.radio("기간 선택", ["어제", "이번달"], horizontal=True)
        df_rank = df_yesterday if period == "어제" else df_this_month

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🥇 판매 TOP 10")
            seller_rank = df_rank.groupby('판매자ID').agg(
                거래건수=('거래금액', 'count'),
                총거래금액=('거래금액', 'sum')
            ).reset_index().sort_values('총거래금액', ascending=False).head(10)
            seller_rank['총거래금액'] = seller_rank['총거래금액'].apply(lambda x: f"{int(x):,}원")
            seller_rank.insert(0, '순위', range(1, len(seller_rank) + 1))
            st.dataframe(seller_rank, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("🥈 구매 TOP 10")
            buyer_rank = df_rank.groupby('구매자ID').agg(
                거래건수=('거래금액', 'count'),
                총거래금액=('거래금액', 'sum')
            ).reset_index().sort_values('총거래금액', ascending=False).head(10)
            buyer_rank['총거래금액'] = buyer_rank['총거래금액'].apply(lambda x: f"{int(x):,}원")
            buyer_rank.insert(0, '순위', range(1, len(buyer_rank) + 1))
            st.dataframe(buyer_rank, use_container_width=True, hide_index=True)

        with col3:
            st.subheader("🥉 수수료 TOP 10")
            fee_rank = df_rank.groupby('판매자ID').agg(
                거래건수=('수수료', 'count'),
                총수수료=('수수료', 'sum')
            ).reset_index().sort_values('총수수료', ascending=False).head(10)
            fee_rank['총수수료'] = fee_rank['총수수료'].apply(lambda x: f"{int(x):,}원")
            fee_rank.insert(0, '순위', range(1, len(fee_rank) + 1))
            st.dataframe(fee_rank, use_container_width=True, hide_index=True)

    # ======================================================
    # TAB 3 : 게임별 현황
    # ======================================================
    with tab3:
        st.header("🎮 게임별 현황")

        period2 = st.radio("기간 선택", ["어제", "이번달"], horizontal=True, key="period2")
        df_game = df_yesterday if period2 == "어제" else df_this_month

        game_summary = df_game.groupby('게임명').agg(
            거래건수=('거래금액', 'count'),
            총거래금액=('거래금액', 'sum'),
            총수수료=('수수료', 'sum')
        ).reset_index().sort_values('총거래금액', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 게임별 거래금액 순위")
            fig3 = px.bar(
                game_summary.head(15),
                x='총거래금액',
                y='게임명',
                orientation='h',
                color='총거래금액',
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig3.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.subheader("📦 게임별 거래건수 순위")
            fig4 = px.bar(
                game_summary.head(15),
                x='거래건수',
                y='게임명',
                orientation='h',
                color='거래건수',
                color_continuous_scale='Greens',
                text_auto=True
            )
            fig4.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        st.subheader("📋 게임별 상세 테이블")
        game_summary_display = game_summary.copy()
        game_summary_display.insert(0, '순위', range(1, len(game_summary_display) + 1))
        game_summary_display['총거래금액'] = game_summary_display['총거래금액'].apply(lambda x: f"{int(x):,}원")
        game_summary_display['총수수료'] = game_summary_display['총수수료'].apply(lambda x: f"{int(x):,}원")
        st.dataframe(game_summary_display, use_container_width=True, hide_index=True)

    # ======================================================
    # TAB 4 : 이번달 추이
    # ======================================================
    with tab4:
        st.header(f"📈 이번달 일별 추이 ({current_month.strftime('%Y년 %m월')})")

        if df_this_month.empty:
            st.warning("이번달 데이터가 없습니다.")
        else:
            daily = df_this_month.groupby('날짜').agg(
                거래금액=('거래금액', 'sum'),
                거래건수=('거래금액', 'count'),
                수수료=('수수료', 'sum')
            ).reset_index()

            avg_amount = daily['거래금액'].mean()
            avg_count = daily['거래건수'].mean()
            yesterday_row = daily[daily['날짜'] == yesterday]

            col1, col2 = st.columns(2)
            with col1:
                if not yesterday_row.empty:
                    delta = yesterday_row['거래금액'].values[0] - avg_amount
                    st.metric(
                        "어제 거래금액 vs 이번달 일평균",
                        f"{fmt(yesterday_row['거래금액'].values[0])}원",
                        delta=f"{fmt(delta)}원"
                    )
            with col2:
                if not yesterday_row.empty:
                    delta_c = yesterday_row['거래건수'].values[0] - avg_count
                    st.metric(
                        "어제 거래건수 vs 이번달 일평균",
                        f"{fmt(yesterday_row['거래건수'].values[0])}건",
                        delta=f"{fmt(delta_c)}건"
                    )

            st.divider()

            st.subheader("💰 일별 거래금액 추이")
            fig5 = px.line(
                daily,
                x='날짜',
                y='거래금액',
                markers=True,
                color_discrete_sequence=['#4C72B0']
            )
            fig5.add_hline(
                y=avg_amount,
                line_dash="dash",
                line_color="red",
                annotation_text=f"월평균: {fmt(avg_amount)}원"
            )
            fig5.update_layout(xaxis_title="날짜", yaxis_title="거래금액(원)")
            st.plotly_chart(fig5, use_container_width=True)

            st.subheader("📦 일별 거래건수 추이")
            fig6 = px.bar(
                daily,
                x='날짜',
                y='거래건수',
                color_discrete_sequence=['#55A868']
            )
            fig6.add_hline(
                y=avg_count,
                line_dash="dash",
                line_color="red",
                annotation_text=f"월평균: {fmt(avg_count)}건"
            )
            fig6.update_layout(xaxis_title="날짜", yaxis_title="거래건수")
            st.plotly_chart(fig6, use_container_width=True)

    # ======================================================
    # TAB 5 : AI 분석
    # ======================================================
    with tab5:
        st.header("🤖 AI 데이터 분석")
        st.info("데이터에 대해 자유롭게 질문하세요! (예: 이번달 게임별 거래금액 순위 알려줘)")

        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
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
                agent_type="tool-calling"
            )

            query = st.chat_input("궁금한 분석 내용을 입력하세요")

            if query:
                with st.chat_message("user"):
                    st.write(query)
                with st.chat_message("assistant"):
                    with st.spinner("분석 중입니다..."):
                        try:
                            response = agent.invoke({"input": query})
                            output = extract_output(response)
                            if output:
                                st.write(output)
                            else:
                                st.warning("응답을 받지 못했습니다. 다시 질문해 주세요.")
                        except Exception as e:
                            st.error(f"❌ 분석 중 오류: {e}")
                            st.exception(e)

        except Exception as e:
            st.error(f"🚨 AI 엔진 초기화 실패: {e}")

else:
    st.warning("transdb 폴더 내에 데이터 파일이 없거나 S3 설정을 확인해야 합니다.")
