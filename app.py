import streamlit as st
import pandas as pd
import io
import time
import requests
import json
import re

# ==========================================
# 🔑 API 키 로드 (Streamlit Secrets)
# ==========================================
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")
# ------------------------------------------
# ⚠️ 중요: SDK 교체
# 기존 `google.generativeai` (구 SDK, deprecated)를
# 신규 통합 SDK `google-genai` 로 교체합니다.
#   pip uninstall google-generativeai
#   pip install google-genai
# requirements.txt 에도 google-generativeai 를 지우고
# google-genai 를 추가해야 합니다.
# ------------------------------------------
from google import genai as google_genai
from google.genai import types as google_types
from openai import OpenAI

def check_password():
    def password_entered():
        if st.session_state["password"] == APP_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔐 プログラムを使用するためのパスワードを入力してください", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔐 プログラムを使用するためのパスワードを入力してください", type="password", on_change=password_entered, key="password")
        st.error("⚠️ パスワードが間違っています。")
        return False

    return True

if not check_password():
    st.stop()

# ⚠️ 검색 결과 매칭(심사)은 항상 이 모델로 고정 실행합니다.
#    (검색/추천 생성 모델과는 별개 — 매칭 품질을 일정하게 유지하기 위함)
JUDGE_MODEL = "gemini-3.6-flash"


# 1. Google Places API에서 데이터 가져오기 (기존과 동일)
def get_google_places_data(search_query):
    url = 'https://places.googleapis.com/v1/places:searchText'
    places_list = []
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_API_KEY,
        'X-Goog-FieldMask': 'places.displayName,places.nationalPhoneNumber,nextPageToken'
    }
    data = {'textQuery': search_query, 'languageCode': 'ja'}
    status_text = st.empty()
    page_count = 1

    while True:
        status_text.text(f"Googleサーバーからデータを取得しています... (現在 {page_count} ページ)")
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result_data = response.json()
            places = result_data.get('places', [])
            for place in places:
                places_list.append({
                    '店舗名': place.get('displayName', {}).get('text', '名前なし'),
                    '電話番号': place.get('nationalPhoneNumber', '**なし**'),
                })
            next_token = result_data.get('nextPageToken')
            if next_token:
                data['pageToken'] = next_token
                time.sleep(2)
                page_count += 1
            else:
                break
        else:
            st.error(f"Google API エラー発生: {response.status_code}")
            break
    status_text.empty()
    return pd.DataFrame(places_list)


# 🌟 2. AI 순수 추천 (사용자 검색어 외 프롬프트 추가 없음 원칙 유지)
#    -> 두 모델 모두 "실시간 웹 검색 그라운딩"이 실제로 켜지도록 수정
def get_ai_pure_recommendation(search_query, selected_model):
    prompt = search_query  # 100% 순수 사용자 검색어만 사용

    if "gpt" in selected_model:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # ⚠️ 기존 chat.completions.create() 는 실시간 웹 검색을 전혀 하지 않습니다.
        #    (학습 데이터로만 답변 -> Google Places 결과와 괴리 발생의 주된 원인)
        # -> Responses API + web_search 툴로 교체해 실제 그라운딩을 켭니다.
        response = client.responses.create(
            model=selected_model,
            tools=[{"type": "web_search"}],
            input=prompt,
        )
        return response.output_text
    else:
        client = google_genai.Client(api_key=GEMINI_API_KEY)
        # ⚠️ 기존 코드는 "google_search_retrieval" (Gemini 2.0 미만 문법)을 사용해
        #    2.0 이상 모델에서 예외가 발생 -> except 에서 검색 없이 조용히 폴백되던 부분.
        # -> Gemini 2.0 이상 모델의 올바른 그라운딩 툴인 GoogleSearch로 교체.
        response = client.models.generate_content(
            model=selected_model,
            contents=prompt,
            config=google_types.GenerateContentConfig(
                temperature=0.7,
                tools=[google_types.Tool(google_search=google_types.GoogleSearch())],
            ),
        )
        return response.text


# 3. AI 심사원 (지점명 무시 + 안정적인 JSON 추출)
#    -> 검색 생성 모델(gpt든 gemini든)과 무관하게 매칭 심사는 항상 JUDGE_MODEL로 고정 실행
#    result_column: 같은 검색어를 여러 번 돌려서 비교할 때 회차별로 다른 컬럼에 저장하기 위한 파라미터
def match_lists_with_ai(df, ai_recommended_text, result_column='AI_推薦(🟢/❌)'):
    shop_names = df['店舗名'].tolist()
    shop_list_text = "\n".join([f"- {name}" for name in shop_names])

    prompt = f"""あなたはデータ照合の専門家です。

【基準テキスト（AIが最初に出力したテキスト）】
{ai_recommended_text}

【対象リスト（Googleマップの検索結果）】
{shop_list_text}

対象リストの各店舗について、基準テキスト内に記載されているいずれかの店舗と「実質的に同じお店（ブランド）」であるか判定してください。

※重要ルールの設定※
1. 「○○店」「本店」「○○支店」「〜館」などの支店名・修飾語は完全に無視してください。
2. コアとなる「メインの店舗名（ブランド名）」が一致していれば "🟢" を付与してください。
3. 基準テキスト内に存在しない全く別の店舗の場合は "❌" を付与してください。

必ず以下のJSON配列形式のみで出力してください。
[
    {{"name": "対象リストにある店舗名1", "result": "🟢"}},
    {{"name": "対象リストにある店舗名2", "result": "❌"}}
]
"""
    client = google_genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=google_types.GenerateContentConfig(temperature=0.0),
    )
    raw_text = response.text

    try:
        clean_text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'```\s*', '', clean_text)

        start_idx = clean_text.find('[')
        end_idx = clean_text.rfind(']') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("JSON配列が見つかりません。")

        json_str = clean_text[start_idx:end_idx]
        judgements = json.loads(json_str)
        judgement_dict = {item['name']: item['result'] for item in judgements}

        df[result_column] = df['店舗名'].apply(lambda x: judgement_dict.get(x, '❌ (判定漏れ)'))
        return df, raw_text

    except Exception as e:
        df[result_column] = '⚠️ 解析エラー'
        return df, raw_text


# N回分の判定を다수결(투표)로 통합
#    - 2択(🟢/❌)이므로 홀수 회차를 권장 (동점 방지)
#    - vote_count: 🟢로 판정된 횟수, total: 유효 판정 총 횟수(⚠️ 해석에러 제외)
def combine_rounds(row, round_columns):
    values = [str(row.get(col, '')) for col in round_columns]
    green_count = sum('🟢' in v for v in values)
    valid_count = sum('⚠️' not in v for v in values)  # 해석 에러는 투표에서 제외

    if valid_count == 0:
        return '⚠️', 0, 0

    # 과반수(다수결)면 최종 🟢
    if green_count > valid_count / 2:
        return '🟢', green_count, valid_count
    return '❌', green_count, valid_count


def highlight_matched_rows(row, target_column):
    val = str(row.get(target_column, ''))
    if '🟢' in val:
        return ['color: #008000; font-weight: bold;'] * len(row)
    elif '⚠️' in val:
        return ['color: #FFA500; font-weight: bold;'] * len(row)
    else:
        return [''] * len(row)


# 5. 全回答（1回〜N回分）をまとめて店舗名だけを整理する
#    -> 複数回検索した結果を必ずすべて反映（1回分だけを見ない）
#    -> 常にJUDGE_MODEL(gemini-3.6-flash)で整理
def summarize_shop_names_only(ai_responses):
    combined_text = "\n\n---\n\n".join(
        f"[{i + 1}回目の回答]\n{text}" for i, text in enumerate(ai_responses)
    )

    prompt = f"""以下は同じ検索キーワードに対してAIが複数回に分けて出力した推薦テキストです（全{len(ai_responses)}回分）。

{combined_text}

上記すべての回答に登場する店舗名を対象に、次のルールで整理してください。
1. 全ての回答を漏れなく確認し、言及されている店舗名を集めてください。
2. 同じ店舗（支店違い含む）が複数回・複数の回答にまたがって出てきても、重複させず1つにまとめてください。
3. 店舗名以外の説明文（営業時間やおすすめ理由など）は含めず、店舗名のみを出力してください。
4. 見やすい番号付きリスト形式で出力してください（例: 1. 店舗名）。
5. 前置きや後書きの文章は一切不要です。リストのみを出力してください。
"""
    client = google_genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=google_types.GenerateContentConfig(temperature=0.0),
    )
    return response.text


# 4. Web UI 構成
st.set_page_config(page_title="地域スポットAI検証システム", layout="wide")
st.title("📍 地域スポットAI検証システム")

col1, col2 = st.columns([3, 1])
with col1:
    search_query_input = st.text_input("検索キーワード", placeholder="例: 浅草 焼肉 店")
with col2:
    model_selection = st.selectbox(
        "推薦リスト作成モデルを選択（照合は常にgemini-3.6-flash固定）",
        options=[
            # GPT-5.6 系列（2026年7月リリース、7/30値下げ）
            "gpt-5.6-luna",   # 最安：$0.20/$1.20 per 1M tokens
            # Gemini 3 系列（2026年8月時点の最新Flashライン）
            "gemini-3.5-flash-lite",  # 最安・低遅延
            "gemini-3.6-flash",       # バランス型
        ]
    )

is_gpt_model = "gpt" in model_selection

# GPTはデフォルト1回（トークン節約）だがスライダーで複数回に増やせる。Geminiはデフォルト3回。
# key にmodel_selectionを含めることで、モデルを切り替えるたびにデフォルト値が正しく再適用される。
default_rounds = 1 if is_gpt_model else 3
num_rounds = st.slider(
    "🔁 AI再検索の回数（多数決で最終判定）",
    min_value=1, max_value=5, value=default_rounds, step=1,
    key=f"num_rounds_{model_selection}",
    help="同じ検索キーワードでAIに複数回問い合わせ、多数決で最終判定を出します。"
         "偶数だと同点が出る可能性があるため奇数が安定的です。"
         "回数を増やすほど精度は上がりますが、API呼び出し回数とコスト・待ち時間も比例して増えます。"
)
if is_gpt_model:
    st.caption("💡 ChatGPTモデルはデフォルト1回（トークン節約）。精度を上げたい場合は回数を増やせます。")

if st.button("🚀 検索および検証を実行", type="primary"):
    if not search_query_input or not search_query_input.strip():
        st.warning("⚠️ 検索キーワードを入力してください！ (검색어를 입력해 주세요!)")
    elif not GOOGLE_API_KEY or not GEMINI_API_KEY or (is_gpt_model and not OPENAI_API_KEY):
        st.error("⚠️ Streamlit Secrets に必要なAPIキーが設定されていません！"
                  "（照合は常にGeminiを使うためGEMINI_API_KEYは必須、GPTモデル選択時はOPENAI_API_KEYも必要です）")
    else:
        try:
            with st.spinner('1️⃣ Google Places APIから場所情報を収集しています...'):
                df_google = get_google_places_data(search_query_input)

            if df_google.empty:
                st.error("Googleの検索結果がありません。")
            else:
                # 同じプロンプトをN回に分けて実行し、全ての結果を取得・比較する
                ai_responses = []
                round_columns = []
                final_df = df_google.copy()

                for i in range(num_rounds):
                    round_no = i + 1
                    round_col = f'{round_no}回目_判定'
                    round_columns.append(round_col)

                    with st.spinner(f'2️⃣-{round_no} {model_selection} がAI回答を生成中... ({round_no}/{num_rounds}回目)'):
                        ai_text = get_ai_pure_recommendation(search_query_input, model_selection)
                        ai_responses.append(ai_text)

                    with st.spinner(f'3️⃣-{round_no} {JUDGE_MODEL} 審査員が照合中... ({round_no}/{num_rounds}回目)'):
                        final_df, _ = match_lists_with_ai(
                            final_df, ai_text, result_column=round_col
                        )

                col_final = f'最終判定(統合🟢/❌・{num_rounds}回中)'

                vote_results = final_df.apply(lambda row: combine_rounds(row, round_columns), axis=1)
                final_df[col_final] = vote_results.apply(lambda x: x[0])
                final_df['一致率'] = vote_results.apply(lambda x: f"{x[1]}/{x[2]}" if x[2] > 0 else "N/A")

                st.success(f"✅ 計 {len(final_df)} 件の検証が完了しました！（{num_rounds}回分の結果を多数決で統合済み）")

                # 全回一致しなかった店舗（判定が割れた＝不確実な店舗）を確認できるようにする
                split_df = final_df[final_df[round_columns].nunique(axis=1) > 1]
                if not split_df.empty:
                    st.warning(f"⚠️ {num_rounds}回の判定が割れた（一致しなかった）店舗が {len(split_df)} 件あります。一致率が低い店舗は下の表で要確認です。")

                try:
                    styled_df = final_df.style.apply(
                        lambda row: highlight_matched_rows(row, col_final), axis=1
                    )
                    st.dataframe(styled_df, use_container_width=True)
                except Exception:
                    st.dataframe(final_df, use_container_width=True)

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='AI_Verification')

                st.download_button(
                    label="📥 Excelファイルをダウンロード (.xlsx)",
                    data=buffer.getvalue(),
                    file_name=f"{search_query_input.replace(' ', '_')}_AI検証結果.xlsx",
                    mime="application/vnd.ms-excel"
                )

                st.subheader("🤖 AIの実際の回答内容（回ごとの生テキスト）")
                cols = st.columns(min(num_rounds, 3))
                for i, ai_text in enumerate(ai_responses):
                    with cols[i % len(cols)]:
                        with st.expander(f"{i + 1}回目の回答"):
                            st.info(ai_text)

                # ⬇️ 가장 아랫부분: 모든 회차(1回〜num_rounds回)의 AI 응답을 gemini-3.6-flash가
                #    통합・중복제거해서 점포명만 정리한 리스트
                with st.spinner(f'4️⃣ {JUDGE_MODEL} が全{num_rounds}回分の回答から店舗名を整理中...'):
                    shop_name_summary = summarize_shop_names_only(ai_responses)

                st.divider()
                st.subheader(f"🏪 店舗名のみ整理リスト（{num_rounds}回分の回答を統合・重複排除／{JUDGE_MODEL}）")
                st.markdown(shop_name_summary)

        except Exception as e:
            # ⚠️ 원인 파악을 위해 실제 예외 내용을 보여줌 (기존엔 조용히 삼켜졌음)
            st.error(f"エラーが発生しました: {e}")
            st.exception(e)
