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


# 3. AI 심사원 (지점명 무시 + 안정적인 JSON 추출) - 이 부분은 검색이 필요 없으므로 그대로 둠
#    result_column: 같은 검색어를 여러 번 돌려서 비교할 때 회차별로 다른 컬럼에 저장하기 위한 파라미터
def match_lists_with_ai(df, ai_recommended_text, selected_model, result_column='AI_推薦(🟢/❌)'):
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
    if "gpt" in selected_model:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_text = response.choices[0].message.content
    else:
        client = google_genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=selected_model,
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


# 2回分の判定を통합: 어느 한 회차에서라도 🟢이면 최종 🟢 (합집합 기준 비교)
def combine_rounds(row, round_columns):
    values = [str(row.get(col, '')) for col in round_columns]
    if any('🟢' in v for v in values):
        return '🟢'
    if any('⚠️' in v for v in values):
        return '⚠️'
    return '❌'


def highlight_matched_rows(row, target_column):
    val = str(row.get(target_column, ''))
    if '🟢' in val:
        return ['color: #008000; font-weight: bold;'] * len(row)
    elif '⚠️' in val:
        return ['color: #FFA500; font-weight: bold;'] * len(row)
    else:
        return [''] * len(row)


# 4. Web UI 構成
st.set_page_config(page_title="地域スポットAI検証システム", layout="wide")
st.title("📍 地域スポットAI検証システム")

col1, col2 = st.columns([3, 1])
with col1:
    search_query_input = st.text_input("検索キーワード", placeholder="例: 浅草 焼肉 店")
with col2:
    model_selection = st.selectbox(
        "推薦リスト作成・審査モデルを選択",
        options=[
            "gpt-4o-mini",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash"
        ]
    )

if st.button("🚀 検索および検証を実行", type="primary"):
    if not search_query_input or not search_query_input.strip():
        st.warning("⚠️ 検索キーワードを入力してください！ (검색어를 입력해 주세요!)")
    elif not GOOGLE_API_KEY or (not GEMINI_API_KEY and not OPENAI_API_KEY):
        st.error("⚠️ Streamlit Secrets に APIキーが設定されていません！")
    else:
        try:
            with st.spinner('1️⃣ Google Places APIから場所情報を収集しています...'):
                df_google = get_google_places_data(search_query_input)

            if df_google.empty:
                st.error("Googleの検索結果がありません。")
            else:
                # 同じプロンプトを2回に分けて実行し、両方の結果を取得・比較する
                with st.spinner(f'2️⃣-1 {model_selection} がAI回答を生成中... (1回目)'):
                    ai_pure_list_1 = get_ai_pure_recommendation(search_query_input, model_selection)

                with st.spinner(f'2️⃣-2 {model_selection} がAI回答を生成中... (2回目)'):
                    ai_pure_list_2 = get_ai_pure_recommendation(search_query_input, model_selection)

                col_1st = '1回目_判定'
                col_2nd = '2回目_判定'
                col_final = '最終判定(統合🟢/❌)'

                with st.spinner(f'3️⃣-1 {model_selection} 審査員が照合中... (1回目結果を基準)'):
                    final_df, raw_ai_response_1 = match_lists_with_ai(
                        df_google, ai_pure_list_1, model_selection, result_column=col_1st
                    )

                with st.spinner(f'3️⃣-2 {model_selection} 審査員が照合中... (2回目結果を基準)'):
                    final_df, raw_ai_response_2 = match_lists_with_ai(
                        final_df, ai_pure_list_2, model_selection, result_column=col_2nd
                    )

                # 두 회차 중 어느 한쪽이라도 🟢이면 최종 🟢 (합집합) — 필요시 기준 조정 가능
                final_df[col_final] = final_df.apply(
                    lambda row: combine_rounds(row, [col_1st, col_2nd]), axis=1
                )

                st.success(f"✅ 計 {len(final_df)} 件の検証が完了しました！（2回分の結果を比較・統合済み）")

                # 1回目と2回目で判定が食い違った店舗（一貫性チェック）
                mismatch_df = final_df[final_df[col_1st] != final_df[col_2nd]]
                if not mismatch_df.empty:
                    st.warning(f"⚠️ 1回目と2回目で判定が異なる店舗が {len(mismatch_df)} 件あります（下の表で確認できます）。")

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

                col_a, col_b = st.columns(2)
                with col_a:
                    with st.expander("🤖 AIの実際の回答内容 - 1回目（生のテキスト）"):
                        st.info(ai_pure_list_1)
                with col_b:
                    with st.expander("🤖 AIの実際の回答内容 - 2回目（生のテキスト）"):
                        st.info(ai_pure_list_2)

        except Exception as e:
            # ⚠️ 원인 파악을 위해 실제 예외 내용을 보여줌 (기존엔 조용히 삼켜졌음)
            st.error(f"エラーが発生しました: {e}")
            st.exception(e)
