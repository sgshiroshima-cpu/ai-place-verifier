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

# ⚠️ 検索結果マッチング(審査)は常にこのモデルで固定実行します。
#    (検索/推薦生成モデルとは別 — マッチング品質を一定に保つため)
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
        # ✅ 修正②: OpenAI API(web_search)とChatGPTアプリの回答が食い違う問題への対策。
        #    完全に同一の出力を保証する公式な方法は存在しませんが、以下の設定で
        #    ChatGPTアプリの検索条件（実際の位置情報・検索範囲の広さ）に近づけます。
        #    - user_location: アプリは実際のユーザー位置を検索に反映するため、
        #      API側にも検索対象地域（例: 東京）を明示的に渡す。
        #      ※ 検索キーワード自体に地名が含まれる場合(例: 新宿)でも、
        #        API はデフォルトで位置情報を持たないため周辺情報の解釈が変わることがある。
        #    - search_context_size="high": 検索結果をより多く参照させ、
        #      アプリの検索結果に近い網羅性を狙う。
        #    - instructions: ユーザープロンプト（100%検索語のみ）は変更せず、
        #      別チャネルのシステム指示として「実在する店舗名のみ・架空店舗禁止」等を追加。
        response = client.responses.create(
            model=selected_model,
            instructions=(
                "あなたはユーザーの検索意図に基づき、実在する店舗名を正確に案内するアシスタントです。"
                "必ずweb検索ツールで最新情報を確認したうえで回答してください。"
                "実在が確認できない店舗名は出力しないでください。"
            ),
            tools=[{
                "type": "web_search",
                "search_context_size": "high",
                "user_location": {
                    "type": "approximate",
                    "country": "JP",
                    "city": "Tokyo",
                    "region": "Tokyo",
                    "timezone": "Asia/Tokyo",
                },
            }],
            input=prompt,
        )
        return response.output_text
    else:
        client = google_genai.Client(api_key=GEMINI_API_KEY)
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


# N回分の判定を統合する
#    ✅ 修正①: 従来は「過半数(green_count > valid_count/2)」で最終🟢を判定していたため、
#       N回中1回だけ🟢が出たケースは❌に丸められてしまっていた。
#       -> 仕様変更: 有効判定のうち1回でも🟢があれば最終的に🟢とする（OR方式）。
#       ※ これにより「精度(誤検出を抑える)」より「再現率(見逃しを減らす)」を優先する設計になります。
#         もし逆に「1回でも❌なら最終❌」にしたい場合は below の条件を反転してください。
def combine_rounds(row, round_columns):
    values = [str(row.get(col, '')) for col in round_columns]
    green_count = sum('🟢' in v for v in values)
    valid_count = sum('⚠️' not in v for v in values)  # 解析エラーは投票から除外

    if valid_count == 0:
        return '⚠️', 0, 0

    # ✅ 1回でも🟢があれば最終🟢（OR方式）
    if green_count > 0:
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
            "gpt-5.6-luna",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
        ]
    )

is_gpt_model = "gpt" in model_selection

default_rounds = 1 if is_gpt_model else 3
num_rounds = st.slider(
    "🔁 AI再検索の回数（1回でも🟢があれば最終🟢）",
    min_value=1, max_value=5, value=default_rounds, step=1,
    key=f"num_rounds_{model_selection}",
    help="同じ検索キーワードでAIに複数回問い合わせます。"
         "いずれか1回でも🟢判定が出れば最終的に🟢になります（OR方式）。"
         "回数を増やすほど見逃し(false negative)は減りますが、API呼び出し回数とコスト・待ち時間も比例して増えます。"
)
if is_gpt_model:
    st.caption("💡 ChatGPTモデルはデフォルト1回（トークン節約）。見逃しを減らしたい場合は回数を増やせます。")
    st.caption("⚠️ 下記「注意」参照: OpenAI APIの回答はChatGPTアプリの回答と完全には一致しません。")

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

                col_final = f'最終判定(OR統合🟢/❌・{num_rounds}回中)'

                vote_results = final_df.apply(lambda row: combine_rounds(row, round_columns), axis=1)
                final_df[col_final] = vote_results.apply(lambda x: x[0])
                final_df['一致率'] = vote_results.apply(lambda x: f"{x[1]}/{x[2]}" if x[2] > 0 else "N/A")

                st.success(f"✅ 計 {len(final_df)} 件の検証が完了しました！（{num_rounds}回分をOR方式で統合済み：1回でも🟢なら🟢）")

                split_df = final_df[final_df[round_columns].nunique(axis=1) > 1]
                if not split_df.empty:
                    st.info(f"ℹ️ {num_rounds}回の判定が割れた店舗が {len(split_df)} 件あります（最終的には🟢が1つでもあれば🟢として採用されています）。")

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

                with st.spinner(f'4️⃣ {JUDGE_MODEL} が全{num_rounds}回分の回答から店舗名を整理中...'):
                    shop_name_summary = summarize_shop_names_only(ai_responses)

                st.divider()
                st.subheader(f"🏪 店舗名のみ整理リスト（{num_rounds}回分の回答を統合・重複排除／{JUDGE_MODEL}）")
                st.markdown(shop_name_summary)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.exception(e)
