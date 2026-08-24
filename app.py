import streamlit as st
import pandas as pd
import io
import time
import requests
import json
import google.generativeai as genai
from openai import OpenAI

# ==========================================
# 🔑 API キー 読み込み (Streamlit Secrets)
# ==========================================
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# 1. Google Places API 連携関数
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

# 2. AI ピュア推薦リスト生成 
def get_ai_pure_recommendation(search_query, selected_model):
    prompt = f"「{search_query}」に関連する、あなたが自信を持っておすすめできる有名で美味しいお店を思いつく限りリストアップしてください。余計な説明は省き、店舗名のみを箇条書きで出力してください。"
    
    if "gpt" in selected_model:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(selected_model)
        return gemini_model.generate_content(prompt, generation_config={"temperature": 0.0}).text

# 3. AI 審査員 スマート照合 (🟢/❌)
def match_lists_with_ai(df, ai_recommended_text, selected_model):
    shop_names = df['店舗名'].tolist()
    shop_list_text = "\n".join([f"- {name}" for name in shop_names])
    
    prompt = f"""あなたはデータ照合の専門家です。

【基準リスト（AIが最初におすすめした店舗）】
{ai_recommended_text}

【対象リスト（Googleマップの検索結果）】
{shop_list_text}

対象リストの各店舗について、基準リストのいずれかの店舗と「同一店舗である（表記揺れを考慮）」と判断できる場合は "🟢"、存在しない場合は "❌" と判定してください。

※重要事項※
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
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(selected_model)
        raw_text = gemini_model.generate_content(prompt, generation_config={"temperature": 0.0}).text
    
    try:
        start_idx = raw_text.find('[')
        end_idx = raw_text.rfind(']') + 1
        json_str = raw_text[start_idx:end_idx]
        
        judgements = json.loads(json_str)
        judgement_dict = {item['name']: item['result'] for item in judgements}
        
        df['AI_推薦(🟢/❌)'] = df['店舗名'].apply(lambda x: judgement_dict.get(x, '❌ (判定漏れ)'))
        return df, raw_text
        
    except Exception as e:
        st.error(f"AIの回答の解析に失敗しました。(エラー: {e})")
        return df, raw_text

# 🌟 4. 표 색상 하이라이트 함수 추가
def highlight_matched_rows(row):
    # '🟢' 기호가 포함된 행의 배경색을 연한 초록색(#e6ffe6)으로 변경합니다.
    if '🟢' in str(row['AI_推薦(🟢/❌)']):
        return ['background-color: #e6ffe6'] * len(row)
    else:
        return [''] * len(row)

# 5. Web UI 構成
st.set_page_config(page_title="地域スポットAI検証システム", layout="wide")
st.title("📍 地域スポットAI検証システム (ピュア推薦マッチング)")

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
                with st.spinner(f'2️⃣ {model_selection} がおすすめリストを作成中...'):
                    ai_pure_list = get_ai_pure_recommendation(search_query_input, model_selection)

                with st.spinner(f'3️⃣ {model_selection} 審査員が照合中...'):
                    final_df, raw_ai_response = match_lists_with_ai(df_google, ai_pure_list, model_selection)

                st.success(f"✅ 計 {len(final_df)} 件の検証が完了しました！")
                
                # 🌟 데이터프레임에 하이라이트 스타일을 적용하여 화면에 출력합니다.
                styled_df = final_df.style.apply(highlight_matched_rows, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                buffer = io.BytesIO()
                # 엑셀 파일에도 스타일을 반영하려면 조금 복잡해지므로, 엑셀은 원본 데이터를 저장합니다.
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='AI_Verification')
                
                st.download_button(
                    label="📥 Excelファイルをダウンロード (.xlsx)",
                    data=buffer.getvalue(),
                    file_name=f"{search_query_input.replace(' ', '_')}_AI検証結果.xlsx",
                    mime="application/vnd.ms-excel"
                )
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
