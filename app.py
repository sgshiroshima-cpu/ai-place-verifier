import streamlit as st
import pandas as pd
import io
import time
import requests
import json
import google.generativeai as genai

# ==========================================
# 🔑 API キー 読み込み (Streamlit Secrets 이용 - 웹 배포용 안전한 방식)
# ==========================================
# 깃허브에 코드가 올라가도 API 키가 노출되지 않도록 st.secrets를 사용합니다.
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# ==========================================
# 1. Google Places API 連携関数 (API 연동 함수)
# ==========================================
def get_google_places_data(search_query):
    url = 'https://places.googleapis.com/v1/places:searchText'
    places_list = []

    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_API_KEY, 
        'X-Goog-FieldMask': 'places.displayName,places.nationalPhoneNumber,nextPageToken'
    }

    data = {
        'textQuery': search_query,
        'languageCode': 'ja'
    }

    status_text = st.empty()
    page_count = 1

    while True:
        status_text.text(f"Googleサーバーからデータを取得しています... (現在 {page_count} ページ、累計 {len(places_list)} 件)")
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result_data = response.json()
            places = result_data.get('places', [])

            for place in places:
                name = place.get('displayName', {}).get('text', '名前なし')
                phone = place.get('nationalPhoneNumber', '**なし**')

                places_list.append({
                    '店舗名': name,      
                    '電話番号': phone,
                })

            next_token = result_data.get('nextPageToken')

            if next_token:
                data['pageToken'] = next_token
                time.sleep(2)
                page_count += 1
            else:
                break
        else:
            st.error(f"Google API エラー発生: {response.status_code} - {response.text}")
            break

    status_text.empty()
    return pd.DataFrame(places_list)

# ==========================================
# 2. AI ピュア推薦リスト生成 (첫 검색 결과 추출 함수)
# ==========================================
def get_ai_pure_recommendation(search_query, selected_model):
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(selected_model)
    
    prompt = f"「{search_query}」に関連する、あなたが自信を持っておすすめできる有名で美味しいお店を思いつく限りリストアップしてください。余計な説明は省き、店舗名のみを箇条書きで出力してください。"
    
    response = gemini_model.generate_content(prompt, generation_config={"temperature": 0.0})
    return response.text

# ==========================================
# 3. AI 審査員 スマート照合 (AI 리스트와 Google 리스트 비교 함수)
# ==========================================
def match_lists_with_ai(df, ai_recommended_text, fixed_model):
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(fixed_model)
    
    shop_names = df['店舗名'].tolist()
    shop_list_text = "\n".join([f"- {name}" for name in shop_names])
    
    prompt = f"""あなたはデータ照合の専門家です。

【基準リスト（AIが最初におすすめした店舗）】
{ai_recommended_text}

【対象リスト（Googleマップの検索結果）】
{shop_list_text}

対象リストの各店舗について、基準リストのいずれかの店舗と「同一店舗である（支店名の有無、ひらがな/漢字の違い、前後の単語の違いなどの表記揺れを考慮）」と判断できる場合は "O"、基準リストに存在しない場合は "X" と判定してください。

※重要事項※
必ず以下のJSON配列形式のみで出力してください。Markdownの記号（```json など）やその他の説明文は絶対に含めないでください。
[
    {{"name": "対象リストにある店舗名1", "result": "O"}},
    {{"name": "対象リストにある店舗名2", "result": "X"}}
]
"""
    response = gemini_model.generate_content(prompt, generation_config={"temperature": 0.0})
    raw_text = response.text
    
    try:
        start_idx = raw_text.find('[')
        end_idx = raw_text.rfind(']') + 1
        json_str = raw_text[start_idx:end_idx]
        
        judgements = json.loads(json_str)
        judgement_dict = {item['name']: item['result'] for item in judgements}
        
        df['Gemini_推薦(O/X)'] = df['店舗名'].apply(lambda x: judgement_dict.get(x, 'X (判定漏れ)'))
        
        return df, raw_text
        
    except Exception as e:
        st.error(f"AIの回答の解析に失敗しました。(エラー: {e})")
        return df, raw_text

# ==========================================
# 4. Web UI 構成 (Streamlit)
# ==========================================
st.set_page_config(page_title="地域スポットAI検証システム", layout="wide")
st.title("📍 地域スポットAI検証システム (ピュア推薦マッチング)")

st.markdown("#### 🔍 検索キーワードおよびAIモデル設定")
st.info("💡 Google検索のように場所やジャンルをスペース区切りで入力してください。（例: 浅草 焼肉 店）")

col1, col2 = st.columns([3, 1])

with col1:
    search_query_input = st.text_input("検索キーワード", value="浅草 焼肉 店")

with col2:
    # 無料枠で安定して動作するFlashモデルのみに整理しました
    model_selection = st.selectbox(
        "推薦リスト作成モデルを選択",
        options=[
            "gemini-3.5-flash-lite", 
            "gemini-3.6-flash"
        ]
    )

if st.button("🚀 検索および検証を実行", type="primary"):
    # Secrets에 키가 제대로 입력되었는지 확인하는 로직으로 변경되었습니다.
    if not GOOGLE_API_KEY or not GEMINI_API_KEY:
        st.error("⚠️ Streamlit Secrets に Google APIキーと Gemini APIキーが設定されていません！ (Advanced settings を確認してください)")
    else:
        try:
            with st.spinner('1️⃣ Google Places APIから場所情報を収集しています...'):
                df_google = get_google_places_data(search_query_input)
            
            if df_google.empty:
                st.error("Googleの検索結果がありません。検索キーワードを変更してみてください。")
            else:
                with st.spinner(f'2️⃣ {model_selection} が先入観なしでおすすめリストを作成中...'):
                    ai_pure_list = get_ai_pure_recommendation(search_query_input, model_selection)

                fixed_judge_model = "gemini-3.6-flash" 
                with st.spinner(f'3️⃣ {fixed_judge_model} 審査員がGoogleリストとAIリストを照合中...'):
                    final_df, raw_ai_response = match_lists_with_ai(df_google, ai_pure_list, fixed_judge_model)

                st.success(f"✅ 計 {len(final_df)} 件の場所の検証が完了しました！")

                st.markdown("#### 📊 最終結果の確認")
                st.dataframe(final_df, use_container_width=True)

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='AI_Verification')
                
                safe_filename = search_query_input.replace(' ', '_')
                file_name = f"{safe_filename}_AI検証結果.xlsx"
                st.download_button(
                    label="📥 Excelファイルをダウンロード (.xlsx)",
                    data=buffer.getvalue(),
                    file_name=file_name,
                    mime="application/vnd.ms-excel"
                )
                
                with st.expander("🤖 Geminiが最初に思いついたピュアなリストを見る"):
                    st.info(ai_pure_list)
                    
                with st.expander("🤖 Geminiの最終判定原文 (JSON) を表示"):
                    st.code(raw_ai_response, language='json')

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")