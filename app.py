import os
os.system("pip uninstall -y opencv-python")

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import base64

from predict import (
    detect_tiles_from_image, 
    analyze_exposed_melds, 
    analyze_concealed_hand, 
    calculate_tai
)

st.set_page_config(page_title="台灣麻將計分系統", page_icon="🀄", layout="centered")

# --- 🧠 1. 設置自定義 CSS 背景圖片 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

local_image_path = 'background.png'

if os.path.exists(local_image_path):
    base64_image = get_base64_of_bin_file(local_image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.error(f"⚠️ 找不到本地背景圖片檔: {local_image_path}，請確認檔名和路徑。")
# --------------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 🧠 --- 狀態記憶 (Session State) 初始化 ---
if 'ai_completed' not in st.session_state:
    st.session_state.ai_completed = False
    st.session_state.base_tai = 0
    st.session_state.details = []
    st.session_state.final_remaining = []
    st.session_state.metrics = {}
    st.session_state.hand_info = {}

st.title("台灣麻將計分輔助系統")
st.markdown("上傳您的麻將牌面，讓 YOLOv8瞬間為您計算台數")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("手牌 (暗牌)")
    st.write("請由正前方拍攝立起來的手牌")
    front_img_file = st.file_uploader("上傳手牌照片", type=['jpg', 'jpeg', 'png'], key="front")

with col2:
    st.subheader("吃碰槓 (明牌)")
    st.write("請由正上方俯拍攤平的吃碰槓 (若門清可不傳)")
    top_img_file = st.file_uploader("上傳吃碰槓照片", type=['jpg', 'jpeg', 'png'], key="top")

if front_img_file or top_img_file:
    st.markdown("### 影像預覽")
    p_col1, p_col2 = st.columns(2)
    if front_img_file:
        p_col1.image(Image.open(front_img_file), caption="手牌預覽", use_container_width=True)
    if top_img_file:
        p_col2.image(Image.open(top_img_file), caption="明牌預覽", use_container_width=True)

st.markdown("---")

# 🚀 --- 2. 結算按鈕與 AI 執行區 ---
if st.button("開始計算台數", use_container_width=True):
    if not front_img_file:
        st.error("⚠️ 請至少上傳一張「手牌」照片才能進行結算！")
    else:
        with st.spinner("正在解析牌型中，請稍候..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_front:
                tmp_front.write(front_img_file.getvalue())
                front_path = tmp_front.name
            
            top_path = "non_existent_file.jpg" 
            if top_img_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_top:
                    tmp_top.write(top_img_file.getvalue())
                    top_path = tmp_top.name

            concealed_hand = detect_tiles_from_image(model, front_path)
            exposed_melds = detect_tiles_from_image(model, top_path)
            is_men_qing = len(exposed_melds) == 0

            exp_seq, exp_tri, exp_quad, exp_rem = analyze_exposed_melds(exposed_melds)
            con_seq, con_tri, con_pair, con_rem = analyze_concealed_hand(concealed_hand)

            concealed_triplets_count = len(con_tri)

            final_sequences = exp_seq + con_seq
            final_triplets = exp_tri + con_tri
            final_quads = exp_quad
            final_pairs = con_pair
            final_remaining = exp_rem + con_rem
            final_hand_zh = concealed_hand + exposed_melds

            os.remove(front_path)
            if top_img_file: os.remove(top_path)

            st.session_state.ai_completed = True
            st.session_state.final_remaining = final_remaining
            
            # 👇 [重點修正] 新增防呆機制：確認牌數 >= 17 才丟給 AI 算台數
            if not final_remaining and len(final_hand_zh) >= 17:
                tai, details = calculate_tai(final_sequences, final_triplets, final_quads, final_pairs, final_hand_zh, is_men_qing, concealed_triplets_count)
                st.session_state.base_tai = tai
                st.session_state.details = details
            else:
                st.session_state.base_tai = 0
                st.session_state.details = []
            
            st.session_state.metrics = {
                "seq": len(final_sequences),
                "tri": len(final_triplets),
                "quad": len(final_quads),
                "pair": len(final_pairs),
                "total_tiles": len(final_hand_zh)
            }
            
            st.session_state.hand_info = {
                "concealed": ", ".join(concealed_hand) if isinstance(concealed_hand, list) else str(concealed_hand),
                "exposed": '無 (門清)' if is_men_qing else (", ".join(exposed_melds) if isinstance(exposed_melds, list) else str(exposed_melds))
            }


# 📊 --- 3. 顯示結果 ---
if st.session_state.ai_completed:
    st.markdown("---")
    
    st.markdown("### 牌型歸類結果")
    st.info(f"**共偵測到 {st.session_state.metrics.get('total_tiles', 0)} 張牌**")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("順子", st.session_state.metrics.get('seq', 0))
    col_b.metric("刻子", st.session_state.metrics.get('tri', 0))
    col_c.metric("槓子", st.session_state.metrics.get('quad', 0))
    col_d.metric("將眼", st.session_state.metrics.get('pair', 0))

    clean_concealed = str(st.session_state.hand_info.get('concealed', '')).replace("[", "").replace("]", "").replace("'", "")
    clean_exposed = str(st.session_state.hand_info.get('exposed', '')).replace("[", "").replace("]", "").replace("'", "")

    st.write(f"**暗牌:** {clean_concealed}")
    st.write(f" **明牌:** {clean_exposed}")

    st.markdown("---")

    st.markdown("### 動態台數自行勾選區 (無法判定之過程)")
    col3, col4 = st.columns(2)

    dynamic_tai = 0
    dynamic_details = []

    with col3:
        if st.checkbox("莊家 (+1台)"): dynamic_tai += 1; dynamic_details.append("莊家 (1台)")
        if st.checkbox("自摸 (+1台)"): dynamic_tai += 1; dynamic_details.append("自摸 (1台)")
        if st.checkbox("獨聽 [邊張/中洞/單吊] (+1台)"): dynamic_tai += 1; dynamic_details.append("獨聽 (1台)")
    with col4:
        if st.checkbox("海底撈月/河底撈魚 (+1台)"): dynamic_tai += 1; dynamic_details.append("海底/河底 (1台)")
        if st.checkbox("槓上開花 (+1台)"): dynamic_tai += 1; dynamic_details.append("槓上開花 (1台)")
        if st.checkbox("全求人 (+2台)"): dynamic_tai += 2; dynamic_details.append("全求人 (2台)")

    st.markdown("---")

    st.markdown("### 最終結算台數")
    
    total_tiles = st.session_state.metrics.get('total_tiles', 0)
    
    # 👇 [重點修正] 在畫面顯示上攔截所有異常狀態
    if total_tiles == 0:
        st.warning("⚠️ 照片中未偵測到任何麻將牌！請確認照片清晰且無遮擋。")
    elif total_tiles < 17:
        st.error(f"❌ **牌數不足！** 台灣麻將胡牌至少需要 17 張牌，目前僅偵測到 {total_tiles} 張。")
    elif st.session_state.final_remaining:
        st.error(f"❌ **牌型尚未完整，無法計算台數。** \n\n剩下的孤張/廢牌: {', '.join(st.session_state.final_remaining)}")
    else:
        total_tai = st.session_state.base_tai + dynamic_tai
        
        if total_tai > 0:
            st.success(f"**總共獲得： {total_tai} 台** (靜態牌型 {st.session_state.base_tai} 台 + 動態勾選 {dynamic_tai} 台)")
            
            for d in st.session_state.details:
                st.write(f"✅ {d}")
                
            for dd in dynamic_details:
                st.markdown(f"✅ {dd} <span style='color: #9E9E9E; font-size: 0.9em;'>(自行勾選)</span>", unsafe_allow_html=True)
        else:
            st.warning("平胡或無特殊牌型 (0台)。")