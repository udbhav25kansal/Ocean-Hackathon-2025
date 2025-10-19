# streamlit_app_LIVE.py - MAXIMUM SPONGEBOB EDITION 🧽🍍

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import requests
from io import BytesIO

from live_data_fetcher import LiveOceanData

# Page config
st.set_page_config(
    page_title="🧽 Bikini Bottom Current Classifier 🍍",
    page_icon="🧽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# MAXIMUM SPONGEBOB CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Titan+One&family=Comic+Neue:wght@700&family=Fredoka+One&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #87CEEB 0%, #4A90E2 30%, #2E5C8A 60%, #1a3a52 100%);
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.1) 0%, transparent 30%);
    }
    
    .main-title {
        font-family: 'Titan One', cursive;
        font-size: 5rem;
        text-align: center;
        color: #FFD700;
        text-shadow: 
            4px 4px 0px #FF6347,
            8px 8px 0px #FF1493,
            12px 12px 20px rgba(0,0,0,0.5);
        animation: bounce 2s infinite;
        margin-bottom: 0;
        letter-spacing: 3px;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0) rotate(-2deg); }
        50% { transform: translateY(-20px) rotate(2deg); }
    }
    
    .subtitle {
        font-family: 'Comic Neue', cursive;
        text-align: center;
        color: #FFD700;
        font-size: 1.8rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .pineapple-card {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        border: 5px solid #8B4513;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        text-align: center;
        transform: rotate(-1deg);
        transition: all 0.3s;
    }
    
    .pineapple-card:hover {
        transform: rotate(1deg) scale(1.05);
    }
    
    .metric-value {
        font-family: 'Fredoka One', cursive;
        font-size: 3rem;
        color: #8B4513;
        text-shadow: 2px 2px 0px #FFE4B5;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-family: 'Comic Neue', cursive;
        color: #8B4513;
        font-size: 1rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .stButton>button {
        font-family: 'Fredoka One', cursive;
        background: linear-gradient(135deg, #FF69B4 0%, #FF1493 100%);
        color: white;
        border: 4px solid #8B008B;
        padding: 1rem 2rem;
        font-size: 1.5rem;
        border-radius: 25px;
        box-shadow: 0 8px 0 #8B008B, 0 12px 20px rgba(0,0,0,0.4);
        width: 100%;
        transition: all 0.1s;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 0 #8B008B, 0 16px 30px rgba(0,0,0,0.5);
    }
    
    .stButton>button:active {
        transform: translateY(4px);
        box-shadow: 0 4px 0 #8B008B, 0 6px 10px rgba(0,0,0,0.3);
    }
    
    .jellfish-badge {
        display: inline-block;
        padding: 1rem 2rem;
        border-radius: 30px;
        font-family: 'Fredoka One', cursive;
        font-size: 2rem;
        border: 4px solid;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(-2deg); }
        50% { transform: translateY(-15px) rotate(2deg); }
    }
    
    .outflow-badge {
        background: linear-gradient(135deg, #00FA9A 0%, #00CED1 100%);
        color: white;
        border-color: #008B8B;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .tides-badge {
        background: linear-gradient(135deg, #4169E1 0%, #1E90FF 100%);
        color: white;
        border-color: #00008B;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .wind-badge {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #8B4513;
        border-color: #FF6347;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.5);
    }
    
    .krusty-krab-box {
        background: linear-gradient(135deg, #FF6347 0%, #FF4500 100%);
        border: 5px solid #8B0000;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        font-family: 'Comic Neue', cursive;
    }
    
    .chum-bucket-box {
        background: linear-gradient(135deg, #98FB98 0%, #90EE90 100%);
        border: 5px solid #228B22;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        font-family: 'Comic Neue', cursive;
    }
    
    h1, h2, h3 {
        font-family: 'Fredoka One', cursive;
        color: #FFD700;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Fredoka One', cursive;
        background: linear-gradient(135deg, #FF69B4 0%, #FF1493 100%);
        border-radius: 15px;
        padding: 1rem 2rem;
        color: white;
        border: 3px solid #8B008B;
        font-size: 1.2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #8B4513;
        border-color: #FF6347;
        transform: scale(1.1);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #FF69B4 0%, #FF1493 100%);
    }
    
    /* Bubble decorations */
    .bubble {
        position: fixed;
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.8), rgba(255,255,255,0.3));
        border-radius: 50%;
        animation: rise 15s infinite ease-in;
        pointer-events: none;
        border: 2px solid rgba(255,255,255,0.5);
    }
    
    @keyframes rise {
        0% {
            bottom: -100px;
            opacity: 0;
        }
        50% {
            opacity: 1;
        }
        100% {
            bottom: 110%;
            opacity: 0;
        }
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        border: 3px solid #8B4513;
        border-radius: 15px;
        font-family: 'Comic Neue', cursive;
        font-weight: bold;
    }
    
    .tartar-sauce {
        font-family: 'Fredoka One', cursive;
        color: #FF1493;
        font-size: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        text-align: center;
        margin: 1rem 0;
    }
</style>

<!-- Floating Bubbles -->
<div class="bubble" style="width: 40px; height: 40px; left: 10%; animation-delay: 0s;"></div>
<div class="bubble" style="width: 60px; height: 60px; left: 30%; animation-delay: 3s;"></div>
<div class="bubble" style="width: 30px; height: 30px; left: 50%; animation-delay: 6s;"></div>
<div class="bubble" style="width: 50px; height: 50px; left: 70%; animation-delay: 9s;"></div>
<div class="bubble" style="width: 35px; height: 35px; left: 85%; animation-delay: 12s;"></div>
<div class="bubble" style="width: 45px; height: 45px; left: 20%; animation-delay: 2s;"></div>
<div class="bubble" style="width: 55px; height: 55px; left: 60%; animation-delay: 7s;"></div>
<div class="bubble" style="width: 38px; height: 38px; left: 80%; animation-delay: 10s;"></div>
""", unsafe_allow_html=True)


class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0)
        self.env_encoder = nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(1792 + 128, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, env_features):
        img_feat = self.backbone(images)
        env_feat = self.env_encoder(env_features)
        combined = torch.cat([img_feat, env_feat], dim=1)
        return self.classifier(combined)


@st.cache_resource
def load_model():
    checkpoint = torch.load('best_model_OPTION3.pth', map_location='cpu')
    model = MultiModalClassifier(num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    label_map = checkpoint['label_map']
    inv_label_map = {v: k for k, v in label_map.items()}
    val_acc = checkpoint.get('val_acc', 0)
    return model, inv_label_map, val_acc


@st.cache_resource
def load_test_results():
    if os.path.exists('test_predictions_FINAL.csv'):
        return pd.read_csv('test_predictions_FINAL.csv')
    return None


model, inv_label_map, model_val_acc = load_model()
test_results = load_test_results()
live_fetcher = LiveOceanData()

# GIANT SPONGEBOB HEADER
st.markdown('''
<h1 class="main-title">🧽 BIKINI BOTTOM 🍍<br>CURRENT CLASSIFIER</h1>
<p class="subtitle">⚓ Powered by Sandy's Super Computer & A Little Bit of Magic! 🐿️✨</p>
''', unsafe_allow_html=True)

st.markdown("---")

# Top Metrics - PINEAPPLE STYLE
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="pineapple-card">
        <div class="metric-label">🎯 ACCURACY</div>
        <div class="metric-value">{model_val_acc:.1f}%</div>
        <div style="font-size: 0.9rem; color: #8B4513; font-weight: bold;">Barnacles!</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="pineapple-card">
        <div class="metric-label">🧠 NEURONS</div>
        <div class="metric-value">18.6M</div>
        <div style="font-size: 0.9rem; color: #8B4513; font-weight: bold;">Brain Coral!</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="pineapple-card">
        <div class="metric-label">⚡ TRAINING</div>
        <div class="metric-value">35 min</div>
        <div style="font-size: 0.9rem; color: #8B4513; font-weight: bold;">Lightning Fast!</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="pineapple-card">
        <div class="metric-label">📊 SAMPLES</div>
        <div class="metric-value">8,168</div>
        <div style="font-size: 0.9rem; color: #8B4513; font-weight: bold;">Holy Shrimp!</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs(
    ["🍔 UPLOAD & CLASSIFY", "🔴 LIVE FROM THE REEF", "🏆 HALL OF FAME"])

# ============================================
# TAB 1: UPLOAD
# ============================================
with tab1:
    st.markdown('<h2 style="text-align: center;">🌊 Drop Your Current Plot Here! 🌊</h2>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown("### 🖼️ Current Plot Upload")
        uploaded_file = st.file_uploader(
            "Choose a .png from your treasure chest! 🏴‍☠️", type=['png'], key='upload')

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)

    with col_right:
        st.markdown("### 🎛️ Ocean Conditions")

        wind_speed = st.slider("💨 Wind Speed (km/h)", 0.0, 60.0,
                               10.0, 1.0, help="How hard is the wind blowing? 🌬️")
        fraser_level = st.slider(
            "🏞️ Fraser River (m)", 0.0, 1.0, 0.3, 0.05, help="Is the river running wild? 🏔️")
        tidal_level = st.slider("🌊 Tide Height (m)", 0.0,
                                6.0, 3.0, 0.1, help="How high is the water? 🌙")

        col_a, col_b = st.columns(2)
        with col_a:
            month = st.selectbox("📅 Month", list(
                range(1, 13)), index=datetime.now().month - 1)
        with col_b:
            pressure = st.number_input("🌡️ Pressure", 980, 1040, 1013, 5)

        is_freshet = int(month in [5, 6, 7])

        if is_freshet:
            st.success(
                "🌸 **SPRING FRESHET!** Gary says 'Meow!' (That means ELEVATED DISCHARGE!)")
        else:
            st.info("❄️ **NORMAL FLOW** - Just another day in paradise! 🏖️")

    st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_file and st.button("🔍 I'M READY! CLASSIFY THIS CURRENT! 🔍", type="primary", use_container_width=True):

        # Prepare image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)

        # Env features
        env_tensor = torch.tensor([[
            wind_speed / 60.0, 180.0 / 360.0, tidal_level / 6.0,
            1.5 / 3.0, 0.0, fraser_level / 1.0,
            pressure / 1050.0, 30.0 / 100.0, float(is_freshet),
            wind_speed / 60.0, 1.5 / 3.0
        ]], dtype=torch.float32)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor, env_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = inv_label_map[pred_idx]
            confidence = probs[pred_idx].item() * 100

        # RESULTS
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<p class="tartar-sauce">⭐ TARTAR SAUCE! HERE ARE YOUR RESULTS! ⭐</p>', unsafe_allow_html=True)

        # Giant Badge
        badge_class = "outflow-badge" if pred_label == 'OUTFLOW' else "tides-badge" if pred_label == 'TIDES' else "wind-badge"
        emoji = "🏞️" if pred_label == 'OUTFLOW' else "🌊" if pred_label == 'TIDES' else "💨"
        st.markdown(
            f'<div style="text-align: center; margin: 2rem 0;"><span class="jellfish-badge {badge_class}">{emoji} {pred_label} {emoji}</span></div>', unsafe_allow_html=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="pineapple-card">
                <div class="metric-label">CONFIDENCE</div>
                <div class="metric-value">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="pineapple-card">
                <div class="metric-label">OUTFLOW</div>
                <div class="metric-value">{probs[0].item()*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="pineapple-card">
                <div class="metric-label">TIDES</div>
                <div class="metric-value">{probs[1].item()*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability Chart - COLORFUL
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 PROBABILITY BREAKDOWN")

        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#87CEEB')
        classes = ['🏞️ OUTFLOW', '🌊 TIDES', '💨 WIND/STORM']
        prob_values = [probs[i].item() * 100 for i in range(3)]
        colors = ['#00FA9A', '#4169E1', '#FFD700']

        bars = ax.barh(classes, prob_values, color=colors,
                       height=0.6, edgecolor='#8B4513', linewidth=4)
        ax.set_xlabel('Probability (%)', fontsize=14,
                      color='#8B4513', fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_facecolor('#B0E0E6')
        ax.tick_params(colors='#8B4513', labelsize=12)
        ax.spines['bottom'].set_color('#8B4513')
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_color('#8B4513')
        ax.spines['left'].set_linewidth(3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, color='white', linewidth=2)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center', color='#8B4513', fontweight='bold', fontsize=13)

        st.pyplot(fig)

        # INFO BOXES
        st.markdown("<br>", unsafe_allow_html=True)

        if pred_label == 'OUTFLOW':
            st.markdown(f"""
            <div class="chum-bucket-box">
                <h3 style="color: #228B22; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🏞️ FRASER RIVER OUTFLOW DETECTED!</h3>
                <p style="color: #006400; font-size: 1.2rem; font-weight: bold;">
                Patrick says: "It's all that fresh water from the mountains!" 🏔️
                </p>
                <p style="color: #228B22; font-size: 1.1rem;">
                <b>Fraser Level:</b> {fraser_level:.2f}m {'🌊 HIGH TIDE!' if fraser_level > 0.4 else '📉 Normal'}<br>
                <b>Season:</b> {'🌸 SPRING FRESHET - Salmon Highway Open!' if is_freshet else '❄️ Regular Flow'}<br>
                <b>Wind:</b> {wind_speed:.1f} km/h<br><br>
                <b>Why This Matters:</b> Salmon migration! 🐟 Estuarine management! Water quality monitoring! 💧
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif pred_label == 'TIDES':
            st.markdown(f"""
            <div class="krusty-krab-box">
                <h3 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">🌊 TIDAL FORCING DOMINANT!</h3>
                <p style="color: #FFE4B5; font-size: 1.2rem; font-weight: bold;">
                Squidward says: "It's basic lunar gravitational physics..." 🌙
                </p>
                <p style="color: white; font-size: 1.1rem;">
                <b>Tide Height:</b> {tidal_level:.2f}m above the Krusty Krab floor 🦀<br>
                <b>Wind:</b> {wind_speed:.1f} km/h (not interfering)<br>
                <b>Moon Phase:</b> Pulling the ocean like taffy! 🌕<br><br>
                <b>Why This Matters:</b> Navigation timing! ⚓ Tidal energy! ⚡ Harbor operations! 🚢
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:  # WIND/STORM
            st.markdown(f"""
            <div class="krusty-krab-box">
                <h3 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">💨 WIND/STORM FORCING ACTIVE!</h3>
                <p style="color: #FFE4B5; font-size: 1.2rem; font-weight: bold;">
                SpongeBob warns: "Batten down the hatches! Time to close the pineapple!" 🍍⛈️
                </p>
                <p style="color: white; font-size: 1.1rem;">
                <b>Wind Speed:</b> {wind_speed:.1f} km/h {'⚠️ STORMY!' if wind_speed > 20 else '🌬️ Breezy'}<br>
                <b>Pressure:</b> {pressure} hPa {'⛈️ LOW PRESSURE!' if pressure < 1005 else '☀️ Fair Weather'}<br>
                <b>Sea Conditions:</b> Choppy! Hold onto your spatula! 🍔<br><br>
                <b>Why This Matters:</b> Search & rescue! 🚨 Oil spill response! 🛢️ Small craft warnings! ⚓
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# TAB 2: LIVE MODE
# ============================================
with tab2:
    st.markdown('<h2 style="text-align: center;">🔴 LIVE FROM BIKINI BOTTOM OBSERVATORY 🔴</h2>',
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #FFD700;">Powered by Plankton\'s Evil Robot API Fetchers! 🤖</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 FETCH FRESH DATA! (Like Krabby Patties!) 🍔", type="primary", use_container_width=True):
        with st.spinner("🐌 Sending Gary to fetch the data..."):
            live_data = live_fetcher.get_all_live_data()

        st.success(
            f"✅ DATA RETRIEVED! Time: {live_data['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="pineapple-card">
                <div class="metric-label">💨 WIND</div>
                <div class="metric-value">{live_data['wind_speed']:.1f}</div>
                <div style="font-size: 0.9rem; color: #8B4513; font-weight: bold;">km/h</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="pineapple-card">
                <div class="metric-label">🌊 TIDE</div>
                <div class="metric-value">{live_data['tidal_level']:.2f}</div>
                <div style="font-size: 0.9rem; color: #8B4513; font-weight: bold;">meters</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="pineapple-card">
                <div class="metric-label">🏞️ FRASER</div>
                <div class="metric-value">{live_data['fraser_level']:.2f}</div>
                <div style="font-size: 0.9rem; color: #8B4513; font-weight: bold;">meters</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="pineapple-card">
                <div class="metric-label">🌡️ PRESSURE</div>
                <div class="metric-value">{live_data['pressure']:.0f}</div>
                <div style="font-size: 0.9rem; color: #8B4513; font-weight: bold;">hPa</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("📸 **NOW UPLOAD A CURRENT PLOT TO CLASSIFY WITH THIS FRESH DATA!**")

        uploaded_live = st.file_uploader("🎣 Hook a plot here!", type=[
                                         'png'], key='live_upload')

        if uploaded_live:
            image = Image.open(uploaded_live).convert('RGB')
            st.image(image, use_container_width=True)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0)

            env_tensor = torch.tensor([[
                live_data['wind_speed'] / 60.0, live_data['wind_dir'] / 360.0,
                live_data['tidal_level'] /
                6.0, abs(live_data['tidal_range']) / 3.0,
                live_data['tidal_velocity'] /
                1.0, live_data['fraser_level'] / 1.0,
                live_data['pressure'] / 1050.0, 30.0 / 100.0,
                float(live_data['is_spring_freshet']),
                live_data['wind_speed'] /
                60.0, abs(live_data['tidal_range']) / 3.0
            ]], dtype=torch.float32)

            with torch.no_grad():
                outputs = model(img_tensor, env_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                pred_label = inv_label_map[pred_idx]
                confidence = probs[pred_idx].item() * 100

            st.markdown("<br>", unsafe_allow_html=True)
            badge_class = "outflow-badge" if pred_label == 'OUTFLOW' else "tides-badge" if pred_label == 'TIDES' else "wind-badge"
            emoji = "🏞️" if pred_label == 'OUTFLOW' else "🌊" if pred_label == 'TIDES' else "💨"
            st.markdown(
                f'<div style="text-align: center;"><span class="jellfish-badge {badge_class}">{emoji} {pred_label} {emoji}</span></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Confidence", f"{confidence:.1f}%")
            with col2:
                st.metric("🏞️ OUTFLOW", f"{probs[0].item()*100:.1f}%")
            with col3:
                st.metric("🌊 TIDES", f"{probs[1].item()*100:.1f}%")

# ============================================
# TAB 3: PERFORMANCE
# ============================================
with tab3:
    st.markdown('<h2 style="text-align: center;">🏆 HALL OF FAME - MODEL PERFORMANCE 🏆</h2>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="chum-bucket-box">
            <h3 style="color: #228B22;">📊 PER-CLASS ACCURACY</h3>
            <p style="color: #006400; font-size: 1.2rem; font-weight: bold;">
            🏞️ OUTFLOW: 68.6%<br>
            🌊 TIDES: 85.4%<br>
            💨 WIND/STORM: 89.4%
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="krusty-krab-box">
            <h3 style="color: white;">⚡ TRAINING DETAILS</h3>
            <p style="color: #FFE4B5; font-size: 1.1rem;">
            <b>Architecture:</b> EfficientNet-B4 + Multi-Modal Fusion 🧠<br>
            <b>GPU:</b> NVIDIA H200 (150GB HBM3) 🖥️<br>
            <b>Training Time:</b> 35 minutes ⚡<br>
            <b>Framework:</b> PyTorch 2.5.1 🔥
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="krusty-krab-box">
            <h3 style="color: white;">📚 DATASET INFO</h3>
            <p style="color: #FFE4B5; font-size: 1.1rem;">
            <b>Total Samples:</b> 8,168 plots 📊<br>
            <b>Timespan:</b> 2 years (2023-2025) 📅<br>
            <b>Resolution:</b> Hourly ⏰<br><br>
            <b>Class Distribution:</b><br>
            🏞️ OUTFLOW: 7.0% (572)<br>
            🌊 TIDES: 40.9% (3,338)<br>
            💨 WIND/STORM: 52.1% (4,258)
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Example Gallery
    if test_results is not None:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            '<h3 style="text-align: center;">🎨 EXAMPLE PREDICTIONS FROM THE VAULT 🎨</h3>', unsafe_allow_html=True)

        show_class = st.selectbox("🐠 Filter by Class:", [
                                  'All', 'OUTFLOW', 'TIDES', 'WIND/STORM'])

        filtered = test_results.copy()
        if show_class != 'All':
            filtered = filtered[filtered['final_label'] == show_class]

        filtered = filtered[filtered['correct']
                            == True].nlargest(6, 'confidence')

        cols = st.columns(3)
        for idx, (_, row) in enumerate(filtered.iterrows()):
            with cols[idx % 3]:
                img_path = os.path.join('plots', row['png_file'])
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                    st.markdown(f"""
                    <div style="text-align: center; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
                                padding: 0.5rem; border-radius: 10px; border: 3px solid #8B4513; margin-top: -5px;">
                        <span style="color: #8B4513; font-weight: bold; font-size: 1.1rem;">
                        ✅ {row['predicted']} ({row['confidence']*100:.0f}%)
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.5rem; color: #FFD700; font-family: "Fredoka One", cursive; text-shadow: 3px 3px 6px rgba(0,0,0,0.7);'>
    <p>🌊 F IS FOR FRIENDS WHO CLASSIFY CURRENTS TOGETHER! 🌊</p>
    <p style="font-size: 1.2rem;">⚓ OceanHack 2024 • Brest, France ⚓</p>
    <p style="font-size: 1rem;">🧽 Noel John | Northeastern University Vancouver 🍍</p>
    <p style="font-size: 0.9rem; font-style: italic;">"The inner machinations of my mind are an enigma." - Patrick Star, Data Scientist 🌟</p>
</div>
""", unsafe_allow_html=True)
