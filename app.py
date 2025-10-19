"""
Bikini Bottom Current Classifier + Ocean Simulator
Run: streamlit run streamlit_app_LIVE.py
"""

from live_data_fetcher import LiveOceanData
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
import warnings
warnings.filterwarnings('ignore')


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
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Fredoka', sans-serif;
    }
    
    /* Main background - darker, professional */
    .stApp {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d5a8c 50%, #1e3a5f 100%);
        background-attachment: fixed;
    }
    
    /* Main content background */
    .main {
        background: transparent;
    }
    
    /* Main title */
    .main-title {
        font-size: 4rem;
        text-align: center;
        color: #FFFFFF;
        font-weight: 700;
        margin-bottom: 0;
        letter-spacing: 2px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #C5E9F0;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    
    /* Cards - clean with proper contrast */
    .pineapple-card {
        background: linear-gradient(135deg, #2d5a8c 0%, #1e3a5f 100%);
        border: 2px solid #4A90E2;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    
    .pineapple-card:hover {
        box-shadow: 0 6px 16px rgba(74, 144, 226, 0.4);
        border-color: #6CB4F5;
    }
    
    /* Metric values - white for contrast */
    .metric-value {
        font-size: 2.5rem;
        color: #FFFFFF;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #C5E9F0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons - solid and clear */
    .stButton>button {
        background: linear-gradient(135deg, #4A90E2 0%, #2E5C8A 100%);
        color: white;
        border: none;
        padding: 0.85rem 1.5rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #6CB4F5 0%, #4A90E2 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(74, 144, 226, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Badges */
    .jellfish-badge {
        display: inline-block;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-size: 1.5rem;
        font-weight: 700;
        border: 2px solid;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .outflow-badge {
        background: linear-gradient(135deg, #00D9A3 0%, #00A877 100%);
        color: white;
        border-color: #00C896;
    }
    
    .tides-badge {
        background: linear-gradient(135deg, #4A90E2 0%, #2E5C8A 100%);
        color: white;
        border-color: #6CB4F5;
    }
    
    .wind-badge {
        background: linear-gradient(135deg, #FFB84D 0%, #FF9500 100%);
        color: white;
        border-color: #FFD699;
    }
    
    /* Info boxes */
    .chum-bucket-box {
        background: linear-gradient(135deg, #1a4d2e 0%, #2d6a47 100%);
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        color: #E8F5E9;
    }
    
    .chum-bucket-box h3 {
        color: #81C784;
    }
    
    .chum-bucket-box p {
        color: #C8E6C9;
    }
    
    .krusty-krab-box {
        background: linear-gradient(135deg, #5d1a1a 0%, #8b3a3a 100%);
        border: 2px solid #E57373;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        color: #FFEBEE;
    }
    
    .krusty-krab-box h3 {
        color: #FFCDD2;
    }
    
    .krusty-krab-box p {
        color: #FFCDD2;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
    }
    
    h2 {
        font-size: 2rem;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        font-size: 1.3rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        color: #C5E9F0;
        border: 2px solid #4A90E2;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4A90E2 0%, #2E5C8A 100%);
        color: white;
        border-color: #6CB4F5;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #4A90E2 0%, #2E5C8A 100%);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(45, 90, 140, 0.5);
        border: 2px dashed #4A90E2;
        border-radius: 12px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background: rgba(45, 90, 140, 0.6) !important;
        border: 2px dashed #4A90E2 !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stFileUploadDropzone"] > section > button {
        color: #C5E9F0;
    }
    
    /* Text inputs and selects */
    .stSelectbox > div > div {
        background: rgba(30, 58, 95, 0.8);
        border: 2px solid #4A90E2;
        border-radius: 8px;
        color: #FFFFFF;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(30, 58, 95, 0.8);
        border: 2px solid #4A90E2;
        border-radius: 8px;
        color: #FFFFFF;
    }
    
    /* Text styling */
    .tartar-sauce {
        color: #FFB84D;
        font-size: 1.3rem;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Dividers */
    hr {
        border: 1px solid #4A90E2;
        margin: 1.5rem 0;
        opacity: 0.5;
    }
    
    /* General text color */
    body, p, span, label {
        color: #E8F4F8 !important;
    }
    
</style>
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

# TABS - NOW 4 TABS
tab1, tab2, tab3, tab4 = st.tabs(
    ["🍔 UPLOAD & CLASSIFY", "🔴 LIVE FROM THE REEF", "🌊 OCEAN SIMULATOR", "🏆 HALL OF FAME"])

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
        wind_direction = st.slider("🧭 Wind Direction (°)", 0, 360, 180, 5,
                                   help="Direction the wind is coming FROM (0°=N, 90°=E, 180°=S, 270°=W) 🌍")
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

        # Use actual discharge data to determine if freshet-like conditions
        discharge_threshold = 0.5
        is_freshet = int(fraser_level > discharge_threshold)

        if is_freshet:
            st.success(
                f"🌸 **ELEVATED DISCHARGE!** Fraser Level: {fraser_level:.2f}m - High discharge detected! Freshet-like conditions! 🏔️")
        else:
            st.info(
                f"📉 **BASE FLOW** - Fraser Level: {fraser_level:.2f}m - Regular seasonal discharge 💧")

    st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_file and st.button("🔍 I'M READY! CLASSIFY THIS CURRENT! 🔍", type="primary", use_container_width=True):

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)

        env_tensor = torch.tensor([[
            wind_speed / 60.0,
            wind_direction / 360.0,
            tidal_level / 6.0,
            1.5 / 3.0,
            0.0,
            fraser_level / 1.0,
            pressure / 1050.0,
            30.0 / 100.0,
            float(is_freshet),
            wind_speed / 60.0,
            1.5 / 3.0
        ]], dtype=torch.float32)

        with torch.no_grad():
            outputs = model(img_tensor, env_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = inv_label_map[pred_idx]
            confidence = probs[pred_idx].item() * 100

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<p class="tartar-sauce">⭐ TARTAR SAUCE! HERE ARE YOUR RESULTS! ⭐</p>', unsafe_allow_html=True)

        badge_class = "outflow-badge" if pred_label == 'OUTFLOW' else "tides-badge" if pred_label == 'TIDES' else "wind-badge"
        emoji = "🏞️" if pred_label == 'OUTFLOW' else "🌊" if pred_label == 'TIDES' else "💨"
        st.markdown(
            f'<div style="text-align: center; margin: 2rem 0;"><span class="jellfish-badge {badge_class}">{emoji} {pred_label} {emoji}</span></div>', unsafe_allow_html=True)

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
                <div class="metric-label">🌊 TIDES</div>
                <div class="metric-value">{probs[1].item()*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="pineapple-card">
                <div class="metric-label">💨 WIND/STORM</div>
                <div class="metric-value">{probs[2].item()*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

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
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        if pred_label == 'OUTFLOW':
            st.markdown(f"""
            <div class="chum-bucket-box">
                <h3 style="color: #228B22;">🏞️ FRASER RIVER OUTFLOW DETECTED!</h3>
                <p style="color: #006400; font-size: 1.2rem; font-weight: bold;">
                Patrick says: "It's all that fresh water from the mountains!" 🏔️
                </p>
                <p style="color: #228B22; font-size: 1.1rem;">
                <b>Fraser Discharge Level:</b> {fraser_level:.2f}m {'🌊 ELEVATED!' if fraser_level > 0.5 else '📉 Baseline'}<br>
                <b>Wind Direction:</b> {wind_direction}° 🧭<br>
                <b>Wind Speed:</b> {wind_speed:.1f} km/h<br>
                <b>Discharge Status:</b> {'🌸 High Freshet-like Conditions!' if fraser_level > 0.5 else '📉 Regular Base Flow'}<br><br>
                <b>Why This Matters:</b> Salmon migration! 🐟 Estuarine management! Water quality monitoring! 💧
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif pred_label == 'TIDES':
            st.markdown(f"""
            <div class="krusty-krab-box">
                <h3 style="color: white;">🌊 TIDAL FORCING DOMINANT!</h3>
                <p style="color: #FFE4B5; font-size: 1.2rem; font-weight: bold;">
                Squidward says: "It's basic lunar gravitational physics..." 🌙
                </p>
                <p style="color: white; font-size: 1.1rem;">
                <b>Tide Height:</b> {tidal_level:.2f}m above the Krusty Krab floor 🦀<br>
                <b>Wind:</b> {wind_speed:.1f} km/h (coming from {wind_direction}°)<br>
                <b>Moon Phase:</b> Pulling the ocean like taffy! 🌕<br><br>
                <b>Why This Matters:</b> Navigation timing! ⚓ Tidal energy! ⚡ Harbor operations! 🚢
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="krusty-krab-box">
                <h3 style="color: white;">💨 WIND/STORM FORCING ACTIVE!</h3>
                <p style="color: #FFE4B5; font-size: 1.2rem; font-weight: bold;">
                SpongeBob warns: "Batten down the hatches! Time to close the pineapple!" 🍍⛈️
                </p>
                <p style="color: white; font-size: 1.1rem;">
                <b>Wind Speed:</b> {wind_speed:.1f} km/h {'⚠️ STORMY!' if wind_speed > 20 else '🌬️ Breezy'}<br>
                <b>Wind Direction:</b> {wind_direction}° (coming from this direction) 🧭<br>
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
            st.session_state.live_data = live_data

        st.success(
            f"✅ DATA RETRIEVED! Time: {live_data['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")

    if 'live_data' in st.session_state:
        live_data = st.session_state.live_data

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

            if st.button("🔍 CLASSIFY THIS PLOT! 🔍", type="primary", use_container_width=True, key="live_classify"):
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [
                                         0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0)

                discharge_threshold = 0.5
                is_freshet = int(
                    live_data['fraser_level'] > discharge_threshold)

                env_tensor = torch.tensor([[
                    live_data['wind_speed'] / 60.0,
                    live_data['wind_dir'] / 360.0,
                    live_data['tidal_level'] / 6.0,
                    abs(live_data['tidal_range']) / 3.0,
                    live_data['tidal_velocity'] / 1.0,
                    live_data['fraser_level'] / 1.0,
                    live_data['pressure'] / 1050.0,
                    30.0 / 100.0,
                    float(is_freshet),
                    live_data['wind_speed'] / 60.0,
                    abs(live_data['tidal_range']) / 3.0
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
                    st.metric("🌊 TIDES", f"{probs[1].item()*100:.1f}%")
                with col3:
                    st.metric("💨 WIND/STORM", f"{probs[2].item()*100:.1f}%")
    else:
        st.info("👆 Click the button above to fetch fresh live data first!")

# ============================================
# TAB 3: OCEAN SIMULATOR (EMBEDDED)
# ============================================
with tab3:
    st.markdown('<h2 style="text-align: center;">🌊 OCEAN CURRENT SIMULATOR 🌊</h2>',
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #FFD700;">Interactive visualization of ocean currents and dynamics</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.components.v1.iframe(
        src="https://udbhav25kansal.github.io/Ocean-current-simulator/",
        height=800,
        scrolling=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("""
    🌍 **Ocean Current Simulator Features:**
    - Interactive visualization of surface currents
    - Seasonal variations
    
    [Open in new tab](https://udbhav25kansal.github.io/Ocean-current-simulator/)
    """)

# ============================================
# TAB 4: PERFORMANCE
# ============================================
with tab4:
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
            <b>Framework:</b> PyTorch 2.5.1 🔥<br>
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
            <b>Resolution:</b> 2 Hours ⏰<br><br>
            <b>Class Distribution:</b><br>
            🏞️ OUTFLOW: 7.0% (572)<br>
            🌊 TIDES: 40.9% (3,338)<br>
            💨 WIND/STORM: 52.1% (4,258)
            </p>
        </div>
        """, unsafe_allow_html=True)

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
    <p style="font-size: 1.2rem;">⚓ OceanHack 2025 • BC ⚓</p>
    <p style="font-size: 1rem;">🧽Team Spongebob</p>
    <p style="font-size: 0.9rem; font-style: italic;">"The inner machinations of my mind are an enigma." - Patrick Star, Data Scientist 🌟</p>
</div>
""", unsafe_allow_html=True)
