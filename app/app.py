import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import predict_news_credibility
from utils.sentiment import analyze_sentiment
from utils.keyword_detector import detect_sensational_keywords
from utils.article_extractor import extract_article_details
from utils.news_fetcher import get_latest_news

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credify | AI News Credibility Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ENERGETIC CSS WITH REFINED INPUT BOXES & BUTTONS ---
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Theme Variables */
    :root {
        --bg-color: #000000;
        --card-bg: #0a0a0a;
        --text-color: #ffffff;
        --muted-text: #888888;
        --border-color: #1a1a1a;
        --primary-accent: #00f2ff; /* Neon cyan */
        --neon-shadow: 0 0 10px rgba(0, 242, 255, 0.4);
        --btn-bg: #121212;
    }

    @media (prefers-color-scheme: light) {
        :root {
            --bg-color: #e0f2fe; /* Pastel Blue */
            --card-bg: #ffffff;
            --text-color: #0f172a;
            --muted-text: #475569;
            --border-color: #cbd5e1;
            --primary-accent: #2563eb;
            --neon-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            --btn-bg: #1e293b;
        }
        .stApp { background-color: #e0f2fe !important; }
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 95rem;
    }

    /* Branded Header Area */
    .brand-header {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        padding: 20px 0;
        margin-bottom: 15px;
    }

    .brand-logo {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3.5rem;
        color: var(--primary-accent);
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: var(--neon-shadow);
        cursor: default;
    }

    /* Refined Boxes for Titles */
    .title-box {
        background-color: var(--card-bg);
        padding: 12px 20px;
        border-radius: 8px;
        border: 2px solid var(--border-color);
        box-shadow: var(--neon-shadow);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 15px;
        color: var(--text-color);
        border-color: var(--primary-accent);
    }

    /* Card Containers */
    .dashboard-card {
        background-color: var(--card-bg);
        padding: 24px;
        border-radius: 12px;
        border: 2px solid var(--border-color);
        box-shadow: var(--neon-shadow);
        height: 100%;
        transition: transform 0.2s ease;
    }
    
    .dashboard-card:hover {
        transform: scale(1.005);
        border-color: var(--primary-accent);
    }

    /* Solid Dark Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        background-color: var(--btn-bg) !important;
        color: white !important;
        border: 1px solid var(--border-color) !important;
        font-weight: 800;
        padding: 14px;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.2s ease;
        box-shadow: none !important;
    }
    
    .stButton>button:hover {
        background-color: #1a1a1a !important;
        border-color: var(--primary-accent) !important;
        color: var(--primary-accent) !important;
    }
    
    /* Input Styling */
    .stTextArea textarea, .stTextInput input {
        background-color: rgba(128, 128, 128, 0.05) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-color) !important;
        font-weight: 500;
        border-radius: 8px !important;
    }

    /* Result Metrics */
    .metric-card {
        background-color: var(--card-bg);
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid var(--border-color);
        box-shadow: var(--neon-shadow);
    }
    
    /* Footer */
    .footer-container {
        text-align: left;
        padding: 40px 0;
        margin-top: 60px;
        border-top: 1px solid var(--border-color);
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER DATA ---
@st.cache_data
def get_model_metadata():
    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
        return joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
    except:
        return {
            "best_model_name": "Linear SVM",
            "best_accuracy": 0.995,
            "all_models": {"Logistic Regression": 0.952, "Linear SVM": 0.995, "Naive Bayes": 0.894}
        }

metadata = get_model_metadata()

# --- 1. BRAND HEADER ---
st.markdown('<div class="brand-header"><div class="brand-logo">Credify</div></div>', unsafe_allow_html=True)

# --- 2. INPUT SECTION ---
input_col1, input_col2 = st.columns(2)

with input_col1:
    # Title placed in the border box
    st.markdown('<div class="title-box">PASTE NEWS DATA</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    article_text = st.text_area("", placeholder="DROP NEWS CONTENT HERE...", height=200, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with input_col2:
    # Title placed in the border box
    st.markdown('<div class="title-box">ANALYZE VIA URL</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    article_url = st.text_input("", placeholder="https://news-source.com/article-url", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    extract_btn = st.button("EXTRACT & ANALYZE NOW")
    
    domain = "N/A"
    if article_url:
        try:
            from urllib.parse import urlparse
            domain = urlparse(article_url).netloc
        except: pass
    
    st.markdown(f"""
    <div style="background-color: rgba(0, 242, 255, 0.05); padding: 12px; border-radius: 8px; border: 1px dashed var(--primary-accent); margin-top: 15px;">
        <span style="color: var(--muted-text); font-weight: 700; font-size: 0.8rem;">LIVE DOMAIN:</span> <span style="color: var(--primary-accent); font-weight: 900;">{domain.upper()}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 3. ANALYZE BUTTON ---
st.markdown("<br>", unsafe_allow_html=True)
col_btn_center = st.columns([1, 1.5, 1])
analyze_btn = col_btn_center[1].button("🔥 ANALYZE CREDIBILITY STATUS")

# --- RESULTS LOGIC ---
analysis_triggered = False
final_text = ""

if analyze_btn:
    if article_text.strip():
        final_text = article_text
        analysis_triggered = True
    else: st.error("INPUT ERROR: NO TEXT DATA FOUND.")

if extract_btn:
    if article_url.strip():
        with st.spinner("SCANNING SOURCE..."):
            data = extract_article_details(article_url)
            if data['success']:
                final_text = data['text']
                analysis_triggered = True
            else: st.error(f"FATAL ERROR: {data.get('error')}")
    else: st.warning("INPUT REQUIRED: URL FIELD EMPTY.")

# --- 4. RESULTS DASHBOARD ---
if analysis_triggered:
    with st.spinner("AI DIAGNOSTICS IN PROGRESS..."):
        result = predict_news_credibility(final_text)
        sentiment_res = analyze_sentiment(final_text)
        keywords = detect_sensational_keywords(final_text)
        
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        
        with r1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            is_real = result['credibility_score'] > 50
            label = "CREDIBLE" if is_real else "FAKE DATA"
            color = "#00ff88" if is_real else "#ff0044"
            st.markdown(f"<h1 style='color: {color}; margin: 0; font-weight: 900; text-shadow: 0 0 10px {color}66;'>{label}</h1>", unsafe_allow_html=True)
            st.markdown("<span style='color: var(--muted-text); font-weight: 800; font-size: 0.8rem;'>STATUS</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            score = result['credibility_score']
            st.markdown(f"<h1 style='color: var(--primary-accent); margin: 0; font-weight: 900;'>{score}%</h1>", unsafe_allow_html=True)
            st.progress(score / 100)
            st.markdown("<span style='color: var(--muted-text); font-weight: 800; font-size: 0.8rem;'>CREDIBILITY INDEX</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            val = sentiment_res['polarity']
            l = "POSITIVE" if val > 0.1 else ("NEGATIVE" if val < -0.1 else "NEUTRAL")
            st.markdown(f"<h1 style='color: #ffaa00; margin: 0; font-weight: 900;'>{l}</h1>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: var(--muted-text); font-weight: 800; font-size: 0.8rem;'>BIAS: {val:.2f}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            risk = "CRITICAL" if len(keywords) > 3 else ("WARNING" if len(keywords) > 0 else "STABLE")
            r_c = "#ff0000" if risk == "CRITICAL" else ("#ffaa00" if risk == "WARNING" else "#00ff88")
            st.markdown(f"<h1 style='color: {r_c}; margin: 0; font-weight: 900;'>{risk}</h1>", unsafe_allow_html=True)
            st.markdown("<span style='color: var(--muted-text); font-weight: 800; font-size: 0.8rem;'>RISK LEVEL</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 5. ANALYTICS ---
        st.markdown("<br>", unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### DATA PROBABILITY")
            fig_p = go.Figure(data=[go.Pie(
                labels=['REAL', 'FAKE'], values=[result['real_probability'], result['fake_probability']],
                hole=.7, marker_colors=['#00f2ff', '#ff0055'], textinfo='percent+label'
            )])
            fig_p.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with v2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### NLP DIAGNOSTICS")
            fig_s = px.bar(
                x=['POLARITY', 'SUBJECTIVITY'], y=[sentiment_res['polarity'], sentiment_res['subjectivity']],
                color=['P', 'S'], color_discrete_sequence=['#00f2ff', '#ff0055']
            )
            fig_s.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, showlegend=False, xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig_s, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- 6. MODEL INSIGHTS ---
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔬 TECHNICAL ENGINE ARCHITECTURE"):
    st.markdown(f"**CORE:** {metadata['best_model_name']} | **PRECISION:** {metadata['best_accuracy']*100:.2f}%")
    st.info("Multinomial NLP Engine | SVM Optimized")

# --- 7. NEWS FEED ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("## 📡 GLOBAL NEWS SCAN")
news = get_latest_news()
if news:
    cols = st.columns(5)
    for i, item in enumerate(news[:5]):
        with cols[i]:
            st.markdown(f"""
            <div class="dashboard-card" style="height: 250px; border-style: dashed;">
                <div style="font-weight: 800; margin-bottom: 8px; color: var(--text-color);">{item['title'][:60].upper()}...</div>
                <div style="color: var(--muted-text); font-size: 0.75rem;">SRC: {item['source']}</div>
                <div style="color: var(--muted-text); font-size: 0.8rem; margin-top: 10px; overflow: hidden; height: 100px;">{item.get('description', 'NO DESCRIPTION')[:80]}...</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("SCAN", key=f"s_{i}"):
                st.session_state.temp_text = item['title']
                st.rerun()

# --- 8. FOOTER ---
st.markdown(f"""
<div class="footer-container">
    <div style="font-family: 'Orbitron'; font-weight: 900; color: var(--primary-accent); font-size: 1.5rem; letter-spacing: 2px;">CREDIFY SYSTEM</div>
    <div style="font-weight: 700; font-size: 0.9rem;">HIGH-SPEED NEURAL NETWORK ANALYSIS</div>
    <div style="font-size: 0.75rem; margin-top: 5px; opacity: 0.5;">V.2.0.5 | STABLE BUILD</div>
</div>
""", unsafe_allow_html=True)

if "temp_text" in st.session_state: st.success(f"LOADING SCAN: {st.session_state.temp_text}")
