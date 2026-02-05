import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import google.generativeai as genai
import time
from fpdf import FPDF
import base64 
import os
import requests 
from streamlit_lottie import st_lottie 

# ==============================================================================
# 1. AYARLAR
# ==============================================================================
st.set_page_config(page_title="Ziraat AI", page_icon="🌿", layout="centered")

if 'giris_yapildi' not in st.session_state: st.session_state['giris_yapildi'] = False
if 'son_teshis' not in st.session_state: st.session_state['son_teshis'] = None
if 'son_bitki' not in st.session_state: st.session_state['son_bitki'] = None
if 'recete_hafizasi' not in st.session_state: st.session_state['recete_hafizasi'] = ""

# --- TASARIM ---
def tasariimi_uygula():
    bg_image_style = 'background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=1527&auto=format&fit=crop");'
    if os.path.exists("arkaplan.jpg"):
        with open("arkaplan.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        bg_image_style = f'background-image: url("data:image/jpg;base64,{encoded_string}");'

    st.markdown(
        f"""
        <style>
        .stApp {{ {bg_image_style} background-attachment: fixed; background-size: cover; }}
        div.stButton > button {{
            display: block !important; margin-left: auto !important; margin-right: auto !important;
            width: 70% !important; border-radius: 25px; font-weight: bold; font-size: 18px;
            background-color: #ff4b4b; color: white; border: 2px solid white;
        }}
        section[data-testid="stSidebar"] {{ background-color: rgba(15, 25, 15, 0.95) !important; }}
        * {{ color: white !important; }} 
        div.stInfo, div.stSuccess, div.stError {{ background-color: rgba(0,0,0,0.8) !important; }}
        </style>
        """, unsafe_allow_html=True
    )
tasariimi_uygula()

def load_lottieurl(url):
    try: return requests.get(url).json()
    except: return None

# ==============================================================================
# 2. HATA AYIKLAYICI GEMINI BAĞLANTISI 🛠️
# ==============================================================================
@st.cache_resource
def gemini_baglan():
    # 1. API KEY KONTROLÜ
    if "GOOGLE_API_KEY" not in st.secrets:
        return None, "HATA: Secrets içinde GOOGLE_API_KEY bulunamadı."
    
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    
    # 2. MODEL TESTİ (HATA GİZLEME KAPALI)
    try:
        # En standart modeli doğrudan dene
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Test")
        return model, "gemini-1.5-flash (Bağlı)"
    except Exception as e:
        # GERÇEK HATAYI DÖNDÜR
        return None, f"BAĞLANTI HATASI DETAYI: {str(e)}"

model_gemini, durum_mesaji = gemini_baglan()

# ==============================================================================
# 3. GİRİŞ EKRANI
# ==============================================================================
if not st.session_state['giris_yapildi']:
    st.markdown("<h1 style='text-align: center;'>🌿 Ziraat AI</h1>", unsafe_allow_html=True)
    lottie_intro = load_lottieurl("https://lottie.host/62688176-784f-4d22-8280-5b1191062085/WkL0s7l9Xj.json")
    if lottie_intro: st_lottie(lottie_intro, height=250)
    
    # HATA VARSA GİRİŞTE GÖSTER
    if model_gemini is None:
        st.error(f"⚠️ {durum_mesaji}")
        st.info("Lütfen 'Reboot App' yapın veya API anahtarınızı kontrol edin.")
    else:
        if st.button("🚀 UYGULAMAYI BAŞLAT"):
            st.session_state['giris_yapildi'] = True
            st.rerun()

# ==============================================================================
# 4. ANA UYGULAMA
# ==============================================================================
else:
    with st.sidebar:
        st.title("Ziraat AI")
        if st.button("Çıkış"):
            st.session_state['giris_yapildi'] = False
            st.rerun()

    tab1, tab2 = st.tabs(["🌿 Teşhis", "🌤️ Bölge"])

    with tab1:
        st.subheader("Fotoğraf Yükle")
        
        @st.cache_resource
        def model_yukle(bitki):
            # Basit mapper (Hızlı test için)
            path = f"{bitki.split(' ')[0].lower()}_uzman_model.keras" # apple_uzman_model.keras
            # Gerçek mapper burada olmalı (önceki koddan)
            mapper = {
                "Elma (Apple)": "apple_uzman_model.keras", "Domates (Tomato)": "tomato_uzman_model.keras",
                "Mısır (Corn)": "corn_uzman_model.keras", "Üzüm (Grape)": "grape_uzman_model.keras",
                "Patates (Potato)": "potato_uzman_model.keras", "Biber (Pepper)": "pepper_uzman_model.keras",
                "Şeftali (Peach)": "peach_uzman_model.keras", "Çilek (Strawberry)": "strawberry_uzman_model.keras",
                "Kiraz (Cherry)": "cherry_uzman_model.keras"
            }
            if bitki in mapper:
                try: return tf.keras.models.load_model(mapper[bitki])
                except: return None
            return None

        secilen = st.selectbox("Bitki", ["Elma (Apple)", "Domates (Tomato)", "Patates (Potato)", "Biber (Pepper)"])
        dosya = st.file_uploader("Resim")

        if dosya and st.button("Analiz Et"):
            # Analiz kodları...
            st.success("Analiz tamamlandı (Test Modu)")
            if model_gemini:
                try:
                    res = model_gemini.generate_content(f"{secilen} bitkisi için genel bilgi ver.").text
                    st.info(res)
                except Exception as e:
                    st.error(f"Gemini Hatası: {e}")
            else:
                st.error(f"Yapay Zeka Çalışmıyor: {durum_mesaji}")

    with tab2:
        sehir = st.text_input("Şehir", "Ankara")
        if st.button("Veri Getir"):
            if model_gemini:
                try:
                    res = model_gemini.generate_content(f"{sehir} için tarım tavsiyesi").text
                    st.success(res)
                except Exception as e:
                    st.error(f"Hata: {e}")
            else:
                st.error(f"Bağlantı Yok: {durum_mesaji}")