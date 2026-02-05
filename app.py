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
# 1. AYARLAR VE GÖRSEL TASARIM
# ==============================================================================
st.set_page_config(page_title="Ziraat AI", page_icon="🌿", layout="centered")

# --- SESSION STATE ---
if 'giris_yapildi' not in st.session_state:
    st.session_state['giris_yapildi'] = False

# --- ARKA PLAN VE CSS ---
def tasariimi_uygula():
    dosya_adi = "arkaplan.jpg"
    bg_image_style = ""
    
    if os.path.exists(dosya_adi):
        with open(dosya_adi, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        bg_image_style = f'background-image: url("data:image/jpg;base64,{encoded_string}");'
    else:
        bg_image_style = 'background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=1527&auto=format&fit=crop");'

    st.markdown(
        f"""
        <style>
        .stApp {{
            {bg_image_style}
            background-attachment: fixed;
            background-size: cover;
        }}
        
        /* --- DÜZELTİLEN KISIM: BUTON ORTALAMA (FLEXBOX) --- */
        /* Butonun kapsayıcısını esnek kutu yap ve ortala */
        .stButton {{
            display: flex;
            justify-content: center;
        }}
        
        /* Butonun kendisinin özellikleri */
        .stButton > button {{
            width: auto !important;     /* Genişlik içeriğe göre olsun */
            min-width: 250px;           /* Ama çok da küçülmesin */
            max-width: 350px;           /* Çok da büyümesin */
            border-radius: 25px;        /* Tam oval kenarlar */
            font-weight: bold;
            font-size: 18px;
            padding: 12px 24px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.4); /* Derinlik gölgesi */
            border: 2px solid white;
            transition: transform 0.2s; /* Tıklama efekti için */
        }}
        
        .stButton > button:active {{
            transform: scale(0.95); /* Tıklayınca hafif küçülsün */
        }}
        /* -------------------------------------------------- */

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 25, 15, 0.95) !important;
            border-right: 3px solid #4CAF50;
        }}
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, p, label {{
            color: white !important;
        }}
        /* Input */
        input[type="text"] {{
            color: white !important;
        }}
        div[data-baseweb="input"] {{
            background-color: rgba(20, 40, 20, 0.8) !important;
            border: 1px solid #4CAF50;
        }}
        /* Sekme ve Expander */
        div[data-testid="stExpander"] {{
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            border-radius: 10px;
        }}
        div[data-testid="stTabs"] button[aria-selected="true"] {{
            background-color: #4CAF50;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

tasariimi_uygula()

# --- ANİMASYON FONKSİYONU ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# ==============================================================================
# 2. GİRİŞ EKRANI (SPLASH SCREEN) 🎯
# ==============================================================================
if not st.session_state['giris_yapildi']:
    st.write("")
    st.write("") 
    
    # Başlıklar
    st.markdown("<h1 style='text-align: center; color: white; font-size: 50px; text-shadow: 3px 3px 6px #000000;'>🌿 Ziraat AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #e8f5e9; text-shadow: 1px 1px 2px #000000;'>Çiftçinin Dijital Asistanı</h3>", unsafe_allow_html=True)
    
    # Animasyon
    lottie_intro = load_lottieurl("https://lottie.host/62688176-784f-4d22-8280-5b1191062085/WkL0s7l9Xj.json")
    if lottie_intro:
        st_lottie(lottie_intro, height=250, key="intro_anim")
    
    st.write("") 
    st.write("") 
    
    # --- BUTON KISMI (SÜTUNSUZ - CSS İLE ORTALANDI) ---
    # Artık columns kullanmıyoruz, CSS otomatik ortalıyor.
    if st.button("🚀 UYGULAMAYI BAŞLAT", type="primary"):
        st.session_state['giris_yapildi'] = True
        st.rerun()
    # --------------------------------------------------

# ==============================================================================
# 3. ANA UYGULAMA 🏗️
# ==============================================================================
else:
    # --- GEMINI BAĞLANTISI ---
    @st.cache_resource
    def gemini_baglan():
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
                genai.configure(api_key=api_key)
                oncelikli = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
                for m in oncelikli:
                    try:
                        test = genai.GenerativeModel(m)
                        test.generate_content("Check") 
                        return test, m 
                    except: continue
                return None, "Model Yok"
            return None, "Anahtar Yok"
        except Exception as e: return None, str(e)

    model_gemini, aktif_model_ismi = gemini_baglan()
    
    # --- YAN MENÜ ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
        st.title("Ziraat AI")
        st.caption(f"Aktif Model: {aktif_model_ismi}")
        
        st.markdown("---")
        if st.button("🔙 Çıkış Yap"):
            st.session_state['giris_yapildi'] = False
            st.rerun()

    # --- ANA BAŞLIK ---
    st.title("🌿 Akıllı Bitki Doktoru")

    # --- SEKMELER ---
    tab1, tab2, tab3 = st.tabs(["🌿 Hastalık Teşhisi", "🌤️ Bölgesel Veriler", "ℹ️ Yardım"])

    # --- SEKME 1: TEŞHİS ---
    with tab1:
        st.markdown("### 📸 Fotoğraf Yükle")
        
        # Model Yükleyici
        @st.cache_resource
        def model_yukle(bitki):
            mapper = {
                "Elma (Apple)": "apple_uzman_model.keras",
                "Domates (Tomato)": "tomato_uzman_model.keras",
                "Mısır (Corn)": "corn_uzman_model.keras",
                "Üzüm (Grape)": "grape_uzman_model.keras",
                "Patates (Potato)": "potato_uzman_model.keras",
                "Biber (Pepper)": "pepper_uzman_model.keras",
                "Şeftali (Peach)": "peach_uzman_model.keras",
                "Çilek (Strawberry)": "strawberry_uzman_model.keras",
                "Kiraz (Cherry)": "cherry_uzman_model.keras"
            }
            if bitki in mapper:
                try: return tf.keras.models.load_model(mapper[bitki])
                except: return None
            return None
        
        def siniflari_al(bitki):
             if bitki == "Elma (Apple)": return ['Kara Leke', 'Kara Çürüklük', 'Pas', 'Sağlıklı']
             return ["Hastalık", "Sağlıklı"]

        col_a, col_b = st.columns(2)
        with col_a:
            secilen_bitki = st.selectbox("Bitki:", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)"])
        with col_b:
            dosya = st.file_uploader("Resim:", type=["jpg","png"])

        if dosya:
            image = Image.open(dosya)
            st.image(image, width=300)
            if st.button("🔍 Analiz Et", type="primary"):
                with st.spinner("İnceleniyor..."):
                    model = model_yukle(secilen_bitki)
                    if model:
                        img = image.resize((160,160))
                        img_arr = np.array(img).astype("float32")
                        if img_arr.ndim==2: img_arr=np.stack((img_arr,)*3, axis=-1)
                        elif img_arr.shape[-1]==4: img_arr=img_arr[:,:,:3]
                        img_arr = img_arr[...,::-1] # BGR
                        input_data = np.expand_dims(img_arr, axis=0)
                        
                        try:
                            tahmin = model.predict(input_data)
                            idx = np.argmax(tahmin)
                            siniflar = siniflari_al(secilen_bitki)
                            sonuc = siniflar[idx] if idx < len(siniflar) else "Tespit Edildi"
                            
                            if "Sağlıklı" in sonuc:
                                st.success(f"**Durum:** {sonuc}")
                                st.balloons()
                            else:
                                st.error(f"**Durum:** {sonuc}")
                                if model_gemini:
                                    res = model_gemini.generate_content(f"{secilen_bitki} bitkisinde {sonuc} hastalığı için kısa tedavi önerisi yaz.")
                                    st.info(res.text)
                                    
                            st.session_state['son_teshis'] = sonuc
                            st.session_state['son_bitki'] = secilen_bitki
                        except: st.error("Model tahmin hatası.")

        # Sohbet
        if 'son_teshis' in st.session_state and model_gemini:
            st.markdown("---")
            soru = st.text_input("Asistana sor:")
            if st.button("Sor"):
                res = model_gemini.generate_content(f"Bitki: {st.session_state['son_bitki']}, Hastalık: {st.session_state['son_teshis']}, Soru: {soru}")
                st.write(res.text)

    # --- SEKME 2: BÖLGE ---
    with tab2:
        st.header("🌤️ Bölgesel Veriler")
        sehir = st.text_input("Şehir:", value="Antalya")
        if st.button("Getir"):
             try:
                geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={sehir}&count=1").json()
                lat = geo["results"][0]["latitude"]
                lon = geo["results"][0]["longitude"]
                w = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m").json()["current"]
                
                c1, c2 = st.columns(2)
                c1.metric("Sıcaklık", f"{w['temperature_2m']} °C")
                c2.metric("Rüzgar", f"{w['wind_speed_10m']} km/s")
                
                if model_gemini:
                    takvim = model_gemini.generate_content(f"Şu an {time.strftime('%B')} ayındayız, yer {sehir}. Çiftçiler ne yapmalı? Kısa özet.")
                    st.success(takvim.text)
             except: st.error("Veri alınamadı.")

    # --- SEKME 3: YARDIM ---
    with tab3:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.9); padding: 25px; border-radius: 15px; border-left: 5px solid #4CAF50; color: black;">
            <h2 style="color: #1b5e20; margin-top: 0;">❓ Nasıl Kullanılır?</h2>
            <p style="font-size: 16px;">
                <b>Adım 1:</b> <code>Teşhis</code> sekmesinden bitkiyi seçin.<br>
                <b>Adım 2:</b> Fotoğraf yükleyin.<br>
                <b>Adım 3:</b> <b>"Analiz Et"</b> butonuna basın.<br>
                <hr>
                <b>Not:</b> Çıkış yapmak için soldaki menüyü kullanabilirsiniz.
            </p>
        </div>
        """, unsafe_allow_html=True)