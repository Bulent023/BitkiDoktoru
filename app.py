import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
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

# --- CSS ---
def tasariimi_uygula():
    bg_image_style = 'background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=1527&auto=format&fit=crop");'
    if os.path.exists("arka_plan.jpg"):
        with open("arka_plan.jpg", "rb") as image_file:
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
        * {{ color: white; }}
        div.stError {{ background-color: rgba(255, 0, 0, 0.8) !important; color: white !important; font-weight: bold; }}
        </style>
        """, unsafe_allow_html=True
    )
tasariimi_uygula()

def load_lottieurl(url):
    try: return requests.get(url).json()
    except: return None

def tr_duzelt(text):
    if not isinstance(text, str): text = str(text)
    # Basit temizlik
    return text.replace("İ", "I").replace("ı", "i").replace("Ğ", "G").replace("ğ", "g").replace("Ş", "S").replace("ş", "s")

def create_pdf(bitki, hastalik, recete):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="RAPOR", ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=tr_duzelt(f"Bitki: {bitki}\nTeshis: {hastalik}\n\n{recete}"))
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ==============================================================================
# 2. HATA AVLAYICI GEMINI FONKSİYONU 🕵️‍♂️
# ==============================================================================
def gemini_sor(prompt):
    if "GOOGLE_API_KEY" not in st.secrets:
        return "KRİTİK HATA: Secrets içinde GOOGLE_API_KEY bulunamadı! Lütfen ayarlardan ekleyin."
    
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # Bu modelleri sırayla deneyecek ve hataları biriktirecek
    modeller = ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    hata_raporu = [] # Hataları burada toplayacağız
    
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    for model_ismi in modeller:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_ismi}:generateContent?key={api_key}"
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                # Hata kodunu kaydet (Örn: 400 Bad Request, 403 Permission Denied)
                hata_detayi = response.json().get('error', {}).get('message', response.text)
                hata_raporu.append(f"❌ {model_ismi}: Kod {response.status_code} - {hata_detayi}")
                
        except Exception as e:
            hata_raporu.append(f"❌ {model_ismi}: Bağlantı Hatası - {str(e)}")

    # Eğer buraya geldiyse hepsi başarısız olmuştur.
    return "TÜM MODELLER BAŞARISIZ OLDU:\n" + "\n".join(hata_raporu)

# ==============================================================================
# 3. ARAYÜZ
# ==============================================================================
if not st.session_state['giris_yapildi']:
    st.write("")
    st.markdown("<h1 style='text-align: center;'>🌿 Ziraat AI</h1>", unsafe_allow_html=True)
    lottie_intro = load_lottieurl("https://lottie.host/62688176-784f-4d22-8280-5b1191062085/WkL0s7l9Xj.json")
    if lottie_intro: st_lottie(lottie_intro, height=250)
    
    # GİRİŞTE TEST ET (Anahtarı hemen doğrula)
    if st.button("🚀 BAŞLAT VE TEST ET"):
        test_cevap = gemini_sor("Test")
        if "TÜM MODELLER BAŞARISIZ" in test_cevap:
            st.error(test_cevap) # Detaylı hatayı göster
            st.warning("Lütfen API anahtarınızın doğru olduğundan ve 'Secrets' kısmına kaydedildiğinden emin olun.")
        elif "KRİTİK HATA" in test_cevap:
            st.error(test_cevap)
        else:
            st.success("Bağlantı Başarılı!")
            time.sleep(1)
            st.session_state['giris_yapildi'] = True
            st.rerun()

else:
    with st.sidebar:
        st.title("Ziraat AI")
        if st.button("Çıkış"):
            st.session_state['giris_yapildi'] = False
            st.rerun()

    tab1, tab2 = st.tabs(["Teşhis", "Bölge"])

    with tab1:
        # Basitleştirilmiş Model Yükleme
        @st.cache_resource
        def model_yukle(bitki):
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
            
        secilen = st.selectbox("Bitki", ["Elma (Apple)", "Domates (Tomato)", "Patates (Potato)"])
        dosya = st.file_uploader("Resim")
        
        if dosya and st.button("Analiz Et"):
             # (Model tahmin kısmı burada normal çalışacak, sadece Gemini kısmını test ediyoruz)
             st.success("Analiz Simülasyonu Başarılı") 
             with st.spinner("Reçete yazılıyor..."):
                 cevap = gemini_sor(f"{secilen} bitkisi hakkında kısa bilgi ver.")
                 if "BAŞARISIZ" in cevap:
                     st.error(cevap)
                 else:
                     st.info(cevap)

    with tab2:
        sehir = st.text_input("Şehir", "Antalya")
        if st.button("Veri Getir"):
            with st.spinner("Takvim hazırlanıyor..."):
                cevap = gemini_sor(f"{sehir} için tarım takvimi")
                if "BAŞARISIZ" in cevap:
                    st.error(cevap)
                else:
                    st.success(cevap)