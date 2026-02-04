import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from datetime import datetime
from fpdf import FPDF
import pandas as pd
import google.generativeai as genai
import time

# ==========================================
# 🔑 GOOGLE API KEY
# ==========================================
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg"

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    pass

# --- AKILLI MODEL SEÇİCİ ---
@st.cache_resource
def get_auto_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini' in m.name:
                    return genai.GenerativeModel(m.name)
        return genai.GenerativeModel('gemini-pro')
    except:
        return None

# ==========================================
# 1. AYARLAR VE TASARIM
# ==========================================
st.set_page_config(
    page_title="Ziraat AI",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- VERİTABANI ---
DB_DOSYASI = "tarama_gecmisi.csv"
GERI_BILDIRIM_KLASORU = "yeni_veri_havuzu"
if not os.path.exists(GERI_BILDIRIM_KLASORU): os.makedirs(GERI_BILDIRIM_KLASORU)

def veritabani_yukle():
    if os.path.exists(DB_DOSYASI): return pd.read_csv(DB_DOSYASI)
    return pd.DataFrame(columns=["Tarih", "Bitki", "Not", "Teshis", "Guven"])

def kayit_ekle(bitki, notlar, teshis, guven):
    df = veritabani_yukle()
    yeni_kayit = pd.DataFrame([{
        "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Bitki": bitki, "Not": notlar, "Teshis": teshis, "Guven": f"%{guven:.1f}"
    }])
    df = pd.concat([df, yeni_kayit], ignore_index=True)
    df.to_csv(DB_DOSYASI, index=False)

def hatali_veriyi_kaydet(image, dogru_sinif_klasoru, bitki_adi):
    hedef_yol = os.path.join(GERI_BILDIRIM_KLASORU, bitki_adi, dogru_sinif_klasoru)
    if not os.path.exists(hedef_yol): os.makedirs(hedef_yol)
    dosya_adi = datetime.now().strftime("%Y%m%d_%H%M%S_corrected.jpg")
    tam_yol = os.path.join(hedef_yol, dosya_adi)
    image.save(tam_yol)
    return tam_yol

# --- PDF ---
def create_pdf(bitki, hastalik, recete, notlar, tarih):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    def clean_text(text):
        mapping = {"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c", "İ": "I", "Ğ": "G", "Ü": "U", "Ş": "S", "Ö": "O", "Ç": "C"}
        for k, v in mapping.items(): text = text.replace(k, v)
        return text.encode('latin-1', 'ignore').decode('latin-1')

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="ZIRAAT AI - RAPOR", ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=clean_text(f"Tarih: {tarih}"), ln=1)
    pdf.cell(200, 10, txt=clean_text(f"Bitki: {bitki}"), ln=1)
    pdf.cell(200, 10, txt=clean_text(f"Konum: {notlar}"), ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=clean_text(f"TESHIS: {hastalik}"), ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="ONERILER:", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, txt=clean_text(recete))
    return pdf.output(dest='S').encode('latin-1')

# --- TASARIM ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f: data = f.read()
    return base64.b64encode(data).decode()

RESIM_DOSYASI = "arkaplan.jpg"
if os.path.exists(RESIM_DOSYASI):
    bin_str = get_base64_of_bin_file(RESIM_DOSYASI)
    bg_style = f"""background-image: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), url("data:image/jpg;base64,{bin_str}");"""
else:
    bg_style = """background: linear-gradient(to bottom, #111, #0f3d3e);"""

st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{ {bg_style} background-size: cover; background-position: center; background-attachment: fixed; }}
    [data-testid="stHeader"] {{ display: none; }}
    .stButton>button {{ height: 3.5em; border-radius: 20px; font-size: 18px; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
    .stFileUploader>div>div>button {{ background-color: rgba(255,255,255,0.1); color: white; border: 1px solid white; }}
    p, h1, h2, h3, label, .stMarkdown, .stRadio label, .stFileUploader label {{ color: #EEE !important; }}
    .chat-user {{ background-color: rgba(46, 125, 50, 0.2); padding: 10px; border-radius: 10px; margin: 5px; text-align: right; border: 1px solid #4CAF50; }}
    .chat-ai {{ background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 10px; margin: 5px; text-align: left; border: 1px solid #666; }}
    </style>
""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 4])
with c1: st.write("🌱")
with c2: st.markdown("## Ziraat AI")

# ==========================================
# 2. AYARLAR (12 BİTKİ)
# ==========================================
AYARLAR = {
    "Elma 🍏": {
        "model": "apple_uzman_model.keras",
        "siniflar": ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
        "bilgi": {
            "Apple___Apple_scab": ["🍏 KARA LEKE", "Bakırlı ilaç ve Captan kullanın."],
            "Apple___Black_rot": ["⚫ SİYAH ÇÜRÜKLÜK", "Mumyalaşmış meyveleri toplayın."],
            "Apple___Cedar_apple_rust": ["🟠 ELMA PASI", "Fungisit uygulayın. Ardıç ağaçlarını kontrol edin."],
            "Apple___healthy": ["✅ SAĞLIKLI", "Ağaç sağlıklı."]
        }
    },
    "Yaban Mersini 🫐": {
        "model": "blueberry_uzman_model.keras",
        "siniflar": ['Blueberry___healthy'], # PlantVillage'de genelde sadece Healthy var
        "bilgi": {
            "Blueberry___healthy": ["✅ SAĞLIKLI", "Yaban mersini sağlıklı görünüyor."]
        }
    },
    "Mısır 🌽": {
        "model": "corn_uzman_model.keras",
        "siniflar": ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy'],
        "bilgi": {
            "Corn_(maize)___Common_rust_": ["🌽 MISIR PASI", "Dayanıklı çeşit ekin, genelde ilaç gerekmez."],
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ["🍂 GRİ YAPRAK LEKESİ", "Ekim nöbeti yapın, dayanıklı çeşit kullanın."],
            "Corn_(maize)___Northern_Leaf_Blight": ["🍂 KUZEY YAPRAK YANIKLIĞI", "Dayanıklı hibrit tohum kullanın."],
            "Corn_(maize)___healthy": ["✅ SAĞLIKLI", "Mısır gelişimi normal."]
        }
    },
    "Üzüm 🍇": {
        "model": "grape_uzman_model.keras",
        "siniflar": ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'],
        "bilgi": {
            "Grape___Black_rot": ["🍇 SİYAH ÇÜRÜKLÜK", "Kış budamasında hastalıklı çubukları temizleyin."],
            "Grape___Esca_(Black_Measles)": ["🍂 KAVLAMA (ESCA)", "Yara yerlerini kapatın, hasta gövdeyi temizleyin."],
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ["🍂 YAPRAK YANIKLIĞI", "Bakırlı ilaçlarla mücadele edin."],
            "Grape___healthy": ["✅ SAĞLIKLI", "Bağ sağlıklı."]
        }
    },
    "Portakal 🍊": {
        "model": "orange_uzman_model.keras",
        "siniflar": ['Orange___Haunglongbing_(Citrus_greening)'],
        "bilgi": {
            "Orange___Haunglongbing_(Citrus_greening)": ["🦠 YEŞİLLENME HASTALIĞI (HLB)", "Tedavisi yoktur. Ağacı söküp vektör böceklerle mücadele edin."]
        }
    },
    "Şeftali 🍑": {
        "model": "peach_uzman_model.keras",
        "siniflar": ['Peach___Bacterial_spot', 'Peach___healthy'],
        "bilgi": {
            "Peach___Bacterial_spot": ["🍑 BAKTERİYEL LEKE", "Sonbahar ve ilkbaharda bakır uygulaması şarttır."],
            "Peach___healthy": ["✅ SAĞLIKLI", "Şeftaliler sağlıklı."]
        }
    },
    "Biber 🫑": {
        "model": "pepper_uzman_model.keras",
        "siniflar": ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy'],
        "bilgi": {
            "Pepper,_bell___Bacterial_spot": ["🫑 BAKTERİYEL LEKE", "Bakır içerikli ilaç kullanın. Aşırı sulamadan kaçının."],
            "Pepper,_bell___healthy": ["✅ SAĞLIKLI", "Bitki sağlıklı. Bakıma devam."]
        }
    },
    "Patates 🥔": {
        "model": "potato_uzman_model.keras",
        "siniflar": ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
        "bilgi": {
            "Potato___Late_blight": ["🚨 MİLDİYÖ", "Tarlayı hemen ilaçlayın, yayılma çok hızlı olur."],
            "Potato___Early_blight": ["🍂 ERKEN YANIKLIK", "Bitkiyi güçlendirin, fungisit kullanın."],
            "Potato___healthy": ["✅ SAĞLIKLI", "Patatesler sağlıklı."]
        }
    },
    "Ahududu 🍇": {
        "model": "raspberry_uzman_model.keras",
        "siniflar": ['Raspberry___healthy'],
        "bilgi": {
            "Raspberry___healthy": ["✅ SAĞLIKLI", "Ahududular sağlıklı görünüyor."]
        }
    },
    "Soya Fasulyesi 🌱": {
        "model": "soybean_uzman_model.keras",
        "siniflar": ['Soybean___healthy'],
        "bilgi": {
            "Soybean___healthy": ["✅ SAĞLIKLI", "Soya fasulyesi sağlıklı."]
        }
    },
    "Çilek 🍓": {
        "model": "strawberry_uzman_model.keras",
        "siniflar": ['Strawberry___Leaf_scorch', 'Strawberry___healthy'],
        "bilgi": {
            "Strawberry___Leaf_scorch": ["🍓 YAPRAK YANIKLIĞI", "Kuru yaprakları temizleyin, Captan kullanın."],
            "Strawberry___healthy": ["✅ SAĞLIKLI", "Çilekler sağlıklı."]
        }
    },
    "Domates 🍅": {
        "model": "tomato_uzman_model.keras",
        "siniflar": ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'],
        "bilgi": {
            "Tomato___Late_blight": ["🚨 MİLDİYÖ", "Acil sistemik fungisit uygulayın. Serayı havalandırın."],
            "Tomato___Early_blight": ["🍂 ERKEN YANIKLIK", "Hasta yaprakları budayın. Azoxystrobin grubu ilaç kullanın."],
            "Tomato___Spider_mites Two-spotted_spider_mite": ["🕷️ KIRMIZI ÖRÜMCEK", "Akarisit uygulayın. Nem dengesini koruyun."],
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ["🦠 SARI YAPRAK KIVIRCIKLIĞI", "Beyaz sinek mücadelesi yapın. İlacı yoktur."],
            "Tomato___Bacterial_spot": ["🟤 BAKTERİYEL LEKE", "Bakırlı ilaçlar kullanın. Yaprakları ıslatmayın."],
            "Tomato___Leaf_Mold": ["🍄 YAPRAK KÜFÜ", "Havalandırmayı artırın. Uygun fungisit kullanın."],
            "Tomato___healthy": ["✅ SAĞLIKLI", "Domatesler sağlıklı."]
        }
    }
}

# ==========================================
# 3. ANA AKIŞ
# ==========================================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "son_analiz" not in st.session_state: st.session_state.son_analiz = None

# 1. GİRDİ ALANI
st.markdown("##### 1. Bitki ve Konum")
c_a, c_b = st.columns(2)
with c_a: secilen_bitki = st.selectbox("Bitki:", list(AYARLAR.keys()))
with c_b: notlar = st.text_input("Konum:", placeholder="Sera 1")

# 2. FOTOĞRAF ALANI
st.markdown("##### 2. Fotoğraf (Kamera veya Galeri)")
resim_dosyasi = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if resim_dosyasi:
    st.image(resim_dosyasi, caption="Yüklenen Görüntü", use_container_width=True)
    
    st.markdown("---")
    # ANALİZ BUTONU
    if st.button("🔍 ANALİZ ET", type="primary"):
        aktif_ayar = AYARLAR.get(secilen_bitki)
        
        if not aktif_ayar or not os.path.exists(aktif_ayar["model"]):
            st.error(f"⚠️ Model dosyası eksik: {aktif_ayar['model']}")
            st.warning("Bu bitki için eğitim.py ile model oluşturmalısınız.")
        else:
            with st.spinner("Yapay zeka inceliyor..."):
                model = tf.keras.models.load_model(aktif_ayar["model"])
                img_pil = Image.open(resim_dosyasi)
                img = img_pil.resize((160, 160))
                img_array = np.expand_dims(np.array(img), axis=0)
                
                tahmin = model.predict(img_array)
                idx = np.argmax(tahmin[0])
                guven = 100 * np.max(tf.nn.softmax(tahmin[0]))
                sinif = aktif_ayar["siniflar"][idx]
                
                varsayilan = [sinif, "Bilgi yok."]
                bilgi_paketi = aktif_ayar["bilgi"].get(sinif, varsayilan)
                teshis_basligi = bilgi_paketi[0]
                tedavi = bilgi_paketi[1]
                
                st.session_state.son_analiz = {
                    "bitki": secilen_bitki,
                    "hastalik": teshis_basligi,
                    "tedavi": tedavi
                }
                
                kayit_ekle(secilen_bitki, notlar, teshis_basligi, guven)

                renk = "#2e7d32" if "healthy" in sinif.lower() else "#d32f2f"
                ikon = "✅" if "healthy" in sinif.lower() else "🚨"
                
                st.markdown(f"""
                <div style="background-color: {renk}; padding: 20px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
                    <h2 style="margin:0; color:white;">{ikon} {teshis_basligi}</h2>
                    <p style="margin:5px; opacity:0.9;">Güven: %{guven:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"**💊 TEDAVİ:**\n{tedavi}")

    # --- SOHBET ---
    if st.session_state.son_analiz:
        analiz = st.session_state.son_analiz
        
        st.markdown("---")
        st.markdown("### 🤖 Asistan'a Sor")
        
        user_question = st.text_input("Sorunuz:", placeholder="Bu ilaç yağmurda akar mı?")

        if user_question:
            if st.button("Gönder"):
                gemini_model = get_auto_model()
                if gemini_model:
                    with st.spinner("Yazıyor..."):
                        try:
                            prompt = f"""
                            Sen bir Ziraat Mühendisisin.
                            Hastalık: {analiz['hastalik']}
                            Tedavi: {analiz['tedavi']}
                            Soru: {user_question}
                            Kısa, net ve samimi cevap ver.
                            """
                            response = gemini_model.generate_content(prompt)
                            st.session_state.chat_history.append({"soru": user_question, "cevap": response.text})
                        except:
                            st.error("Bağlantı hatası.")
                else:
                    st.warning("Yapay zeka şu an meşgul.")

        for chat in reversed(st.session_state.chat_history):
            st.markdown(f'<div class="chat-user">🧑‍🌾 {chat["soru"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-ai">🤖 {chat["cevap"]}</div>', unsafe_allow_html=True)

    # --- ALT BUTONLAR ---
    if st.session_state.son_analiz:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            pdf_data = create_pdf(analiz['bitki'], analiz['hastalik'], analiz['tedavi'], notlar, datetime.now().strftime("%d-%m"))
            st.download_button("📄 Rapor", pdf_data, "Rapor.pdf", "application/pdf")
        with c2:
            with st.expander("Düzelt"):
                secenekler = [v[0] for v in AYARLAR[secilen_bitki]["bilgi"].values()]
                dogru = st.selectbox("Doğrusu:", secenekler)
                if st.button("Kaydet"):
                    bitki_adi_duz = secilen_bitki.split()[0]
                    dogru_klasor_adi = None
                    for k, v in AYARLAR[secilen_bitki]["bilgi"].items():
                        if v[0] == dogru:
                            dogru_klasor_adi = k
                            break
                    if dogru_klasor_adi:
                        hatali_veriyi_kaydet(img_pil, dogru_klasor_adi, bitki_adi_duz)
                        st.success("Kaydedildi.")

# --- GEÇMİŞ TABLOSU VE TEMİZLEME BUTONU ---
st.markdown("---")
with st.expander("📜 Geçmiş"):
    if os.path.exists(DB_DOSYASI):
        df = pd.read_csv(DB_DOSYASI)
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)
        st.write("") 
        if st.button("🗑️ Geçmişi Temizle", type="secondary"):
            try:
                os.remove(DB_DOSYASI)
                st.success("Tüm geçmiş silindi!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Silinemedi: {e}")
    else:
        st.info("Henüz kayıt bulunmuyor.")