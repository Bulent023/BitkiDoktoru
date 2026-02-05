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
# 1. AYARLAR VE GÖRSEL TASARIM
# ==============================================================================
st.set_page_config(page_title="Ziraat AI", page_icon="🌿", layout="centered")

if 'giris_yapildi' not in st.session_state: st.session_state['giris_yapildi'] = False
if 'son_teshis' not in st.session_state: st.session_state['son_teshis'] = None
if 'son_bitki' not in st.session_state: st.session_state['son_bitki'] = None
if 'recete_hafizasi' not in st.session_state: st.session_state['recete_hafizasi'] = ""

# --- CSS TASARIMI ---
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
            box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
        }}
        div.stButton > button:hover {{ border-color: #ff4b4b; color: #ff4b4b; background-color: white; }}
        section[data-testid="stSidebar"] {{ background-color: rgba(15, 25, 15, 0.95) !important; border-right: 3px solid #4CAF50; }}
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, p, label {{ color: white !important; }}
        input[type="text"] {{ color: white !important; }}
        div[data-baseweb="input"] {{ background-color: rgba(20, 40, 20, 0.8) !important; border: 1px solid #4CAF50; }}
        div[data-testid="stExpander"] {{ background-color: rgba(0, 0, 0, 0.8); color: white; border-radius: 10px; }}
        div[data-testid="stTabs"] button[aria-selected="true"] {{ background-color: #4CAF50; color: white; }}
        div.stInfo {{ background-color: rgba(0, 0, 0, 0.7) !important; color: white !important; border: 1px solid #2196F3; }}
        div.stSuccess {{ background-color: rgba(0, 50, 0, 0.7) !important; color: white !important; }}
        div.stError {{ background-color: rgba(50, 0, 0, 0.8) !important; color: white !important; }}
        </style>
        """, unsafe_allow_html=True
    )
tasariimi_uygula()

def load_lottieurl(url):
    try: return requests.get(url).json()
    except: return None

# --- GÜÇLENDİRİLMİŞ KARAKTER TEMİZLEME FONKSİYONU ---
def tr_duzelt(text):
    if not isinstance(text, str):
        text = str(text)
        
    # 1. Türkçe karakterleri İngilizce karşılıklarına çevir
    source = "şŞıİğĞüÜöÖçÇ"
    target = "sSiIgGuUoOcC"
    translation_table = str.maketrans(source, target)
    text = text.translate(translation_table)
    
    # 2. Emojileri ve desteklenmeyen sembolleri sil (ZORUNLU)
    # Bu satır metni 'latin-1' formatına sığdırır, sığmayanları (emoji vb.) siler.
    text = text.encode('latin-1', 'ignore').decode('latin-1')
    
    return text

def create_pdf(bitki, hastalik, recete):
    pdf = FPDF()
    pdf.add_page()
    
    # Font Ekleme (Varsayılan Arial kullanılır)
    pdf.set_font("Arial", 'B', 16)
    
    # Başlık
    pdf.cell(200, 10, txt=tr_duzelt("ZIRAAT AI - TESHIS RAPORU"), ln=1, align='C')
    pdf.ln(10)
    
    # Bilgiler
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=tr_duzelt(f"Tarih: {time.strftime('%d-%m-%Y')}"), ln=1)
    pdf.cell(200, 10, txt=tr_duzelt(f"Bitki: {bitki}"), ln=1)
    pdf.cell(200, 10, txt=tr_duzelt(f"Teshis: {hastalik}"), ln=1)
    
    pdf.ln(10)
    
    # Reçete Bölümü
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=tr_duzelt("DETAYLI BILGI VE RECETE:"), ln=1)
    
    pdf.set_font("Arial", size=11)
    # Reçete metni çok uzun olabileceği için multi_cell kullanıyoruz
    # ve tr_duzelt ile emojilerden arındırıyoruz
    pdf.multi_cell(0, 10, txt=tr_duzelt(recete))
    
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ==============================================================================
# 2. MANUEL GEMINI BAĞLANTISI (KOTA DOSTU MOD) 🛠️
# ==============================================================================
def gemini_sor(prompt):
    if "GOOGLE_API_KEY" not in st.secrets:
        return "HATA: API Anahtarı bulunamadı."
    
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    ucretsiz_modeller = [
        "gemini-1.5-flash", 
        "gemini-1.5-pro", 
        "gemini-pro"
    ]
    
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    for model_ismi in ucretsiz_modeller:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_ismi}:generateContent?key={api_key}"
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            elif response.status_code in [429, 404, 503]:
                continue 
            else:
                return f"Hata ({model_ismi}): {response.status_code} - {response.text}"
        except Exception as e:
            continue

    return "⚠️ Üzgünüz, tüm modellerin günlük kotası dolmuş olabilir. Lütfen yarın tekrar deneyin veya yeni bir API anahtarı alın."

# ==============================================================================
# 3. GİRİŞ EKRANI
# ==============================================================================
if not st.session_state['giris_yapildi']:
    st.write("")
    st.write("") 
    st.markdown("<h1 style='text-align: center; color: white; font-size: 50px; text-shadow: 3px 3px 6px #000000;'>🌿 Ziraat AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #e8f5e9; text-shadow: 1px 1px 2px #000000;'>Çiftçinin Dijital Asistanı</h3>", unsafe_allow_html=True)
    
    lottie_intro = load_lottieurl("https://lottie.host/62688176-784f-4d22-8280-5b1191062085/WkL0s7l9Xj.json")
    if lottie_intro: st_lottie(lottie_intro, height=250, key="intro_anim")
    
    st.write("") 
    if st.button("🚀 UYGULAMAYI BAŞLAT", key="baslat_butonu"):
        st.session_state['giris_yapildi'] = True
        st.rerun()

# ==============================================================================
# 4. ANA UYGULAMA
# ==============================================================================
else:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
        st.title("Ziraat AI")
        st.success("Mod: Kota Dostu (Flash/Pro)")
        
        if st.button("🔙 Çıkış Yap"):
            st.session_state['giris_yapildi'] = False
            st.rerun()

    st.title("🌿 Akıllı Bitki Doktoru")
    tab1, tab2, tab3 = st.tabs(["🌿 Hastalık Teşhisi & Reçete", "🌤️ Bölgesel Hava Durumu ve Takvim", "ℹ️ Yardım"])

    # --- SEKME 1: TEŞHİS & REÇETE ---
    with tab1:
        st.markdown("### 📸 Fotoğraf Yükle")
        
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
        
        def siniflari_al(bitki):
             if bitki == "Elma (Apple)": return ['Elma Kara Leke', 'Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı']
             elif bitki == "Domates (Tomato)": return ['Bakteriyel Leke', 'Erken Yanıklık', 'Geç Yanıklık', 'Yaprak Küfü', 'Septoria Yaprak Lekesi', 'Örümcek Akarları', 'Hedef Leke', 'Sarı Yaprak Kıvırcıklığı', 'Mozaik Virüsü', 'Sağlıklı']
             elif bitki == "Mısır (Corn)": return ['Mısır Gri Yaprak Lekesi', 'Mısır Yaygın Pas', 'Mısır Kuzey Yaprak Yanıklığı', 'Mısır Sağlıklı']
             elif bitki == "Patates (Potato)": return ['Patates Erken Yanıklık', 'Patates Geç Yanıklık', 'Patates Sağlıklı']
             elif bitki == "Üzüm (Grape)": return ['Üzüm Kara Çürüklüğü', 'Üzüm Siyah Kızamık (Esca)', 'Üzüm Yaprak Yanıklığı', 'Üzüm Sağlıklı']
             elif bitki == "Biber (Pepper)": return ['Biber Bakteriyel Leke', 'Biber Sağlıklı']
             elif bitki == "Şeftali (Peach)": return ['Şeftali Bakteriyel Leke', 'Şeftali Sağlıklı']
             elif bitki == "Çilek (Strawberry)": return ['Çilek Yaprak Yanıklığı', 'Çilek Sağlıklı']
             elif bitki == "Kiraz (Cherry)": return ['Kiraz Külleme', 'Kiraz Sağlıklı']
             return ["Hastalık", "Sağlıklı"]

        col_a, col_b = st.columns(2)
        with col_a: secilen_bitki = st.selectbox("Bitki:", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)", "Kiraz (Cherry)"])
        with col_b: dosya = st.file_uploader("Resim:", type=["jpg","png"])

        if dosya:
            image = Image.open(dosya)
            st.image(image, width=300)
            
            if st.button("🔍 Analiz Et", type="primary"):
                with st.spinner("Analiz yapılıyor..."):
                    model = model_yukle(secilen_bitki)
                    if model:
                        img = image.resize((160,160))
                        img_arr = np.array(img).astype("float32")
                        if img_arr.ndim==2: img_arr=np.stack((img_arr,)*3, axis=-1)
                        elif img_arr.shape[-1]==4: img_arr=img_arr[:,:,:3]
                        img_arr = img_arr[...,::-1] 
                        input_data = np.expand_dims(img_arr, axis=0)
                        
                        try:
                            tahmin = model.predict(input_data)
                            idx = np.argmax(tahmin)
                            siniflar = siniflari_al(secilen_bitki)
                            sonuc = siniflar[idx] if idx < len(siniflar) else f"Tespit: {idx}"
                            
                            st.session_state['son_teshis'] = sonuc
                            st.session_state['son_bitki'] = secilen_bitki
                            
                            if "Sağlıklı" in sonuc:
                                st.success(f"✅ **Durum:** {sonuc}")
                                st.balloons()
                                st.session_state['recete_hafizasi'] = "Bitki sağlıklı. Koruyucu önlem olarak düzenli bakım yapınız."
                            else:
                                st.error(f"⚠️ **Tespit:** {sonuc}")
                                # --- GEMINI REÇETE ---
                                prompt = f"Bitki: {secilen_bitki}, Hastalık: {sonuc}. Bu hastalık için 3 başlıkta bilgi ver: 1-Nedir, 2-Kültürel Önlem, 3-İlaçlı Mücadele."
                                recete = gemini_sor(prompt)
                                st.session_state['recete_hafizasi'] = recete
                        except Exception as e: st.error(f"Hata: {e}")

            if st.session_state['son_teshis']:
                st.markdown("---")
                with st.expander("📋 Reçete ve Tedavi (Tıkla)", expanded=True):
                    st.markdown(st.session_state['recete_hafizasi'])
                
                # PDF Hatası için güvenli oluşturma
                try:
                    pdf_data = create_pdf(st.session_state['son_bitki'], st.session_state['son_teshis'], st.session_state['recete_hafizasi'])
                    st.download_button(label="📄 Raporu İndir (PDF)", data=pdf_data, file_name="rapor.pdf", mime="application/pdf")
                except Exception as e:
                    st.warning("PDF oluşturulamadı (Karakter hatası). Lütfen ekran görüntüsü alınız.")
                
                st.markdown("---")
                st.subheader("💬 Asistan")
                soru = st.text_input("Sorunuz var mı?")
                if st.button("Sor"):
                    if soru:
                        with st.spinner("Cevaplanıyor..."):
                            cevap = gemini_sor(f"Konu: {st.session_state['son_teshis']}, Soru: {soru}")
                            st.write(cevap)

    # --- SEKME 2: BÖLGE VE TAKVİM ---
    with tab2:
        st.header("🌤️ Bölgesel Tarım Verileri")
        sehir = st.text_input("Şehir Giriniz:", value="Antalya")
        
        if st.button("Verileri Getir", type="primary"):
             try:
                geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={sehir}&count=1").json()
                if "results" in geo:
                    lat = geo["results"][0]["latitude"]
                    lon = geo["results"][0]["longitude"]
                    w = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m").json()["current"]
                    
                    st.subheader(f"📍 {sehir.upper()}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Sıcaklık", f"{w['temperature_2m']} °C")
                    c2.metric("Nem", f"%{w['relative_humidity_2m']}")
                    c3.metric("Rüzgar", f"{w['wind_speed_10m']} km/s")
                    
                    st.markdown("---")
                    st.subheader("📅 Akıllı Takvim")
                    aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
                    simdiki_ay = aylar[int(time.strftime("%m")) - 1]
                    
                    with st.spinner(f"{simdiki_ay} ayı analiz ediliyor..."):
                        prompt_takvim = f"{simdiki_ay} ayında {sehir} ilinde tarımsal olarak ne yapılır? 4 maddede özetle."
                        takvim_cevap = gemini_sor(prompt_takvim)
                        st.info(f"**{simdiki_ay} Ayı Tavsiyeleri:**\n\n" + takvim_cevap)
                else:
                    st.error("Şehir bulunamadı.")
             except Exception as e: st.error(f"Bağlantı Hatası: {e}")

    # --- SEKME 3: YARDIM ---
    with tab3:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; color: black;">
            <h3 style="color: #1b5e20;">❓ Yardım</h3>
            <p>1. <b>Teşhis</b> sekmesinden fotoğraf yükleyin.<br>
            2. Analiz sonrası reçeteniz otomatik oluşur.<br>
            3. <b>Bölge</b> sekmesinden şehrinizin verilerine bakın.</p>
        </div>
        """, unsafe_allow_html=True)