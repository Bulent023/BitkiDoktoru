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
if 'calisan_model_ismi' not in st.session_state: st.session_state['calisan_model_ismi'] = None

# ==============================================================================
# 2. MODERN TASARIM (CSS) 🎨
# ==============================================================================
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
        
        /* --- MODERN BUTONLAR --- */
        div.stButton > button {{
            background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%); /* Modern Yeşil Gradyan */
            color: white;
            border: none;
            border-radius: 12px; /* Yumuşak köşeler */
            padding: 12px 24px;
            font-weight: 600;
            font-size: 16px;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            width: 100%;
        }}
        
        div.stButton > button:hover {{
            transform: translateY(-2px); /* Tıklama hissi */
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
            background: linear-gradient(135deg, #1b5e20 0%, #388e3c 100%);
            color: white;
            border-color: transparent;
        }}
        
        div.stButton > button:active {{
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        /* ----------------------- */

        section[data-testid="stSidebar"] {{ background-color: rgba(20, 30, 20, 0.95) !important; border-right: 1px solid #4CAF50; }}
        * {{ color: white; }}
        input[type="text"] {{ 
            background-color: rgba(255, 255, 255, 0.1) !important; 
            color: white !important; 
            border: 1px solid #4CAF50 !important; 
            border-radius: 8px;
        }}
        div[data-testid="stExpander"] {{ 
            background-color: rgba(0, 0, 0, 0.6); 
            border-radius: 10px; 
            border: 1px solid #4CAF50;
        }}
        /* Metrik Kutuları */
        div[data-testid="stMetric"] {{
            background-color: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        div[data-testid="stMetricLabel"] {{ color: #e0e0e0 !important; }}
        div[data-testid="stMetricValue"] {{ color: #4CAF50 !important; font-weight: bold; }}
        
        </style>
        """, unsafe_allow_html=True
    )
tasariimi_uygula()

def load_lottieurl(url):
    try: return requests.get(url).json()
    except: return None

def tr_duzelt(text):
    if not isinstance(text, str): text = str(text)
    source = "şŞıİğĞüÜöÖçÇ"
    target = "sSiIgGuUoOcC"
    translation_table = str.maketrans(source, target)
    text = text.translate(translation_table)
    text = text.encode('latin-1', 'ignore').decode('latin-1')
    return text

def ruzgar_yonu_bul(derece):
    """Rüzgar derecesini (0-360) yön ismine çevirir."""
    yonler = ["Kuzey", "Kuzeydoğu", "Doğu", "Güneydoğu", "Güney", "Güneybatı", "Batı", "Kuzeybatı"]
    index = round(derece / 45) % 8
    return yonler[index]

def create_pdf(bitki, hastalik, recete):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="RAPOR", ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=tr_duzelt(f"Bitki: {bitki}\nDurum: {hastalik}\n\n{recete}"))
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ==============================================================================
# 3. AKILLI MODEL BULUCU & API
# ==============================================================================
def model_bul_ve_getir(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            modeller = response.json().get('models', [])
            uygun_modeller = []
            for m in modeller:
                if 'generateContent' in m.get('supportedGenerationMethods', []):
                    if 'flash' in m['name']: return m['name']
                    uygun_modeller.append(m['name'])
            if uygun_modeller: return uygun_modeller[0]
        return None
    except: return None

def gemini_sor(prompt):
    if "GOOGLE_API_KEY" not in st.secrets: return "HATA: API Anahtarı Yok."
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    if st.session_state['calisan_model_ismi'] is None:
        bulunan = model_bul_ve_getir(api_key)
        if bulunan: st.session_state['calisan_model_ismi'] = bulunan
        else: st.session_state['calisan_model_ismi'] = "models/gemini-1.5-flash"

    model_id = st.session_state['calisan_model_ismi']
    if not model_id.startswith("models/"): model_id = f"models/{model_id}"

    url = f"https://generativelanguage.googleapis.com/v1beta/{model_id}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        elif response.status_code == 404:
            st.session_state['calisan_model_ismi'] = None
            return "Model anlık bulunamadı, tekrar deneyiniz."
        else:
            return f"HATA ({model_id}): {response.status_code}"
    except Exception as e: return f"Bağlantı Hatası: {e}"

# ==============================================================================
# 4. GİRİŞ EKRANI
# ==============================================================================
if not st.session_state['giris_yapildi']:
    st.write("")
    st.markdown("<h1 style='text-align: center;'>🌿 Ziraat AI</h1>", unsafe_allow_html=True)
    lottie_intro = load_lottieurl("https://lottie.host/62688176-784f-4d22-8280-5b1191062085/WkL0s7l9Xj.json")
    if lottie_intro: st_lottie(lottie_intro, height=250)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 BAŞLAT (MODEL TARA)"):
            if "GOOGLE_API_KEY" in st.secrets:
                test = model_bul_ve_getir(st.secrets["GOOGLE_API_KEY"])
                if test:
                    st.session_state['calisan_model_ismi'] = test
                    st.success(f"Bağlantı: {test}")
                    time.sleep(1)
                    st.session_state['giris_yapildi'] = True
                    st.rerun()
                else: st.error("API Anahtarı ile model bulunamadı.")
            else: st.error("API Anahtarı Yok")

# ==============================================================================
# 5. ANA EKRAN
# ==============================================================================
else:
    with st.sidebar:
        st.title("Ziraat AI")
        if st.session_state['calisan_model_ismi']:
            temiz_isim = st.session_state['calisan_model_ismi'].replace("models/", "")
            st.info(f"🤖 {temiz_isim}")
        if st.button("Çıkış"):
            st.session_state['giris_yapildi'] = False
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["Akıllı Hastalık Teşhisi", "Bölgesel Veriler", "Yardım"])

    # --- TAB 1: TEŞHİS ---
    with tab1:
        st.subheader("Fotoğraf Analizi")
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

        c1, c2 = st.columns(2)
        with c1: secilen = st.selectbox("Bitki", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)", "Kiraz (Cherry)"])
        with c2: dosya = st.file_uploader("Resim")

        if dosya and st.button("Analiz Et"):
            with st.spinner("İnceleniyor..."):
                model = model_yukle(secilen)
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
                        siniflar = siniflari_al(secilen)
                        sonuc = siniflar[idx] if idx < len(siniflar) else f"Tespit: {idx}"
                        st.session_state['son_teshis'] = sonuc
                        st.session_state['son_bitki'] = secilen
                        
                        if "Sağlıklı" in sonuc:
                            st.success(f"✅ {sonuc}")
                            st.balloons()
                            st.session_state['recete_hafizasi'] = "Bitki sağlıklı."
                        else:
                            st.error(f"⚠️ {sonuc}")
                            prompt = f"Bitki: {secilen}, Hastalık: {sonuc}. Bu hastalık için 3 başlıkta bilgi ver: 1-Nedir, 2-Kültürel Önlem, 3-İlaçlı Mücadele."
                            st.session_state['recete_hafizasi'] = gemini_sor(prompt)
                    except Exception as e: st.error(f"Hata: {e}")

        if st.session_state['son_teshis']:
            st.markdown("---")
            with st.expander("Reçete", expanded=True):
                st.write(st.session_state['recete_hafizasi'])
            try:
                pdf_data = create_pdf(st.session_state['son_bitki'], st.session_state['son_teshis'], st.session_state['recete_hafizasi'])
                st.download_button("Rapor İndir", pdf_data, "rapor.pdf", "application/pdf")
            except: pass
            
            st.markdown("---")
            soru = st.text_input("Asistana Sor:")
            if st.button("Gönder"):
                with st.spinner("..."):
                    st.write(gemini_sor(f"Konu: {st.session_state['son_teshis']}, Soru: {soru}"))

    # --- TAB 2: BÖLGE VE DETAYLI HAVA DURUMU ---
    with tab2:
        st.header("🌤️ Bölgesel Tarım Verileri")
        sehir = st.text_input("Şehir Giriniz:", value="Antalya")
        
        if st.button("Verileri Getir"):
            try:
                # 1. Hava Durumu (API Güncellendi)
                geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={sehir}&count=1").json()
                if "results" in geo:
                    lat = geo["results"][0]["latitude"]
                    lon = geo["results"][0]["longitude"]
                    # YENİ API ÇAĞRISI: Rüzgar yönü ve nem eklendi
                    w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
                    w = requests.get(w_url).json()["current"]
                    
                    st.subheader(f"📍 {sehir.upper()} Anlık Durum")
                    
                    # 4 SÜTUNLU MODERN GÖSTERİM
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Sıcaklık", f"{w['temperature_2m']} °C")
                    with col2:
                        st.metric("Nem", f"%{w['relative_humidity_2m']}")
                    with col3:
                        st.metric("Rüzgar Hızı", f"{w['wind_speed_10m']} km/s")
                    with col4:
                        yon_isim = ruzgar_yonu_bul(w['wind_direction_10m'])
                        st.metric("Rüzgar Yönü", f"{yon_isim}")
                    
                    st.markdown("---")
                    aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
                    simdiki_ay = aylar[int(time.strftime("%m")) - 1]
                    with st.spinner("Takvim hazırlanıyor..."):
                        takvim = gemini_sor(f"{simdiki_ay} ayında {sehir} tarım takvimi")
                        st.info(takvim)
                else: st.error("Şehir bulunamadı.")
            except: st.error("Veri alınamadı.")

    with tab3:
        st.markdown("""
        <div style="background-color: rgba(0, 0, 0, 0.5); padding: 20px; border-radius: 10px;">
            <h3>Yardım</h3>
            <p>1. <b>Teşhis</b> sekmesinden fotoğraf yükleyin.</p>
            <p>2. <b>Bölge</b> sekmesinden detaylı hava durumuna bakın.</p>
        </div>
        """, unsafe_allow_html=True)