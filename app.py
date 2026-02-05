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

# ==============================================================================
# 1. AYARLAR VE GÖRSEL TASARIM (CSS DÜZELTİLDİ) 🎨
# ==============================================================================
st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿", layout="centered")

# --- ARKA PLAN VE SIDEBAR TASARIMI ---
def tasariimi_uygula():
    # 1. Arka Plan Resmini Ayarla
    dosya_adi = "arkaplan.jpg"
    bg_image_style = ""
    
    if os.path.exists(dosya_adi):
        with open(dosya_adi, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        bg_image_style = f'background-image: url("data:image/jpg;base64,{encoded_string}");'
    else:
        # Yedek resim
        bg_image_style = 'background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=1527&auto=format&fit=crop");'

    # 2. CSS İle Renkleri Zorla (Okunabilirlik İçin)
    st.markdown(
        f"""
        <style>
        /* Ana Arka Plan */
        .stApp {{
            {bg_image_style}
            background-attachment: fixed;
            background-size: cover;
        }}
        
        /* Sidebar (Yan Menü) Arka Planı - KOYU VE OKUNAKLI */
        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 25, 15, 0.95) !important; /* Çok koyu yeşil/siyah */
            border-right: 3px solid #4CAF50; /* Sağ tarafa şık bir çizgi */
        }}
        
        /* Sidebar'daki Yazıların Rengini BEYAZ Yap */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] p {{
            color: #ffffff !important;
            text-shadow: 1px 1px 2px black; /* Yazı gölgesi ile netlik */
        }}
        
        /* Input (Giriş) Kutularının İçi */
        div[data-baseweb="input"] {{
            background-color: rgba(255, 255, 255, 0.9) !important;
        }}
        input[type="text"] {{
            color: black !important;
        }}

        /* Ana ekrandaki kutucuklar (Expander) */
        div[data-testid="stExpander"] {{
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

tasariimi_uygula()

# KOTA AYARLARI
SORU_LIMITI = 20        
BEKLEME_SURESI = 15     

# ==============================================================================
# 2. HAVA DURUMU MODÜLÜ (SIDEBAR) 🌤️
# ==============================================================================
def hava_durumu_getir(sehir):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={sehir}&count=1&language=tr&format=json"
        geo_response = requests.get(geo_url).json()
        
        if "results" in geo_response:
            lat = geo_response["results"][0]["latitude"]
            lon = geo_response["results"][0]["longitude"]
            
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto"
            w_response = requests.get(weather_url).json()
            
            return w_response["current"]
        return None
    except:
        return None

with st.sidebar:
    st.title("🌤️ Hava Durumu") # Header yerine Title daha büyük durur
    st.write("Bölgenizdeki tarımsal veriler:")
    
    sehir_secimi = st.text_input("Şehir Giriniz:", value="Ankara")
    
    if st.button("Verileri Getir", type="primary"): # Butonu vurguladık
        veri = hava_durumu_getir(sehir_secimi)
        if veri:
            st.success(f"📍 {sehir_secimi.upper()}")
            
            # Metrikleri daha şık gösterelim
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sıcaklık", f"{veri['temperature_2m']} °C")
            with col2:
                st.metric("Nem", f"%{veri['relative_humidity_2m']}")
            
            st.metric("Rüzgar", f"{veri['wind_speed_10m']} km/s")
            
            if veri['wind_speed_10m'] > 15:
                st.warning("⚠️ Rüzgar sert! İlaçlama yapmayınız.")
            elif veri['relative_humidity_2m'] > 80:
                st.info("💧 Nem yüksek. Mantar riski!")
            else:
                st.info("✅ Hava şartları ilaçlama için uygun.")
        else:
            st.error("Şehir bulunamadı.")
            
    st.markdown("---")
    st.caption("🌿 **Ziraat AI** v1.2")

st.title("🌿 Ziraat AI - Akıllı Bitki Doktoru")
st.markdown("**Yapay Zeka Destekli Hastalık Teşhisi ve Tedavi Uzmanı**")

# ==============================================================================
# 3. YARDIMCI FONKSİYONLAR (PDF İÇİN)
# ==============================================================================
def tr_duzelt(text):
    source = "şŞıİğĞüÜöÖçÇ"
    target = "sSiIgGuUoOcC"
    translation_table = str.maketrans(source, target)
    return text.translate(translation_table)

def rapor_olustur(bitki, hastalik, recete):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="ZIRAAT AI - TESHIS RAPORU", ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=tr_duzelt(f"Tarih: {time.strftime('%d-%m-%Y')}"), ln=1)
    pdf.cell(200, 10, txt=tr_duzelt(f"Analiz Edilen Bitki: {bitki}"), ln=1)
    pdf.cell(200, 10, txt=tr_duzelt(f"Tespit Edilen Durum: {hastalik}"), ln=1)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="YAPAY ZEKA ONERISI VE RECETE:", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=tr_duzelt(recete))
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, txt="Bu rapor yapay zeka tarafindan uretilmistir. Kesin teshis icin uzmana danisiniz.", align='C')
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ==============================================================================
# 4. GEMINI BAĞLANTISI
# ==============================================================================
@st.cache_resource
def gemini_baglan():
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=api_key)
            oncelikli_modeller = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-pro']
            for m in oncelikli_modeller:
                try:
                    test_model = genai.GenerativeModel(m)
                    test_model.generate_content("System check") 
                    return test_model, m 
                except: continue
            
            tum_modeller = genai.list_models()
            for m in tum_modeller:
                if 'generateContent' in m.supported_generation_methods:
                    if 'gemini-2.5' in m.name: continue 
                    try:
                        yedek_model = genai.GenerativeModel(m.name)
                        yedek_model.generate_content("System check")
                        return yedek_model, m.name
                    except: continue
            return None, "Model Bulunamadı"
        return None, "Anahtar Yok"
    except Exception as e:
        return None, str(e)

model_gemini, aktif_model_ismi = gemini_baglan()

if model_gemini:
    st.caption(f"✅ Sistem Hazır: `{aktif_model_ismi}`")
else:
    st.error(f"⚠️ Bağlantı Hatası: {aktif_model_ismi}")

st.markdown("---")

# ==============================================================================
# 5. TEŞHİS MODELİ YÜKLEME
# ==============================================================================
@st.cache_resource
def model_yukle(bitki_tipi):
    mapper = {
        "Elma (Apple)": "apple_uzman_model.keras",
        "Domates (Tomato)": "tomato_uzman_model.keras",
        "Mısır (Corn)": "corn_uzman_model.keras",
        "Üzüm (Grape)": "grape_uzman_model.keras",
        "Şeftali (Peach)": "peach_uzman_model.keras",
        "Biber (Pepper)": "pepper_uzman_model.keras",
        "Patates (Potato)": "potato_uzman_model.keras",
        "Çilek (Strawberry)": "strawberry_uzman_model.keras",
        "Kiraz (Cherry)": "cherry_uzman_model.keras",
        "Yaban Mersini": "blueberry_uzman_model.keras",
        "Ahududu": "raspberry_uzman_model.keras",
        "Soya Fasulyesi": "soybean_uzman_model.keras",
        "Kabak": "squash_uzman_model.keras",
        "Portakal": "orange_uzman_model.keras"
    }
    if bitki_tipi in mapper:
        try:
            return tf.keras.models.load_model(mapper[bitki_tipi])
        except:
            return None
    return None

def siniflari_getir(bitki_tipi):
    if bitki_tipi == "Elma (Apple)":
        return ['Elma Kara Leke', 'Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı']
    elif bitki_tipi == "Domates (Tomato)":
        return ['Bakteriyel Leke', 'Erken Yanıklık', 'Geç Yanıklık', 'Yaprak Küfü', 'Septoria Yaprak Lekesi', 'Örümcek Akarları', 'Hedef Leke', 'Sarı Yaprak Kıvırcıklığı', 'Mozaik Virüsü', 'Sağlıklı']
    elif bitki_tipi == "Mısır (Corn)":
        return ['Mısır Gri Yaprak Lekesi', 'Mısır Yaygın Pas', 'Mısır Kuzey Yaprak Yanıklığı', 'Mısır Sağlıklı']
    elif bitki_tipi == "Patates (Potato)":
        return ['Patates Erken Yanıklık', 'Patates Geç Yanıklık', 'Patates Sağlıklı']
    elif bitki_tipi == "Üzüm (Grape)":
        return ['Üzüm Kara Çürüklüğü', 'Üzüm Siyah Kızamık (Esca)', 'Üzüm Yaprak Yanıklığı', 'Üzüm Sağlıklı']
    return ["Hastalık", "Sağlıklı"]

# ==============================================================================
# 6. KULLANICI OTURUM TAKİBİ
# ==============================================================================
if 'soru_sayaci' not in st.session_state: st.session_state['soru_sayaci'] = 0
if 'son_soru_zamani' not in st.session_state: st.session_state['son_soru_zamani'] = 0
if 'rapor_hazir' not in st.session_state: st.session_state['rapor_hazir'] = None

# ==============================================================================
# 7. ARAYÜZ VE ANALİZ
# ==============================================================================
secilen_bitki = st.selectbox("🌿 Hangi bitkiyi analiz edelim?", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)"])
yuklenen_dosya = st.file_uploader("📸 Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if yuklenen_dosya:
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    if st.button("🔍 Hastalığı Analiz Et ve Raporla", type="primary"):
        with st.spinner('Yapay zeka analiz ediyor ve reçete yazıyor...'):
            model = model_yukle(secilen_bitki)
            if model:
                hedef_boyut = (160, 160)
                img = image.resize(hedef_boyut) 
                img_array = np.array(img).astype("float32")
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
                img_array = img_array[..., ::-1] 
                input_data = np.expand_dims(img_array, axis=0)
                
                try:
                    tahmin = model.predict(input_data)
                    olasiliklar = tf.nn.softmax(tahmin).numpy()[0]
                    indeks = np.argmax(olasiliklar)
                    guven = olasiliklar[indeks] * 100
                    siniflar = siniflari_getir(secilen_bitki)
                    
                    if indeks < len(siniflar):
                        sonuc_ismi = siniflar[indeks]
                        recete_metni = "Hastalık sağlıklı olduğu için tedavi gerekmez."
                        if "Sağlıklı" in sonuc_ismi:
                            st.success(f"**Teşhis:** {sonuc_ismi}")
                            st.balloons()
                        else:
                            st.error(f"**Teşhis:** {sonuc_ismi}")
                            if model_gemini:
                                prompt_rapor = f"Bitki: {secilen_bitki}. Hastalık: {sonuc_ismi}. Bu hastalık için çiftçiye uygulanabilir, maddeler halinde kısa bir tedavi reçetesi ve ilaç önerisi yaz. Türkçe karakter kullanma (ornek: ş yerine s yaz)."
                                try:
                                    response = model_gemini.generate_content(prompt_rapor)
                                    recete_metni = response.text
                                except: recete_metni = "Yapay zeka reçete oluştururken bir hata oluştu."

                        st.info(f"**Güven Oranı:** %{guven:.2f}")
                        pdf_data = rapor_olustur(secilen_bitki, sonuc_ismi, recete_metni)
                        st.session_state['rapor_hazir'] = pdf_data
                        st.session_state['son_teshis'] = sonuc_ismi
                        st.session_state['son_bitki'] = secilen_bitki
                    else: st.error("Liste hatası.")
                except Exception as e: st.error(f"Tahmin hatası: {e}")

    if st.session_state['rapor_hazir']:
        st.download_button(label="📄 PDF Raporunu İndir", data=st.session_state['rapor_hazir'], file_name="ziraat_ai_rapor.pdf", mime="application/pdf", type="secondary")

# ==============================================================================
# 8. SOHBET MODU
# ==============================================================================
if 'son_teshis' in st.session_state and model_gemini:
    st.markdown("---")
    st.subheader(f"🤖 Ziraat Asistanı ile Konuşun")
    kalan_hak = SORU_LIMITI - st.session_state['soru_sayaci']
    st.progress(st.session_state['soru_sayaci'] / SORU_LIMITI, text=f"Günlük Soru Hakkı: {kalan_hak} kaldı")
    st.write(f"**Durum:** {st.session_state['son_bitki']} - {st.session_state['son_teshis']}")
    soru = st.text_input("Sorunuzu buraya yazın...")
    
    if st.button("Soruyu Gönder"):
        if st.session_state['soru_sayaci'] >= SORU_LIMITI:
            st.error("🚫 Bu oturumdaki soru limitiniz doldu! Yarın tekrar bekleriz.")
        elif (time.time() - st.session_state['son_soru_zamani']) < BEKLEME_SURESI:
            kalan_sure = int(BEKLEME_SURESI - (time.time() - st.session_state['son_soru_zamani']))
            st.warning(f"⏳ Biraz yavaşlayalım! Lütfen {kalan_sure} saniye daha bekle.")
        elif soru:
            with st.spinner('Cevaplanıyor...'):
                prompt = f"Sen uzman bir ziraat mühendisisin. Bitki: {st.session_state['son_bitki']}. Hastalık: {st.session_state['son_teshis']}. Soru: '{soru}'. Kısa cevap ver."
                try:
                    cevap = model_gemini.generate_content(prompt)
                    st.write(cevap.text)
                    st.session_state['soru_sayaci'] += 1
                    st.session_state['son_soru_zamani'] = time.time()
                except Exception as e: st.error(f"Hata: {e}")
elif 'son_teshis' in st.session_state and not model_gemini:
     st.warning("⚠️ Sohbet sistemi şu an mola verdi.")