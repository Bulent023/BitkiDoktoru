import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import google.generativeai as genai

# ==============================================================================
# 1. AYARLAR VE API ANAHTARI
# ==============================================================================
# 👇 BURAYA KENDİ API KEY'İNİ MUTLAKA YAZ! 👇
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg" 

st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

# GEMINI CHATBOT (Hatasız Sürüm)
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash') 
    chatbot_aktif = True
except:
    chatbot_aktif = False

st.title("🌿 Ziraat AI - Akıllı Bitki Doktoru")
st.markdown("---")

# ==============================================================================
# 2. MODEL YÜKLEME
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

# ==============================================================================
# 3. SINIF LİSTESİ (DÜZELTİLMİŞ DOĞRU SIRALAMA) ✅
# ==============================================================================
def siniflari_getir(bitki_tipi):
    if bitki_tipi == "Elma (Apple)":
        # Hata analizine göre KESİNLEŞMİŞ doğru sıralama:
        # 0: Scab (Leke), 1: Black Rot (Çürük), 2: Rust (Pas), 3: Healthy (Sağlıklı)
        return ['Elma Kara Leke', 'Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı']
        
    elif bitki_tipi == "Domates (Tomato)":
        return ['Bakteriyel Leke', 'Geç Yanıklık', 'Erken Yanıklık', 'Yaprak Küfü', 'Septoria Yaprak Lekesi', 'Örümcek Akarları', 'Hedef Leke', 'Sarı Yaprak Kıvırcıklığı', 'Mozaik Virüsü', 'Sağlıklı']
    elif bitki_tipi == "Mısır (Corn)":
        return ['Mısır Gri Yaprak Lekesi', 'Mısır Yaygın Pas', 'Mısır Kuzey Yaprak Yanıklığı', 'Mısır Sağlıklı']
    elif bitki_tipi == "Patates (Potato)":
        return ['Patates Erken Yanıklık', 'Patates Geç Yanıklık', 'Patates Sağlıklı']
    elif bitki_tipi == "Üzüm (Grape)":
        return ['Üzüm Kara Çürüklüğü', 'Üzüm Siyah Kızamık (Esca)', 'Üzüm Yaprak Yanıklığı', 'Üzüm Sağlıklı']
    
    return ["Hastalık", "Sağlıklı"]

# ==============================================================================
# 4. ARAYÜZ VE ANALİZ
# ==============================================================================
secilen_bitki = st.selectbox("🌿 Hangi bitkiyi analiz edelim?", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)"])
yuklenen_dosya = st.file_uploader("📸 Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if yuklenen_dosya:
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    if st.button("🔍 Hastalığı Analiz Et", type="primary"):
        with st.spinner('Yapay zeka analiz ediyor...'):
            model = model_yukle(secilen_bitki)
            if model:
                # 1. BOYUT: 160x160 (Röntgen Sonucuna Göre)
                hedef_boyut = (160, 160)
                img = image.resize(hedef_boyut) 
                
                # Array'e çevir
                img_array = np.array(img).astype("float32")
                
                # Kanal kontrolü
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]

                # -------------------------------------------------------------
                # 🧪 OTOMATİK RENK SEÇİMİ (RGB vs BGR)
                # -------------------------------------------------------------
                
                # 1. Seçenek: Normal RGB (0-255 arası)
                input_rgb = np.expand_dims(img_array, axis=0)
                
                # 2. Seçenek: Normal RGB (0-1 arası)
                input_rgb_norm = np.expand_dims(img_array / 255.0, axis=0)

                # Tahminleri al
                pred_rgb = model.predict(input_rgb)
                pred_norm = model.predict(input_rgb_norm)
                
                # Güven skorlarını hesapla
                conf_rgb = np.max(tf.nn.softmax(pred_rgb).numpy()[0])
                conf_norm = np.max(tf.nn.softmax(pred_norm).numpy()[0])
                
                # Hangisi daha yüksekse onu kullan
                if conf_rgb > conf_norm:
                    final_pred = pred_rgb
                    final_conf = conf_rgb * 100
                else:
                    final_pred = pred_norm
                    final_conf = conf_norm * 100

                # SONUCU YAZDIR
                indeks = np.argmax(final_pred)
                siniflar = siniflari_getir(secilen_bitki)
                
                if indeks < len(siniflar):
                    sonuc_ismi = siniflar[indeks]
                    
                    if "Sağlıklı" in sonuc_ismi:
                        st.success(f"**Teşhis:** {sonuc_ismi}")
                        st.balloons()
                    else:
                        st.error(f"**Teşhis:** {sonuc_ismi}")
                    
                    st.info(f"**Güven Oranı:** %{final_conf:.2f}")
                    
                    # Session Kaydı
                    st.session_state['son_teshis'] = sonuc_ismi
                    st.session_state['son_bitki'] = secilen_bitki
                else:
                    st.error("Liste hatası.")

# ==============================================================================
# 5. SOHBET MODU (GEMINI 1.5 FLASH)
# ==============================================================================
if 'son_teshis' in st.session_state and chatbot_aktif:
    st.markdown("---")
    st.subheader(f"🤖 Ziraat Asistanı ile Konuşun")
    st.write(f"**Durum:** {st.session_state['son_bitki']} - {st.session_state['son_teshis']}")
    
    soru = st.text_input("Sorunuzu buraya yazın...")
    
    if st.button("Soruyu Gönder"):
        if soru:
            with st.spinner('Cevaplanıyor...'):
                prompt = f"Sen ziraat uzmanısın. Bitki: {st.session_state['son_bitki']}, Hastalık: {st.session_state['son_teshis']}. Soru: {soru}. Kısa cevap ver."
                try:
                    cevap = model_gemini.generate_content(prompt)
                    st.write(cevap.text)
                except Exception as e:
                    st.error(f"Hata: {e}")