import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# ==============================================================================
# 1. AYARLAR VE API ANAHTARI
# ==============================================================================
# BURAYA KENDİ API KEY'İNİ MUTLAKA YAZ!
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg" 

st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

# --- YENİ: OTOMATİK MODEL SEÇİCİ ---
# Bu fonksiyon hesabındaki modelleri tarar ve çalışan bir tanesini seçer.
def gemini_modelini_baslat():
    if not GOOGLE_API_KEY or "BURAYA" in GOOGLE_API_KEY:
        return None, False

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # 1. Hesaptaki tüm uygun modelleri listele
        uygun_modeller = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                uygun_modeller.append(m.name)
        
        # 2. Öncelik sırasına göre seçim yap
        secilen_model_adi = ""
        oncelikler = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        
        # Önce favori modellerimizi kontrol et
        for oncelik in oncelikler:
            if oncelik in uygun_modeller:
                secilen_model_adi = oncelik
                break
        
        # Eğer favoriler yoksa, listedeki ilk uygun modeli al
        if not secilen_model_adi and uygun_modeller:
            secilen_model_adi = uygun_modeller[0]
            
        if secilen_model_adi:
            # Modeli başlat
            return genai.GenerativeModel(secilen_model_adi), True
        else:
            return None, False
            
    except Exception as e:
        return None, False

# Modeli kurmaya çalış
model_gemini, chatbot_aktif = gemini_modelini_baslat()

st.title("🌿 Ziraat AI - Akıllı Bitki Doktoru")
st.markdown("---")

# ==============================================================================
# 2. HASTALIK MODELİ YÜKLEME (RAM DOSTU)
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
# 3. SINIF İSİMLERİ (DÜZELTİLMİŞ LİSTE)
# ==============================================================================
def siniflari_getir(bitki_tipi):
    # Domates sıralaması (Alfabetik)
    if bitki_tipi == "Domates (Tomato)":
        return ['Bakteriyel Leke', 'Erken Yanıklık', 'Geç Yanıklık', 'Yaprak Küfü', 'Septoria Yaprak Lekesi', 'Örümcek Akarları', 'Hedef Leke', 'Sarı Yaprak Kıvırcıklığı', 'Mozaik Virüsü', 'Sağlıklı']
    elif bitki_tipi == "Elma (Apple)":
        return ['Elma Kara Leke', 'Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı']
    elif bitki_tipi == "Mısır (Corn)":
        return ['Mısır Gri Yaprak Lekesi', 'Mısır Yaygın Pas', 'Mısır Kuzey Yaprak Yanıklığı', 'Mısır Sağlıklı']
    elif bitki_tipi == "Patates (Potato)":
        return ['Patates Erken Yanıklık', 'Patates Geç Yanıklık', 'Patates Sağlıklı']
    elif bitki_tipi == "Üzüm (Grape)":
        return ['Üzüm Kara Çürüklüğü', 'Üzüm Siyah Kızamık (Esca)', 'Üzüm Yaprak Yanıklığı', 'Üzüm Sağlıklı']
    elif bitki_tipi == "Biber (Pepper)":
        return ['Biber Bakteriyel Leke', 'Biber Sağlıklı']
    elif bitki_tipi == "Şeftali (Peach)":
        return ['Şeftali Bakteriyel Leke', 'Şeftali Sağlıklı']
    elif bitki_tipi == "Çilek (Strawberry)":
        return ['Çilek Yaprak Yanıklığı', 'Çilek Sağlıklı']
    return ["Hastalık Tespit Edildi", "Sağlıklı", "Bilinmiyor"]

# ==============================================================================
# 4. ARAYÜZ
# ==============================================================================
secilen_bitki = st.selectbox("🌿 Hangi bitkiyi analiz edelim?", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)"])
yuklenen_dosya = st.file_uploader("📸 Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if yuklenen_dosya:
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    if st.button("🔍 Hastalığı Analiz Et", type="primary"):
        with st.spinner('Yapay zeka inceliyor...'):
            model = model_yukle(secilen_bitki)
            if model:
                # Akıllı Boyutlandırma
                try:
                    shape = model.input_shape
                    boyut = (shape[1], shape[2]) if shape and shape[1] else (256, 256)
                except:
                    boyut = (256, 256)
                
                img = image.resize(boyut)
                img_array = np.array(img).astype("float32") / 255.0
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
                img_array = np.expand_dims(img_array, axis=0)
                
                tahmin = model.predict(img_array)
                indeks = np.argmax(tahmin)
                guven = np.max(tahmin) * 100
                siniflar = siniflari_getir(secilen_bitki)
                
                if indeks < len(siniflar):
                    hastalik_ismi = siniflar[indeks]
                    st.success(f"**Teşhis:** {hastalik_ismi}")
                    st.info(f"**Eminlik:** %{guven:.2f}")
                    st.session_state['son_teshis'] = hastalik_ismi
                    st.session_state['son_bitki'] = secilen_bitki
                else:
                    st.error("Hata: Sınıf listesi uyumsuz.")

# ==============================================================================
# 5. SOH