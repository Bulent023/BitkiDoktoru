import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# ==============================================================================
# 1. AYARLAR VE API ANAHTARI
# ==============================================================================
# BURAYA KENDİ API KEY'İNİ DİKKATLİCE YAPIŞTIR (Tırnaklar kalacak)
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg" 

st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

# --- HATA AYIKLAYICI MODEL BAŞLATMA ---
def gemini_modelini_baslat():
    # 1. Kontrol: Anahtar girilmiş mi?
    if not GOOGLE_API_KEY or "BURAYA" in GOOGLE_API_KEY:
        return None, "Lütfen app.py dosyasındaki GOOGLE_API_KEY kısmına şifrenizi yazın."

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Direkt flash modelini zorlayalım
        return genai.GenerativeModel('gemini-1.5-flash'), "OK"
    except Exception as e:
        return None, f"Google Bağlantı Hatası: {str(e)}"

# Modeli başlatmayı dene
model_gemini, chatbot_durumu = gemini_modelini_baslat()

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
        except Exception as e:
            st.error(f"Model dosyası yüklenemedi! Hata: {e}")
            return None
    return None

# ==============================================================================
# 3. SINIF İSİMLERİ
# ==============================================================================
def siniflari_getir(bitki_tipi):
    if bitki_tipi == "Domates (Tomato)":
        # PlantVillage Standart Sırası (Bacterial, Early, Late...)
        return [
            'Bakteriyel Leke',           # 0
            'Erken Yanıklık',            # 1
            'Geç Yanıklık',              # 2
            'Yaprak Küfü',               # 3
            'Septoria Yaprak Lekesi',    # 4
            'Örümcek Akarları',          # 5
            'Hedef Leke',                # 6
            'Sarı Yaprak Kıvırcıklığı',  # 7
            'Mozaik Virüsü',             # 8
            'Sağlıklı'                   # 9
        ]
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
                # --- SABİT BOYUTLANDIRMA (224x224) ---
                # Otomatik algılamayı kaldırdık, standart boyuta zorluyoruz.
                boyut = (224, 224) 
                
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
                    # Eminlik oranı düşükse sarı, yüksekse yeşil göster
                    if guven < 40:
                        st.warning(f"**Teşhis:** {hastalik_ismi} (Emin Değilim)")
                        st.write("⚠️ Model bu fotoğraftan çok emin olamadı. Lütfen daha net veya yakından bir fotoğraf deneyin.")
                    else:
                        st.success(f"**Teşhis:** {hastalik_ismi}")
                    
                    st.info(f"**Eminlik:** %{guven:.2f}")
                    st.session_state['son_teshis'] = hastalik_ismi
                    st.session_state['son_bitki'] = secilen_bitki
                else:
                    st.error("Hata: Sınıf listesi uyumsuz.")

# ==============================================================================
# 5. SOHBET MODU (HATA GÖSTERGELİ)
# ==============================================================================
st.markdown("---")
st.subheader("🤖 Ziraat Asistanı")

# Eğer model başarıyla yüklendiyse sohbeti aç
if chatbot_durumu == "OK":
    if 'son_teshis' in st.session_state:
        st.write(f"**Konu:** {st.session_state['son_bitki']} - {st.session_state['son_teshis']}")
        soru = st.text_input("Sorunuzu buraya yazın (Örn: İlaç önerisi nedir?)")
        
        if st.button("Soruyu Gönder"):
            if soru:
                with st.spinner('Asistan cevaplıyor...'):
                    prompt = f"Sen uzman bir ziraat mühendisisin. Bitki: {st.session_state['son_bitki']}, Hastalık: {st.session_state['son_teshis']}. Soru: '{soru}'. Kısa ve net cevap ver."
                    try:
                        cevap = model_gemini.generate_content(prompt)
                        st.markdown(f"**Cevap:** {cevap.text}")
                    except Exception as e:
                        st.error(f"Cevap alınırken hata: {e}")
    else:
        st.info("Sohbet etmek için önce yukarıdan bir bitki analiz etmelisiniz.")
else:
    # Eğer hata varsa sebebini ekrana KIRMIZI olarak bas
    st.error(f"⚠️ Sohbet Modu Çalışmadı. Sebep: {chatbot_durumu}")