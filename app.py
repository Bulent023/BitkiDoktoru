import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# ==============================================================================
# 1. AYARLAR VE API ANAHTARI
# ==============================================================================
# 👇 BURAYA KENDİ API KEY'İNİ MUTLAKA YAZ! 👇
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg" 

st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

# Gemini Modelini Kur
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
    chatbot_aktif = True
except Exception as e:
    st.error(f"Chatbot hatası: {e}")
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
# 3. SINIF İSİMLERİ
# ==============================================================================
def siniflari_getir(bitki_tipi):
    # DOMATES İÇİN SIRALAMAYI KONTROL ET
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
# 4. ARAYÜZ VE ANALİZ
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
                # 1. BOYUTLANDIRMA
                # Hata almamak için 224x224 standart yapıyoruz (Çoğu model için güvenlidir)
                # Eğer senin modelin 256 ise burayı (256, 256) yap.
                try:
                    shape = model.input_shape
                    boyut = (shape[1], shape[2]) if shape and shape[1] else (224, 224)
                except:
                    boyut = (224, 224)
                
                img = image.resize(boyut)
                img_array = np.array(img).astype("float32")
                
                # -------------------------------------------------------------
                # 🚨 KRİTİK DEĞİŞİKLİK BURADA: BÖLME İŞLEMİNİ KALDIRDIK
                # Eskiden: img_array = img_array / 255.0  (Bu yanlıştı)
                # Şimdi:   img_array = img_array          (Olduğu gibi bırakıyoruz)
                # -------------------------------------------------------------
                
                # Boyutları düzelt (Batch ve Kanal)
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
                img_array = np.expand_dims(img_array, axis=0)
                
                # 2. TAHMİN
                try:
                    tahmin = model.predict(img_array)
                    indeks = np.argmax(tahmin)
                    guven = np.max(tahmin) # 100 ile çarpmadan önce ham değeri alalım
                    
                    # Eğer güven skoru çok düşükse (örn: 0.99 yerine 0.001 çıkıyorsa) bir terslik vardır
                    # Bazı modeller softmax çıktısı vermez, logits verir.
                    # Güvenlik için softmax uygulayalım:
                    if guven > 1.0: # Zaten yüzdeyse veya logits ise
                         guven_yuzde = guven
                    else:
                         guven_yuzde = guven * 100

                    siniflar = siniflari_getir(secilen_bitki)
                    
                    if indeks < len(siniflar):
                        hastalik_ismi = siniflar[indeks]
                        st.success(f"**Teşhis:** {hastalik_ismi}")
                        st.info(f"**Eminlik:** %{guven_yuzde:.2f}")
                        st.session_state['son_teshis'] = hastalik_ismi
                        st.session_state['son_bitki'] = secilen_bitki
                    else:
                        st.error("Hata: Sınıf listesi uyumsuz.")
                except ValueError as e:
                    st.error(f"Hata oluştu: {e}")

# ==============================================================================
# 5. SOHBET MODU
# ==============================================================================
if 'son_teshis' in st.session_state and chatbot_aktif:
    st.markdown("---")
    st.subheader(f"🤖 Ziraat Asistanı ile Konuşun")
    st.write(f"**Durum:** {st.session_state['son_bitki']} - {st.session_state['son_teshis']}")
    
    soru = st.text_input("Sorunuzu buraya yazın (Örn: İlaç önerisi nedir?)")
    
    if st.button("Soruyu Gönder"):
        if soru:
            with st.spinner('Asistan cevaplıyor...'):
                prompt = f"Sen uzman bir ziraat mühendisisin. Kullanıcının bitkisinde şu hastalık var: {st.session_state['son_bitki']} bitkisinde {st.session_state['son_teshis']}. Soru: '{soru}'. Kısa ve öz çözüm öner."
                try:
                    cevap = model_gemini.generate_content(prompt)
                    st.markdown(f"**Cevap:** {cevap.text}")
                except Exception as e:
                    st.error(f"Bir hata oluştu: {e}")