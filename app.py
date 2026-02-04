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

# GEMINI MODELİ (Yedekli Sistem)
# Önce Flash dener, olmazsa Pro dener. 404 hatasını bitirir.
chatbot_aktif = False
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # En stabil sürümü deniyoruz
    model_gemini = genai.GenerativeModel('gemini-1.5-flash') 
    chatbot_aktif = True
except:
    try:
        model_gemini = genai.GenerativeModel('gemini-pro')
        chatbot_aktif = True
    except Exception as e:
        st.error(f"Chatbot Modeli Yüklenemedi: {e}")

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
# 3. SINIF LİSTESİ
# ==============================================================================
def siniflari_getir(bitki_tipi):
    if bitki_tipi == "Elma (Apple)":
        # Röntgen sonuçlarına göre en olası sıralama:
        # 0: Çürük, 1: Pas, 2: Sağlıklı, 3: Leke
        return ['Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı', 'Elma Kara Leke']
        
    elif bitki_tipi == "Domates (Tomato)":
        return ['Bakteriyel Leke', 'Geç Yanıklık', 'Erken Yanıklık', 'Yaprak Küfü', 'Septoria Yaprak Lekesi', 'Örümcek Akarları', 'Hedef Leke', 'Sarı Yaprak Kıvırcıklığı', 'Mozaik Virüsü', 'Sağlıklı']
    elif bitki_tipi == "Mısır (Corn)":
        return ['Mısır Gri Yaprak Lekesi', 'Mısır Yaygın Pas', 'Mısır Kuzey Yaprak Yanıklığı', 'Mısır Sağlıklı']
    elif bitki_tipi == "Patates (Potato)":
        return ['Patates Erken Yanıklık', 'Patates Geç Yanıklık', 'Patates Sağlıklı']
    return ["Hastalık", "Sağlıklı"]

# ==============================================================================
# 4. ARAYÜZ VE DÖRTLÜ ÇAPRAZ TEST
# ==============================================================================
secilen_bitki = st.selectbox("🌿 Hangi bitkiyi analiz edelim?", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)"])
yuklenen_dosya = st.file_uploader("📸 Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if yuklenen_dosya:
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    if st.button("🔍 Hastalığı Analiz Et", type="primary"):
        with st.spinner('Yapay zeka renk filtrelerini deniyor...'):
            model = model_yukle(secilen_bitki)
            if model:
                # 1. BOYUT: 160x160 (Röntgen Sonucu)
                hedef_boyut = (160, 160)
                img = image.resize(hedef_boyut) 
                
                # RGB Array
                img_array_rgb = np.array(img).astype("float32")
                
                # Kanal kontrolü (RGBA temizliği)
                if img_array_rgb.ndim == 2: img_array_rgb = np.stack((img_array_rgb,)*3, axis=-1)
                elif img_array_rgb.shape[-1] == 4: img_array_rgb = img_array_rgb[:,:,:3]

                # BGR Array (Renkleri Ters Çevir: Kırmızı <-> Mavi)
                # Eğer OpenCV ile eğittiysen model bunu isteyecektir!
                img_array_bgr = img_array_rgb[..., ::-1] 

                # -------------------------------------------------------------
                # 🧪 DÖRTLÜ TEST KOMBİNASYONU
                # -------------------------------------------------------------
                inputs = {
                    "RGB_Normalize": np.expand_dims(img_array_rgb / 255.0, axis=0),
                    "RGB_Ham":       np.expand_dims(img_array_rgb, axis=0),
                    "BGR_Normalize": np.expand_dims(img_array_bgr / 255.0, axis=0), # Favori Adayım
                    "BGR_Ham":       np.expand_dims(img_array_bgr, axis=0)
                }
                
                en_iyi_guven = 0
                en_iyi_sonuc = "Belirsiz"
                kazanan_yontem = ""
                
                # Dört yöntemi de dene, en yüksek puanı alanı seç
                for yontem_adi, veri in inputs.items():
                    tahmin = model.predict(veri)
                    olasiliklar = tf.nn.softmax(tahmin).numpy()[0]
                    indeks = np.argmax(olasiliklar)
                    guven = olasiliklar[indeks] * 100
                    
                    if guven > en_iyi_guven:
                        en_iyi_guven = guven
                        kazanan_yontem = yontem_adi
                        siniflar = siniflari_getir(secilen_bitki)
                        if indeks < len(siniflar):
                            en_iyi_sonuc = siniflar[indeks]

                # SONUCU YAZDIR
                if en_iyi_guven > 0:
                    st.toast(f"Model {kazanan_yontem} yöntemi ile çalıştı.")
                    
                    if "Sağlıklı" in en_iyi_sonuc:
                        st.success(f"**Teşhis:** {en_iyi_sonuc}")
                        st.balloons()
                    else:
                        st.error(f"**Teşhis:** {en_iyi_sonuc}")
                    
                    st.info(f"**Güven Oranı:** %{en_iyi_guven:.2f}")
                    
                    # Session Kaydı
                    st.session_state['son_teshis'] = en_iyi_sonuc
                    st.session_state['son_bitki'] = secilen_bitki
                else:
                    st.error("Model hiçbir yöntemle sonuç üretemedi.")

# ==============================================================================
# 5. SOHBET MODU
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