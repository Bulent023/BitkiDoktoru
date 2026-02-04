import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import google.generativeai as genai

st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")
st.title("🌿 Ziraat AI - Akıllı Bitki Doktoru")

# ==============================================================================
# 🔍 TANI KOYMA MODU (DEBUG)
# ==============================================================================
chatbot_aktif = False

# 1. KASA KONTROLÜ
if "GOOGLE_API_KEY" in st.secrets:
    st.toast("✅ Kasa Bağlantısı Başarılı: Anahtar bulundu.")
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # 2. ANAHTAR SAĞLAM MI KONTROLÜ
    try:
        genai.configure(api_key=api_key)
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        response = model_gemini.generate_content("Merhaba")
        chatbot_aktif = True
        st.toast("✅ Google Gemini Bağlantısı Başarılı!")
    except Exception as e:
        st.error(f"🚨 ANAHTAR HATASI: Kasa dolu ama anahtar çalışmıyor. Google'dan gelen hata: {e}")
        chatbot_aktif = False
else:
    st.error("🚨 KASA HATASI: 'Secrets' içinde 'GOOGLE_API_KEY' bulunamadı.")
    st.info("Lütfen Streamlit panelindeki 'Secrets' ayarını kontrol et.")
    chatbot_aktif = False

st.markdown("---")

# ==============================================================================
# MODEL YÜKLEME VE TEŞHİS (Standart Kod)
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

secilen_bitki = st.selectbox("🌿 Hangi bitkiyi analiz edelim?", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)"])
yuklenen_dosya = st.file_uploader("📸 Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if yuklenen_dosya:
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    if st.button("🔍 Hastalığı Analiz Et", type="primary"):
        with st.spinner('Yapay zeka analiz ediyor...'):
            model = model_yukle(secilen_bitki)
            if model:
                hedef_boyut = (160, 160)
                img = image.resize(hedef_boyut) 
                img_array = np.array(img).astype("float32")
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
                
                # RENK DÜZELTME (BGR)
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
                        if "Sağlıklı" in sonuc_ismi:
                            st.success(f"**Teşhis:** {sonuc_ismi}")
                        else:
                            st.error(f"**Teşhis:** {sonuc_ismi}")
                        st.info(f"**Güven Oranı:** %{guven:.2f}")
                        st.session_state['son_teshis'] = sonuc_ismi
                        st.session_state['son_bitki'] = secilen_bitki
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")

# SOHBET KISMI
if 'son_teshis' in st.session_state and chatbot_aktif:
    st.markdown("---")
    st.subheader("🤖 Ziraat Asistanı")
    soru = st.text_input("Sorunuzu yazın...")
    if st.button("Soruyu Gönder"):
        if soru:
            try:
                cevap = model_gemini.generate_content(f"Bitki: {st.session_state['son_bitki']}, Durum: {st.session_state['son_teshis']}. Soru: {soru}")
                st.write(cevap.text)
            except Exception as e:
                st.error(f"Hata: {e}")
elif not chatbot_aktif:
    st.warning("⚠️ Chatbot devre dışı. Lütfen yukarıdaki kırmızı hata mesajını okuyun.")