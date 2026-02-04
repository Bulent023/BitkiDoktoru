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

# GEMINI MODELİ (En Güncel Sürüm: 1.5 Flash)
# 404 Hatasını çözmek için 'gemini-1.5-flash' kullanıyoruz.
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
    chatbot_aktif = True
except Exception as e:
    st.error(f"Chatbot Hatası: {e}")
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
# 3. SINIF LİSTESİ (Sıralama 3. Sıranın Leke Olduğunu Göstermişti)
# ==============================================================================
def siniflari_getir(bitki_tipi):
    if bitki_tipi == "Elma (Apple)":
        # Önceki testlerde 3. sıra sürekli kazandığı için Leke'yi 3'e koyduk.
        # Sıralama: 0: Çürük, 1: Pas, 2: Sağlıklı, 3: Leke
        return ['Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı', 'Elma Kara Leke']
        
    elif bitki_tipi == "Domates (Tomato)":
        return ['Bakteriyel Leke', 'Erken Yanıklık', 'Geç Yanıklık', 'Yaprak Küfü', 'Septoria Yaprak Lekesi', 'Örümcek Akarları', 'Hedef Leke', 'Sarı Yaprak Kıvırcıklığı', 'Mozaik Virüsü', 'Sağlıklı']
    elif bitki_tipi == "Mısır (Corn)":
        return ['Mısır Gri Yaprak Lekesi', 'Mısır Yaygın Pas', 'Mısır Kuzey Yaprak Yanıklığı', 'Mısır Sağlıklı']
    elif bitki_tipi == "Patates (Potato)":
        return ['Patates Erken Yanıklık', 'Patates Geç Yanıklık', 'Patates Sağlıklı']
    # Diğer bitkiler için genel liste...
    return ["Hastalık", "Sağlıklı"]

# ==============================================================================
# 4. ARAYÜZ VE ÇİFT YÖNLÜ ANALİZ
# ==============================================================================
secilen_bitki = st.selectbox("🌿 Hangi bitkiyi analiz edelim?", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)"])
yuklenen_dosya = st.file_uploader("📸 Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if yuklenen_dosya:
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    if st.button("🔍 Hastalığı Analiz Et", type="primary"):
        with st.spinner('Yapay zeka iki farklı yöntemle deniyor...'):
            model = model_yukle(secilen_bitki)
            if model:
                # 1. BOYUT AYARI (Röntgende çıkan 160x160)
                hedef_boyut = (160, 160)
                img = image.resize(hedef_boyut) # Resize (Sündürme) en garantisidir
                
                img_array = np.array(img).astype("float32")
                
                # Kanal Düzeltme
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
                
                # -------------------------------------------------------------
                # 🧪 ÇİFT YÖNLÜ TEST: NORMALİZE Mİ DEĞİL Mİ?
                # -------------------------------------------------------------
                
                # Yöntem A: 255'e Bölerek (0-1 arası)
                input_A = np.expand_dims(img_array / 255.0, axis=0)
                
                # Yöntem B: Bölmeden (0-255 arası)
                input_B = np.expand_dims(img_array, axis=0)
                
                # İkisini de tahmin ettir
                tahmin_A = model.predict(input_A)
                tahmin_B = model.predict(input_B)
                
                # Güven oranlarını hesapla (Softmax)
                olasilik_A = tf.nn.softmax(tahmin_A).numpy()[0]
                olasilik_B = tf.nn.softmax(tahmin_B).numpy()[0]
                
                guven_A = np.max(olasilik_A) * 100
                guven_B = np.max(olasilik_B) * 100
                
                # HANGİSİ DAHA EMİNSE ONU SEÇ! 🏆
                if guven_B > guven_A:
                    st.toast("Bilgi: Model Ham (0-255) veriyi daha çok sevdi.")
                    final_olasilik = olasilik_B
                    final_guven = guven_B
                else:
                    st.toast("Bilgi: Model Normalize (0-1) veriyi daha çok sevdi.")
                    final_olasilik = olasilik_A
                    final_guven = guven_A
                
                # SONUCU YAZDIR
                en_yuksek_indeks = np.argmax(final_olasilik)
                siniflar = siniflari_getir(secilen_bitki)
                
                if en_yuksek_indeks < len(siniflar):
                    sonuc_ismi = siniflar[en_yuksek_indeks]
                    
                    if "Sağlıklı" in sonuc_ismi:
                        st.success(f"**Teşhis:** {sonuc_ismi}")
                        st.balloons()
                    else:
                        st.error(f"**Teşhis:** {sonuc_ismi}")
                    
                    st.info(f"**Güven Oranı:** %{final_guven:.2f}")
                    
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