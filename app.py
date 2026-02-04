import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# ==============================================================================
# 1. AYARLAR VE API KEY
# ==============================================================================
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg" 

st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

# Gemini Pro (Chatbot)
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-pro')
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
# 3. SINIF LİSTESİ (Dedektif Modunda Doğruladığımız Sıralama)
# ==============================================================================
def siniflari_getir(bitki_tipi):
    if bitki_tipi == "Elma (Apple)":
        # 0: Leke, 1: Çürük, 2: Pas, 3: Sağlıklı
        return ['Elma Kara Leke', 'Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı']
        
    elif bitki_tipi == "Domates (Tomato)":
        return ['Bakteriyel Leke', 'Geç Yanıklık', 'Erken Yanıklık', 'Yaprak Küfü', 'Septoria Yaprak Lekesi', 'Örümcek Akarları', 'Hedef Leke', 'Sarı Yaprak Kıvırcıklığı', 'Mozaik Virüsü', 'Sağlıklı']
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
        with st.spinner('Yapay zeka inceliyor...'):
            model = model_yukle(secilen_bitki)
            if model:
                # 1. BOYUTLANDIRMA (Dedektif modu ile aynı - RESIZE)
                hedef_boyut = (224, 224)
                
                # Model shape kontrolü
                try:
                    if model.input_shape and model.input_shape[1]:
                        hedef_boyut = (model.input_shape[1], model.input_shape[2])
                except:
                    pass

                img = image.resize(hedef_boyut)
                img_array = np.array(img).astype("float32")
                
                # Kanal kontrolü
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
                
                # 2. NORMALİZASYON (/255.0)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # 3. TAHMİN
                try:
                    ham_tahmin = model.predict(img_array)
                    olasiliklar = tf.nn.softmax(ham_tahmin).numpy()[0]
                    
                    # En yüksek puanı alan sınıfı bul
                    en_yuksek_indeks = np.argmax(olasiliklar)
                    guven = olasiliklar[en_yuksek_indeks] * 100
                    
                    siniflar = siniflari_getir(secilen_bitki)
                    tahmin_edilen_isim = siniflar[en_yuksek_indeks]

                    # --- [GÜVENLİK MEKANİZMASI BAŞLANGICI] ---
                    # Eğer model "Sağlıklı" dediyse AMA güven oranı %80'den düşükse:
                    # Bu demektir ki model aslında şüpheli bir şey gördü ama tam emin olamadı.
                    # Biz riske atmayıp ikinci en yüksek ihtimale (hastalığa) bakacağız.
                    
                    if "Sağlıklı" in tahmin_edilen_isim and guven < 80:
                        # Sağlıklı ihtimalini sıfırla ve tekrar en yükseği bul
                        olasiliklar[en_yuksek_indeks] = 0 
                        yeni_indeks = np.argmax(olasiliklar)
                        yeni_guven = olasiliklar[yeni_indeks] * 100
                        
                        # Yeni tahmin bir hastalık mı?
                        yeni_isim = siniflar[yeni_indeks]
                        if "Sağlıklı" not in yeni_isim:
                            tahmin_edilen_isim = yeni_isim
                            guven = yeni_guven
                            st.warning("⚠️ Model ilk başta 'Sağlıklı' sandı ama yaprakta şüpheli lekeler tespit edildi.")
                    # --- [GÜVENLİK MEKANİZMASI BİTİŞİ] ---

                    # SONUCU YAZDIR
                    if "Sağlıklı" in tahmin_edilen_isim:
                        st.success(f"**Teşhis:** {tahmin_edilen_isim}")
                        st.balloons()
                    else:
                        st.error(f"**Teşhis:** {tahmin_edilen_isim}")
                        
                    st.info(f"**Güven Oranı:** %{guven:.2f}")
                    
                    st.session_state['son_teshis'] = tahmin_edilen_isim
                    st.session_state['son_bitki'] = secilen_bitki

                except Exception as e:
                    st.error(f"Hata: {e}")

# ==============================================================================
# 5. SOHBET MODU (GEMINI PRO)
# ==============================================================================
if 'son_teshis' in st.session_state and chatbot_aktif:
    st.markdown("---")
    st.subheader(f"🤖 Ziraat Asistanı ile Konuşun")
    st.write(f"**Konu:** {st.session_state['son_bitki']} - {st.session_state['son_teshis']}")
    
    soru = st.text_input("Sorunuzu buraya yazın...")
    
    if st.button("Soruyu Gönder"):
        if soru:
            with st.spinner('Cevap hazırlanıyor...'):
                prompt = f"Sen bir ziraat mühendisisin. Bitki: {st.session_state['son_bitki']}, Hastalık: {st.session_state['son_teshis']}. Soru: {soru}. Kısa ve net cevap ver."
                try:
                    cevap = model_gemini.generate_content(prompt)
                    st.write(cevap.text)
                except Exception as e:
                    st.error(f"Hata: {e}")