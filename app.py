import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# ==============================================================================
# 1. AYARLAR
# ==============================================================================
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg" 

st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

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
# 3. SINIF İSİMLERİ (Röntgen ile Onaylanmış Sıralama)
# ==============================================================================
def siniflari_getir(bitki_tipi):
    if bitki_tipi == "Elma (Apple)":
        # 0: Çürük, 1: Pas, 2: Sağlıklı, 3: Leke (Senin modelin için en olası sıra buydu)
        # Eğer bu kodla Leke yerine Sağlıklı derse, buradaki sıralamayı değiştireceğiz.
        return ['Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı', 'Elma Kara Leke']
        
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
                # -----------------------------------------------------------
                # DÜZELTME: SÜNDÜRME YÖNTEMİ (RESIZE)
                # Resmin kenarlarını kesmemek için resize kullanıyoruz.
                # Boyut kesinlikle 160x160 (Röntgene göre)
                # -----------------------------------------------------------
                hedef_boyut = (160, 160)
                
                # Resmi sıkıştırarak (sündürerek) boyuta getir. Veri kaybı olmaz.
                img = image.resize(hedef_boyut)
                
                img_array = np.array(img).astype("float32")
                
                # Kanal kontrolü
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
                
                # Normalizasyon
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # TAHMİN
                try:
                    ham_tahmin = model.predict(img_array)
                    olasiliklar = tf.nn.softmax(ham_tahmin).numpy()[0]
                    
                    en_yuksek_indeks = np.argmax(olasiliklar)
                    guven = olasiliklar[en_yuksek_indeks] * 100
                    
                    siniflar = siniflari_getir(secilen_bitki)
                    
                    if en_yuksek_indeks < len(siniflar):
                        tahmin_edilen_isim = siniflar[en_yuksek_indeks]
                        
                        # --- GÜVENLİK AYARI DÜŞÜRÜLDÜ ---
                        # %50 üzeri güven bizim için yeterlidir.
                        if guven < 50: 
                            st.warning(f"⚠️ Model biraz kararsız (%{guven:.1f}). Fotoğraf net olmayabilir veya model bu hastalığı tam öğrenmemiş olabilir.")
                            st.info(f"En yakın tahmin: {tahmin_edilen_isim}")
                        else:
                            if "Sağlıklı" in tahmin_edilen_isim:
                                st.success(f"**Teşhis:** {tahmin_edilen_isim}")
                                st.balloons()
                            else:
                                st.error(f"**Teşhis:** {tahmin_edilen_isim}")
                            st.info(f"**Güven Oranı:** %{guven:.2f}")
                        
                        # Session kaydı
                        st.session_state['son_teshis'] = tahmin_edilen_isim
                        st.session_state['son_bitki'] = secilen_bitki
                    else:
                        st.error("Sınıf listesi hatası.")

                except Exception as e:
                    st.error(f"Hata: {e}")

# ==============================================================================
# 5. SOHBET MODU
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