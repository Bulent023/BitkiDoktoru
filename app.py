import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# ==============================================================================
# 1. AYARLAR VE API ANAHTARI (BURAYI DOLDUR!)
# ==============================================================================
# Buraya kendi Gemini API Key'ini tırnak içine yazmalısın.
# Eğer Streamlit Secrets kullanıyorsan oradan da çekebilirsin.
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg" 

# Sayfa Ayarları
st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

# Gemini Modelini Kur
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-pro')
    chatbot_aktif = True
except:
    chatbot_aktif = False

st.title("🌿 Ziraat AI - Akıllı Bitki Doktoru")
st.markdown("---")

# ==============================================================================
# 2. MODEL YÜKLEME VE OPTİMİZASYON (RAM DOSTU)
# ==============================================================================
@st.cache_resource
def model_yukle(bitki_tipi):
    # Dosya eşleştirmeleri
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
# 3. SINIF İSİMLERİ (BU SIRALAMA ÇOK ÖNEMLİ!)
# ==============================================================================
def siniflari_getir(bitki_tipi):
    # DOMATES İÇİN STANDART SIRALAMA (Alfabetik: Bacterial, Early, Late, Leaf Mold...)
    # Eğer sonucun yanlış çıkıyorsa buradaki sırayı eğitim klasörlerine göre değiştir.
    if bitki_tipi == "Domates (Tomato)":
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
        
    # Diğerleri için varsayılan (Hata almamak için)
    return ["Hastalık Tespit Edildi", "Sağlıklı", "Bilinmiyor"]

# ==============================================================================
# 4. ARAYÜZ VE İŞLEMLER
# ==============================================================================
secilen_bitki = st.selectbox(
    "🌿 Hangi bitkiyi analiz edelim?",
    ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)"]
)

yuklenen_dosya = st.file_uploader("📸 Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if yuklenen_dosya:
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    if st.button("🔍 Hastalığı Analiz Et", type="primary"):
        with st.spinner('Yapay zeka inceliyor...'):
            # 1. Modeli Yükle
            model = model_yukle(secilen_bitki)
            
            if model:
                # 2. Resmi Hazırla (Akıllı Boyutlandırma)
                try:
                    shape = model.input_shape
                    boyut = (shape[1], shape[2]) if shape and shape[1] else (256, 256)
                except:
                    boyut = (256, 256)
                
                img = image.resize(boyut)
                img_array = np.array(img).astype("float32") / 255.0
                
                # Boyut düzeltme
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
                img_array = np.expand_dims(img_array, axis=0)
                
                # 3. Tahmin
                tahmin = model.predict(img_array)
                indeks = np.argmax(tahmin)
                guven = np.max(tahmin) * 100
                
                siniflar = siniflari_getir(secilen_bitki)
                
                if indeks < len(siniflar):
                    hastalik_ismi = siniflar[indeks]
                    st.success(f"**Teşhis:** {hastalik_ismi}")
                    st.info(f"**Eminlik:** %{guven:.2f}")
                    
                    # Sonucu Session State'e kaydet (Sohbet için lazım)
                    st.session_state['son_teshis'] = hastalik_ismi
                    st.session_state['son_bitki'] = secilen_bitki
                else:
                    st.error("Hata: Sınıf listesi uyumsuz.")

# ==============================================================================
# 5. YAPAY ZEKA SOHBET MODU (YENİ EKLENEN KISIM)
# ==============================================================================
if 'son_teshis' in st.session_state and chatbot_aktif:
    st.markdown("---")
    st.subheader(f"🤖 Ziraat Asistanı ile Konuşun")
    st.write(f"**Teşhis edilen durum:** {st.session_state['son_bitki']} - {st.session_state['son_teshis']}")
    st.write("Bu hastalıkla ilgili tedavi yöntemlerini, ilaçları veya kültürel önlemleri sorabilirsiniz.")

    soru = st.text_input("Sorunuzu buraya yazın (Örn: Bu hastalık için hangi ilacı kullanmalıyım?)")
    
    if st.button("Soruyu Gönder"):
        if soru:
            with st.spinner('Asistan cevaplıyor...'):
                prompt = f"Sen uzman bir ziraat mühendisisin. Kullanıcının bitkisinde şu hastalık tespit edildi: {st.session_state['son_bitki']} bitkisinde {st.session_state['son_teshis']}. Kullanıcının sorusu şu: '{soru}'. Buna göre bilimsel, pratik ve çözüm odaklı kısa bir cevap ver."
                try:
                    cevap = model_gemini.generate_content(prompt)
                    st.markdown(f"**Cevap:** {cevap.text}")
                except Exception as e:
                    st.error(f"Hata oluştu: {e}. Lütfen API anahtarınızı kontrol edin.")
    
elif not chatbot_aktif:
    st.warning("⚠️ Sohbet özelliğini kullanmak için kodun en başına geçerli bir 'GOOGLE_API_KEY' eklemelisiniz.")