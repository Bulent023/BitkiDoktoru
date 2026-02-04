import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import google.generativeai as genai

# ==============================================================================
# 1. API ANAHTARI
# ==============================================================================
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg" 

st.set_page_config(page_title="Ziraat AI - Kalibrasyon", page_icon="🔧")

# ==============================================================================
# 2. GEMINI MODELİNİ (404 HATASINA KARŞI) GÜVENLİ YÜKLEME
# ==============================================================================
model_gemini = None
chatbot_aktif = False

if GOOGLE_API_KEY != "BURAYA_KENDI_API_KEYINI_YAPISTIR":
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Sırasıyla modelleri dener, hangisi çalışırsa onu seçer
    modeller = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']
    
    for m in modeller:
        try:
            test_model = genai.GenerativeModel(m)
            # Ufak bir test sorusu soralım
            test_model.generate_content("Test")
            model_gemini = test_model
            chatbot_aktif = True
            print(f"✅ Başarılı Model: {m}")
            break
        except:
            continue

if not chatbot_aktif:
    st.warning("⚠️ Gemini modellerine erişilemedi. API Key'i veya bölgeyi kontrol et.")

st.title("🔧 Ziraat AI - Model Kalibrasyonu")
st.info("Bu mod, modelin hangi hastalığa hangi numarayı verdiğini çözmek içindir.")

# ==============================================================================
# 3. MODEL YÜKLEME
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
            st.error(f"Model yüklenemedi: {e}")
            return None
    return None

# ==============================================================================
# 4. ARAYÜZ VE KALİBRASYON ANALİZİ
# ==============================================================================
secilen_bitki = st.selectbox("🌿 Hangi bitkiyi test edelim?", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)"])
yuklenen_dosya = st.file_uploader("📸 Fotoğraf Yükle (Pas veya Leke)", type=["jpg", "png", "jpeg"])

if yuklenen_dosya and st.button("🧠 Modelin Beynini Oku"):
    model = model_yukle(secilen_bitki)
    
    if model:
        image = Image.open(yuklenen_dosya)
        st.image(image, caption='Yüklenen Fotoğraf', width=200)
        
        # 1. BOYUT: 160x160 (Röntgende çıkan kesin boyut)
        hedef_boyut = (160, 160)
        img = image.resize(hedef_boyut)
        
        # Array işlemleri
        img_array = np.array(img).astype("float32")
        
        # Kanal kontrolü
        if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]

        # -------------------------------------------------------------
        # ÇİFT TEST: Hem Normalize (0-1) Hem Ham (0-255)
        # -------------------------------------------------------------
        
        inputs = {
            "Giriş A (0-1 arası)": np.expand_dims(img_array / 255.0, axis=0),
            "Giriş B (0-255 arası)": np.expand_dims(img_array, axis=0)
        }
        
        st.write("### 📊 Modelin Verdiği Cevaplar (İsimsiz)")
        
        for ad, veri in inputs.items():
            tahmin = model.predict(veri)
            olasiliklar = tf.nn.softmax(tahmin).numpy()[0]
            
            st.write(f"--- **{ad}** Sonuçları ---")
            
            # Sadece numaraları yazdırıyoruz, isimleri değil!
            for i, skor in enumerate(olasiliklar):
                yuzde = skor * 100
                cubuk = "🟦" * int(yuzde / 5)
                st.write(f"**SINIF {i}:** %{yuzde:.2f} {cubuk}")
            
            kazanan = np.argmax(olasiliklar)
            st.info(f"🏆 Bu ayarla Kazanan: **SINIF {kazanan}**")

        st.warning("""
        **LÜTFEN BANA ŞUNU YAZ:**
        1. Yüklediğin fotoğraf neydi? (Örn: Pas)
        2. Hangi Sınıf Numarası kazandı? (Örn: Sınıf 2)
        """)

# ==============================================================================
# 5. SOHBET MODU
# ==============================================================================
if chatbot_aktif:
    st.markdown("---")
    st.subheader("🤖 Sohbet Testi")
    soru = st.text_input("Bot çalışıyor mu diye bir şey yaz:")
    if st.button("Gönder"):
        if soru:
            try:
                cevap = model_gemini.generate_content(soru)
                st.success(cevap.text)
            except Exception as e:
                st.error(f"Hata: {e}")