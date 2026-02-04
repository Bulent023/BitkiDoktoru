import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# ==============================================================================
# AYARLAR
# ==============================================================================
GOOGLE_API_KEY = "AIzaSyC25FnENO9YyyPAlvfWTRyDHfrpii4Pxqg" 

st.set_page_config(page_title="Ziraat AI - Dedektif Modu", page_icon="🕵️‍♂️")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-pro')
    chatbot_aktif = True
except:
    chatbot_aktif = False

st.title("🕵️‍♂️ Ziraat AI - Sıralama Testi")
st.warning("Bu mod, hangi hastalığın hangi sırada olduğunu bulmak içindir.")

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

# ŞİMDİLİK BU LİSTE ÖNEMLİ DEĞİL, ÇÜNKÜ TÜM OLASILIKLARI GÖRECEĞİZ
def siniflari_getir(bitki_tipi):
    if bitki_tipi == "Elma (Apple)":
        return ['Elma Kara Leke', 'Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı']
    # Diğerleri aynı kalabilir...
    return ["Sınıf 1", "Sınıf 2", "Sınıf 3", "Sınıf 4", "Sınıf 5", "Sınıf 6", "Sınıf 7", "Sınıf 8", "Sınıf 9", "Sınıf 10"]

secilen_bitki = st.selectbox("Bitki Seçin", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)"])
yuklenen_dosya = st.file_uploader("Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if yuklenen_dosya and st.button("🔍 Detaylı Analiz Et"):
    with st.spinner('Modelin beyni okunuyor...'):
        model = model_yukle(secilen_bitki)
        image = Image.open(yuklenen_dosya)
        st.image(image, caption='Yüklenen Resim', width=300)

        if model:
            # GÖRÜNTÜ İŞLEME (Senin %99 aldığın ayarlar)
            img = image.resize((224, 224))
            img_array = np.array(img).astype("float32") / 255.0
            if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
            img_array = np.expand_dims(img_array, axis=0)

            # TAHMİN
            preds = model.predict(img_array)
            olasiliklar = tf.nn.softmax(preds).numpy()[0] # Softmax ile yüzdeleri düzelt

            st.write("### 📊 Modelin Aklındaki Tüm Sıralama:")
            
            # Tüm sınıfların yüzdelerini tek tek yazdırıyoruz
            mevcut_liste = siniflari_getir(secilen_bitki)
            
            for i, skor in enumerate(olasiliklar):
                yuzde = skor * 100
                cubuk = "🟩" * int(yuzde / 5)
                # Eğer listede isim varsa yaz, yoksa Sınıf X yaz
                isim = mevcut_liste[i] if i < len(mevcut_liste) else f"Sınıf {i}"
                
                st.write(f"**Sıra {i} ({isim}):** %{yuzde:.2f}  {cubuk}")

            st.info("👆 Lütfen yukarıdaki listede EN YÜKSEK (yeşil çubuğu en uzun) olanın 'Sıra Numarasını' bana söyle.")