import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# ==============================================================================
# 1. AYARLAR
# ==============================================================================
# 👇 BURAYA KENDİ API KEY'İNİ MUTLAKA YAZ! 👇
GOOGLE_API_KEY = "BURAYA_KENDI_API_KEYINI_YAPISTIR" 

st.set_page_config(page_title="Ziraat AI - Röntgen Modu", page_icon="🧬")

st.title("🧬 Model Röntgen Cihazı")
st.warning("Bu mod, modelin hangi hastalığa hangi sayıyı verdiğini bulmak içindir.")

# ==============================================================================
# 2. MODEL YÜKLEME
# ==============================================================================
@st.cache_resource
def model_yukle(bitki_tipi):
    # Sadece Elma üzerinden test yapıyoruz şimdilik
    if bitki_tipi == "Elma (Apple)":
        return tf.keras.models.load_model("apple_uzman_model.keras")
    return None

# ==============================================================================
# 3. ANALİZ EKRANI
# ==============================================================================
secilen_bitki = st.selectbox("Test Edilecek Bitki", ["Elma (Apple)"])
yuklenen_dosya = st.file_uploader("Fotoğraf Yükle (Pas veya Külleme)", type=["jpg", "png", "jpeg"])

if yuklenen_dosya and st.button("🧠 Modelin Beynini Oku"):
    model = model_yukle(secilen_bitki)
    
    if model:
        image = Image.open(yuklenen_dosya)
        st.image(image, width=250, caption="Yüklenen Resim")
        
        # --- RESMİ HAZIRLA ---
        # 1. Standart Boyut (Genelde 256 veya 224)
        hedef_boyut = (224, 224)
        try:
            if model.input_shape and model.input_shape[1]:
                hedef_boyut = (model.input_shape[1], model.input_shape[2])
        except:
            pass
            
        st.info(f"Model şu boyutta istiyor: {hedef_boyut}")

        img = image.resize(hedef_boyut)
        img_array = np.array(img).astype("float32")
        
        # 2. Normalizasyon (ÖNEMLİ: Senin modelin 255'e bölmeli mi bölmemeli mi?)
        # Bunu test etmek için hem bölerek hem bölmeyerek bakacağız.
        img_array_norm = img_array / 255.0  # Normalize edilmiş
        img_array_raw = img_array           # Normalize edilmemiş
        
        # Boyut Ekle
        if img_array_norm.ndim == 2: 
            img_array_norm = np.stack((img_array_norm,)*3, axis=-1)
            img_array_raw = np.stack((img_array_raw,)*3, axis=-1)
        elif img_array_norm.shape[-1] == 4: 
            img_array_norm = img_array_norm[:,:,:3]
            img_array_raw = img_array_raw[:,:,:3]
            
        input_norm = np.expand_dims(img_array_norm, axis=0)
        # input_raw = np.expand_dims(img_array_raw, axis=0) # Gerekirse bunu da deneriz

        # --- TAHMİN ---
        tahmin = model.predict(input_norm)
        
        # --- SONUÇLARI DÖK ---
        st.write("---")
        st.subheader("📊 Model Çıktısı (Ham Skorlar)")
        
        cikis_sayisi = len(tahmin[0])
        st.write(f"**Modelin Bildiği Hastalık Sayısı:** {cikis_sayisi}")
        
        # Ham değerleri Softmax'e sokalım ki yüzde görelim
        olasiliklar = tf.nn.softmax(tahmin).numpy()[0]
        
        for i in range(cikis_sayisi):
            yuzde = olasiliklar[i] * 100
            cubuk = "🟩" * int(yuzde / 5)
            
            # Burada tahmini isimler YAZMIYORUM, sadece SIRA NUMARASI yazıyorum.
            # Böylece hangisinin hangisi olduğunu sen söyleyeceksin.
            st.write(f"**Sıra {i} (Neuron {i}):** %{yuzde:.2f}  {cubuk}")
            
        en_yuksek = np.argmax(olasiliklar)
        st.error(f"🏆 KAZANAN: **Sıra {en_yuksek}**")
        
        st.markdown("""
        ### 🕵️‍♂️ Şimdi Ne Yapacağız?
        1. Eğer yüklediğin resim **PAS** ise ve kazanan **Sıra 2** ise -> Listede 2. sıraya 'Pas' yazacağız.
        2. Eğer yüklediğin resim **KÜLLEME** ise ve model saçmalıyorsa (düşük puanlar) -> Model Külleme bilmiyor demektir.
        """)