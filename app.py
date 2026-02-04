import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------
# 1. AYARLAR VE BAŞLIK
# ---------------------------------------------------------------------
st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

st.title("🌿 Ziraat AI - Akıllı Bitki Hastalığı Tespitçisi")
st.markdown("Bitkinin fotoğrafını yükleyin, yapay zeka hastalığı teşhis etsin.")

# ---------------------------------------------------------------------
# 2. MODEL YÜKLEME FONKSİYONU (LAZY LOADING & CACHING)
# Bu kısım RAM tasarrufu sağlar. Sadece seçilen bitkinin modelini yükler.
# ---------------------------------------------------------------------
@st.cache_resource
def model_yukle(bitki_tipi):
    model_yolu = ""
    
    # Dosya isimlerinin GitHub'daki isimlerle BİREBİR aynı olduğundan emin ol
    if bitki_tipi == "Elma (Apple)":
        model_yolu = "apple_uzman_model.keras"
    elif bitki_tipi == "Yaban Mersini (Blueberry)":
        model_yolu = "blueberry_uzman_model.keras"
    elif bitki_tipi == "Kiraz (Cherry)":
        model_yolu = "cherry_uzman_model.keras"
    elif bitki_tipi == "Mısır (Corn)":
        model_yolu = "corn_uzman_model.keras"
    elif bitki_tipi == "Üzüm (Grape)":
        model_yolu = "grape_uzman_model.keras"
    elif bitki_tipi == "Portakal (Orange)":
        model_yolu = "orange_uzman_model.keras"
    elif bitki_tipi == "Şeftali (Peach)":
        model_yolu = "peach_uzman_model.keras"
    elif bitki_tipi == "Biber (Pepper)":
        model_yolu = "pepper_uzman_model.keras"
    elif bitki_tipi == "Patates (Potato)":
        model_yolu = "potato_uzman_model.keras"
    elif bitki_tipi == "Ahududu (Raspberry)":
        model_yolu = "raspberry_uzman_model.keras"
    elif bitki_tipi == "Soya Fasulyesi (Soybean)":
        model_yolu = "soybean_uzman_model.keras"
    elif bitki_tipi == "Kabak (Squash)":
        model_yolu = "squash_uzman_model.keras"
    elif bitki_tipi == "Çilek (Strawberry)":
        model_yolu = "strawberry_uzman_model.keras"
    elif bitki_tipi == "Domates (Tomato)":
        model_yolu = "tomato_uzman_model.keras"
        
    if model_yolu:
        try:
            model = tf.keras.models.load_model(model_yolu)
            return model
        except Exception as e:
            st.error(f"Model yüklenirken hata oluştu: {e}")
            return None
    return None

# ---------------------------------------------------------------------
# 3. SINIF İSİMLERİ (ÖNEMLİ: Kendi Eğitim Sıralana Göre Düzenle!)
# Burası modelin verdiği 0, 1, 2 sayılarını isme çevirir.
# ---------------------------------------------------------------------
def siniflari_getir(bitki_tipi):
    if bitki_tipi == "Elma (Apple)":
        return ['Elma Kara Leke (Scab)', 'Elma Kara Çürüklüğü (Black Rot)', 'Elma Sedir Pası (Cedar Rust)', 'Elma Sağlıklı']
    
    elif bitki_tipi == "Domates (Tomato)":
        return [
            'Bakteriyel Leke', 'Erken Yanıklık', 'Geç Yanıklık', 'Yaprak Küfü', 
            'Septoria Yaprak Lekesi', 'Örümcek Akarları', 'Hedef Leke', 
            'Sarı Yaprak Kıvırcıklığı Virüsü', 'Mozaik Virüsü', 'Sağlıklı'
        ]
    
    elif bitki_tipi == "Mısır (Corn)":
        return ['Mısır Gri Yaprak Lekesi', 'Mısır Yaygın Pas', 'Mısır Kuzey Yaprak Yanıklığı', 'Mısır Sağlıklı']
    
    elif bitki_tipi == "Üzüm (Grape)":
        return ['Üzüm Kara Çürüklüğü', 'Üzüm Siyah Kızamık (Esca)', 'Üzüm Yaprak Yanıklığı', 'Üzüm Sağlıklı']
    
    elif bitki_tipi == "Patates (Potato)":
        return ['Patates Erken Yanıklık', 'Patates Geç Yanıklık', 'Patates Sağlıklı']
    
    elif bitki_tipi == "Biber (Pepper)":
        return ['Biber Bakteriyel Leke', 'Biber Sağlıklı']
        
    elif bitki_tipi == "Şeftali (Peach)":
        return ['Şeftali Bakteriyel Leke', 'Şeftali Sağlıklı']
    
    elif bitki_tipi == "Çilek (Strawberry)":
        return ['Çilek Yaprak Yanıklığı', 'Çilek Sağlıklı']

    # Diğer bitkiler için varsayılan basit liste (Eğer eğitimde farklıysa burayı düzelt)
    else:
        return ["Hastalık Tespit Edildi", "Sağlıklı"] 

# ---------------------------------------------------------------------
# 4. GÖRÜNTÜ İŞLEME VE ARAYÜZ
# ---------------------------------------------------------------------

# Kullanıcıdan Bitki Seçimi
secilen_bitki = st.selectbox(
    "Lütfen analiz edilecek bitkiyi seçin:",
    [
        "Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Üzüm (Grape)", 
        "Patates (Potato)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)",
        "Yaban Mersini (Blueberry)", "Kiraz (Cherry)", "Portakal (Orange)", 
        "Ahududu (Raspberry)", "Soya Fasulyesi (Soybean)", "Kabak (Squash)"
    ]
)

# Fotoğraf Yükleme Alanı
yuklenen_dosya = st.file_uploader("Bir yaprak fotoğrafı yükleyin...", type=["jpg", "jpeg", "png"])

if yuklenen_dosya is not None:
    # Resmi Göster
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    # Analiz Butonu
    if st.button("Hastalığı Analiz Et"):
        with st.spinner('Yapay Zeka Modeli Yükleniyor ve Analiz Ediliyor...'):
            
            # 1. Seçilen bitkiye uygun modeli yükle
            model = model_yukle(secilen_bitki)
            
            if model:
                # 2. Resmi modele uygun hale getir (224x224 boyutlandırma ve normalize etme)
                img = image.resize((224, 224))
                img_array = np.array(img)
                
                # Eğer resim gri tonlamalıysa (tek kanallıysa) RGB'ye çevir
                if img_array.ndim == 2:
                    img_array = np.stack((img_array,)*3, axis=-1)
                # Eğer resim PNG ise ve 4 kanallıysa (RGBA), sadece ilk 3 kanalı al (RGB)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                    
                img_array = img_array / 255.0  # Normalize et (0-1 arası)
                img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle (1, 224, 224, 3)

                # 3. Tahmin Yap
                tahminler = model.predict(img_array)
                en_yuksek_skor_index = np.argmax(tahminler)
                
                # 4. Sonucu Yazdır
                sinif_listesi = siniflari_getir(secilen_bitki)
                
                # Eğer sınıf listesi modelin çıktısıyla uyuşmuyorsa hata vermemesi için önlem
                if en_yuksek_skor_index < len(sinif_listesi):
                    sonuc = sinif_listesi[en_yuksek_skor_index]
                    guven_orani = tahminler[0][en_yuksek_skor_index] * 100
                    
                    st.success(f"**Sonuç:** {sonuc}")
                    st.info(f"**Doğruluk Oranı:** %{guven_orani:.2f}")
                    
                    # Eğer sağlık oranı düşükse uyarı ver
                    if "Sağlıklı" not in sonuc:
                        st.warning("⚠️ Bitkinizde hastalık tespit edildi. Ziraat mühendisine danışmanız önerilir.")
                    else:
                        st.balloons()
                else:
                    st.error("Hata: Sınıf listesi ile model çıktı sayısı uyuşmuyor. Lütfen app.py içindeki listeyi kontrol edin.")