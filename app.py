import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------
# 1. SAYFA AYARLARI
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Ziraat AI - Bitki Doktoru",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Ziraat AI - Akıllı Bitki Hastalığı Tespitçisi")
st.markdown("---")
st.write("Bitki türünü seçin ve yaprağın fotoğrafını yükleyin. Yapay zeka saniyeler içinde analiz etsin.")

# ---------------------------------------------------------------------
# 2. AKILLI MODEL YÜKLEME (Hafıza Dostu)
# ---------------------------------------------------------------------
@st.cache_resource
def model_yukle(bitki_tipi):
    """
    Seçilen bitkiye göre ilgili modeli hafızaya yükler.
    GitHub'daki dosya isimleriyle birebir aynı olmalı.
    """
    model_yolu = ""
    
    # Dosya isimleri GitHub reponuzdaki .keras dosyalarıyla AYNI olmalı
    mapper = {
        "Elma (Apple)": "apple_uzman_model.keras",
        "Yaban Mersini (Blueberry)": "blueberry_uzman_model.keras",
        "Kiraz (Cherry)": "cherry_uzman_model.keras",
        "Mısır (Corn)": "corn_uzman_model.keras",
        "Üzüm (Grape)": "grape_uzman_model.keras",
        "Portakal (Orange)": "orange_uzman_model.keras",
        "Şeftali (Peach)": "peach_uzman_model.keras",
        "Biber (Pepper)": "pepper_uzman_model.keras",
        "Patates (Potato)": "potato_uzman_model.keras",
        "Ahududu (Raspberry)": "raspberry_uzman_model.keras",
        "Soya Fasulyesi (Soybean)": "soybean_uzman_model.keras",
        "Kabak (Squash)": "squash_uzman_model.keras",
        "Çilek (Strawberry)": "strawberry_uzman_model.keras",
        "Domates (Tomato)": "tomato_uzman_model.keras"
    }

    if bitki_tipi in mapper:
        model_yolu = mapper[bitki_tipi]
    
    if model_yolu:
        try:
            # Modeli yükle
            model = tf.keras.models.load_model(model_yolu)
            return model
        except Exception as e:
            st.error(f"⚠️ Model dosyası yüklenemedi: {model_yolu}\nHata: {e}")
            return None
    return None

# ---------------------------------------------------------------------
# 3. SINIF İSİMLERİ (ÖNEMLİ: Eğitim Sırasına Göre!)
# Burası modelin çıktısını (0, 1, 2) Türkçeye çevirir.
# ---------------------------------------------------------------------
def siniflari_getir(bitki_tipi):
    # NOT: Eğer sonuçlar yanlış çıkarsa (örn: Pas yerine Leke diyorsa)
    # buradaki sıralamayı eğitim klasörlerindeki sıralamayla aynı yapın.
    
    if bitki_tipi == "Elma (Apple)":
        return ['Elma Kara Leke', 'Elma Kara Çürüklüğü', 'Elma Sedir Pası', 'Elma Sağlıklı']
    
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

    # Diğerleri için genel bir güvenlik önlemi
    return ["Hastalık Tespit Edildi", "Sağlıklı", "Bilinmiyor"] 

# ---------------------------------------------------------------------
# 4. ARAYÜZ VE İŞLEMLER
# ---------------------------------------------------------------------

# 1. Adım: Bitki Seçimi
secilen_bitki = st.selectbox(
    "🌿 Hangi bitkiyi analiz edelim?",
    [
        "Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Üzüm (Grape)", 
        "Patates (Potato)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)",
        "Yaban Mersini (Blueberry)", "Kiraz (Cherry)", "Portakal (Orange)", 
        "Ahududu (Raspberry)", "Soya Fasulyesi (Soybean)", "Kabak (Squash)"
    ]
)

# 2. Adım: Fotoğraf Yükleme
yuklenen_dosya = st.file_uploader("📸 Yaprak fotoğrafını buraya yükleyin", type=["jpg", "jpeg", "png"])

if yuklenen_dosya is not None:
    # Resmi Ekrana Bas
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Analiz Edilecek Fotoğraf', use_container_width=True)
    
    # Buton
    if st.button("🔍 Hastalığı Analiz Et", type="primary"):
        
        with st.spinner('Yapay zeka motoru çalışıyor...'):
            # Modeli çağır
            model = model_yukle(secilen_bitki)
            
            if model:
                # --- [KRİTİK BÖLÜM] AKILLI BOYUTLANDIRMA ---
                # Modelin giriş boyutunu (Input Shape) otomatik öğreniyoruz.
                try:
                    # Model şekli genelde (None, 256, 256, 3) döner.
                    # Biz buradan 256, 256 kısmını alacağız.
                    input_shape = model.input_shape
                    
                    # Eğer shape (None, None, None, 3) gibi belirsiz gelirse varsayılan 256 yapalım.
                    if input_shape and len(input_shape) >= 3:
                        yukseklik = input_shape[1] if input_shape[1] is not None else 256
                        genislik = input_shape[2] if input_shape[2] is not None else 256
                        hedef_boyut = (yukseklik, genislik)
                    else:
                        hedef_boyut = (256, 256)
                        
                    # Bilgi mesajı (Geliştirici için, isterseniz silebilirsiniz)
                    # st.info(f"Model bu resmi {hedef_boyut} boyutuna dönüştürerek işliyor.")
                    
                except:
                    # Herhangi bir hata olursa standart boyuta dön
                    hedef_boyut = (256, 256)

                # Resmi Hazırla
                img = image.resize(hedef_boyut)
                img_array = np.array(img)

                # Renk kanallarını düzelt (RGBA -> RGB veya Gri -> RGB)
                if img_array.ndim == 2:  # Gri ise
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4:  # PNG (Şeffaf) ise
                    img_array = img_array[:, :, :3]
                
                # Normalize Et (0 ile 1 arasına sıkıştır)
                img_array = img_array.astype("float32") / 255.0
                
                # Batch boyutu ekle (Örn: (256,256,3) -> (1,256,256,3))
                img_array = np.expand_dims(img_array, axis=0)

                # TAHMİN YAP
                try:
                    tahminler = model.predict(img_array)
                    en_yuksek_skor_index = np.argmax(tahminler)
                    guven_orani = np.max(tahminler) * 100
                    
                    # Sonucu İsimlendir
                    sinif_listesi = siniflari_getir(secilen_bitki)
                    
                    # Liste uzunluk kontrolü
                    if en_yuksek_skor_index < len(sinif_listesi):
                        sonuc_ismi = sinif_listesi[en_yuksek_skor_index]
                        
                        st.success(f"**Teşhis:** {sonuc_ismi}")
                        st.progress(int(guven_orani))
                        st.write(f"**Eminlik Oranı:** %{guven_orani:.2f}")
                        
                        if "Sağlıklı" in sonuc_ismi:
                            st.balloons()
                            st.write("Harika! Bitkiniz gayet sağlıklı görünüyor. 🥳")
                        else:
                            st.warning("⚠️ Tedavi yöntemleri için bir uzmana danışmanızı öneririz.")
                    else:
                        st.error(f"Sınıf listesi kısa geldi. Model {en_yuksek_skor_index} numaralı sınıfı buldu ama listede o kadar isim yok.")
                        
                except Exception as e:
                    st.error(f"Tahmin sırasında teknik bir hata oluştu: {e}")
                    st.write("Öneri: Modellerinizin girdi boyutunu (input shape) kontrol edin.")