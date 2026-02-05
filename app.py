import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import google.generativeai as genai
import time

# ==============================================================================
# 1. AYARLAR
# ==============================================================================
st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

# KOTA AYARLARI
SORU_LIMITI = 20        # Kullanıcı başına günlük soru hakkı
BEKLEME_SURESI = 15     # Spam koruması (saniye)

st.title("🌿 Ziraat AI - Akıllı Bitki Doktoru")

# ==============================================================================
# 2. GEMINI BAĞLANTISI (YASAKLI MODELLER ENGELLENDİ) 🛡️
# ==============================================================================
@st.cache_resource
def gemini_baglan():
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=api_key)
            
            # SADECE BU MODELLERİ KULLAN (Diğerleri yasak)
            # 2.5-flash gibi düşük kotalı modelleri listeye almıyoruz.
            izin_verilen_modeller = [
                'gemini-1.5-flash',          # ÖNCELİK 1: En yüksek kota (1500/gün)
                'gemini-1.5-flash-latest',   # ÖNCELİK 2: Alternatif sürüm
                'gemini-1.5-pro',            # ÖNCELİK 3: Pro sürüm
                'gemini-1.0-pro'             # ÖNCELİK 4: Eski ama sağlam sürüm
            ]
            
            # Sadece listedekileri dene. Bulamazsan hata ver (Düşük kotalıya gitme).
            for m in izin_verilen_modeller:
                try:
                    test_model = genai.GenerativeModel(m)
                    test_model.generate_content("System check") 
                    return test_model, m # Çalışan modeli ve ismini döndür
                except:
                    continue
            
            return None, "Uygun Model Bulunamadı"
                    
        return None, "Anahtar Yok"
    except Exception as e:
        return None, str(e)

# Bağlantıyı Başlat
model_gemini, aktif_model_ismi = gemini_baglan()

# Durum Bildirimi
if model_gemini:
    st.caption(f"✅ Yapay Zeka Hazır: `{aktif_model_ismi}` (Yüksek Kota)")
else:
    st.error("⚠️ Yapay Zeka Bağlantı Hatası: Yüksek kotalı modellerden hiçbirine erişilemedi.")

st.markdown("---")

# ==============================================================================
# 3. TEŞHİS MODELİ YÜKLEME
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

# ==============================================================================
# 4. KULLANICI OTURUM TAKİBİ
# ==============================================================================
if 'soru_sayaci' not in st.session_state:
    st.session_state['soru_sayaci'] = 0

if 'son_soru_zamani' not in st.session_state:
    st.session_state['son_soru_zamani'] = 0

# ==============================================================================
# 5. ARAYÜZ VE ANALİZ
# ==============================================================================
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

                # BGR DÖNÜŞÜMÜ
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
                            st.balloons()
                        else:
                            st.error(f"**Teşhis:** {sonuc_ismi}")
                        
                        st.info(f"**Güven Oranı:** %{guven:.2f}")
                        st.session_state['son_teshis'] = sonuc_ismi
                        st.session_state['son_bitki'] = secilen_bitki
                    else:
                        st.error("Liste hatası.")
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")

# ==============================================================================
# 6. SOHBET MODU
# ==============================================================================
if 'son_teshis' in st.session_state and model_gemini:
    st.markdown("---")
    st.subheader(f"🤖 Ziraat Asistanı ile Konuşun")
    
    kalan_hak = SORU_LIMITI - st.session_state['soru_sayaci']
    st.progress(st.session_state['soru_sayaci'] / SORU_LIMITI, text=f"Günlük Soru Hakkı: {kalan_hak} kaldı")
    
    st.write(f"**Durum:** {st.session_state['son_bitki']} - {st.session_state['son_teshis']}")
    
    soru = st.text_input("Sorunuzu buraya yazın...")
    
    if st.button("Soruyu Gönder"):
        if st.session_state['soru_sayaci'] >= SORU_LIMITI:
            st.error("🚫 Bu oturumdaki soru limitiniz doldu! Yarın tekrar bekleriz.")
        
        elif (time.time() - st.session_state['son_soru_zamani']) < BEKLEME_SURESI:
            kalan_sure = int(BEKLEME_SURESI - (time.time() - st.session_state['son_soru_zamani']))
            st.warning(f"⏳ Biraz yavaşlayalım! Lütfen {kalan_sure} saniye daha bekle.")
            
        elif soru:
            with st.spinner('Cevaplanıyor...'):
                prompt = f"Sen uzman bir ziraat mühendisisin. Kullanıcının bitkisi: {st.session_state['son_bitki']}. Teşhis edilen hastalık: {st.session_state['son_teshis']}. Kullanıcı sorusu: '{soru}'. Bu soruya kısa, öz ve çiftçi dostu bir dille cevap ver. Tedavi yöntemlerinden bahset."
                try:
                    cevap = model_gemini.generate_content(prompt)
                    st.write(cevap.text)
                    st.session_state['soru_sayaci'] += 1
                    st.session_state['son_soru_zamani'] = time.time()
                except Exception as e:
                    st.error(f"Hata: {e}")
                    
elif 'son_teshis' in st.session_state and not model_gemini:
     st.warning("⚠️ Sohbet sistemi şu an mola verdi (Kota Limiti).")