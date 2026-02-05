import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import google.generativeai as genai
import time
from fpdf import FPDF # PDF kütüphanesi

# ==============================================================================
# 1. AYARLAR
# ==============================================================================
st.set_page_config(page_title="Ziraat AI - Bitki Doktoru", page_icon="🌿")

# KOTA AYARLARI
SORU_LIMITI = 20        
BEKLEME_SURESI = 15     

st.title("🌿 Ziraat AI - Akıllı Bitki Doktoru")

# ==============================================================================
# 2. YARDIMCI FONKSİYONLAR (PDF İÇİN)
# ==============================================================================
# PDF kütüphanesi Türkçe karakterlerde (Ş,Ğ,İ) sorun çıkarabilir.
# Bu fonksiyon raporun bozuk görünmemesi için karakterleri düzeltir.
def tr_duzelt(text):
    source = "şŞıİğĞüÜöÖçÇ"
    target = "sSiIgGuUoOcC"
    translation_table = str.maketrans(source, target)
    return text.translate(translation_table)

def rapor_olustur(bitki, hastalik, recete):
    pdf = FPDF()
    pdf.add_page()
    
    # Başlık
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="ZIRAAT AI - TESHIS RAPORU", ln=1, align='C')
    pdf.ln(10) # Boşluk
    
    # Bilgiler
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=tr_duzelt(f"Tarih: {time.strftime('%d-%m-%Y')}"), ln=1)
    pdf.cell(200, 10, txt=tr_duzelt(f"Analiz Edilen Bitki: {bitki}"), ln=1)
    pdf.cell(200, 10, txt=tr_duzelt(f"Tespit Edilen Durum: {hastalik}"), ln=1)
    pdf.ln(10)
    
    # Yapay Zeka Tavsiyesi
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="YAPAY ZEKA ONERISI VE RECETE:", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=tr_duzelt(recete))
    
    # Alt Bilgi
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, txt="Bu rapor yapay zeka tarafindan uretilmistir. Kesin teshis icin uzmana danisiniz.", align='C')
    
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ==============================================================================
# 3. GEMINI BAĞLANTISI
# ==============================================================================
@st.cache_resource
def gemini_baglan():
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=api_key)
            
            oncelikli_modeller = [
                'gemini-1.5-flash',
                'gemini-1.5-flash-latest',
                'gemini-1.5-pro',
                'gemini-1.0-pro',
                'gemini-pro'
            ]
            
            for m in oncelikli_modeller:
                try:
                    test_model = genai.GenerativeModel(m)
                    test_model.generate_content("System check") 
                    return test_model, m 
                except:
                    continue
            
            # Yedek plan (Yasaklı modeller hariç)
            tum_modeller = genai.list_models()
            for m in tum_modeller:
                if 'generateContent' in m.supported_generation_methods:
                    if 'gemini-2.5' in m.name: continue 
                    try:
                        yedek_model = genai.GenerativeModel(m.name)
                        yedek_model.generate_content("System check")
                        return yedek_model, m.name
                    except:
                        continue

            return None, "Model Bulunamadı"
        return None, "Anahtar Yok"
    except Exception as e:
        return None, str(e)

model_gemini, aktif_model_ismi = gemini_baglan()

if model_gemini:
    st.caption(f"✅ Sistem Hazır: `{aktif_model_ismi}`")
else:
    st.error(f"⚠️ Bağlantı Hatası: {aktif_model_ismi}")

st.markdown("---")

# ==============================================================================
# 4. TEŞHİS MODELİ YÜKLEME
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
# 5. KULLANICI OTURUM TAKİBİ
# ==============================================================================
if 'soru_sayaci' not in st.session_state:
    st.session_state['soru_sayaci'] = 0
if 'son_soru_zamani' not in st.session_state:
    st.session_state['son_soru_zamani'] = 0
if 'rapor_hazir' not in st.session_state:
    st.session_state['rapor_hazir'] = None # PDF verisini burada tutacağız

# ==============================================================================
# 6. ARAYÜZ VE ANALİZ
# ==============================================================================
secilen_bitki = st.selectbox("🌿 Hangi bitkiyi analiz edelim?", ["Elma (Apple)", "Domates (Tomato)", "Mısır (Corn)", "Patates (Potato)", "Üzüm (Grape)", "Biber (Pepper)", "Şeftali (Peach)", "Çilek (Strawberry)"])
yuklenen_dosya = st.file_uploader("📸 Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if yuklenen_dosya:
    image = Image.open(yuklenen_dosya)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    if st.button("🔍 Hastalığı Analiz Et ve Raporla", type="primary"):
        with st.spinner('Yapay zeka analiz ediyor ve reçete yazıyor...'):
            model = model_yukle(secilen_bitki)
            if model:
                hedef_boyut = (160, 160)
                img = image.resize(hedef_boyut) 
                img_array = np.array(img).astype("float32")
                if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[-1] == 4: img_array = img_array[:,:,:3]

                img_array = img_array[..., ::-1] # BGR Düzeltmesi
                input_data = np.expand_dims(img_array, axis=0)
                
                try:
                    tahmin = model.predict(input_data)
                    olasiliklar = tf.nn.softmax(tahmin).numpy()[0]
                    indeks = np.argmax(olasiliklar)
                    guven = olasiliklar[indeks] * 100
                    siniflar = siniflari_getir(secilen_bitki)
                    
                    if indeks < len(siniflar):
                        sonuc_ismi = siniflar[indeks]
                        
                        # --- OTOMATİK RAPOR OLUŞTURMA KISMI ---
                        recete_metni = "Hastalık sağlıklı olduğu için tedavi gerekmez."
                        
                        if "Sağlıklı" in sonuc_ismi:
                            st.success(f"**Teşhis:** {sonuc_ismi}")
                            st.balloons()
                        else:
                            st.error(f"**Teşhis:** {sonuc_ismi}")
                            
                            # Hastalık varsa Gemini'den reçete iste (Kotadan düşmez, sistem kullanır)
                            if model_gemini:
                                prompt_rapor = f"Bitki: {secilen_bitki}. Hastalık: {sonuc_ismi}. Bu hastalık için çiftçiye uygulanabilir, maddeler halinde kısa bir tedavi reçetesi ve ilaç önerisi yaz. Türkçe karakter kullanma (ornek: ş yerine s yaz)."
                                try:
                                    response = model_gemini.generate_content(prompt_rapor)
                                    recete_metni = response.text
                                except:
                                    recete_metni = "Yapay zeka reçete oluştururken bir hata oluştu."

                        st.info(f"**Güven Oranı:** %{guven:.2f}")
                        
                        # PDF Oluştur ve Hafızaya Al
                        pdf_data = rapor_olustur(secilen_bitki, sonuc_ismi, recete_metni)
                        st.session_state['rapor_hazir'] = pdf_data
                        
                        st.session_state['son_teshis'] = sonuc_ismi
                        st.session_state['son_bitki'] = secilen_bitki
                    else:
                        st.error("Liste hatası.")
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")

    # --- PDF İNDİRME BUTONU ---
    if st.session_state['rapor_hazir']:
        st.download_button(
            label="📄 PDF Raporunu İndir",
            data=st.session_state['rapor_hazir'],
            file_name="ziraat_ai_rapor.pdf",
            mime="application/pdf",
            type="secondary"
        )

# ==============================================================================
# 7. SOHBET MODU
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
                prompt = f"Sen uzman bir ziraat mühendisisin. Bitki: {st.session_state['son_bitki']}. Hastalık: {st.session_state['son_teshis']}. Soru: '{soru}'. Kısa cevap ver."
                try:
                    cevap = model_gemini.generate_content(prompt)
                    st.write(cevap.text)
                    st.session_state['soru_sayaci'] += 1
                    st.session_state['son_soru_zamani'] = time.time()
                except Exception as e:
                    st.error(f"Hata: {e}")
                    
elif 'son_teshis' in st.session_state and not model_gemini:
     st.warning("⚠️ Sohbet sistemi şu an mola verdi.")