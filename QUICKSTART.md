# FC26 Squad Optimizer - Hızlı Başlangıç Rehberi

## 🚀 İlk Çalıştırma (5 Dakika)

### Adım 1: Kurulum Kontrolü

```powershell
# Python versiyonunu kontrol edin
python --version
# Çıktı: Python 3.8+ olmalı

# Proje klasörüne gidin
cd d:\fc26_squad_optimizer

# Virtual environment aktif mi kontrol edin
# Terminalde (venv) yazısını görmeli siniz
```

### Adım 2: Paketleri Yükleyin (İlk Kez)

```powershell
pip install -r requirements.txt
```

**Beklenen Süre:** 2-3 dakika

**Yüklenen Paketler:**
- ✅ streamlit (Web GUI)
- ✅ pandas, numpy (Veri işleme)
- ✅ scikit-learn, xgboost (ML)
- ✅ matplotlib, seaborn, plotly (Grafikler)

### Adım 3: Web GUI'yi Başlatın

```powershell
streamlit run app.py
```

**Otomatik olarak tarayıcınızda açılır:**
- URL: http://localhost:8501
- Port: 8501

### Adım 4: Veri Yükleyin

1. Sol menüden **"📂 Veri Yükle"** butonuna tıklayın
2. 18,000+ oyuncu verisi yüklenecek
3. ✅ "Veri Yüklendi" mesajını görmelisiniz

### Adım 5: Keşfetmeye Başlayın!

**Önerilen Sıra:**
1. 📊 **Veri Analizi** - Oyuncuları keşfedin
2. 🤖 **Makine Öğrenmesi** - Model eğitin (5-10 dk)
3. 🧬 **Takım Optimizasyonu** - Takımınızı kurun (1-2 dk)

---

## 💻 Komut Satırı Kullanımı

Eğer terminal üzerinden çalışmayı tercih ediyorsanız:

```powershell
python main.py
```

**İnteraktif Menü:**
- Veri yükleme
- ML modeli eğitme (e/h)
- Sinerji NN eğitme (e/h)
- Takım optimizasyonu
- Sonuçları kaydetme

---

## 🎯 Hızlı Test Senaryoları

### Test 1: Basit Takım Kurma (2 dakika)

**Web GUI:**
1. Veri Yükle
2. Takım Optimizasyonu sayfasına git
3. Bütçe: €50M
4. Formasyon: 4-3-3
5. "Optimizasyonu Başlat" ✅

**Komut Satırı:**
```powershell
python main.py
# Enter tuşuna basın (varsayılan değerler)
# ML: h (hayır)
# Sinerji: h (hayır)
# Bütçe: 50000000 (veya Enter)
# Formasyon: 1 (4-3-3)
```

### Test 2: ML ile Gelişmiş (10 dakika)

**Web GUI:**
1. Makine Öğrenmesi → Model Eğit
2. Test oranı: 0.2
3. "Model Eğitimini Başlat" (5 dk bekle)
4. Değeri düşük oyuncular bul
5. Takım Optimizasyonu → ML kullan ✅

### Test 3: Full Özellikli (15 dakika)

**Tüm özellikleri aktif:**
1. Veri yükle
2. ML model eğit
3. Sinerji NN eğit
4. Takım optimize et (ML + Sinerji aktif)
5. Tüm sonuçları kaydet

---

## 🐛 Sık Karşılaşılan Sorunlar

### ❌ "streamlit: command not found"

**Çözüm:**
```powershell
pip install streamlit
# veya
python -m pip install streamlit
```

### ❌ "ModuleNotFoundError: No module named 'plotly'"

**Çözüm:**
```powershell
pip install plotly
```

### ❌ "FileNotFoundError: players.csv"

**Çözüm:**
- CSV dosyasını `data/players.csv` konumuna koyun
- Dosya adının tam olarak `players.csv` olduğundan emin olun

### ❌ Port 8501 zaten kullanımda

**Çözüm:**
```powershell
# Farklı port kullanın
streamlit run app.py --server.port 8502
```

### ❌ Streamlit beyaz ekran gösteriyor

**Çözüm:**
```powershell
# Cache'i temizleyin
streamlit cache clear
# Tarayıcı cache'ini temizleyin (Ctrl+F5)
```

---

## ⚙️ Özelleştirme

### Varsayılan Bütçeyi Değiştirme

`app.py` dosyasında:
```python
budget = st.number_input(
    "💰 Bütçe (EUR)",
    value=50000000,  # ← Burası değiştir
    ...
)
```

### Nesil Sayısını Artırma

```python
generations = st.slider(
    "🔄 Nesil Sayısı", 
    10, 100, 30  # ← Son değeri değiştir (varsayılan)
)
```

### Yeni Formasyon Ekleme

`src/genetic_algorithm.py` dosyasında:
```python
self.formations = {
    '433': ['GK', 'LB', 'CB', 'CB', 'RB', 'CM', 'CM', 'CM', 'LW', 'ST', 'RW'],
    '442': [...],
    '532': ['GK', 'CB', 'CB', 'CB', 'LM', 'RM', 'CM', 'CM', 'ST', 'ST'],  # YENİ
}
```

---

## 📊 Performans İpuçları

### Hızlı Optimizasyon İçin:
- Nesil: 20
- Popülasyon: 30
- ML: Devre dışı
- Sinerji: Devre dışı
- **Süre:** ~30 saniye

### Dengeli Optimizasyon İçin:
- Nesil: 30
- Popülasyon: 50
- ML: Aktif
- Sinerji: Devre dışı
- **Süre:** ~2 dakika

### En İyi Sonuç İçin:
- Nesil: 50-100
- Popülasyon: 100
- ML: Aktif
- Sinerji: Aktif
- **Süre:** ~5-10 dakika

---

## 🎨 GUI Özellikleri

### Ana Sayfa 🏠
- Hızlı istatistikler
- Kayıtlı modeller
- Son takımlar

### Veri Analizi 📊
- İnteraktif filtreler
- 4 farklı sekme
- 10+ grafik türü
- Oyuncu arama

### Makine Öğrenmesi 🤖
- 3 model karşılaştırma
- Gerçek zamanlı metrikler
- Değeri düşük oyuncu bulma
- Model kaydetme/yükleme

### Takım Optimizasyonu 🧬
- Bütçe/formasyon seçimi
- İlerleme göstergesi
- Detaylı kadro listesi
- Sinerji analizi

---

## 📝 Komutlar Özeti

```powershell
# Kurulum
pip install -r requirements.txt

# Web GUI (Önerilen)
streamlit run app.py

# Komut Satırı
python main.py

# Cache Temizleme
streamlit cache clear

# Paket Güncelleme
pip install --upgrade streamlit pandas numpy scikit-learn

# Virtual Environment
python -m venv venv
venv\Scripts\activate

# Deaktive
deactivate
```

---

## 🌐 Tarayıcı Önerileri

**En İyi Deneyim:**
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Edge 90+
- ⚠️ Safari (bazı animasyonlar çalışmayabilir)

---

## 📱 Ekran Çözünürlüğü

**Önerilen:**
- Minimum: 1366x768
- Optimal: 1920x1080
- 4K: Tam destek

**Mobil:**
- Responsive design
- Tablet: Tam destek
- Telefon: Kısıtlı destek

---

## 🔒 Güvenlik Notları

- Uygulama **localhost** (127.0.0.1) üzerinde çalışır
- Dışarıdan erişim kapalıdır
- Veri sadece lokal olarak işlenir
- Model dosyaları lokal olarak saklanır

**Internet paylaşımı için:**
```powershell
streamlit run app.py --server.address 0.0.0.0
# DİKKAT: Güvenlik riski! Sadece güvenli ağlarda kullanın
```

---

## 📞 Yardım Alma

1. **Dokumentasyon:** README.md ve SETUP_GUIDE.md okuyun
2. **Hata Mesajları:** Tam hata metnini kopyalayın
3. **GitHub Issues:** Detaylı açıklama ile sorun bildirin
4. **Loglar:** Streamlit terminalindeki tüm çıktıyı paylaşın

---

## ✅ Başarı Kontrol Listesi

Hazırsınız! Şunları tamamladıysanız:

- [x] Python 3.8+ kurulu
- [x] Paketler yüklendi (`pip install -r requirements.txt`)
- [x] `data/players.csv` dosyası var
- [x] `streamlit run app.py` çalışıyor
- [x] Tarayıcıda http://localhost:8501 açıldı
- [x] Sol menüden "Veri Yükle" tıklandı
- [x] ✅ "Veri Yüklendi" mesajı görüldü

🎉 **Tebrikler! Kullanıma hazırsınız!**

---

**Made with ❤️ - Have fun optimizing! ⚽**
