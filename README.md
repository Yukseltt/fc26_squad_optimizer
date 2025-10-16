# ⚽ FC26 Squad Optimizer

Modern yapay zeka teknikleri kullanarak optimal futbol takımı oluşturma sistemi.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Özellikler

- **📊 Veri Analizi**: 18,000+ oyuncu verisini interaktif grafiklerle keşfedin
- **🤖 Makine Öğrenmesi**: 3 farklı ML modeli ile oyuncu değeri tahmini
- **🧠 Neural Network**: Takım sinerjisini öğrenen 3 katmanlı yapay zeka
- **🧬 Genetik Algoritma**: Evrimsel optimizasyon ile en iyi takımı bulun
- **💰 Bütçe Yönetimi**: Gerçekçi transfer bütçeleri ile optimizasyon
- **⚽ Çoklu Formasyon**: 4-3-3, 4-4-2, 3-5-2, 4-2-3-1 formasyonları
- **🎨 Modern Web GUI**: Streamlit ile kullanıcı dostu arayüz

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler

- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)

### 2. Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/Yukseltt/fc26_squad_optimizer.git
cd fc26_squad_optimizer

# Virtual environment oluşturun (opsiyonel ama önerilen)
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

### 3. Veri Hazırlığı

FC26 oyuncu CSV dosyanızı `data/players.csv` olarak kaydedin.

### 4. Uygulamayı Çalıştırma

#### Web GUI ile (Önerilen) 🌐

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` açılacaktır.

#### Komut Satırı ile 💻

```bash
python main.py
```

## 📖 Kullanım Kılavuzu

### Web GUI Kullanımı

1. **Ana Sayfa** 🏠
   - Projeye genel bakış
   - Hızlı istatistikler
   - Kayıtlı modeller ve takımlar

2. **Veri Analizi** 📊
   - Oyuncu istatistikleri
   - İnteraktif grafikler
   - Overall/yaş/değer dağılımları
   - Milliyet ve lig analizi
   - Oyuncu arama ve filtreleme

3. **Makine Öğrenmesi** 🤖
   - Model eğitimi (Random Forest, Gradient Boosting, XGBoost)
   - Model performans karşılaştırması
   - Değeri düşük oyuncu bulma
   - Model kaydetme ve yükleme

4. **Takım Optimizasyonu** 🧬
   - Bütçe ve formasyon seçimi
   - Genetik algoritma parametreleri
   - ML ve NN entegrasyonu
   - Optimal takım oluşturma
   - Sinerji analizi

### Komut Satırı Kullanımı

```bash
python main.py
```

Program sizi adım adım yönlendirecektir:

1. Veri yükleme
2. ML model eğitimi (opsiyonel)
3. Neural Network eğitimi (opsiyonel)
4. Genetik algoritma optimizasyonu
5. Sonuçları kaydetme

## 🏗️ Proje Yapısı

```
FC26_SQUAD_OPTIMIZER/
│
├── app.py                      # Streamlit web uygulaması (YENİ!)
├── main.py                     # Komut satırı uygulaması
├── requirements.txt            # Python bağımlılıkları
├── README.md                   # Bu dosya
├── SETUP_GUIDE.md             # Detaylı kurulum rehberi
│
├── data/
│   └── players.csv            # Oyuncu verileri
│
├── models/
│   └── trained_models/        # Eğitilmiş ML modelleri
│
├── results/
│   ├── best_squads/          # Oluşturulan takımlar
│   └── ml_results_*.png      # ML performans grafikleri
│
├── notebooks/
│   ├── data_analysis.py      # Veri analizi scripti
│   ├── ml_models_script.py   # ML eğitim scripti
│   └── optimization_script.py # Optimizasyon scripti
│
└── src/
    ├── __init__.py
    ├── data_loader.py         # Veri yükleme ve temizleme
    ├── ml_models.py           # ML modelleri
    ├── team_synergy_nn.py     # Neural Network sinerjisi
    ├── genetic_algorithm.py   # Genetik algoritma
    └── utils.py               # Yardımcı fonksiyonlar
```

## 🛠️ Teknolojiler

### Veri İşleme
- **pandas**: Veri manipülasyonu ve analizi
- **numpy**: Sayısal hesaplamalar

### Makine Öğrenmesi
- **scikit-learn**: ML modelleri ve metrikler
- **xgboost**: Gelişmiş gradient boosting

### Görselleştirme
- **matplotlib**: Statik grafikler
- **seaborn**: İstatistiksel görselleştirme
- **plotly**: İnteraktif grafikler

### Web Uygulaması
- **streamlit**: Modern web GUI framework

### Diğer
- **joblib**: Model serialization

## 🧠 Algoritmalar

### 1. Makine Öğrenmesi Modelleri

#### Random Forest Regressor
- Ensemble learning yaklaşımı
- 100 decision tree
- Overfitting'e karşı robust

#### Gradient Boosting Regressor
- Sequential ensemble method
- Hataları minimize eden yaklaşım

#### XGBoost Regressor
- Extreme gradient boosting
- En yüksek performans
- Regularization ile overfitting önleme

### 2. Neural Network (Team Synergy)

```
Input Layer (28+ features)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Hidden Layer 3 (32 neurons, ReLU)
    ↓
Output Layer (Synergy Score)
```

**Sinerji Faktörleri:**
- Temel istatistikler (overall, potential)
- Yaş dengesi (prime age, young players)
- Kimya skorları (milliyet, lig, kulüp)
- Oyun stili uyumu (pace, physic, technical)

### 3. Genetik Algoritma

**Evrimsel Optimizasyon:**
1. **İlk Popülasyon**: Rastgele takımlar oluştur
2. **Fitness Değerlendirme**: Her takımı skorla
3. **Seçilim**: En iyi takımları seç (elitism)
4. **Çaprazlama**: İki takımı birleştir
5. **Mutasyon**: Rastgele oyuncu değişimi
6. **Yeni Nesil**: Süreci tekrarla

**Fitness Fonksiyonu:**
```
Total Fitness = 
    (Position Score × 0.3 + Overall × 0.7) +
    (Chemistry Bonus × 0.5) +
    (Synergy NN Score × 2.0)
```

## 📊 Model Performansı

Tipik performans metrikleri (test seti):

| Model | R² Score | MAE (EUR) | RMSE (EUR) |
|-------|----------|-----------|------------|
| Random Forest | 0.87-0.90 | €3-5M | €6-8M |
| Gradient Boosting | 0.88-0.91 | €2.5-4M | €5-7M |
| XGBoost | 0.89-0.92 | €2-3.5M | €4-6M |

## 💡 Kullanım Senaryoları

### Senaryo 1: Düşük Bütçe ile Rekabetçi Takım
```
Bütçe: €20-30M
Strateji: ML ile değeri düşük oyuncuları bul
Hedef: Overall 80+ takım kur
```

### Senaryo 2: Galaktik Kadro
```
Bütçe: €500M+
Strateji: En yüksek overall oyuncular
Hedef: 85+ overall takım, maksimum sinerji
```

### Senaryo 3: Gençlik Projesi
```
Bütçe: €50M
Strateji: Potential > Overall oyuncular
Hedef: Genç yıldızlar (18-23 yaş)
```

## 🔍 Örnek Kullanım

### Python Scripti

```python
from src.data_loader import DataLoader
from src.genetic_algorithm import GeneticSquadOptimizer

# Veri yükle
loader = DataLoader('data/players.csv')
df = loader.load_data()
df = loader.clean_data()

# Genetik algoritma
optimizer = GeneticSquadOptimizer(df, formation='433')

result = optimizer.optimize(
    budget=50000000,  # €50M
    population_size=50,
    generations=30
)

# Sonuçları göster
optimizer.print_squad(result)
```

## 🎓 Öğrenme Kaynakları

Bu proje şu konuları öğrenmek için idealdir:

- ✅ Makine Öğrenmesi (Supervised Learning)
- ✅ Neural Networks (Deep Learning)
- ✅ Genetik Algoritmalar (Optimization)
- ✅ Veri Analizi ve Görselleştirme
- ✅ Web Uygulaması Geliştirme
- ✅ Python OOP (Object-Oriented Programming)
- ✅ Feature Engineering
- ✅ Model Evaluation & Cross-Validation

## 🐛 Sorun Giderme

### Problem: ModuleNotFoundError

**Çözüm:**
```bash
pip install -r requirements.txt
```

### Problem: FileNotFoundError: players.csv

**Çözüm:**
- CSV dosyasının `data/players.csv` konumunda olduğundan emin olun
- Dosya adının tam olarak `players.csv` olduğunu kontrol edin

### Problem: Streamlit çalışmıyor

**Çözüm:**
```bash
pip install --upgrade streamlit
streamlit --version
```

### Problem: Optimizasyon çok uzun sürüyor

**Çözüm:**
- Nesil sayısını azaltın (örn: 30 → 20)
- Popülasyon boyutunu azaltın (örn: 50 → 30)
- ML/NN özelliklerini devre dışı bırakın

## 📈 Gelecek Geliştirmeler

- [ ] **API Entegrasyonu**: Canlı piyasa verileri
- [ ] **Daha Fazla Formasyon**: 5-3-2, 4-1-4-1, vb.
- [ ] **Veritabanı**: SQLite/PostgreSQL entegrasyonu
- [ ] **Deep Learning**: LSTM/Transformer modelleri
- [ ] **Multi-objective Optimization**: Pareto cephesi
- [ ] **Docker Konteyner**: Kolay dağıtım
- [ ] **Unit Tests**: pytest ile test coverage
- [ ] **CI/CD Pipeline**: GitHub Actions
- [ ] **Mobile Support**: Responsive design
- [ ] **Cloud Deployment**: Heroku/AWS

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları takip edin:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'e push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👨‍💻 Yazar

**Yukseltt**
- GitHub: [@Yukseltt](https://github.com/Yukseltt)

## 🌟 Teşekkürler

- FIFA/EA Sports - Oyuncu verileri için
- Scikit-learn & XGBoost ekipleri
- Streamlit community
- Python community

---

**Made with ❤️ using Python & Streamlit**

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!
