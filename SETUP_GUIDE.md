# FC26 Squad Optimizer - Kurulum Rehberi 🚀

## ✅ Tamamlanması Gereken Adımlar

### 1. Dosya Yapısını Oluşturma

Aşağıdaki klasör yapısını oluşturun:

```
FC26_SQUAD_OPTIMIZER/
│
├── data/
│   └── players.csv          ← SİZİN CSV DOSYANIZ
│
├── models/
│   └── trained_models/      ← Boş klasör
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_ml_models.ipynb
│   └── 03_optimization.ipynb
│
├── results/
│   └── best_squads/         ← Boş klasör
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── ml_models.py
│   ├── genetic_algorithm.py
│   ├── team_synergy_nn.py   ← YENİ!
│   └── utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── SETUP_GUIDE.md (bu dosya)
```

### 2. VSCode'da Terminal Açma

1. VSCode'u açın
2. `Ctrl + ö` (veya View → Terminal)
3. Proje klasörüne gidin:
   ```bash
   cd C:\path\to\FC26_SQUAD_OPTIMIZER
   ```

### 3. Python Versiyonunu Kontrol Etme

```bash
python --version
```

**Gerekli:** Python 3.8 veya üzeri

Eğer yüklü değilse: https://www.python.org/downloads/

### 4. Sanal Ortam Oluşturma (Önerilen)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

Terminal'de `(venv)` yazısını görmelisiniz.

### 5. Kütüphaneleri Yükleme

```bash
pip install -r requirements.txt
```

Bu işlem birkaç dakika sürebilir. Yüklenen kütüphaneler:
- ✅ pandas (veri işleme)
- ✅ numpy (hesaplamalar)
- ✅ scikit-learn (ML modelleri)
- ✅ xgboost (gelişmiş ML)
- ✅ matplotlib & seaborn (grafikler)
- ✅ jupyter (notebook'lar)

### 6. CSV Dosyanızı Yerleştirme

1. FC26 oyuncu CSV dosyanızı bulun
2. `data` klasörüne kopyalayın
3. Adını **tam olarak** `players.csv` yapın

**ÖNEMLI:** Dosya yolu: `data/players.csv`

### 7. İlk Çalıştırma

```bash
python main.py
```

## 🎯 Hızlı Test

### Basit Test (5 dakika)

```bash
python
```

```python
from src.data_loader import DataLoader

# Veriyi yükle
loader = DataLoader('data/players.csv')
df = loader.load_data()
df = loader.clean_data()

# İstatistikleri göster
stats = loader.get_statistics()
print(stats)

# İlk 5 oyuncuyu göster
print(df[['short_name', 'overall', 'value_eur']].head())
```

Çıkış yapın: `exit()`

### Tam Test (ML olmadan - 2 dakika)

```bash
python main.py
```

1. **Veri yükleme** → Enter
2. **ML eğitimi?** → `h` (hayır)
3. **Sinerji NN?** → `h` (hayır)
4. **Bütçe** → Enter (50M)
5. **Formasyon** → `1` (4-3-3)
6. Sonuç gelir! ✅

### Tam Test (ML ile - 10 dakika)

```bash
python main.py
```

1. **ML eğitimi?** → `e` (evet)
2. **Grafik kaydet?** → `e`
3. **Model kaydet?** → `e`
4. **Değeri düşük oyuncular?** → `e`
5. **Sinerji NN?** → `e` ⭐
6. **Sentetik takım sayısı** → Enter (1000)
7. **Sinerji kaydet?** → `e`
8. **Bütçe** → `100000000`
9. **Formasyon** → `1`
10. **ML kullan?** → `e`
11. **Sinerji kullan?** → `e` 🧠
12. Muhteşem sonuç! 🎉

## 📊 Jupyter Notebook Kullanımı

### Notebook'ları Başlatma

```bash
jupyter notebook
```

Tarayıcınızda otomatik açılır.

### Hangi Notebook'u Kullanmalıyım?

1. **`01_data_analysis.ipynb`** 
   - Veriyi keşfetmek için
   - Grafikler ve istatistikler
   - 10 dakika

2. **`02_ml_models.ipynb`**
   - ML modellerini eğitmek için
   - Feature importance
   - 15 dakika

3. **`03_optimization.ipynb`**
   - Genetik algoritma testleri
   - Farklı bütçe/formasyon denemeleri
   - 20 dakika

## 🐛 Sorun Giderme

### Problem 1: "ModuleNotFoundError"

**Çözüm:**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib jupyter
```

### Problem 2: "FileNotFoundError: players.csv"

**Kontrol listesi:**
- ✅ CSV dosyası `data` klasöründe mi?
- ✅ Dosya adı tam olarak `players.csv` mi?
- ✅ Terminal doğru klasörde mi? (`pwd` veya `cd` ile kontrol)

### Problem 3: "KeyError: 'gk'" veya pozisyon hatası

**Çözüm:**
CSV'nizde şu sütunlar olmalı:
- `gk`, `lb`, `cb`, `rb`, `lwb`, `rwb`
- `cdm`, `cm`, `cam`, `lm`, `rm`
- `lw`, `rw`, `st`, `cf`, `lf`, `rf`

CSV'nizi açıp kontrol edin.

### Problem 4: "MemoryError" veya çok yavaş

**Çözüm 1:** Daha az sentetik veri
```bash
# main.py çalıştırırken
Sentetik takım sayısı: 500  # 1000 yerine
```

**Çözüm 2:** Parametreleri azalt
```python
# main.py içinde düzenle
result = optimizer.optimize(
    budget=budget,
    population_size=30,    # 50 yerine
    generations=20,        # 30 yerine
    ...
)
```

### Problem 5: GPU/CUDA hatası (XGBoost)

**Çözüm:**
```bash
pip uninstall xgboost
pip install xgboost
```

### Problem 6: Jupyter başlamıyor

**Çözüm:**
```bash
pip install --upgrade jupyter notebook ipykernel
python -m ipykernel install --user
```

## 🎓 İlk Kez Python Kullanıyorum

### Adım 1: Python Kurulumu

1. https://www.python.org/downloads/
2. "Download Python 3.x" butonuna tıklayın
3. İndirdiğiniz .exe'yi çalıştırın
4. ✅ **"Add Python to PATH"** işaretleyin!
5. "Install Now" tıklayın

### Adım 2: VSCode Kurulumu

1. https://code.visualstudio.com/
2. İndirip kurun
3. Python extension yükleyin (Ctrl+Shift+X → "Python")

### Adım 3: Terminal Kullanımı

**Windows:**
- Terminal açma: Ctrl + ö
- Klasör değiştirme: `cd C:\Users\...\FC26_SQUAD_OPTIMIZER`
- Dosya listeleme: `dir`

**Mac/Linux:**
- Terminal açma: Ctrl + ö
- Klasör değiştirme: `cd ~/Desktop/FC26_SQUAD_OPTIMIZER`
- Dosya listeleme: `ls`

## 🚀 Gelişmiş Kullanım

### Kendi Script'inizi Yazma

`my_optimizer.py` dosyası oluşturun:

```python
from src.data_loader import DataLoader
from src.genetic_algorithm import GeneticSquadOptimizer
from src.team_synergy_nn import TeamSynergyPredictor

# Veriyi yükle
loader = DataLoader('data/players.csv')
df = loader.load_data()
df = loader.clean_data()

# Sinerji NN eğit
synergy = TeamSynergyPredictor()
synergy.train(df, n_samples=500)

# Farklı bütçelerle test et
budgets = [10_000_000, 50_000_000, 100_000_000]

for budget in budgets:
    optimizer = GeneticSquadOptimizer(df, formation='433')
    result = optimizer.optimize(
        budget=budget,
        use_synergy=True,
        synergy_predictor=synergy
    )
    print(f"\nBütçe {budget:,}: Fitness={result['fitness']:.1f}")
```

Çalıştırma:
```bash
python my_optimizer.py
```

### Modeli Kaydet ve Yükle

```python
# Eğit ve kaydet
synergy = TeamSynergyPredictor()
synergy.train(df, n_samples=1000)
synergy.save_model('models/trained_models/my_synergy.pkl')

# Sonra yükle
synergy2 = TeamSynergyPredictor()
synergy2.load_model('models/trained_models/my_synergy.pkl')
```

## 📈 Performans İyileştirme

### 1. Daha Hızlı Eğitim

```python
# main.py içinde
POPULATION_SIZE = 30      # 50 yerine
GENERATIONS = 20          # 30 yerine
n_samples = 500           # 1000 yerine
```

### 2. Daha İyi Sonuçlar

```python
# main.py içinde
POPULATION_SIZE = 100     # Daha fazla
GENERATIONS = 50          # Daha uzun
n_samples = 2000          # Daha çok veri
```

### 3. Paralel İşleme

```python
# ml_models.py içinde RandomForestRegressor için
RandomForestRegressor(n_estimators=100, n_jobs=-1)  # Tüm CPU'ları kullan
```

## 🎯 Sonraki Adımlar

✅ **Şimdi yapın:**
1. CSV dosyanızı `data/players.csv` olarak kaydedin
2. `pip install -r requirements.txt` çalıştırın
3. `python main.py` ile teste başlayın

✅ **Sonra yapın:**
1. Jupyter notebook'ları inceleyin
2. Farklı formasyonları deneyin
3. ML modellerini eğitin
4. **Sinerji NN'ini deneyin!** 🧠

✅ **İleri seviye:**
1. Kendi script'lerinizi yazın
2. Parametreleri optimize edin
3. Yeni özellikler ekleyin

## 📞 Yardım

Herhangi bir sorunla karşılaşırsanız:

1. Bu dosyayı tekrar okuyun
2. Hata mesajını Google'da aratın
3. README.md dosyasına bakın
4. GitHub issue açın (varsa)

## 🎉 Başarılar!

Artık hazırsınız! Keyifli optimizasyonlar! ⚽🤖

---

**Son Güncelleme:** Takım Sinerjisi Neural Network eklendi 🧠✨