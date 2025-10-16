# FC26 AI Squad Optimizer ⚽

Bu proje, makine öğrenmesi, sinir ağları ve genetik algoritmaları bir araya getirerek bütçe ve formasyon kısıtları altında en iyi futbol takımını oluşturan interaktif bir web uygulamasıdır.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<p align="center">
    <img src="https://drive.google.com/file/d/1Be3SesbESj9xyHHq5CvWFK6Jc5LZsYC_/view?usp=sharing" alt="Uygulama Arayüzü" width="800">
    <br/>
    <i>Streamlit arayüzünden ML/NN eğitim ve GA optimizasyon akışı</i>
  
</p>

---

## Proje Hakkında

FC26 AI Squad Optimizer, FC26 oyuncu verilerini analiz eder, oyuncu değerlerini makine öğrenmesi ile tahmin eder, 11 oyunculuk takım sinerjisini bir sinir ağı ile puanlar ve en sonunda genetik algoritma ile belirlenen bütçeye en uygun (fitness’ı en yüksek) takımı oluşturur. Uygulama Streamlit ile interaktiftir ve TR/EN çoklu dil desteği içerir.

## Temel Özellikler

- İnteraktif Veri Analizi: Plotly ile 18,000+ oyuncuyu filtreleyip keşfedin
- Değer Tahmini (ML): Random Forest, Gradient Boosting ve XGBoost ile oyuncu piyasa değerini tahmin edin; undervalued oyuncuları yakalayın
- Sinerji Tahmini (NN): 11 oyuncunun kimya, yaş dengesi ve oyun stili gibi 40+ özelliğinden 0–100 sinerji skoru üretin
- Takım Optimizasyonu (GA): Bütçe ve formasyona göre en yüksek fitness skorlu takımı evrimsel olarak bulun
- Dinamik Hiperparametre: Hem ML hem NN için katman/ağaç sayısı, derinlik, öğrenme oranı vb. ayarları arayüzden değiştirin

## Kullanılan Teknolojiler

- Ana Çatı & Arayüz: Python, Streamlit
- Veri İşleme: Pandas, NumPy
- Makine Öğrenmesi: scikit-learn, XGBoost
- Görselleştirme: Plotly (GUI), Matplotlib/Seaborn (grafik kaydı)
- Model Kaydı: Joblib

## Kurulum

Windows PowerShell için adımlar:

```powershell
# Depoyu klonlayın
git clone https://github.com/Yukseltt/fc26_squad_optimizer.git
cd fc26_squad_optimizer

# (Opsiyonel) Sanal ortam oluşturup etkinleştirin
python -m venv venv
venv\Scripts\activate

# Bağımlılıkları kurun
pip install -r requirements.txt
```

Veri: Oyuncu CSV dosyanızı `data/players.csv` yoluna koyun.

Uygulamayı başlatın (GUI):

```powershell
streamlit run app.py
```

Komut satırı alternatifi:

```powershell
python main.py
```



---

## Kullanım

1) Sol menüden “Veri Yükle” butonuyla `data/players.csv` yüklenir/temizlenir.
2) “Makine Öğrenmesi” sekmesinde ML modellerini eğitin ve isterseniz kaydedin.
3) “Sinerji NN” sekmesinde sinerji modelini sentetik verilerle eğitin ve kaydedin.
4) “Takım Optimizasyonu” sekmesinde bütçe, formasyon, GA parametrelerini seçin; “Gelişmiş Seçenekler”den eğitilmiş ML/NN’leri dahil edin.
5) “Optimizasyonu Başlat” ile en iyi takım ekrana gelir; CSV olarak kaydedebilirsiniz.

---

## ML: Oyuncu Değeri Tahmini (Detaylı)

Bu bölüm `src/ml_models.py` içindeki `PlayerValuePredictor` sınıfına dayanır ve arayüzdeki “Makine Öğrenmesi” sayfası ile entegredir.

- Girdi özellikleri (DataLoader.get_features_for_ml):
    - overall, potential, age, height_cm, weight_kg
    - pace, shooting, passing, dribbling, defending, physic
    - weak_foot, skill_moves, international_reputation
- Hedef: value_eur (oyuncu piyasa değeri)
- Ön işleme: Eksikler sütun ortalaması ile doldurulur; StandardScaler ile X ölçeklenir
- Veri bölme: train_test_split(test_size UI’dan ayarlanır; varsayılan 0.2)
- Değerlendirme metrikleri: R², MAE, RMSE (test setinde gösterilir)

Desteklenen modeller ve hiperparametreler (UI’dan dinamik):

- RandomForestRegressor
    - n_estimators, max_depth, random_state=42, n_jobs=-1
- GradientBoostingRegressor
    - n_estimators, max_depth, learning_rate, random_state=42
- XGBRegressor
    - n_estimators, max_depth, learning_rate, random_state=42, n_jobs=-1

Eğitimden sonra en iyi model R² skoruna göre seçilir; ağaç tabanlı modeller için feature_importances_ görsellenebilir. “Değeri Düşük Oyuncular” sekmesi, tahmin/gerçek oranına göre undervalued oyuncuları listeler (UI’daki eşik ile kontrol edilir).

Model Kaydet/Yükle:

- Kaydet: `models/trained_models/ml_model_YYYYMMDD_HHMMSS.pkl`
- İçerik: { model, scaler, model_name, feature_importance }
- Uygulama açılırken “en son kaydedilen” model otomatik yüklenmeye çalışılır.

---

## NN: Takım Sinerjisi Tahmini (Detaylı)

Bu bölüm `src/team_synergy_nn.py` içindeki `TeamSynergyPredictor` sınıfına dayanır. 11 oyuncudan takım düzeyinde 0–100 arası sinerji skoru üretir.

Özellik Mühendisliği (11 oyuncudan 40+ öznitelik; yaklaşık 46 özellik):

- Temel takım istatistikleri (6):
    - overall ort/Std/medyan/min/max, 80+ oyuncu sayısı
- Yaş dengesi (4):
    - yaş ort/Std, 24–28 prime sayısı, <23 genç sayısı
- Kimya (6):
    - milliyet çeşitliliği, en çok tekrar eden milliyet ve dağılımı
    - lig çeşitliliği, en çok tekrar eden lig ve dağılımı
- Oyun stili (12):
    - pace, shooting, passing, dribbling, defending, physic için ortalama ve Std
- Pozisyon uyumu (11):
    - formasyondaki her mevki için oyuncunun ilgili pozisyon reytingi (örn. ST→st)
- Hücum–savunma dengesi (4):
    - hücum ve savunma ortalamaları, farkı, hücum oranı
- Değer dağılımı (3):
    - takım değer ort/Std, max/ortalama oranı

Sentetik Eğitim Verisi Üretimi:

- Formasyonlardan (433, 442, 352) 1000+ rastgele takım örneklenir (UI’dan sayı ayarlanır)
- Uygun mevkilere uygun oyuncular seçilir (player_positions eşleşmesi ve kısıtlar)
- Gerçek etiket olmayan “true synergy” bir sezgisel fonksiyonla (overall dağılımı, yaş dengesi, milliyet/lig kümelenmesi vb.) 0–100 arası skalanır ve küçük gürültü eklenir

Model Mimarisi ve Eğitim:

- Skaler: StandardScaler
- Model: scikit-learn MLPRegressor
- Varsayılan hiperparametreler (UI’dan değiştirilebilir):
    - hidden_layer_sizes=(128, 64, 32)
    - activation='relu', solver='adam'
    - max_iter=500, learning_rate='adaptive'
    - early_stopping=True, n_iter_no_change=20
- Train/test ayrımı: 0.8/0.2
- Metrikler: R² ve MSE (train/test)

Tahmin ve Açıklama:

- predict_synergy(squad, positions) → [0,100]
- explain_synergy(...) → skor + özet metrikler (avg_overall, avg_age, çeşitlilikler) ve yıldız derecelendirmesi

Model Kaydet/Yükle:

- Kaydet: `models/trained_models/synergy_nn_YYYYMMDD_HHMMSS.pkl`
- İçerik: { model, scaler, trained }

---

## Genetik Algoritma (Detaylı)

Kaynak: `src/genetic_algorithm.py` – `GeneticSquadOptimizer`.

- Formasyonlar: 4-3-3, 4-4-2, 3-5-2, 4-2-3-1 (positions haritaları hazır)
- Bütçe ayrımı: %70 ilk 11, %30 yedekler; oyuncu başı aşırı pahalı/ucuzlara karşı korumalar
- Uygunluk: player_positions ile asıl mevkii kontrolü; ilgili pozisyon reytingine (örn. cm/cb/st) alt eşik
- Fitness:
    - Σ[(pozisyon skoru×0.3 + overall×0.7)]
    - + kimya bonusu (milliyet/lig/klüp eşleşmelerine dayalı proxy) ×0.5
    - + (opsiyonel) NN sinerji ×2.0
    - + (opsiyonel) ML tahmini değer → overall’a log-ölçekli bonus
- Evrimsel adımlar: elitizm, turnuva seçimi, tek-nokta çaprazlama, mutasyon (pozisyon bazlı oyuncu değişimi)
- UI parametreleri: population_size, generations; ML/NN kullanım onayları
- Çıktı: en iyi takım, yedekler (opsiyonel), fitness, kimya, ort. overall, pozisyon sırası ve jenerasyon bazlı ilerleme
- Dışa aktarma: `results/best_squads/squad_{formation}_YYYYMMDD_HHMMSS.csv`

Not: `src/ml_optimizer.py` içinde GA’sız hızlı/greedy bir alternatif (MLSquadOptimizer) de bulunmaktadır.

---

## Proje Yapısı

```
fc26_squad_optimizer/
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── texts.py
├── data/
│   └── players.csv
├── models/
│   └── trained_models/
├── notebooks/
│   ├── data_analysis.py
│   ├── ml_models_script.py
│   └── optimization_script.py
├── results/
│   └── best_squads/
└── src/
        ├── __init__.py
        ├── data_loader.py
        ├── genetic_algorithm.py
        ├── ml_models.py
        ├── ml_optimizer.py
        ├── team_synergy_nn.py
        └── utils.py
```

---

## Sık Karşılaşılan Sorular / Sorun Giderme

- Streamlit komutu bulunamadı → `pip install streamlit`; ardından `streamlit run app.py`
- ModuleNotFoundError → `pip install -r requirements.txt`
- `data/players.csv` bulunamadı → dosyayı belirtilen yola ekleyin (ad tam olarak players.csv olmalı)
- Port 8501 dolu → `streamlit run app.py --server.port 8502`
- Optimizasyon yavaş → nesil/popülasyon değerlerini düşürün; ML/NN seçimini kapatın

---

## Katkı ve Lisans

- Katkılar PR ile memnuniyetle kabul edilir. Issue/pull request açmaktan çekinmeyin.
- Lisans: MIT

Yazar: @Yukseltt

—

Made with ❤️ using Python & Streamlit

