"""
FC26 Makine Öğrenmesi Modelleri Script
notebooks/ml_models_script.py
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_loader import DataLoader
from src.ml_models import PlayerValuePredictor

print("="*70)
print(" "*15 + "FC26 MAKİNE ÖĞRENMESİ MODELLERİ")
print("="*70)

# Veriyi yükle
print("\n📂 Veri yükleniyor...")
loader = DataLoader('../data/players.csv')
df = loader.load_data()
df = loader.clean_data()

# ML için özellikleri hazırla
print("\n🎯 ML özellikleri hazırlanıyor...")
X, y = loader.get_features_for_ml()

print(f"   Özellik sayısı: {X.shape[1]}")
print(f"   Örnek sayısı: {X.shape[0]}")
print(f"   Özellikler: {list(X.columns)}")

# Model eğitimi
print("\n🤖 MODELLER EĞİTİLİYOR...")
print("-" * 70)

predictor = PlayerValuePredictor()
results = predictor.train(X, y, test_size=0.2)

# Sonuçları tablo olarak göster
print("\n📊 MODEL PERFORMANS KARŞILAŞTIRMASI:")
print("-" * 70)

results_data = []
for model_name, metrics in results.items():
    results_data.append({
        'Model': model_name.upper(),
        'MAE (EUR)': f"{metrics['mae']:,.0f}",
        'RMSE (EUR)': f"{metrics['rmse']:,.0f}",
        'R² Score': f"{metrics['r2']:.4f}"
    })

results_df = pd.DataFrame(results_data)
print(results_df.to_string(index=False))

print(f"\n🏆 En İyi Model: {predictor.best_model_name.upper()}")
print(f"   R² Score: {results[predictor.best_model_name]['r2']:.4f}")

# Feature Importance
print("\n🔍 ÖNEMLİ ÖZELLİKLER (Feature Importance):")
print("-" * 70)

if predictor.feature_importance is not None:
    print(f"\nEn önemli 10 özellik ({predictor.best_model_name.upper()}):")
    print(predictor.feature_importance.head(10).to_string(index=False))
    
    # Feature importance grafiği
    plt.figure(figsize=(10, 6))
    top_features = predictor.feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
    plt.xlabel('Önem Skoru', fontsize=12)
    plt.title('En Önemli 10 Özellik', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Grafik kaydedildi: results/feature_importance.png")
    plt.show()

# Görselleştirme
print("\n📈 GÖRSELLEŞTIRMELER OLUŞTURULUYOR...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = f'../results/ml_results_{timestamp}.png'
predictor.plot_results(save_path=plot_path)

# Örnek tahminler
print("\n🎲 ÖRNEK TAHMİNLER:")
print("-" * 70)

sample_indices = np.random.choice(X.index, size=10, replace=False)
sample_X = X.loc[sample_indices]
sample_y = y.loc[sample_indices]

predictions = predictor.predict(sample_X)

comparison = pd.DataFrame({
    'Oyuncu': df.loc[sample_indices, 'short_name'].values,
    'Overall': df.loc[sample_indices, 'overall'].values,
    'Gerçek Değer': sample_y.values,
    'Tahmin': predictions,
    'Fark': predictions - sample_y.values,
    'Hata %': ((predictions - sample_y.values) / sample_y.values * 100).round(2)
})

print("\n10 rastgele oyuncu tahmini:")
print(comparison.to_string(index=False))

# Değeri düşük oyuncular
print("\n💎 DEĞERİ DÜŞÜK OYUNCULAR:")
print("-" * 70)

undervalued = predictor.find_undervalued_players(X, y, threshold=0.5)

if len(undervalued) > 0:
    print(f"\n✅ {len(undervalued)} değeri düşük oyuncu bulundu!")
    
    # Oyuncu bilgilerini ekle
    undervalued_with_info = undervalued.copy()
    undervalued_with_info['player_name'] = df.loc[undervalued.index, 'short_name'].values
    undervalued_with_info['overall'] = df.loc[undervalued.index, 'overall'].values
    undervalued_with_info['potential'] = df.loc[undervalued.index, 'potential'].values
    undervalued_with_info['age'] = df.loc[undervalued.index, 'age'].values
    undervalued_with_info['position'] = df.loc[undervalued.index, 'player_positions'].values
    
    # Sırala ve göster
    cols = ['player_name', 'overall', 'potential', 'age', 'position',
            'actual_value', 'predicted_value', 'value_diff']
    
    print("\nİlk 20 oyuncu (tahmin edilen değer > gerçek değer):")
    print(undervalued_with_info[cols].head(20).to_string(index=False))
    
    # CSV olarak kaydet
    output_path = f'../results/undervalued_players_{timestamp}.csv'
    undervalued_with_info[cols].to_csv(output_path, index=False)
    print(f"\n✅ Tüm liste kaydedildi: results/undervalued_players_{timestamp}.csv")
    
    # Grafik
    plt.figure(figsize=(12, 6))
    top_20 = undervalued_with_info.head(20)
    plt.barh(range(len(top_20)), top_20['value_diff']/1_000_000, color='gold', edgecolor='black')
    plt.yticks(range(len(top_20)), top_20['player_name'].values, fontsize=9)
    plt.xlabel('Değer Farkı (Milyon EUR)', fontsize=12)
    plt.title('En Değeri Düşük 20 Oyuncu', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/undervalued_players.png', dpi=300, bbox_inches='tight')
    print("✅ Grafik kaydedildi: results/undervalued_players.png")
    plt.show()
else:
    print("❌ Değeri düşük oyuncu bulunamadı.")

# Model kaydetme
print("\n💾 MODEL KAYDETME:")
print("-" * 70)

save_choice = input("\nEğitilmiş modeli kaydetmek ister misiniz? (e/h): ").lower()

if save_choice == 'e':
    model_path = f'../models/trained_models/ml_model_{timestamp}.pkl'
    predictor.save_model(model_path)
    print(f"✅ Model başarıyla kaydedildi!")
else:
    print("Model kaydedilmedi.")

# Özet
print("\n" + "="*70)
print(" "*20 + "ML EĞİTİMİ TAMAMLANDI!")
print("="*70)

print("\n📁 Oluşturulan dosyalar:")
print(f"   - results/ml_results_{timestamp}.png")
print(f"   - results/feature_importance.png")
if len(undervalued) > 0:
    print(f"   - results/undervalued_players_{timestamp}.csv")
    print(f"   - results/undervalued_players.png")
if save_choice == 'e':
    print(f"   - models/trained_models/ml_model_{timestamp}.pkl")

print("\n📊 Model Özeti:")
print(f"   En İyi Model: {predictor.best_model_name.upper()}")
print(f"   Test R² Score: {results[predictor.best_model_name]['r2']:.4f}")
print(f"   Test MAE: {results[predictor.best_model_name]['mae']:,.0f} EUR")

print("\n✅ Tüm işlemler başarıyla tamamlandı!")    