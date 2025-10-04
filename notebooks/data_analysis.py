"""
FC26 Veri Analizi Script
notebooks/data_analysis.py
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import DataLoader
from src.utils import DataAnalyzer, VisualizationHelper

print("="*70)
print(" "*20 + "FC26 VERİ ANALİZİ")
print("="*70)

# Veriyi yükle
print("\n📂 Veri yükleniyor...")
loader = DataLoader('../data/players.csv')
df = loader.load_data()
df = loader.clean_data()

# Temel istatistikler
print("\n📊 TEMEL İSTATİSTİKLER:")
print("-" * 70)
stats = loader.get_statistics()
for key, value in stats.items():
    if isinstance(value, float):
        print(f"   {key}: {value:,.2f}")
    else:
        print(f"   {key}: {value:,}")

# Overall dağılımı
print("\n📈 Overall Dağılımı:")
overall_counts = df['overall'].value_counts().sort_index()
print(f"   En düşük: {df['overall'].min()}")
print(f"   En yüksek: {df['overall'].max()}")
print(f"   Ortalama: {df['overall'].mean():.1f}")
print(f"   80+ oyuncu: {len(df[df['overall'] >= 80])}")
print(f"   85+ oyuncu: {len(df[df['overall'] >= 85])}")
print(f"   90+ oyuncu: {len(df[df['overall'] >= 90])}")

# Grafik 1: Overall Dağılımı
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['overall'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Overall', fontsize=12)
plt.ylabel('Oyuncu Sayısı', fontsize=12)
plt.title('Overall Dağılımı', fontsize=14, fontweight='bold')
plt.axvline(df['overall'].mean(), color='r', linestyle='--', 
            label=f'Ortalama: {df["overall"].mean():.1f}')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(np.log10(df['value_eur']), bins=30, edgecolor='black', 
         alpha=0.7, color='green')
plt.xlabel('Log10(Değer EUR)', fontsize=12)
plt.ylabel('Oyuncu Sayısı', fontsize=12)
plt.title('Oyuncu Değer Dağılımı (Log)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/overall_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafik kaydedildi: results/overall_distribution.png")
plt.show()

# Pozisyon analizi
print("\n📍 POZİSYON ANALİZİ:")
print("-" * 70)
analyzer = DataAnalyzer()

position_dist = analyzer.analyze_position_distribution(df)
print("\nEn popüler 10 pozisyon:")
print(position_dist.head(10).to_string(index=False))

value_by_pos = analyzer.analyze_value_by_position(df)
print("\nPozisyona göre değer analizi:")
print(value_by_pos.to_string(index=False))

# Grafik 2: Pozisyon Değerleri
plt.figure(figsize=(12, 6))
plt.bar(value_by_pos['position'], value_by_pos['avg_value']/1_000_000, 
        color='coral', edgecolor='black')
plt.xlabel('Pozisyon', fontsize=12)
plt.ylabel('Ortalama Değer (Milyon EUR)', fontsize=12)
plt.title('Pozisyonlara Göre Ortalama Oyuncu Değeri', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/position_values.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafik kaydedildi: results/position_values.png")
plt.show()

# En iyi değer/performans oranı
print("\n💰 EN İYİ DEĞER/PERFORMANS ORANI:")
print("-" * 70)
best_value_players = analyzer.find_best_value_players(df, top_n=20)
print("\nEn uygun fiyatlı 20 oyuncu:")
print(best_value_players[['short_name', 'overall', 'potential', 'value_eur', 
                          'age']].to_string(index=False))

# Lig karşılaştırması
print("\n🏆 LİG KARŞILAŞTIRMASI:")
print("-" * 70)
league_comp = analyzer.league_comparison(df)
print("\nEn iyi 10 lig:")
print(league_comp.head(10).to_string())

# Grafik 3: Top Ligler
plt.figure(figsize=(12, 6))
top_leagues = league_comp.head(10)
x = range(len(top_leagues))
plt.barh(x, top_leagues['avg_overall'], color='purple', alpha=0.7)
plt.yticks(x, top_leagues.index, fontsize=10)
plt.xlabel('Ortalama Overall', fontsize=12)
plt.title('En İyi 10 Lig (Ortalama Overall)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/top_leagues.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafik kaydedildi: results/top_leagues.png")
plt.show()

# Yaş analizi
print("\n👤 YAŞ ANALİZİ:")
print("-" * 70)
print(f"   Ortalama yaş: {df['age'].mean():.1f}")
print(f"   En genç: {df['age'].min()}")
print(f"   En yaşlı: {df['age'].max()}")
print(f"   23 yaş altı: {len(df[df['age'] < 23])}")
print(f"   Prime (24-28): {len(df[(df['age'] >= 24) & (df['age'] <= 28)])}")
print(f"   30 yaş üzeri: {len(df[df['age'] >= 30])}")

# Korelasyon analizi
print("\n🔗 KORELASYON ANALİZİ:")
print("-" * 70)
numeric_cols = ['overall', 'potential', 'value_eur', 'age', 'pace', 
                'shooting', 'passing', 'dribbling', 'defending', 'physic']
correlation = df[numeric_cols].corr()

print("\nOverall ile en yüksek korelasyon:")
overall_corr = correlation['overall'].sort_values(ascending=False)
print(overall_corr.head(6).to_string())

# Grafik 4: Korelasyon Matrisi
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, cbar_kws={'shrink': 0.8})
plt.title('Özellik Korelasyon Matrisi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafik kaydedildi: results/correlation_matrix.png")
plt.show()

# Özet
print("\n" + "="*70)
print(" "*25 + "ANALİZ TAMAMLANDI!")
print("="*70)
print("\n📁 Oluşturulan dosyalar:")
print("   - results/overall_distribution.png")
print("   - results/position_values.png")
print("   - results/top_leagues.png")
print("   - results/correlation_matrix.png")
print("\n✅ Tüm analizler başarıyla tamamlandı!")