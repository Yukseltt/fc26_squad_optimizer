"""
FC26 Genetik Algoritma Optimizasyonu Script
notebooks/optimization_script.py
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_loader import DataLoader
from src.genetic_algorithm import GeneticSquadOptimizer
from src.team_synergy_nn import TeamSynergyPredictor
from src.utils import VisualizationHelper

print("="*70)
print(" "*12 + "FC26 GENETİK ALGORİTMA OPTİMİZASYONU")
print("="*70)

# Veriyi yükle
print("\n📂 Veri yükleniyor...")
loader = DataLoader('../data/players.csv')
df = loader.load_data()
df = loader.clean_data()

print(f"   Toplam oyuncu: {len(df)}")

# Parametreler
BUDGET = 50_000_000  # EUR
FORMATION = '433'     # 4-3-3
POPULATION_SIZE = 50
GENERATIONS = 30

print(f"\n⚙️ PARAMETRELER:")
print(f"   Bütçe: {BUDGET:,} EUR")
print(f"   Formasyon: {FORMATION}")
print(f"   Popülasyon: {POPULATION_SIZE}")
print(f"   Jenerasyon: {GENERATIONS}")

# Sinerji NN (opsiyonel)
print("\n🧠 SİNERJİ NEURAL NETWORK:")
print("-" * 70)

use_synergy = input("\nSinerji NN'ini eğitmek ister misiniz? (e/h): ").lower()

synergy_predictor = None

if use_synergy == 'e':
    try:
        print("\n🔬 Sinerji NN eğitiliyor...")
        synergy_predictor = TeamSynergyPredictor()
        synergy_predictor.train(df, n_samples=500)  # Hızlı test için 500
        print("✅ Sinerji NN başarıyla eğitildi!")
    except Exception as e:
        print(f"⚠️ Hata: {e}")
        synergy_predictor = None

# Optimizasyon
print("\n🧬 OPTİMİZASYON BAŞLIYOR...")
print("-" * 70)

optimizer = GeneticSquadOptimizer(df, formation=FORMATION)

result = optimizer.optimize(
    budget=BUDGET,
    population_size=POPULATION_SIZE,
    generations=GENERATIONS,
    elite_size=5,
    use_ml=False,
    use_synergy=(synergy_predictor is not None),
    synergy_predictor=synergy_predictor
)

# Sonuçları göster
optimizer.print_squad(result)

# Sinerji açıklaması
if synergy_predictor:
    print("\n🧠 SİNERJİ ANALİZİ:")
    print("-" * 70)
    explanation = synergy_predictor.explain_synergy(result['squad'], result['positions'])
    print(f"   Sinerji Skoru: {explanation['synergy_score']:.1f}/100")
    print(f"   Değerlendirme: {explanation['rating']}")
    print(f"   Ortalama Overall: {explanation['avg_overall']:.1f}")
    print(f"   Overall Std: {explanation['overall_std']:.1f}")
    print(f"   Ortalama Yaş: {explanation['avg_age']:.1f}")
    print(f"   Yaş Std: {explanation['age_std']:.1f}")
    print(f"   Milliyet Çeşitliliği: {explanation['nation_diversity']} ülke")
    print(f"   Lig Çeşitliliği: {explanation['league_diversity']} lig")

# Görselleştirme
print("\n📊 GÖRSELLEŞTİRMELER:")
print("-" * 70)

viz = VisualizationHelper()

# 1. İlerleme grafiği
viz.plot_optimization_progress(
    result['generation_progress'],
    save_path='../results/optimization_progress.png'
)

# 2. Takım istatistikleri
viz.plot_squad_stats(
    result['squad'],
    result['positions'],
    save_path='../results/squad_stats.png'
)

# 3. Saha üzerinde
viz.plot_squad_on_field(
    result['squad'],
    result['positions'],
    save_path='../results/squad_field.png'
)

# Takımı kaydet
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f'../results/best_squads/squad_{FORMATION}_{timestamp}.csv'
optimizer.export_squad(result, output_path)

# Farklı bütçelerle karşılaştırma
print("\n💰 FARKLI BÜTÇELERLE KARŞILAŞTIRMA:")
print("-" * 70)

budgets = [10_000_000, 30_000_000, 50_000_000, 100_000_000]
budget_results = []

print("\nFarklı bütçelerle takımlar oluşturuluyor...")

for budget in budgets:
    print(f"\n   Bütçe: {budget:,} EUR")
    
    opt = GeneticSquadOptimizer(df, formation=FORMATION)
    res = opt.optimize(
        budget=budget,
        population_size=30,
        generations=20,
        use_synergy=(synergy_predictor is not None),
        synergy_predictor=synergy_predictor
    )
    
    budget_results.append({
        'Bütçe (M EUR)': budget / 1_000_000,
        'Maliyet (M EUR)': res['cost'] / 1_000_000,
        'Avg Overall': res['avg_overall'],
        'Kimya': res['chemistry'],
        'Fitness': res['fitness']
    })

# Karşılaştırma tablosu
comparison_df = pd.DataFrame(budget_results)
print("\n📊 BÜTÇE KARŞILAŞTIRMASI:")
print(comparison_df.to_string(index=False))

# Karşılaştırma grafikleri
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(comparison_df['Bütçe (M EUR)'], comparison_df['Avg Overall'], 
            marker='o', linewidth=2, markersize=8, color='blue')
axes[0].set_xlabel('Bütçe (Milyon EUR)', fontsize=12)
axes[0].set_ylabel('Ortalama Overall', fontsize=12)
axes[0].set_title('Bütçe vs Overall', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].plot(comparison_df['Bütçe (M EUR)'], comparison_df['Kimya'], 
            marker='s', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('Bütçe (Milyon EUR)', fontsize=12)
axes[1].set_ylabel('Kimya Skoru', fontsize=12)
axes[1].set_title('Bütçe vs Kimya', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)

axes[2].plot(comparison_df['Bütçe (M EUR)'], comparison_df['Fitness'], 
            marker='^', linewidth=2, markersize=8, color='orange')
axes[2].set_xlabel('Bütçe (Milyon EUR)', fontsize=12)
axes[2].set_ylabel('Fitness Skoru', fontsize=12)
axes[2].set_title('Bütçe vs Fitness', fontsize=13, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/budget_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafik kaydedildi: results/budget_comparison.png")
plt.show()

# Farklı formasyonlar
print("\n⚽ FARKLI FORMASYONLARLA KARŞILAŞTIRMA:")
print("-" * 70)

formations = ['433', '442', '352', '4231']
formation_results = []

print("\nFarklı formasyonlar test ediliyor...")

for formation in formations:
    print(f"\n   Formasyon: {formation}")
    
    opt = GeneticSquadOptimizer(df, formation=formation)
    res = opt.optimize(
        budget=50_000_000,
        population_size=30,
        generations=20,
        use_synergy=(synergy_predictor is not None),
        synergy_predictor=synergy_predictor
    )
    
    formation_results.append({
        'Formasyon': formation,
        'Maliyet (M EUR)': res['cost'] / 1_000_000,
        'Avg Overall': res['avg_overall'],
        'Kimya': res['chemistry'],
        'Fitness': res['fitness']
    })

# Formasyon karşılaştırması
formation_df = pd.DataFrame(formation_results)
print("\n📊 FORMASYON KARŞILAŞTIRMASI:")
print(formation_df.to_string(index=False))

# Formasyon grafiği
plt.figure(figsize=(12, 6))
x = range(len(formation_df))
width = 0.2

plt.bar([i - width for i in x], formation_df['Avg Overall'], 
        width, label='Avg Overall', color='skyblue')
plt.bar([i for i in x], formation_df['Kimya']/2, 
        width, label='Kimya/2', color='lightgreen')
plt.bar([i + width for i in x], formation_df['Fitness']/20, 
        width, label='Fitness/20', color='coral')

plt.xlabel('Formasyon', fontsize=12)
plt.ylabel('Değer', fontsize=12)
plt.title('Formasyon Karşılaştırması', fontsize=14, fontweight='bold')
plt.xticks(x, formation_df['Formasyon'])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/formation_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafik kaydedildi: results/formation_comparison.png")
plt.show()

# Özet
print("\n" + "="*70)
print(" "*20 + "OPTİMİZASYON TAMAMLANDI!")
print("="*70)

print("\n📁 Oluşturulan dosyalar:")
print(f"   - results/best_squads/squad_{FORMATION}_{timestamp}.csv")
print("   - results/optimization_progress.png")
print("   - results/squad_stats.png")
print("   - results/squad_field.png")
print("   - results/budget_comparison.png")
print("   - results/formation_comparison.png")

print("\n🏆 EN İYİ TAKIM ÖZETİ:")
print(f"   Formasyon: {FORMATION}")
print(f"   Toplam Maliyet: {result['cost']:,.0f} EUR")
print(f"   Ortalama Overall: {result['avg_overall']:.1f}")
print(f"   Kimya Skoru: {result['chemistry']:.0f}")
print(f"   Fitness Skoru: {result['fitness']:.1f}")
if synergy_predictor:
    print(f"   🧠 Sinerji Skoru: {explanation['synergy_score']:.1f}/100")

print("\n💡 ÖNERİLER:")
print("   - Daha uzun jenerasyon sayısı (50-100) daha iyi sonuçlar verebilir")
print("   - Farklı formasyonları test edin")
print("   - Sinerji NN kullanımı takım kalitesini artırır")
print("   - ML tahminleri ile değeri düşük oyuncular bulunabilir")

print("\n✅ Tüm işlemler başarıyla tamamlandı!")