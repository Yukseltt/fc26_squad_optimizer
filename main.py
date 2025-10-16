"""
FC26 Takım Kurma Optimizasyonu - Ana Program
Makine Öğrenmesi + Genetik Algoritma + Neural Network
"""

import os
import sys
from datetime import datetime

# Proje modüllerini import et
from src.data_loader import DataLoader
from src.ml_models import PlayerValuePredictor
from src.genetic_algorithm import GeneticSquadOptimizer
from src.team_synergy_nn import TeamSynergyPredictor

def create_directories():
    """Gerekli klasörleri oluştur"""
    dirs = ['data', 'models/trained_models', 'results/best_squads', 'notebooks']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def print_header():
    """Program başlığını yazdır"""
    print("\n" + "="*70)
    print(" "*15 + "FC26 TAKIM KURMA OPTİMİZASYONU")
    print(" "*10 + "Makine Öğrenmesi + Genetik Algoritma + Neural Network")
    print("="*70 + "\n")

def main():
    """Ana program akışı"""
    create_directories()
    print_header()
    
    # ========== 1. VERİ YÜKLEME ==========
    print("📋 ADIM 1: Veri Yükleme ve Hazırlık")
    print("-" * 70)
    
    csv_path = 'data/players.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ HATA: {csv_path} dosyası bulunamadı!")
        print("\nLütfen CSV dosyanızı 'data/players.csv' olarak kaydedin.")
        return
    
    # Veriyi yükle
    loader = DataLoader(csv_path)
    df = loader.load_data()
    df = loader.clean_data()
    
    # İstatistikleri göster
    stats = loader.get_statistics()
    print(f"\n📊 Veri Seti İstatistikleri:")
    print(f"   Toplam Oyuncu: {stats['total_players']}")
    print(f"   Ortalama Overall: {stats['avg_overall']:.1f}")
    print(f"   Ortalama Değer: {stats['avg_value']:,.0f} EUR")
    print(f"   En Yüksek Overall: {stats['max_overall']}")
    print(f"   Değer Aralığı: {stats['min_value']:,.0f} - {stats['max_value']:,.0f} EUR")
    
    input("\n✅ Veri yüklendi. Devam etmek için Enter'a basın...")
    
    # ========== 2. MAKİNE ÖĞRENMESİ ==========
    print("\n\n📋 ADIM 2: Makine Öğrenmesi Model Eğitimi")
    print("-" * 70)
    
    print("\nMakine öğrenmesi ile oyuncu değerlerini tahmin edeceğiz.")
    print("Bu sayede düşük değerli ama potansiyeli yüksek oyuncuları bulabiliriz.\n")
    
    train_ml = input("Makine öğrenmesi modelini eğitmek ister misiniz? (e/h): ").lower()
    
    ml_predictor = None
    
    if train_ml == 'e':
        # ML özellikleri hazırla
        X, y = loader.get_features_for_ml()
        
        # Model eğit
        ml_predictor = PlayerValuePredictor()
        results = ml_predictor.train(X, y)
        
        # Sonuçları göster
        print("\n📈 Model Performansı:")
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"   MAE: {metrics['mae']:,.0f} EUR")
            print(f"   RMSE: {metrics['rmse']:,.0f} EUR")
            print(f"   R² Score: {metrics['r2']:.4f}")
        
        # Grafikleri kaydet
        save_plots = input("\nModel performans grafiklerini kaydetmek ister misiniz? (e/h): ").lower()
        if save_plots == 'e':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f'results/ml_results_{timestamp}.png'
            ml_predictor.plot_results(save_path=plot_path)
        
        # Modeli kaydet
        save_model = input("\nEğitilmiş modeli kaydetmek ister misiniz? (e/h): ").lower()
        if save_model == 'e':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'models/trained_models/ml_model_{timestamp}.pkl'
            ml_predictor.save_model(model_path)
        
        # Değeri düşük oyuncuları bul
        find_undervalued = input("\nDeğerinin altındaki oyuncuları bulmak ister misiniz? (e/h): ").lower()
        if find_undervalued == 'e':
            print("\n🔍 Değeri düşük oyuncular aranıyor...")
            undervalued = ml_predictor.find_undervalued_players(X, y, loader.df, threshold=0.5)
            
            if len(undervalued) > 0:
                print(f"\n✅ {len(undervalued)} değeri düşük oyuncu bulundu!")
                print("\nİlk 10 oyuncu:")
                
                # Detaylı sütunlar göster
                display_cols = ['short_name', 'overall', 'potential', 'age', 'player_positions',
                               'actual_value', 'predicted_value', 'value_diff', 'value_ratio',
                               'club_name', 'nationality_name']
                available_cols = [col for col in display_cols if col in undervalued.columns]
                
                if available_cols:
                    print(undervalued[available_cols].head(10).to_string())
                else:
                    print(undervalued.head(10).to_string())
            else:
                print("❌ Değeri düşük oyuncu bulunamadı.")
    
    # ========== 2.5. TAKIM SİNERJİSİ NEURAL NETWORK ==========
    print("\n\n📋 ADIM 2.5: Takım Sinerjisi Neural Network 🧠")
    print("-" * 70)
    
    print("\n🧠 Neural Network ile takım sinerjisini tahmin edebiliriz.")
    print("Bu, 11 oyuncunun birlikte nasıl oynayacağını öğrenir.")
    print("Bu GERÇEK bir yapay zeka uygulamasıdır!\n")
    
    train_synergy = input("Takım Sinerjisi NN'ini eğitmek ister misiniz? (e/h): ").lower()
    
    synergy_predictor = None
    
    if train_synergy == 'e':
        try:
            n_samples = input("Kaç sentetik takım oluşturulsun? [varsayılan: 1000]: ").strip()
            n_samples = int(n_samples) if n_samples else 1000
            
            synergy_predictor = TeamSynergyPredictor()
            results = synergy_predictor.train(loader.df, n_samples=n_samples)
            
            print(f"\n✅ Neural Network başarıyla eğitildi!")
            print(f"   Test R² Skoru: {results['test_r2']:.4f}")
            print(f"   Test MSE: {results['test_mse']:.2f}")
            
            # Modeli kaydet
            save_synergy = input("\nSinerji modelini kaydetmek ister misiniz? (e/h): ").lower()
            if save_synergy == 'e':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                synergy_path = f'models/trained_models/synergy_nn_{timestamp}.pkl'
                synergy_predictor.save_model(synergy_path)
        
        except Exception as e:
            print(f"\n⚠️ Sinerji modeli eğitilemedi: {str(e)}")
            synergy_predictor = None
    
    input("\n✅ Tüm ML aşamaları tamamlandı. Devam etmek için Enter'a basın...")
    
    # ========== 3. GENETİK ALGORİTMA OPTİMİZASYONU ==========
    print("\n\n📋 ADIM 3: Genetik Algoritma ile Takım Optimizasyonu")
    print("-" * 70)
    
    print("\nŞimdi genetik algoritma kullanarak optimal takımı oluşturacağız.")
    
    # Kullanıcı parametreleri
    try:
        budget_input = input("\nBütçenizi girin (EUR) [varsayılan: 50000000]: ").strip()
        budget = float(budget_input) if budget_input else 50000000
        
        print("\nMevcut Formasyonlar:")
        print("1. 4-3-3 (Klasik)")
        print("2. 4-4-2 (Dengeli)")
        print("3. 3-5-2 (Hücum)")
        print("4. 4-2-3-1 (Modern)")
        
        formation_choice = input("\nFormasyon seçin (1-4) [varsayılan: 1]: ").strip()
        formation_map = {'1': '433', '2': '442', '3': '352', '4': '4231'}
        formation = formation_map.get(formation_choice, '433')
        
        use_ml_in_ga = False
        use_synergy = False
        
        if ml_predictor:
            use_ml = input("\nGenetik algoritmada ML tahminlerini kullanmak ister misiniz? (e/h): ").lower()
            use_ml_in_ga = (use_ml == 'e')
        
        if synergy_predictor:
            use_syn = input("\nGenetik algoritmada Sinerji NN'ini kullanmak ister misiniz? (e/h) [ÖNERİLİR!]: ").lower()
            use_synergy = (use_syn == 'e')
        
    except ValueError:
        print("❌ Geçersiz değer! Varsayılan değerler kullanılıyor.")
        budget = 50000000
        formation = '433'
        use_ml_in_ga = False
        use_synergy = False
    
    # Genetik algoritma
    optimizer = GeneticSquadOptimizer(loader.df, formation=formation)
    
    try:
        result = optimizer.optimize(
            budget=budget,
            population_size=50,
            generations=30,
            elite_size=5,
            use_ml=use_ml_in_ga,
            ml_predictor=ml_predictor,
            use_synergy=use_synergy,
            synergy_predictor=synergy_predictor
        )
        
        # Sonuçları göster
        optimizer.print_squad(result)
        
        # Sinerji açıklaması
        if use_synergy and synergy_predictor:
            print("\n🧠 Takım Sinerjisi Analizi:")
            explanation = synergy_predictor.explain_synergy(result['squad'], result['positions'])
            print(f"   Sinerji Skoru: {explanation['synergy_score']:.1f}/100")
            print(f"   Değerlendirme: {explanation['rating']}")
            print(f"   Ortalama Overall: {explanation['avg_overall']:.1f}")
            print(f"   Ortalama Yaş: {explanation['avg_age']:.1f}")
            print(f"   Milliyet Çeşitliliği: {explanation['nation_diversity']} farklı ülke")
        
        # Takımı kaydet
        save_squad = input("\nBu takımı kaydetmek ister misiniz? (e/h): ").lower()
        if save_squad == 'e':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            squad_path = f'results/best_squads/squad_{formation}_{timestamp}.csv'
            optimizer.export_squad(result, squad_path)
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        print("\nÖneriler:")
        print("- Bütçeyi artırın")
        print("- Farklı bir formasyon deneyin")
        print("- Veri setinin yeterli oyuncu içerdiğinden emin olun")
    
    # ========== 4. BİTİŞ ==========
    print("\n\n" + "="*70)
    print(" "*25 + "PROGRAM TAMAMLANDI!")
    print("="*70)
    
    print("\n📁 Oluşturulan Dosyalar:")
    print("   - models/trained_models/    → Eğitilmiş ML modelleri")
    print("   - results/best_squads/      → Oluşturulan takımlar")
    print("   - results/ml_results_*.png  → ML performans grafikleri")
    
    print("\n💡 İpucu: 'notebooks' klasöründeki Jupyter notebook'ları da")
    print("   inceleyerek detaylı analiz yapabilirsiniz!")
    
    print("\n🧠 NEURAL NETWORK ile takım sinerjisi öğrenildi!")
    print("   Bu gerçek bir yapay zeka uygulamasıdır! 🎓")
    
    print("\n✅ Teşekkürler!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Program kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {str(e)}")
        import traceback
        traceback.print_exc()