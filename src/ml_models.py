import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict

class PlayerValuePredictor:
    """Oyuncu değerini tahmin eden ML modeli"""
    
    def __init__(self):
        # GÜNCELLEME: Modeller artık burada sabit olarak oluşturulmuyor.
        # `train` metodu içinde dinamik olarak oluşturulacaklar.
        self.models = {} 
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.results = {}
        
    # GÜNCELLEME: Fonksiyon imzasına `model_params` eklendi.
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, model_params: Dict = None) -> Dict:
        """Modelleri eğit ve en iyisini seç"""
        print("\n🤖 Makine öğrenmesi modelleri eğitiliyor...\n")
        
        # GÜNCELLEME: Arayüzden gelen parametreleri kullanmak için model tanımlamaları buraya taşındı.
        if model_params is None:
            model_params = {} # Parametre gelmezse boş sözlük kullan

        # Arayüzden gelen anahtarları kod içi anahtarlarla eşleştir
        params_key_map = {
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'xgboost': 'XGBoost'
        }

        # Her model için arayüzden gelen parametreleri al, yoksa boş sözlük kullan
        rf_params = model_params.get(params_key_map['random_forest'], {})
        gb_params = model_params.get(params_key_map['gradient_boosting'], {})
        xgb_params = model_params.get(params_key_map['xgboost'], {})

        # Modelleri alınan veya varsayılan parametrelerle başlat
        self.models = {
            'random_forest': RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(**gb_params, random_state=42),
            'xgboost': xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)
        }

        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Ölçekleme
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_score = -np.inf
        
        # Her modeli eğit ve değerlendir
        for name, model in self.models.items():
            print(f"📊 {name.upper()} eğitiliyor...")
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Metrikler
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # GÜNCELLEME: Model isimlerini arayüzde görünecek hale getiriyoruz.
            display_name = params_key_map.get(name, name)
            self.results[display_name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"   MAE: {mae:,.0f} EUR")
            print(f"   RMSE: {rmse:,.0f} EUR")
            print(f"   R²: {r2:.4f}")
            print()
            
            # En iyi modeli seç (R² skoruna göre)
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = name
        
        print(f"✅ En iyi model: {self.best_model_name.upper()} (R² = {best_score:.4f})")
        
        # Feature importance hesapla
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self.results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Yeni veriler için tahmin yap"""
        if self.best_model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def find_undervalued_players(self, X: pd.DataFrame, y: pd.Series, 
                                     original_df: pd.DataFrame = None,
                                     threshold: float = 0.7) -> pd.DataFrame:
        """Değerinin altında olan oyuncuları bul"""
        predictions = self.predict(X)
        
        # Tahmin / Gerçek değer oranı
        # 0'a bölme hatasını önlemek için küçük bir değer (epsilon) ekleyelim
        y_values = y.values
        y_values[y_values == 0] = 1e-6 # Sıfır değerlerini çok küçük bir sayıyla değiştir
        value_ratio = predictions / y_values
        
        # Değerinin altında olanlar (tahmin, gerçek değerin belirli bir orandan fazlası)
        undervalued_mask = value_ratio > (1 / threshold) # Eşiği daha sezgisel hale getirdik. Örn: 0.5 ise tahmin gerçek değerin 2 katı
        
        undervalued_df = pd.DataFrame({
            'actual_value': y[undervalued_mask].values,
            'predicted_value': predictions[undervalued_mask],
            'value_diff': predictions[undervalued_mask] - y[undervalued_mask].values,
            'value_ratio': value_ratio[undervalued_mask]
        }, index=y[undervalued_mask].index)
        
        # Orijinal dataframe'den ek bilgileri ekle
        if original_df is not None:
            info_cols = ['short_name', 'overall', 'potential', 'age', 
                         'nationality_name', 'league_name', 'club_name', 'player_positions']
            
            # Mevcut olan sütunları al
            available_cols = [col for col in info_cols if col in original_df.columns]
            
            # Bilgileri birleştir
            undervalued_df = undervalued_df.join(original_df[available_cols], how='left')

        return undervalued_df.sort_values('value_diff', ascending=False)

    def plot_results(self, save_path: str = None):
        """Sonuçları görselleştir"""
        if not self.results:
            print("❌ Henüz eğitilmiş model yok!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        best_model_display_name = [k for k,v in self.results.items() if self.best_model_name in k.lower().replace(' ', '_')][0]

        # 1. Model karşılaştırması
        ax1 = axes[0, 0]
        model_names = list(self.results.keys())
        r2_scores = [self.results[m]['r2'] for m in model_names]
        colors = ['#2ecc71' if m == best_model_display_name else '#95a5a6' for m in model_names]
        
        ax1.bar(model_names, r2_scores, color=colors)
        ax1.set_title('Model Karşılaştırması (R² Skoru)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Skoru')
        ax1.set_ylim([max(0, min(r2_scores)-0.05), min(1, max(r2_scores)+0.05)])
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Tahmin vs Gerçek (En iyi model)
        ax2 = axes[0, 1]
        best_results = self.results[best_model_display_name]
        ax2.scatter(best_results['actual'], best_results['predictions'], 
                      alpha=0.5, s=10)
        ax2.plot([best_results['actual'].min(), best_results['actual'].max()],
                 [best_results['actual'].min(), best_results['actual'].max()],
                 'r--', lw=2)
        ax2.set_xlabel('Gerçek Değer (EUR)')
        ax2.set_ylabel('Tahmin Edilen Değer (EUR)')
        ax2.set_title(f'Tahmin vs Gerçek ({best_model_display_name})', 
                      fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # 3. Feature Importance
        ax3 = axes[1, 0]
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            ax3.barh(top_features['feature'], top_features['importance'])
            ax3.set_xlabel('Önem Skoru')
            ax3.set_title('En Önemli 10 Özellik', fontsize=14, fontweight='bold')
            ax3.invert_yaxis()
            ax3.grid(axis='x', alpha=0.3)
        
        # 4. Hata dağılımı
        ax4 = axes[1, 1]
        errors = best_results['predictions'] - best_results['actual']
        ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Tahmin Hatası (EUR)')
        ax4.set_ylabel('Frekans')
        ax4.set_title('Tahmin Hatası Dağılımı', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")
        
        plt.show()

    def save_model(self, path: str):
        """Modeli ve ilgili nesneleri kaydet"""
        if self.best_model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        # Kaydedilecek veriyi bir sözlükte topla
        data_to_save = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance
        }
        joblib.dump(data_to_save, path)
        print(f"✅ Model kaydedildi: {path}")

    @staticmethod
    def load_model(path: str):
        """Kaydedilmiş modeli yükle"""
        # Bu metodun Streamlit tarafındaki `auto_load_latest_model` fonksiyonu ile
        # senkronize çalıştığından emin ol. Streamlit tarafında yükleme yapıldığı için
        # bu metodun doğrudan kullanılması gerekmeyebilir.
        try:
            data = joblib.load(path)
            
            # Modern format (dict)
            if isinstance(data, dict):
                predictor = PlayerValuePredictor()
                predictor.best_model = data['model']
                predictor.scaler = data['scaler']
                predictor.best_model_name = data['model_name']
                predictor.feature_importance = data.get('feature_importance')
                predictor.results = {}
                print(f"✅ Model yüklendi: {predictor.best_model_name}")
                return predictor
            # Eski format (tüm sınıf)
            elif isinstance(data, PlayerValuePredictor):
                print(f"✅ Model yüklendi (eski format): {data.best_model_name}")
                return data
            else:
                raise TypeError("Tanınmayan model formatı.")
        except Exception as e:
            print(f"❌ Model yüklenirken hata oluştu: {e}")
            return None


class PerformancePredictor:
    """Oyuncu performansını tahmin eden model"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Performans tahmin modelini eğit"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        print(f"Performans Tahmini R² Skoru: {r2:.4f}")
        
        return r2
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Performans tahmini yap"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)