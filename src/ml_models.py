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
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.results = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """Modelleri eğit ve en iyisini seç"""
        print("\n🤖 Makine öğrenmesi modelleri eğitiliyor...\n")
        
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
            
            self.results[name] = {
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
                                 threshold: float = 0.7) -> pd.DataFrame:
        """Değerinin altında olan oyuncuları bul"""
        predictions = self.predict(X)
        
        # Tahmin / Gerçek değer oranı
        value_ratio = predictions / y.values
        
        # Değerinin altında olanlar (tahmin > gerçek değer)
        undervalued_mask = value_ratio > (1 + threshold)
        
        undervalued_df = pd.DataFrame({
            'actual_value': y[undervalued_mask].values,
            'predicted_value': predictions[undervalued_mask],
            'value_diff': predictions[undervalued_mask] - y[undervalued_mask].values,
            'value_ratio': value_ratio[undervalued_mask]
        })
        
        return undervalued_df.sort_values('value_diff', ascending=False)
    
    def plot_results(self, save_path: str = None):
        """Sonuçları görselleştir"""
        if not self.results:
            print("❌ Henüz eğitilmiş model yok!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model karşılaştırması
        ax1 = axes[0, 0]
        model_names = list(self.results.keys())
        r2_scores = [self.results[m]['r2'] for m in model_names]
        colors = ['#2ecc71' if m == self.best_model_name else '#95a5a6' for m in model_names]
        
        ax1.bar(model_names, r2_scores, color=colors)
        ax1.set_title('Model Karşılaştırması (R² Skoru)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Skoru')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Tahmin vs Gerçek (En iyi model)
        ax2 = axes[0, 1]
        best_results = self.results[self.best_model_name]
        ax2.scatter(best_results['actual'], best_results['predictions'], 
                   alpha=0.5, s=10)
        ax2.plot([best_results['actual'].min(), best_results['actual'].max()],
                [best_results['actual'].min(), best_results['actual'].max()],
                'r--', lw=2)
        ax2.set_xlabel('Gerçek Değer (EUR)')
        ax2.set_ylabel('Tahmin Edilen Değer (EUR)')
        ax2.set_title(f'Tahmin vs Gerçek ({self.best_model_name.upper()})', 
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
        errors = best_results['predictions'] - best_results['actual'].values
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
        """Modeli kaydet"""
        if self.best_model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance
        }, path)
        print(f"✅ Model kaydedildi: {path}")
    
    def load_model(self, path: str):
        """Modeli yükle"""
        data = joblib.load(path)
        self.best_model = data['model']
        self.scaler = data['scaler']
        self.best_model_name = data['model_name']
        self.feature_importance = data.get('feature_importance')
        print(f"✅ Model yüklendi: {self.best_model_name}")


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