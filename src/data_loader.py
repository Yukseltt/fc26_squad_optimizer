import pandas as pd
import numpy as np
from typing import Tuple, List

class DataLoader:
    """FC26 oyuncu verilerini yükler ve temizler"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.position_columns = ['gk', 'lb', 'cb', 'rb', 'lwb', 'rwb', 'cdm', 
                                'cm', 'cam', 'lm', 'rm', 'lw', 'rw', 'st', 
                                'cf', 'lf', 'rf']
        
    def load_data(self) -> pd.DataFrame:
        """CSV dosyasını yükle"""
        print("📂 Veri yükleniyor...")
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ {len(self.df)} oyuncu yüklendi")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Veriyi temizle ve hazırla"""
        print("\n🧹 Veri temizleniyor...")
        
        # Eksik değerleri kontrol et
        print(f"Eksik değer sayısı: {self.df.isnull().sum().sum()}")
        
        # Gerekli sütunları kontrol et
        required_cols = ['overall', 'value_eur', 'player_positions', 'age']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"❌ Gerekli sütun bulunamadı: {col}")
        
        # Temizleme işlemleri
        self.df = self.df[self.df['overall'].notna()]
        self.df = self.df[self.df['value_eur'].notna()]
        self.df = self.df[self.df['value_eur'] > 0]
        
        # Pozisyon sütunlarını sayıya çevir
        for pos_col in self.position_columns:
            if pos_col in self.df.columns:
                self.df[pos_col] = pd.to_numeric(self.df[pos_col], errors='coerce').fillna(50)
        
        print(f"✅ Temizleme tamamlandı. Kalan oyuncu: {len(self.df)}")
        return self.df
    
    def get_features_for_ml(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Makine öğrenmesi için özellikleri hazırla"""
        print("\n🎯 ML özellikleri hazırlanıyor...")
        
        # Hedef değişken: value_eur
        y = self.df['value_eur']
        
        # Özellikler
        feature_cols = [
            'overall', 'potential', 'age', 'height_cm', 'weight_kg',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
            'weak_foot', 'skill_moves', 'international_reputation'
        ]
        
        # Mevcut sütunları al
        available_cols = [col for col in feature_cols if col in self.df.columns]
        X = self.df[available_cols].copy()
        
        # Eksik değerleri doldur
        X = X.fillna(X.mean())
        
        print(f"✅ {len(available_cols)} özellik hazırlandı")
        return X, y
    
    def get_position_features(self, position: str) -> List[str]:
        """Belirli bir pozisyon için önemli özellikleri döndür"""
        position_importance = {
            'GK': ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_positioning', 'goalkeeping_reflexes'],
            'CB': ['defending', 'physic', 'heading_accuracy', 'marking'],
            'LB': ['pace', 'defending', 'stamina', 'crossing'],
            'RB': ['pace', 'defending', 'stamina', 'crossing'],
            'CDM': ['defending', 'passing', 'stamina', 'interceptions'],
            'CM': ['passing', 'stamina', 'vision', 'ball_control'],
            'CAM': ['passing', 'dribbling', 'shooting', 'vision'],
            'LW': ['pace', 'dribbling', 'crossing', 'finishing'],
            'RW': ['pace', 'dribbling', 'crossing', 'finishing'],
            'ST': ['shooting', 'finishing', 'positioning', 'heading_accuracy']
        }
        return position_importance.get(position, [])
    
    def get_statistics(self) -> dict:
        """Veri seti istatistiklerini döndür"""
        stats = {
            'total_players': len(self.df),
            'avg_overall': self.df['overall'].mean(),
            'avg_value': self.df['value_eur'].mean(),
            'max_overall': self.df['overall'].max(),
            'min_value': self.df['value_eur'].min(),
            'max_value': self.df['value_eur'].max(),
            'avg_age': self.df['age'].mean() if 'age' in self.df.columns else 0
        }
        return stats
    
    def filter_by_budget(self, max_budget: float) -> pd.DataFrame:
        """Bütçeye göre oyuncuları filtrele"""
        return self.df[self.df['value_eur'] <= max_budget].copy()
    
    def filter_by_position(self, position: str, min_score: int = 60) -> pd.DataFrame:
        """Pozisyona göre oyuncuları filtrele"""
        pos_col = position.lower()
        if pos_col in self.df.columns:
            return self.df[self.df[pos_col] >= min_score].copy()
        return pd.DataFrame()
    
    def export_clean_data(self, output_path: str):
        """Temizlenmiş veriyi kaydet"""
        self.df.to_csv(output_path, index=False)
        print(f"✅ Temiz veri kaydedildi: {output_path}")