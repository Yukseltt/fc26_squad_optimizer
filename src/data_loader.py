import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class DataLoader:
    """
    FC26 oyuncu verilerini yükler, temizler ve çoklu dil desteği ile sunar.
    Loads and cleans FC26 player data with multi-language support.
    """
    
    # Sabitler
    # Constants
    DEFAULT_POSITION_RATING = 50
    DEFAULT_MIN_POSITION_SCORE = 60

    # TR/EN metinleri
    # TR/EN texts
    TEXTS = {
        'tr': {
            'loading_data': "Veri yükleniyor...",
            'data_loaded': "✅ {count} oyuncu yüklendi.",
            'cleaning_data': "\nVeri temizleniyor...",
            'missing_values': "Eksik değer sayısı: {count}",
            'required_col_missing': "❌ Gerekli sütun bulunamadı: {col}",
            'cleaning_complete': "✅ Temizleme tamamlandı. Kalan oyuncu: {count}",
            'preparing_ml_features': "\nML özellikleri hazırlanıyor...",
            'features_ready': "✅ {count} özellik hazırlandı.",
            'data_saved': "✅ Temiz veri kaydedildi: {path}"
        },
        'en': {
            'loading_data': "Loading data...",
            'data_loaded': "✅ {count} players loaded.",
            'cleaning_data': "\nCleaning data...",
            'missing_values': "Missing value count: {count}",
            'required_col_missing': "❌ Required column not found: {col}",
            'cleaning_complete': "✅ Cleaning complete. Players remaining: {count}",
            'preparing_ml_features': "\nPreparing ML features...",
            'features_ready': "✅ {count} features prepared.",
            'data_saved': "✅ Clean data saved to: {path}"
        }
    }
    
    def __init__(self, csv_path: str, language: str = 'en'):
        """
        Args:
            csv_path (str): Veri setinin dosya yolu. / File path of the dataset.
            language (str): Kullanılacak dil ('tr' veya 'en'). / Language to use ('tr' or 'en').
        """
        self.csv_path = csv_path
        self.df = None
        self.position_columns = [
            'gk', 'lb', 'cb', 'rb', 'lwb', 'rwb', 'cdm', 
            'cm', 'cam', 'lm', 'rm', 'lw', 'rw', 'st', 
            'cf', 'lf', 'rf'
        ]
        
        if language not in self.TEXTS:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.TEXTS.keys())}")
        self.texts = self.TEXTS[language]
        
    def load_data(self) -> pd.DataFrame:
        """CSV dosyasını yükler. / Loads the CSV file."""
        print(self.texts['loading_data'])
        self.df = pd.read_csv(self.csv_path)
        print(self.texts['data_loaded'].format(count=len(self.df)))
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Veriyi temizler ve hazırlar. / Cleans and prepares the data."""
        print(self.texts['cleaning_data'])
        
        # Eksik değerleri kontrol et / Check for missing values
        print(self.texts['missing_values'].format(count=self.df.isnull().sum().sum()))
        
        # Gerekli sütunları kontrol et / Check for required columns
        required_cols = ['overall', 'value_eur', 'player_positions', 'age']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(self.texts['required_col_missing'].format(col=col))
        
        # Temizleme işlemleri / Cleaning operations
        self.df = self.df[self.df['overall'].notna()]
        self.df = self.df[self.df['value_eur'].notna()]
        self.df = self.df[self.df['value_eur'] > 0]
        
        # Pozisyon sütunlarını sayıya çevir / Convert position columns to numeric
        for pos_col in self.position_columns:
            if pos_col in self.df.columns:
                self.df[pos_col] = pd.to_numeric(self.df[pos_col], errors='coerce').fillna(self.DEFAULT_POSITION_RATING)
        
        print(self.texts['cleaning_complete'].format(count=len(self.df)))
        return self.df
    
    def get_features_for_ml(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Makine öğrenmesi için özellikleri hazırlar. / Prepares features for machine learning."""
        print(self.texts['preparing_ml_features'])
        
        y = self.df['value_eur']
        
        feature_cols = [
            'overall', 'potential', 'age', 'height_cm', 'weight_kg',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
            'weak_foot', 'skill_moves', 'international_reputation'
        ]
        
        available_cols = [col for col in feature_cols if col in self.df.columns]
        X = self.df[available_cols].copy()
        
        X = X.fillna(X.mean())
        
        print(self.texts['features_ready'].format(count=len(available_cols)))
        return X, y
    
    def get_position_features(self, position: str) -> List[str]:
        """Belirli bir pozisyon için önemli özellikleri döndürür. / Returns important features for a specific position."""
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
        return position_importance.get(position.upper(), [])
    
    def get_statistics(self) -> Dict[str, float]:
        """Veri seti istatistiklerini döndürür. / Returns dataset statistics."""
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
        """Bütçeye göre oyuncuları filtreler. / Filters players by budget."""
        return self.df[self.df['value_eur'] <= max_budget].copy()
    
    def filter_by_position(self, position: str, min_score: int = DEFAULT_MIN_POSITION_SCORE) -> pd.DataFrame:
        """Pozisyona göre oyuncuları filtreler. / Filters players by position."""
        pos_col = position.lower()
        if pos_col in self.df.columns:
            return self.df[self.df[pos_col] >= min_score].copy()
        return pd.DataFrame()
    
    def export_clean_data(self, output_path: str):
        """Temizlenmiş veriyi kaydeder. / Saves the cleaned data."""
        self.df.to_csv(output_path, index=False)
        print(self.texts['data_saved'].format(path=output_path))