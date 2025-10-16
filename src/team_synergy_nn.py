"""
Takım Sinerjisi Neural Network
11 oyuncunun birlikte nasıl oynayacağını tahmin eder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import List, Tuple, Dict

class TeamSynergyPredictor:
    """
    Neural Network ile takım sinerjisini tahmin eder
    
    Sinerji Faktörleri:
    - Kimya (aynı lig, kulüp, milliyet)
    - Yaş dengesi
    - Oyun stili uyumu (pace, physic, teknik)
    - Pozisyon komşuluğu
    """
    
    def __init__(self):
        # GÜNCELLEME: Model artık burada sabit olarak oluşturulmuyor.
        # `train` metodu içinde, gelen parametrelere göre dinamik olarak oluşturulacak.
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        
    def extract_squad_features(self, squad: List[pd.Series], 
                                 positions: List[str]) -> np.ndarray:
        """11 oyuncudan takım özelliklerini çıkar"""
        features = []
        
        # 1. Temel İstatistikler (6)
        overalls = [float(p.get('overall', 70)) for p in squad]
        features.extend([
            np.mean(overalls),      # Ortalama overall
            np.std(overalls),       # Overall std sapması
            np.max(overalls),       # En yüksek overall
            np.min(overalls),       # En düşük overall
            np.median(overalls),    # Medyan overall
            len([o for o in overalls if o >= 80])  # 80+ oyuncu sayısı
        ])
        
        # 2. Yaş Dengesi (4)
        ages = [float(p.get('age', 25)) for p in squad]
        features.extend([
            np.mean(ages),
            np.std(ages),
            len([a for a in ages if 24 <= a <= 28]),  # Prime yaş
            len([a for a in ages if a < 23])  # Genç oyuncu
        ])
        
        # 3. Kimya Skorları (6)
        nations = [p.get('nationality_name', 'Unknown') for p in squad]
        nation_counts = pd.Series(nations).value_counts()
        features.extend([
            len(nation_counts),  # Farklı milliyet sayısı
            nation_counts.max() if len(nation_counts) > 0 else 0,  # En çok tekrar
            np.std(list(nation_counts.values)) if len(nation_counts) > 1 else 0
        ])
        
        leagues = [p.get('league_name', 'Unknown') for p in squad]
        league_counts = pd.Series(leagues).value_counts()
        features.extend([
            len(league_counts),
            league_counts.max() if len(league_counts) > 0 else 0,
            np.std(list(league_counts.values)) if len(league_counts) > 1 else 0
        ])
        
        # 4. Oyun Stili (12)
        pace_vals = [float(p.get('pace', 70)) for p in squad if pd.notna(p.get('pace'))]
        shooting_vals = [float(p.get('shooting', 70)) for p in squad if pd.notna(p.get('shooting'))]
        passing_vals = [float(p.get('passing', 70)) for p in squad if pd.notna(p.get('passing'))]
        dribbling_vals = [float(p.get('dribbling', 70)) for p in squad if pd.notna(p.get('dribbling'))]
        defending_vals = [float(p.get('defending', 70)) for p in squad if pd.notna(p.get('defending'))]
        physic_vals = [float(p.get('physic', 70)) for p in squad if pd.notna(p.get('physic'))]
        
        for vals in [pace_vals, shooting_vals, passing_vals, dribbling_vals, defending_vals, physic_vals]:
            if len(vals) > 0:
                features.extend([np.mean(vals), np.std(vals)])
            else:
                features.extend([70, 0])
        
        # 5. Pozisyon Uyumu (11)
        position_mapping = {
            'GK': 'gk', 'LB': 'lb', 'CB': 'cb', 'RB': 'rb',
            'LWB': 'lwb', 'RWB': 'rwb', 'CDM': 'cdm', 'CM': 'cm',
            'CAM': 'cam', 'LM': 'lm', 'RM': 'rm', 'LW': 'lw',
            'RW': 'rw', 'ST': 'st', 'CF': 'cf', 'LCB': 'cb',
            'RCB': 'cb', 'LCM': 'cm', 'RCM': 'cm', 'LAM': 'cam',
            'RAM': 'cam', 'LS': 'st', 'RS': 'st', 'LDM': 'cdm',
            'RDM': 'cdm', 'LF': 'lw', 'RF': 'rw'
        }
        
        for player, position in zip(squad, positions):
            pos_col = position_mapping.get(position.upper(), 'cm')
            pos_score = float(player.get(pos_col, 50)) if pd.notna(player.get(pos_col)) else 50
            features.append(pos_score)
        
        # 6. Hücum-Savunma Dengesi (4)
        attack_positions = ['LW', 'RW', 'ST', 'CF', 'CAM', 'LF', 'RF']
        defense_positions = ['CB', 'LB', 'RB', 'CDM', 'LCB', 'RCB']
        
        attack_overalls = [float(squad[i].get('overall', 70)) for i, pos in enumerate(positions) if pos.upper() in attack_positions]
        defense_overalls = [float(squad[i].get('overall', 70)) for i, pos in enumerate(positions) if pos.upper() in defense_positions]
        
        features.extend([
            np.mean(attack_overalls) if attack_overalls else 70,
            np.mean(defense_overalls) if defense_overalls else 70,
            abs(np.mean(attack_overalls) - np.mean(defense_overalls)) if attack_overalls and defense_overalls else 0,
            len(attack_overalls) / len(positions)  # Hücum oranı
        ])
        
        # 7. Değer Dağılımı (3)
        values = [float(p.get('value_eur', 0)) for p in squad]
        features.extend([
            np.mean(values),
            np.std(values),
            np.max(values) / (np.mean(values) + 1)  # En pahalı/ortalama oranı
        ])
        
        return np.array(features)

    def generate_synthetic_training_data(self, df: pd.DataFrame, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        # ... Bu fonksiyon içeriği aynı kalabilir, değişiklik gerekmiyor ...
        print(f"🔬 {n_samples} sentetik takım oluşturuluyor...")
        
        X_train = []
        y_train = []
        
        formations = {
            '433': ['GK', 'LB', 'CB', 'CB', 'RB', 'CM', 'CM', 'CM', 'LW', 'ST', 'RW'],
            '442': ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CM', 'CM', 'RM', 'ST', 'ST'],
            '352': ['GK', 'CB', 'CB', 'CB', 'LM', 'CM', 'CM', 'CM', 'RM', 'ST', 'ST']
        }
        
        for i in range(n_samples):
            if len(X_train) >= n_samples: break
            if i % 200 == 0 and i > 0: print(f"  İlerleme: {len(X_train)}/{n_samples}")
            
            formation_name = np.random.choice(list(formations.keys()))
            positions = formations[formation_name]
            
            squad = []
            used_ids = set()
            
            # Daha basit ve etkili bir oyuncu seçme mantığı
            try:
                for pos in positions:
                    base_pos = pos.replace('L','').replace('R','')
                    eligible_players = df[
                        (~df['player_id'].isin(used_ids)) &
                        (df['player_positions'].str.contains(base_pos, na=False))
                    ]
                    if eligible_players.empty:
                        raise StopIteration # Bu takımı atla
                    
                    player = eligible_players.sample(1).iloc[0]
                    squad.append(player)
                    used_ids.add(player['player_id'])
            
            except StopIteration:
                continue

            if len(squad) != 11: continue
            
            features = self.extract_squad_features(squad, positions)
            synergy_score = self._calculate_true_synergy(squad, positions)
            
            X_train.append(features)
            y_train.append(synergy_score)
            
        print(f"✅ {len(X_train)} takım oluşturuldu")
        return np.array(X_train), np.array(y_train)

    def _calculate_true_synergy(self, squad: List[pd.Series], positions: List[str]) -> float:
        # ... Bu fonksiyon içeriği aynı kalabilir, değişiklik gerekmiyor ...
        score = 0.0
        
        overalls = [float(p.get('overall', 70)) for p in squad]
        avg_overall = np.mean(overalls)
        std_overall = np.std(overalls)
        score += (avg_overall - 50) * 0.6
        if std_overall > 8: score -= (std_overall - 8) * 0.5
        
        ages = [float(p.get('age', 25)) for p in squad]
        avg_age = np.mean(ages)
        prime_count = len([a for a in ages if 24 <= a <= 28])
        score += prime_count * 1.5
        if 24 <= avg_age <= 28: score += 5
        
        nations = [p.get('nationality_name', 'Unknown') for p in squad]
        nation_counts = pd.Series(nations).value_counts()
        for count in nation_counts:
            if count >= 3: score += count * 1.0
            
        leagues = [p.get('league_name', 'Unknown') for p in squad]
        league_counts = pd.Series(leagues).value_counts()
        for count in league_counts:
            if count >= 3: score += count * 0.8
            
        return max(0, min(100, score + np.random.normal(0, 2)))

    # GÜNCELLEME: Fonksiyon imzasına `nn_params` eklendi.
    def train(self, df: pd.DataFrame, n_samples: int = 1000, nn_params: Dict = None):
        """Neural Network'ü eğit"""
        print("\n🧠 Takım Sinerjisi Neural Network Eğitimi Başlıyor...")
        
        X, y = self.generate_synthetic_training_data(df, n_samples)
        
        if len(X) < 10:
            raise ValueError("Eğitim için yeterli sentetik veri oluşturulamadı.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Özellik sayısı: {X_train.shape[1]}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # GÜNCELLEME: Modeli arayüzden gelen veya varsayılan parametrelerle oluştur.
        if nn_params is None:
            nn_params = {}
            
        default_params = {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 500,
            'learning_rate': 'adaptive',
            'early_stopping': True,
            'n_iter_no_change': 20
        }
        # Kullanıcının verdiği parametreleri varsayılanların üzerine yaz
        final_params = {**default_params, **nn_params}
        
        print("\n🔄 Neural Network eğitiliyor...")
        print(f"   Parametreler: {final_params}")
        
        self.model = MLPRegressor(**final_params, random_state=42, verbose=False)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred_test = self.model.predict(X_test_scaled)
        y_pred_train = self.model.predict(X_train_scaled)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print("\n📈 Model Performansı:")
        print(f"   Test R²: {test_r2:.4f}")
        
        self.trained = True
        
        return {
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_r2': train_r2, 'test_r2': test_r2
        }
    
    def predict_synergy(self, squad: List[pd.Series], positions: List[str]) -> float:
        # ... Bu fonksiyon içeriği aynı kalabilir, değişiklik gerekmiyor ...
        if not self.trained:
            raise ValueError("Model henüz eğitilmedi! Önce train() metodunu çağırın.")
        
        features = self.extract_squad_features(squad, positions)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        synergy = self.model.predict(features_scaled)[0]
        return max(0, min(100, synergy))
    
    def save_model(self, path: str):
        # ... Bu fonksiyon içeriği aynı kalabilir, değişiklik gerekmiyor ...
        if not self.trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        data_to_save = {
            'model': self.model,
            'scaler': self.scaler,
            'trained': self.trained
        }
        joblib.dump(data_to_save, path)
        print(f"✅ Takım Sinerjisi modeli kaydedildi: {path}")

    @staticmethod
    def load_model(path: str):
        # ... Bu fonksiyon içeriği aynı kalabilir, değişiklik gerekmiyor ...
        data = joblib.load(path)
        
        if isinstance(data, dict): # Modern format
            predictor = TeamSynergyPredictor()
            predictor.model = data['model']
            predictor.scaler = data['scaler']
            predictor.trained = data.get('trained', False) # Eski versiyonlarla uyumluluk
            print(f"✅ Takım Sinerjisi modeli yüklendi: {path}")
            return predictor
        elif isinstance(data, TeamSynergyPredictor): # Eski format (tüm sınıf)
             print(f"✅ Takım Sinerjisi modeli yüklendi (eski format): {path}")
             return data
        else:
            raise TypeError("Tanınmayan model formatı.")

    
    def explain_synergy(self, squad: List[pd.Series], 
                       positions: List[str]) -> Dict:
        """Sinerji skorunun detaylı açıklaması"""
        synergy_score = self.predict_synergy(squad, positions)
        
        # Detaylı analiz
        overalls = [float(p.get('overall', 70)) for p in squad]
        ages = [float(p.get('age', 25)) for p in squad]
        
        nations = [p.get('nationality_name', '') for p in squad]
        nation_diversity = len(set(nations))
        
        leagues = [p.get('league_name', '') for p in squad]
        league_diversity = len(set(leagues))
        
        explanation = {
            'synergy_score': synergy_score,
            'avg_overall': np.mean(overalls),
            'overall_std': np.std(overalls),
            'avg_age': np.mean(ages),
            'age_std': np.std(ages),
            'nation_diversity': nation_diversity,
            'league_diversity': league_diversity,
            'rating': self._get_rating(synergy_score)
        }
        
        return explanation
    
    def _get_rating(self, score: float) -> str:
        """Sinerji skoruna göre değerlendirme"""
        if score >= 90:
            return "⭐⭐⭐⭐⭐ Mükemmel Sinerji!"
        elif score >= 80:
            return "⭐⭐⭐⭐ Çok İyi Sinerji"
        elif score >= 70:
            return "⭐⭐⭐ İyi Sinerji"
        elif score >= 60:
            return "⭐⭐ Orta Sinerji"
        else:
            return "⭐ Zayıf Sinerji"