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
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),  # 3 katmanlı NN
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=True
        )
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
        # Milliyet
        nations = [p.get('nationality_name', 'Unknown') for p in squad]
        nation_counts = pd.Series(nations).value_counts()
        features.extend([
            len(nation_counts),  # Farklı milliyet sayısı
            nation_counts.max() if len(nation_counts) > 0 else 0,  # En çok tekrar
            np.std(list(nation_counts.values)) if len(nation_counts) > 1 else 0
        ])
        
        # Lig
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
        
        for vals in [pace_vals, shooting_vals, passing_vals, 
                     dribbling_vals, defending_vals, physic_vals]:
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
            pos_col = position_mapping.get(position, 'cm')
            pos_score = float(player.get(pos_col, 50)) if pd.notna(player.get(pos_col)) else 50
            features.append(pos_score)
        
        # 6. Hücum-Savunma Dengesi (4)
        attack_positions = ['LW', 'RW', 'ST', 'CF', 'CAM', 'LF', 'RF']
        defense_positions = ['CB', 'LB', 'RB', 'CDM', 'LCB', 'RCB']
        
        attack_overalls = [float(squad[i].get('overall', 70)) 
                          for i, pos in enumerate(positions) if pos in attack_positions]
        defense_overalls = [float(squad[i].get('overall', 70)) 
                           for i, pos in enumerate(positions) if pos in defense_positions]
        
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
    
    def generate_synthetic_training_data(self, df: pd.DataFrame, 
                                        n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sentetik eğitim verisi oluştur
        Gerçek takım verileri olmadığı için simüle ediyoruz
        """
        print(f"🔬 {n_samples} sentetik takım oluşturuluyor...")
        
        X_train = []
        y_train = []
        
        formations = {
            '433': ['GK', 'LB', 'CB', 'CB', 'RB', 'CM', 'CM', 'CM', 'LW', 'ST', 'RW'],
            '442': ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CM', 'CM', 'RM', 'ST', 'ST'],
            '352': ['GK', 'CB', 'CB', 'CB', 'LM', 'CM', 'CM', 'CM', 'RM', 'ST', 'ST']
        }
        
        position_mapping = {
            'GK': 'gk', 'LB': 'lb', 'CB': 'cb', 'RB': 'rb',
            'CM': 'cm', 'LW': 'lw', 'RW': 'rw', 'ST': 'st',
            'LM': 'lm', 'RM': 'rm'
        }
        
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"  İlerleme: {i}/{n_samples}")
            
            # Rastgele formasyon seç
            formation = np.random.choice(list(formations.keys()))
            positions = formations[formation]
            
            # Rastgele oyuncular seç
            squad = []
            used_ids = set()
            
            for position in positions:
                pos_col = position_mapping.get(position, 'cm')
                
                candidates = df[
                    (~df['player_id'].isin(used_ids)) &
                    (df[pos_col] >= 50) &
                    (df['value_eur'] > 0)
                ]
                
                if len(candidates) == 0:
                    continue
                
                # Bazen iyi, bazen kötü takımlar oluştur
                if np.random.random() < 0.3:  # %30 kötü takım
                    candidates = candidates[candidates['overall'] < 75]
                elif np.random.random() < 0.3:  # %30 çok iyi takım
                    candidates = candidates[candidates['overall'] >= 80]
                
                if len(candidates) > 0:
                    player = candidates.sample(n=1).iloc[0]
                    squad.append(player)
                    used_ids.add(player['player_id'])
            
            if len(squad) != 11:
                continue
            
            # Özellikleri çıkar
            features = self.extract_squad_features(squad, positions)
            
            # Sinerji skorunu hesapla (gerçek bir fonksiyon)
            synergy_score = self._calculate_true_synergy(squad, positions)
            
            X_train.append(features)
            y_train.append(synergy_score)
        
        print(f"✅ {len(X_train)} takım oluşturuldu")
        return np.array(X_train), np.array(y_train)
    
    def _calculate_true_synergy(self, squad: List[pd.Series], 
                               positions: List[str]) -> float:
        """
        Gerçek sinerji skorunu hesapla (eğitim için)
        Bu, takımın ne kadar iyi oynayacağının tahmini
        """
        score = 0
        
        # 1. Temel güç (40%)
        overalls = [float(p.get('overall', 70)) for p in squad]
        score += np.mean(overalls) * 4
        
        # 2. Kimya bonusu (25%)
        nations = [p.get('nationality_name', '') for p in squad]
        leagues = [p.get('league_name', '') for p in squad]
        clubs = [p.get('club_name', '') for p in squad]
        
        nation_bonus = sum([nations.count(n) * 2 for n in set(nations)]) * 0.5
        league_bonus = sum([leagues.count(l) * 1.5 for l in set(leagues)]) * 0.5
        club_bonus = sum([clubs.count(c) * 3 for c in set(clubs)]) * 0.5
        
        score += (nation_bonus + league_bonus + club_bonus) * 0.25
        
        # 3. Pozisyon uyumu (20%)
        position_mapping = {
            'GK': 'gk', 'LB': 'lb', 'CB': 'cb', 'RB': 'rb',
            'CM': 'cm', 'LW': 'lw', 'RW': 'rw', 'ST': 'st',
            'LM': 'lm', 'RM': 'rm'
        }
        
        pos_scores = []
        for player, position in zip(squad, positions):
            pos_col = position_mapping.get(position, 'cm')
            pos_score = float(player.get(pos_col, 50)) if pd.notna(player.get(pos_col)) else 50
            pos_scores.append(pos_score)
        
        score += np.mean(pos_scores) * 2
        
        # 4. Yaş dengesi (10%)
        ages = [float(p.get('age', 25)) for p in squad]
        avg_age = np.mean(ages)
        
        if 24 <= avg_age <= 28:
            age_bonus = 10
        elif 22 <= avg_age < 24 or 28 < avg_age <= 30:
            age_bonus = 7
        else:
            age_bonus = 4
        
        score += age_bonus
        
        # 5. Oyun stili uyumu (5%)
        pace_vals = [float(p.get('pace', 70)) for p in squad if pd.notna(p.get('pace'))]
        physic_vals = [float(p.get('physic', 70)) for p in squad if pd.notna(p.get('physic'))]
        
        # Pace standardı (hızlı takım bonusu)
        if pace_vals and np.mean(pace_vals) >= 75:
            score += 5
        
        # Fiziksel güç (güçlü takım bonusu)
        if physic_vals and np.mean(physic_vals) >= 75:
            score += 5
        
        # Rastgele varyasyon ekle (gerçekçilik için)
        score += np.random.normal(0, 3)
        
        # 0-100 arası normalize et
        return max(0, min(100, score))
    
    def train(self, df: pd.DataFrame, n_samples: int = 1000):
        """Neural Network'ü eğit"""
        print("\n🧠 Takım Sinerjisi Neural Network Eğitimi Başlıyor...")
        print("="*60)
        
        # Sentetik veri oluştur
        X, y = self.generate_synthetic_training_data(df, n_samples)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Eğitim verisi: {X_train.shape}")
        print(f"📊 Test verisi: {X_test.shape}")
        print(f"📊 Özellik sayısı: {X_train.shape[1]}")
        
        # Ölçekleme
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model eğitimi
        print("\n🔄 Neural Network eğitiliyor...")
        print("   Katmanlar: 128 → 64 → 32 → 1")
        print("   Aktivasyon: ReLU")
        print("   Optimizer: Adam")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Değerlendirme
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print("\n📈 Model Performansı:")
        print(f"   Eğitim MSE: {train_mse:.2f}")
        print(f"   Test MSE: {test_mse:.2f}")
        print(f"   Eğitim R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        
        self.trained = True
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def predict_synergy(self, squad: List[pd.Series], 
                       positions: List[str]) -> float:
        """Bir takımın sinerji skorunu tahmin et"""
        if not self.trained:
            raise ValueError("Model henüz eğitilmedi! Önce train() metodunu çağırın.")
        
        features = self.extract_squad_features(squad, positions)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        synergy = self.model.predict(features_scaled)[0]
        
        # 0-100 arası sınırla
        return max(0, min(100, synergy))
    
    def predict_multiple(self, squads: List[List[pd.Series]], 
                        positions_list: List[List[str]]) -> np.ndarray:
        """Birden fazla takımın sinerji skorlarını tahmin et"""
        if not self.trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        features_list = []
        for squad, positions in zip(squads, positions_list):
            features = self.extract_squad_features(squad, positions)
            features_list.append(features)
        
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        
        # 0-100 arası sınırla
        return np.clip(predictions, 0, 100)
    
    def get_feature_count(self) -> int:
        """Özellik sayısını döndür"""
        # 6 + 4 + 6 + 12 + 11 + 4 + 3 = 46 özellik
        return 46
    
    def save_model(self, path: str):
        """Modeli kaydet"""
        if not self.trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'trained': self.trained
        }, path)
        print(f"✅ Takım Sinerjisi modeli kaydedildi: {path}")
    
    def load_model(self, path: str):
        """Modeli yükle"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.trained = data['trained']
        print(f"✅ Takım Sinerjisi modeli yüklendi: {path}")
    
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