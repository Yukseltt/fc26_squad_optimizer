import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import random

class GeneticSquadOptimizer:
    """Genetik algoritma ile takım optimizasyonu"""
    
    def __init__(self, players_df: pd.DataFrame, formation: str = '433'):
        self.players_df = players_df
        self.formation = formation
        self.formations = {
            '433': ['GK', 'LB', 'CB', 'CB', 'RB', 'CM', 'CM', 'CM', 'LW', 'ST', 'RW'],
            '442': ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CM', 'CM', 'RM', 'ST', 'ST'],
            '352': ['GK', 'CB', 'CB', 'CB', 'LM', 'CM', 'CM', 'CM', 'RM', 'ST', 'ST'],
            '4231': ['GK', 'LB', 'CB', 'CB', 'RB', 'CDM', 'CDM', 'CAM', 'CAM', 'CAM', 'ST']
        }
        self.position_mapping = {
            'GK': 'gk', 'LB': 'lb', 'CB': 'cb', 'RB': 'rb',
            'LWB': 'lwb', 'RWB': 'rwb', 'CDM': 'cdm', 'CM': 'cm',
            'CAM': 'cam', 'LM': 'lm', 'RM': 'rm', 'LW': 'lw',
            'RW': 'rw', 'ST': 'st', 'CF': 'cf'
        }
        
    def get_position_score(self, player: pd.Series, position: str) -> float:
        """Oyuncunun pozisyon uygunluk skorunu al"""
        pos_col = self.position_mapping.get(position, 'cm')
        score = player.get(pos_col, 50)
        return float(score) if pd.notna(score) else 50.0
    
    def calculate_chemistry(self, squad: List[pd.Series]) -> float:
        """Takım kimyasını hesapla"""
        chemistry = 0
        
        # Milliyet kimyası
        nations = {}
        for player in squad:
            nation = player.get('nationality_name', 'Unknown')
            nations[nation] = nations.get(nation, 0) + 1
        
        for count in nations.values():
            chemistry += count * 3  # Her aynı milletten oyuncu +3
        
        # Lig kimyası
        leagues = {}
        for player in squad:
            league = player.get('league_name', 'Unknown')
            leagues[league] = leagues.get(league, 0) + 1
        
        for count in leagues.values():
            chemistry += count * 2  # Her aynı ligden oyuncu +2
        
        # Kulüp kimyası
        clubs = {}
        for player in squad:
            club = player.get('club_name', 'Unknown')
            clubs[club] = clubs.get(club, 0) + 1
        
        for count in clubs.values():
            chemistry += count * 5  # Her aynı kulüpten oyuncu +5
        
        return chemistry
    
    def calculate_fitness(self, squad: List[pd.Series], positions: List[str], 
                         use_ml_predictions: bool = False, 
                         ml_predictor=None,
                         use_synergy: bool = False,
                         synergy_predictor=None) -> float:
        """Takım fitness skorunu hesapla"""
        total_fitness = 0
        
        # 1. Pozisyon uyumu ve overall skorları
        for player, position in zip(squad, positions):
            position_score = self.get_position_score(player, position)
            overall = float(player.get('overall', 70))
            
            # ML tahmini varsa kullan
            if use_ml_predictions and ml_predictor is not None:
                try:
                    predicted_value = ml_predictor.predict(player.to_frame().T)[0]
                    # Yüksek tahmin edilen değer = daha yüksek fitness
                    value_bonus = np.log10(predicted_value) * 2
                    overall += value_bonus
                except:
                    pass
            
            total_fitness += (position_score * 0.3) + (overall * 0.7)
        
        # 2. Kimya bonusu
        chemistry = self.calculate_chemistry(squad)
        total_fitness += chemistry * 0.5
        
        # 3. NEURAL NETWORK SİNERJİ BONUSU (YENİ!)
        if use_synergy and synergy_predictor is not None:
            try:
                synergy_score = synergy_predictor.predict_synergy(squad, positions)
                # Sinerji skoru 0-100 arası, bunu fitness'a ekle
                total_fitness += synergy_score * 2  # Ağırlık: 2x
                print(f"   🧠 Sinerji Skoru: {synergy_score:.1f}")
            except Exception as e:
                print(f"   ⚠️ Sinerji hesaplanamadı: {str(e)}")
        
        # 4. Yaş dengesi bonusu
        ages = [float(player.get('age', 25)) for player in squad]
        avg_age = np.mean(ages)
        if 24 <= avg_age <= 28:  # İdeal yaş aralığı
            total_fitness += 50
        
        # 5. Pace dengesi (hücum hattı için)
        attack_positions = ['LW', 'ST', 'RW', 'CF']
        attack_pace = []
        for player, position in zip(squad, positions):
            if position in attack_positions:
                pace = float(player.get('pace', 70))
                attack_pace.append(pace)
        
        if attack_pace and np.mean(attack_pace) >= 80:
            total_fitness += 30
        
        return total_fitness
    
    def create_random_squad(self, positions: List[str], max_budget: float) -> Dict:
        """Rastgele geçerli bir takım oluştur"""
        squad = []
        total_cost = 0
        used_players = set()
        
        for position in positions:
            pos_col = self.position_mapping.get(position, 'cm')
            
            # Pozisyona uygun oyuncuları filtrele
            candidates = self.players_df[
                (~self.players_df['player_id'].isin(used_players)) &
                (self.players_df['value_eur'] > 0) &
                (self.players_df[pos_col] >= 50)
            ].copy()
            
            # Bütçeye uygun olanları filtrele
            candidates = candidates[
                (total_cost + candidates['value_eur']) <= max_budget
            ]
            
            if len(candidates) == 0:
                return None  # Geçerli takım oluşturulamadı
            
            # En iyi 50'den rastgele seç
            top_candidates = candidates.nlargest(50, pos_col)
            if len(top_candidates) == 0:
                return None
            
            selected = top_candidates.sample(n=1).iloc[0]
            
            squad.append(selected)
            used_players.add(selected['player_id'])
            total_cost += float(selected['value_eur'])
        
        return {
            'squad': squad,
            'cost': total_cost
        }
    
    def crossover(self, parent1: List[pd.Series], parent2: List[pd.Series], 
                  positions: List[str]) -> List[pd.Series]:
        """İki ebeveyni çaprazla"""
        child = []
        used_players = set()
        split_point = len(positions) // 2
        
        for i, position in enumerate(positions):
            # İlk yarı parent1'den, ikinci yarı parent2'den
            player = parent1[i] if i < split_point else parent2[i]
            
            # Aynı oyuncu zaten kullanıldıysa alternatif bul
            if player['player_id'] in used_players:
                pos_col = self.position_mapping.get(position, 'cm')
                candidates = self.players_df[
                    (~self.players_df['player_id'].isin(used_players)) &
                    (self.players_df[pos_col] >= 50)
                ]
                
                if len(candidates) > 0:
                    player = candidates.sample(n=1).iloc[0]
                else:
                    continue
            
            child.append(player)
            used_players.add(player['player_id'])
        
        return child if len(child) == len(positions) else None
    
    def mutate(self, squad: List[pd.Series], positions: List[str], 
               mutation_rate: float = 0.2) -> List[pd.Series]:
        """Rastgele mutasyon uygula"""
        if random.random() > mutation_rate:
            return squad
        
        mutated_squad = squad.copy()
        mutate_idx = random.randint(0, len(squad) - 1)
        position = positions[mutate_idx]
        
        pos_col = self.position_mapping.get(position, 'cm')
        used_ids = {p['player_id'] for p in squad}
        
        candidates = self.players_df[
            (~self.players_df['player_id'].isin(used_ids)) &
            (self.players_df[pos_col] >= 50)
        ]
        
        if len(candidates) > 0:
            new_player = candidates.sample(n=1).iloc[0]
            mutated_squad[mutate_idx] = new_player
        
        return mutated_squad
    
    def optimize(self, budget: float, population_size: int = 50, 
                generations: int = 30, elite_size: int = 5,
                use_ml: bool = False, ml_predictor=None) -> Dict:
        """Genetik algoritma ile optimizasyon"""
        print(f"\n🧬 Genetik algoritma başlatılıyor...")
        print(f"   Formasyon: {self.formation}")
        print(f"   Bütçe: {budget:,.0f} EUR")
        print(f"   Popülasyon: {population_size}")
        print(f"   Jenerasyon: {generations}")
        print(f"   ML Kullanımı: {'Evet' if use_ml else 'Hayır'}\n")
        
        positions = self.formations[self.formation]
        
        # İlk popülasyonu oluştur
        population = []
        attempts = 0
        max_attempts = population_size * 10
        
        while len(population) < population_size and attempts < max_attempts:
            squad_data = self.create_random_squad(positions, budget)
            if squad_data:
                population.append(squad_data)
            attempts += 1
        
        if len(population) == 0:
            raise ValueError("❌ Belirtilen bütçe ile takım oluşturulamadı!")
        
        print(f"✅ İlk popülasyon oluşturuldu: {len(population)} takım\n")
        
        best_overall = None
        generation_best = []
        
        # Jenerasyonlar
        for gen in range(generations):
            # Fitness hesapla
            for individual in population:
                individual['fitness'] = self.calculate_fitness(
                    individual['squad'], 
                    positions,
                    use_ml,
                    ml_predictor
                )
            
            # Sırala
            population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # En iyiyi kaydet
            if best_overall is None or population[0]['fitness'] > best_overall['fitness']:
                best_overall = {
                    'squad': [p.copy() for p in population[0]['squad']],
                    'cost': population[0]['cost'],
                    'fitness': population[0]['fitness']
                }
            
            generation_best.append(population[0]['fitness'])
            
            # İlerleme
            if (gen + 1) % 5 == 0:
                print(f"Jenerasyon {gen + 1}/{generations} - En İyi Fitness: {population[0]['fitness']:.1f}")
            
            # Yeni popülasyon
            elite = [{'squad': [p.copy() for p in ind['squad']], 
                     'cost': ind['cost']} for ind in population[:elite_size]]
            new_population = elite
            
            # Crossover ve mutasyon
            while len(new_population) < population_size:
                # Ebeveyn seçimi (tournament)
                parent1 = population[random.randint(0, min(19, len(population)-1))]
                parent2 = population[random.randint(0, min(19, len(population)-1))]
                
                # Crossover
                child_squad = self.crossover(parent1['squad'], parent2['squad'], positions)
                
                if child_squad:
                    # Mutasyon
                    child_squad = self.mutate(child_squad, positions)
                    
                    # Bütçe kontrolü
                    child_cost = sum(float(p['value_eur']) for p in child_squad)
                    
                    if child_cost <= budget:
                        new_population.append({
                            'squad': child_squad,
                            'cost': child_cost
                        })
            
            population = new_population[:population_size]
        
        # Son değerlendirme
        chemistry = self.calculate_chemistry(best_overall['squad'])
        avg_overall = np.mean([float(p['overall']) for p in best_overall['squad']])
        
        print(f"\n✅ Optimizasyon tamamlandı!")
        print(f"   En İyi Fitness: {best_overall['fitness']:.1f}")
        print(f"   Toplam Maliyet: {best_overall['cost']:,.0f} EUR")
        print(f"   Ortalama Overall: {avg_overall:.1f}")
        print(f"   Kimya Skoru: {chemistry:.0f}")
        
        return {
            'squad': best_overall['squad'],
            'cost': best_overall['cost'],
            'fitness': best_overall['fitness'],
            'chemistry': chemistry,
            'avg_overall': avg_overall,
            'positions': positions,
            'generation_progress': generation_best
        }
    
    def print_squad(self, result: Dict):
        """Takımı güzel bir formatta yazdır"""
        print("\n" + "="*60)
        print(f"{'TAKIM KADROSU':^60}")
        print("="*60)
        print(f"Formasyon: {self.formation}")
        print(f"Toplam Maliyet: {result['cost']:,.0f} EUR")
        print(f"Ortalama Overall: {result['avg_overall']:.1f}")
        print(f"Kimya Skoru: {result['chemistry']:.0f}")
        print(f"Fitness Skoru: {result['fitness']:.1f}")
        print("="*60)
        
        for i, (player, position) in enumerate(zip(result['squad'], result['positions']), 1):
            pos_score = self.get_position_score(player, position)
            print(f"{i:2d}. [{position:>3}] {player['short_name']:<25} "
                  f"OVR: {int(player['overall']):2d} | "
                  f"POS: {int(pos_score):2d} | "
                  f"{float(player['value_eur']):>12,.0f} EUR")
            print(f"      {player.get('club_name', 'N/A'):<30} | "
                  f"{player.get('nationality_name', 'N/A')}")
        
        print("="*60)
    
    def export_squad(self, result: Dict, filename: str):
        """Takımı CSV olarak kaydet"""
        squad_data = []
        for player, position in zip(result['squad'], result['positions']):
            squad_data.append({
                'position': position,
                'player_name': player['short_name'],
                'overall': int(player['overall']),
                'position_score': int(self.get_position_score(player, position)),
                'value_eur': float(player['value_eur']),
                'club': player.get('club_name', 'N/A'),
                'nationality': player.get('nationality_name', 'N/A'),
                'age': int(player.get('age', 0))
            })
        
        df = pd.DataFrame(squad_data)
        df.to_csv(filename, index=False)
        print(f"\n✅ Takım kaydedildi: {filename}")