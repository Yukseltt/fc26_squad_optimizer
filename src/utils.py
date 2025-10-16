"""
Yardımcı fonksiyonlar ve araçlar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from datetime import datetime

# Matplotlib için Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

class VisualizationHelper:
    """Görselleştirme yardımcı fonksiyonları"""
    
    @staticmethod
    def plot_squad_on_field(squad: List[pd.Series], positions: List[str], 
                           save_path: str = None):
        """Takımı saha üzerinde görselleştir"""
        fig, ax = plt.subplots(figsize=(12, 16))
        
        # Saha çizimi
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.set_facecolor('#2d5016')
        
        # Pozisyon koordinatları (4-3-3 için)
        position_coords = {
            'GK': (5, 1),
            'LB': (2, 3.5), 'CB': (4, 3.5), 'RB': (8, 3.5),
            'LCB': (3.5, 3.5), 'RCB': (6.5, 3.5),
            'LWB': (1.5, 4), 'RWB': (8.5, 4),
            'CDM': (5, 5.5), 'LDM': (3.5, 5.5), 'RDM': (6.5, 5.5),
            'CM': (5, 7), 'LCM': (3.5, 7), 'RCM': (6.5, 7),
            'CAM': (5, 9), 'LAM': (3, 9), 'RAM': (7, 9),
            'LM': (2, 7), 'RM': (8, 7),
            'LW': (2, 10.5), 'RW': (8, 10.5),
            'ST': (5, 12), 'LS': (3.5, 12), 'RS': (6.5, 12),
            'CF': (5, 11), 'LF': (3.5, 11), 'RF': (6.5, 11)
        }
        
        # Oyuncuları çiz
        for player, position in zip(squad, positions):
            coords = position_coords.get(position, (5, 7))
            
            # Oyuncu noktası
            circle = plt.Circle(coords, 0.3, color='white', zorder=3)
            ax.add_patch(circle)
            
            # Overall değeri
            overall = int(player.get('overall', 70))
            color = 'gold' if overall >= 85 else 'lightgreen' if overall >= 80 else 'lightblue'
            
            ax.text(coords[0], coords[1], str(overall), 
                   ha='center', va='center', fontsize=10, 
                   fontweight='bold', color='black', zorder=4)
            
            # Oyuncu adı
            name = player.get('short_name', 'Unknown')
            if len(name) > 12:
                name = name[:10] + '.'
            
            ax.text(coords[0], coords[1] - 0.6, name, 
                   ha='center', va='top', fontsize=8, 
                   color='white', zorder=4,
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor=color, alpha=0.7))
        
        # Başlık
        ax.set_title('Takım Kadrosu', fontsize=16, fontweight='bold', 
                    color='white', pad=20)
        
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='#2d5016')
            print(f"✅ Saha görselleştirmesi kaydedildi: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_squad_stats(squad: List[pd.Series], positions: List[str],
                        save_path: str = None):
        """Takım istatistiklerini görselleştir"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall dağılımı
        ax1 = axes[0, 0]
        overalls = [int(p.get('overall', 70)) for p in squad]
        ax1.bar(range(len(overalls)), overalls, color='skyblue', edgecolor='black')
        ax1.axhline(y=np.mean(overalls), color='r', linestyle='--', 
                   label=f'Ortalama: {np.mean(overalls):.1f}')
        ax1.set_xlabel('Oyuncu')
        ax1.set_ylabel('Overall')
        ax1.set_title('Oyuncu Overall Değerleri')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Pozisyon dağılımı
        ax2 = axes[0, 1]
        position_counts = {}
        for pos in positions:
            base_pos = pos.replace('L', '').replace('R', '').replace('C', '')
            position_counts[base_pos] = position_counts.get(base_pos, 0) + 1
        
        ax2.pie(position_counts.values(), labels=position_counts.keys(), 
               autopct='%1.0f%%', startangle=90)
        ax2.set_title('Pozisyon Dağılımı')
        
        # 3. Yaş dağılımı
        ax3 = axes[1, 0]
        ages = [int(p.get('age', 25)) for p in squad]
        ax3.hist(ages, bins=range(min(ages), max(ages)+2), 
                color='lightgreen', edgecolor='black')
        ax3.axvline(x=np.mean(ages), color='r', linestyle='--',
                   label=f'Ortalama: {np.mean(ages):.1f}')
        ax3.set_xlabel('Yaş')
        ax3.set_ylabel('Oyuncu Sayısı')
        ax3.set_title('Yaş Dağılımı')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Radar chart (Takım ortalamaları)
        ax4 = axes[1, 1]
        
        stat_names = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physical']
        stat_keys = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        
        stat_values = []
        for key in stat_keys:
            values = [float(p.get(key, 70)) for p in squad if pd.notna(p.get(key))]
            stat_values.append(np.mean(values) if values else 70)
        
        angles = np.linspace(0, 2 * np.pi, len(stat_names), endpoint=False).tolist()
        stat_values += stat_values[:1]
        angles += angles[:1]
        
        ax4 = plt.subplot(224, projection='polar')
        ax4.plot(angles, stat_values, 'o-', linewidth=2, color='blue')
        ax4.fill(angles, stat_values, alpha=0.25, color='blue')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(stat_names)
        ax4.set_ylim(0, 100)
        ax4.set_title('Takım İstatistik Profili')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ İstatistik grafikleri kaydedildi: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_optimization_progress(generation_progress: List[float],
                                  save_path: str = None):
        """Optimizasyon ilerlemesini görselleştir"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(generation_progress, linewidth=2, color='blue', marker='o')
        plt.xlabel('Jenerasyon', fontsize=12)
        plt.ylabel('En İyi Fitness Skoru', fontsize=12)
        plt.title('Genetik Algoritma İlerleme Grafiği', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # En iyi noktayı işaretle
        best_gen = np.argmax(generation_progress)
        best_fitness = generation_progress[best_gen]
        plt.plot(best_gen, best_fitness, 'r*', markersize=20, 
                label=f'En İyi: {best_fitness:.1f} (Gen {best_gen+1})')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ İlerleme grafiği kaydedildi: {save_path}")
        
        plt.tight_layout()
        plt.show()


class DataAnalyzer:
    """Veri analiz yardımcı fonksiyonları"""
    
    @staticmethod
    def analyze_position_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """Pozisyon dağılımını analiz et"""
        positions = []
        for pos_str in df['player_positions'].dropna():
            positions.extend(pos_str.split(','))
        
        position_counts = pd.Series(positions).value_counts()
        return pd.DataFrame({
            'position': position_counts.index,
            'count': position_counts.values,
            'percentage': (position_counts.values / len(df) * 100).round(2)
        })
    
    @staticmethod
    def analyze_value_by_position(df: pd.DataFrame) -> pd.DataFrame:
        """Pozisyona göre değer analizi"""
        position_stats = []
        
        positions = ['GK', 'CB', 'LB', 'RB', 'CDM', 'CM', 'CAM', 'LW', 'RW', 'ST']
        
        for pos in positions:
            pos_col = pos.lower()
            if pos_col in df.columns:
                mask = df[pos_col] >= 70
                pos_players = df[mask]
                
                if len(pos_players) > 0:
                    position_stats.append({
                        'position': pos,
                        'count': len(pos_players),
                        'avg_overall': pos_players['overall'].mean(),
                        'avg_value': pos_players['value_eur'].mean(),
                        'max_value': pos_players['value_eur'].max()
                    })
        
        return pd.DataFrame(position_stats)
    
    @staticmethod
    def find_best_value_players(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """En iyi değer/performans oranına sahip oyuncuları bul"""
        df_copy = df.copy()
        df_copy['value_per_overall'] = df_copy['value_eur'] / df_copy['overall']
        
        best_value = df_copy.nsmallest(top_n, 'value_per_overall')[
            ['short_name', 'overall', 'potential', 'value_eur', 
             'value_per_overall', 'age', 'player_positions']
        ]
        
        return best_value
    
    @staticmethod
    def league_comparison(df: pd.DataFrame) -> pd.DataFrame:
        """Ligler arası karşılaştırma"""
        league_stats = df.groupby('league_name').agg({
            'overall': ['mean', 'max'],
            'value_eur': ['mean', 'sum'],
            'player_id': 'count'
        }).round(2)
        
        league_stats.columns = ['avg_overall', 'max_overall', 
                               'avg_value', 'total_value', 'player_count']
        
        return league_stats.sort_values('avg_overall', ascending=False).head(20)


class ExportHelper:
    """Dışa aktarma yardımcı fonksiyonları"""
    
    @staticmethod
    def export_squad_to_json(result: Dict, filename: str):
        """Takımı JSON formatında kaydet"""
        export_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'formation': result.get('formation', 'N/A'),
                'total_cost': float(result['cost']),
                'avg_overall': float(result['avg_overall']),
                'chemistry': float(result['chemistry']),
                'fitness': float(result['fitness'])
            },
            'players': []
        }
        
        for player, position in zip(result['squad'], result['positions']):
            export_data['players'].append({
                'position': position,
                'name': player.get('short_name', 'Unknown'),
                'overall': int(player.get('overall', 0)),
                'value': float(player.get('value_eur', 0)),
                'age': int(player.get('age', 0)),
                'club': player.get('club_name', 'Unknown'),
                'nationality': player.get('nationality_name', 'Unknown')
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Takım JSON olarak kaydedildi: {filename}")
    
    @staticmethod
    def export_comparison_report(squads: List[Dict], filename: str):
        """Birden fazla takımı karşılaştırmalı rapor olarak kaydet"""
        comparison_data = []
        
        for i, squad in enumerate(squads, 1):
            comparison_data.append({
                'squad_id': i,
                'total_cost': squad['cost'],
                'avg_overall': squad['avg_overall'],
                'chemistry': squad['chemistry'],
                'fitness': squad['fitness']
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(filename, index=False)
        print(f"✅ Karşılaştırma raporu kaydedildi: {filename}")


class ValidationHelper:
    """Doğrulama yardımcı fonksiyonları"""
    
    @staticmethod
    def validate_squad(squad: List[pd.Series], positions: List[str], 
                      max_budget: float) -> Tuple[bool, List[str]]:
        """Takımın geçerliliğini kontrol et"""
        errors = []
        
        # Oyuncu sayısı kontrolü
        if len(squad) != 11:
            errors.append(f"Takımda {len(squad)} oyuncu var, 11 olmalı")
        
        # Bütçe kontrolü
        total_cost = sum(float(p.get('value_eur', 0)) for p in squad)
        if total_cost > max_budget:
            errors.append(f"Bütçe aşıldı: {total_cost:,.0f} > {max_budget:,.0f}")
        
        # Aynı oyuncu kontrolü
        player_ids = [p.get('player_id') for p in squad]
        if len(player_ids) != len(set(player_ids)):
            errors.append("Takımda aynı oyuncu birden fazla kez var")
        
        # Kaleci kontrolü
        if 'GK' not in positions:
            errors.append("Takımda kaleci yok")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_csv_format(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """CSV formatının geçerliliğini kontrol et"""
        errors = []
        required_columns = ['player_id', 'short_name', 'overall', 'value_eur', 
                          'player_positions']
        
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Gerekli sütun bulunamadı: {col}")
        
        if len(df) < 100:
            errors.append(f"Çok az oyuncu var: {len(df)} (en az 100 olmalı)")
        
        return len(errors) == 0, errors


def format_currency(value: float) -> str:
    """Para birimini formatla"""
    if value >= 1_000_000:
        return f"€{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"€{value/1_000:.1f}K"
    else:
        return f"€{value:.0f}"


def calculate_squad_summary(squad: List[pd.Series]) -> Dict:
    """Takım özetini hesapla"""
    return {
        'total_players': len(squad),
        'avg_overall': np.mean([float(p.get('overall', 70)) for p in squad]),
        'avg_age': np.mean([float(p.get('age', 25)) for p in squad]),
        'total_value': sum(float(p.get('value_eur', 0)) for p in squad),
        'avg_pace': np.mean([float(p.get('pace', 70)) for p in squad 
                            if pd.notna(p.get('pace'))]),
        'avg_shooting': np.mean([float(p.get('shooting', 70)) for p in squad 
                                if pd.notna(p.get('shooting'))]),
        'avg_defending': np.mean([float(p.get('defending', 70)) for p in squad 
                                 if pd.notna(p.get('defending'))])
    }

