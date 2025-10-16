"""
FC26 Squad Optimizer Package
"""

__version__ = '1.0.1'
__author__ = 'Erhan Yuksel Turgut'

from .data_loader import DataLoader
from .ml_models import PlayerValuePredictor, PerformancePredictor
from .genetic_algorithm import GeneticSquadOptimizer
from .team_synergy_nn import TeamSynergyPredictor

__all__ = [
    'DataLoader',
    'PlayerValuePredictor',
    'PerformancePredictor',
    'GeneticSquadOptimizer',
    'TeamSynergyPredictor'
]