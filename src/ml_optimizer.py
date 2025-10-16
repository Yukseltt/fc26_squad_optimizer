import numpy as np
import pandas as pd
from typing import List, Dict


class MLSquadOptimizer:
    """ML ve sezgisel kurallarla hızlı takım seçimi (GA yok).

    Yaklaşım: Pozisyon bazlı, bütçe kısıtlı, greedy seçim.
    Skor bileşenleri:
    - Pozisyon uygunluğu + overall
    - (opsiyonel) ML değer tahmini ile undervaluation bonusu
    - Basit kimya proxysi (milliyet/lig/klüp eşleşmesi)
    - Bütçe fizibilite cezası
    """

    def __init__(self, players_df: pd.DataFrame, formation: str = '433',
                 include_bench: bool = True, bench_size: int = 7):
        self.players_df = players_df.copy()
        self.formation = formation
        self.include_bench = include_bench
        self.bench_size = bench_size

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

        self.bench_positions = self._get_bench_positions()

    def _get_bench_positions(self) -> List[str]:
        unique_positions = list(set(self.formations.get(self.formation, [])))
        bench_priority = {
            'GK': 2, 'CB': 2, 'CM': 2, 'ST': 2,
            'LB': 1, 'RB': 1, 'LW': 1, 'RW': 1, 'CDM': 1, 'CAM': 1, 'LM': 1, 'RM': 1
        }
        bench: List[str] = []
        for pos in unique_positions:
            bench.extend([pos] * bench_priority.get(pos, 1))
        return bench[:self.bench_size]

    def _get_position_score(self, player: pd.Series, position: str) -> float:
        pos_col = self.position_mapping.get(position, 'cm')
        score = player.get(pos_col, 50)
        try:
            return float(score) if pd.notna(score) else 50.0
        except Exception:
            return 50.0

    def _chemistry_increment(self, current_squad: List[pd.Series], candidate: pd.Series) -> float:
        """Basit kimya proxysi: aynı milliyet/lig/klüp eşleşmelerinden artı puan."""
        if not current_squad:
            return 0.0
        inc = 0.0
        cn, cl, cg = (
            candidate.get('nationality_name', 'Unknown'),
            candidate.get('league_name', 'Unknown'),
            candidate.get('club_name', 'Unknown'),
        )
        for p in current_squad:
            if p.get('nationality_name', None) == cn:
                inc += 3
            if p.get('league_name', None) == cl:
                inc += 2
            if p.get('club_name', None) == cg:
                inc += 5
        return inc

    def _score_player(self, player: pd.Series, position: str,
                      current_squad: List[pd.Series],
                      use_ml: bool, ml_predictor,
                      use_synergy: bool, synergy_predictor,
                      remaining_budget: float, remaining_slots: int) -> float:
        position_score = self._get_position_score(player, position)
        overall = float(player.get('overall', 70))
        base = position_score * 0.3 + overall * 0.7

        # ML undervaluation bonus: predicted higher than market value → bonus
        if use_ml and ml_predictor is not None:
            try:
                pred_val = float(ml_predictor.predict(player.to_frame().T)[0])
                market_val = float(player.get('value_eur', 0) or 0)
                if market_val > 0 and pred_val > 0:
                    underval = np.log10(max(pred_val / market_val, 1e-6))
                    base += underval * 5.0
            except Exception:
                pass

        # Chemistry proxy bonus with current squad
        base += self._chemistry_increment(current_squad, player) * 0.5

        # Budget feasibility penalty: if too expensive relative to remaining
        value_eur = float(player.get('value_eur', 0) or 0)
        if remaining_slots > 0:
            avg_affordable = remaining_budget / remaining_slots
            # penalize if significantly over affordable average
            if value_eur > 2.0 * avg_affordable:
                base -= (value_eur / max(avg_affordable, 1))  # strong penalty
        return base

    def _eligible_players(self, used_ids: set, position: str, threshold: float) -> pd.DataFrame:
        pos_col = self.position_mapping.get(position, 'cm')
        df = self.players_df
        eligible = df[
            (~df['player_id'].isin(used_ids)) &
            (df[pos_col] >= threshold)
        ]
        # ASIL MEVKİİ kontrolü
        if 'player_positions' in eligible.columns:
            mask = eligible['player_positions'].astype(str).str.contains(position, case=False, na=False, regex=False)
            eligible = eligible[mask]
        return eligible

    def _build_group(self, positions: List[str], max_budget: float,
                      use_ml: bool, ml_predictor,
                      use_synergy: bool, synergy_predictor,
                      threshold: float) -> List[pd.Series]:
        squad: List[pd.Series] = []
        used_ids: set = set()
        remaining_budget = float(max_budget)
        remaining_slots = len(positions)

        for pos in positions:
            candidates = self._eligible_players(used_ids, pos, threshold)
            if candidates.empty:
                # fallback: relax threshold a bit
                candidates = self._eligible_players(used_ids, pos, max(35.0, threshold - 10))
            if candidates.empty:
                continue

            # Score all candidates
            scores = []
            for _, cand in candidates.iterrows():
                score = self._score_player(
                    cand, pos, squad,
                    use_ml, ml_predictor,
                    use_synergy, synergy_predictor,
                    remaining_budget, remaining_slots
                )
                scores.append(score)
            candidates = candidates.copy()
            candidates['__score__'] = scores

            # Sort by score, prefer mid-range prices to avoid extremes
            candidates.sort_values(['__score__', 'overall'], ascending=[False, False], inplace=True)

            # Try top-K shortlist with budget feasibility
            picked = None
            topk = candidates.head(20)
            for _, row in topk.iterrows():
                value = float(row.get('value_eur', 0) or 0)
                # Feasibility: keep budget for remaining
                if remaining_slots > 0:
                    max_allow = (remaining_budget / remaining_slots) * 2.5
                else:
                    max_allow = remaining_budget
                if value <= remaining_budget and value <= max_allow:
                    picked = row
                    break
            if picked is None:
                # pick best affordable candidate if none passed the heuristic
                affordable = topk[topk['value_eur'] <= remaining_budget]
                if affordable.empty:
                    continue
                picked = affordable.iloc[0]

            squad.append(picked)
            used_ids.add(picked['player_id'])
            remaining_budget -= float(picked.get('value_eur', 0) or 0)
            remaining_slots -= 1

        return squad

    def optimize(self, budget: float,
                 use_ml: bool = False, ml_predictor=None,
                 use_synergy: bool = False, synergy_predictor=None) -> Dict:
        positions = self.formations.get(self.formation, self.formations['433'])

        main_budget = budget * 0.70
        bench_budget = budget * 0.30 if self.include_bench else 0.0

        main_squad = self._build_group(
            positions, main_budget,
            use_ml, ml_predictor,
            use_synergy, synergy_predictor,
            threshold=50.0
        )

        bench: List[pd.Series] = []
        if self.include_bench:
            bench = self._build_group(
                self.bench_positions, bench_budget,
                use_ml, ml_predictor,
                use_synergy, synergy_predictor,
                threshold=45.0
            )

        total_cost = sum(float(p.get('value_eur', 0) or 0) for p in main_squad)
        total_cost += sum(float(p.get('value_eur', 0) or 0) for p in bench)

        # Simple chemistry estimate for display
        chemistry = 0.0
        nations = {}
        leagues = {}
        clubs = {}
        for p in main_squad:
            n = p.get('nationality_name', 'Unknown'); nations[n] = nations.get(n, 0) + 1
            l = p.get('league_name', 'Unknown'); leagues[l] = leagues.get(l, 0) + 1
            c = p.get('club_name', 'Unknown'); clubs[c] = clubs.get(c, 0) + 1
        for cnt in nations.values(): chemistry += cnt * 3
        for cnt in leagues.values(): chemistry += cnt * 2
        for cnt in clubs.values(): chemistry += cnt * 5

        avg_overall = 0.0
        if len(main_squad) > 0:
            avg_overall = float(np.mean([float(p.get('overall', 70)) for p in main_squad]))

        # Fitness proxy for ML optimizer (for UI consistency)
        fitness = sum(
            self._get_position_score(p, pos) * 0.3 + float(p.get('overall', 70)) * 0.7
            for p, pos in zip(main_squad, positions[:len(main_squad)])
        ) + chemistry * 0.5

        return {
            'squad': main_squad,
            'bench': bench,
            'cost': total_cost,
            'fitness': fitness,
            'chemistry': chemistry,
            'avg_overall': avg_overall,
            'positions': positions,
            'generation_progress': []
        }
