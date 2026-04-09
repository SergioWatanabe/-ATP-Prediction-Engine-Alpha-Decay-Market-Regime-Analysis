import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from pathlib import Path

class ClassicElo:
    """Classic Elo rating system with constant K-factor."""
    
    def __init__(self, initial_rating=1500, k_factor=32):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
    
    def expected_score(self, rating_a, rating_b):
        """Calculate expected win probability."""
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update_ratings(self, rating_a, rating_b, score_a):
        """Update ratings after a match."""
        expected_a = self.expected_score(rating_a, rating_b)
        rating_change = self.k_factor * (score_a - expected_a)
        return rating_a + rating_change, rating_b - rating_change

class PlayerStyle:
    """Estimate advanced stats from basic statistics."""
    
    def __init__(self):
        pass
    
    def get_serve_speed_features(self, ace_pct, surface):
        """Estimate serve speed from ace rate and surface."""
        surface_speed_multiplier = {
            'hard': 1.0, 'clay': 0.92, 'grass': 1.08, 'carpet': 1.05
        }
        base_speed = 170
        estimated_serve_speed = base_speed * (1 + ace_pct * 2) * surface_speed_multiplier.get(surface, 1.0)
        return estimated_serve_speed
    
    def get_shot_variety_index(self, first_won_pct_list):
        """Estimate shot variety from service game variance."""
        if len(first_won_pct_list) > 3:
            return np.std(first_won_pct_list)
        return 0.1


class FeatureCalculator:
    """
    Calculate ALL features for ALL matches chronologically.
    This creates a complete feature file that can be sliced for training.
    """
    
    def __init__(self, raw_match_data):
        self.raw_match_data = raw_match_data
        self.classic_elo = ClassicElo(k_factor=32)
        self.player_style = PlayerStyle()
        
        # Current Elo tracking
        self.current_enhanced_elo = defaultdict(lambda: {
            'overall': 1500, 'hard': 1500, 'clay': 1500,
            'grass': 1500, 'carpet': 1500
        })
        
        self.current_classic_elo = defaultdict(lambda: {
            'overall': 1500, 'hard': 1500, 'clay': 1500,
            'grass': 1500, 'carpet': 1500
        })
        
        # Match history for each player
        self.match_history = []
        
        # ==========================================
        # DEPLOYMENT: CATEGORICAL ENCODERS & MAPPINGS
        # ==========================================
        self.ROUND_ORDER = {
            'R128': 1, 'RR': 1, 'R64': 2, 'R32': 3,
            'R16': 4, 'QF': 5, 'SF': 6, 'F': 7
        }
        
        # Standardized Mappings
        self.SURFACE_CODES = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 4}
        self.ROUND_CODES = {'R128': 0, 'RR': 0, 'R64': 1, 'R32': 2, 'R16': 3, 'QF': 4, 'SF': 5, 'F': 6}
        self.LEVEL_WEIGHTS = {'G': 5, 'F': 4, 'M': 3, 'I': 2, 'D': 1, 'O': 0, 'P': 6, 'PM': 6}
        self.HAND_CODES = {'L': 0, 'R': 1, 'U': 0.5}
        
        # Dynamic Encoders for Strings (replaces LabelEncoder)
        self.player_encoder = {}
        self.tourney_encoder = {}
        
        # Check available features
        self.has_hand = 'winner_hand' in raw_match_data.columns
        self.has_age = 'winner_age' in raw_match_data.columns
        self.has_height = 'winner_ht' in raw_match_data.columns
        self.has_serve_stats = 'w_svpt' in raw_match_data.columns
        self.has_round = 'round' in raw_match_data.columns
        self.has_tourney_level = 'tourney_level' in raw_match_data.columns
        self.tournament_store = {}
        self.latest_stats = {}
        self.peak_stats = {}

    def export_simulator_snapshots(self, export_dir=""):
        """Save the Feature Stores to disk for the API to use."""
        with open(f"/Users/sergiowatanabe/Documents/ATPproject/api/{export_dir}tournaments.json", "w") as f:
            json.dump(self.tournament_store, f, indent=4)
            
        with open(f"/Users/sergiowatanabe/Documents/ATPproject/api/{export_dir}latest_stats.json", "w") as f:
            json.dump(self.latest_stats, f, indent=4)
            
        with open(f"/Users/sergiowatanabe/Documents/ATPproject/api/{export_dir}peak_stats.json", "w") as f:
            json.dump(self.peak_stats, f, indent=4)
            
        print(f"\n  ✓ Exported Feature Stores (Tournaments, Latest, Peak) to disk!")

    def _update_snapshots(self, player_name, current_date, age, height, hand):
        """Called inside the chronological loop to update the API JSON files."""
        # Get baseline features (as if playing on Hard, just to get base form/stats)
        features = self._get_player_features(player_name, current_date, 'hard', age, height, hand)
        
        # We must explicitly save all surface Elos so the API can simulate any surface
        all_surface_elos = {
            "Hard": self.current_classic_elo[player_name]['hard'],
            "Clay": self.current_classic_elo[player_name]['clay'],
            "Grass": self.current_classic_elo[player_name]['grass'],
            "Carpet": self.current_classic_elo[player_name]['carpet']
        }
        
        snapshot = {
            "last_match_date": current_date.strftime("%Y-%m-%d"),
            "age": float(age) if pd.notna(age) else 25.0,
            "height": float(height) if pd.notna(height) else 180.0,
            "hand": features['hand'],
            "classic_elo_overall": self.current_classic_elo[player_name]['overall'],
            "surface_elos": all_surface_elos,
            "form": {
                "win_pct_l5": features['win_pct_l5'],
                "win_pct_l10": features['win_pct_l10'],
                "momentum_score": features['momentum_score'],
                "upset_wins_l10": features['upset_wins_l10']
            },
            "serve_stats": {
                "ace_pct": features['ace_pct'],
                "df_pct": features['df_pct'],
                "first_won_pct": features['first_won_pct'],
                "second_won_pct": features['second_won_pct'],
                "serve_speed": features['serve_speed'],
                "shot_variety": features['shot_variety']
            }
        }

        # 1. Update Latest
        self.latest_stats[player_name] = snapshot

        # 2. Check and Update Peak
        current_peak = self.peak_stats.get(player_name, {}).get("classic_elo_overall", 0)
        if snapshot["classic_elo_overall"] >= current_peak:
            self.peak_stats[player_name] = snapshot
        

    def _get_encoded_id(self, encoder_dict, value):
        """Helper to dynamically encode string variables into integer IDs."""
        if pd.isna(value) or value is None:
            return -1
        value_str = str(value)
        if value_str not in encoder_dict:
            encoder_dict[value_str] = len(encoder_dict)
        return encoder_dict[value_str]

    def export_mappings(self, filepath="deployment_mappings.json"):
        """
        Exports all categorical encoders to a JSON file.
        The API will load this JSON to map raw string inputs back to correct integers.
        """
        mappings = {
            "surface_codes": self.SURFACE_CODES,
            "round_codes": self.ROUND_CODES,
            "tourney_level_weights": self.LEVEL_WEIGHTS,
            "hand_codes": self.HAND_CODES,
            "player_encoder": self.player_encoder,
            "tourney_encoder": self.tourney_encoder
        }
        with open(filepath, 'w') as f:
            json.dump(mappings, f, indent=4)
        print(f"\n  ✓ Deployment mappings successfully exported to: {filepath}")

    def calculate_all_features(self):
        """
        Calculate features for ALL matches chronologically.
        
        Returns:
            DataFrame: Complete feature set for all matches
        """
        print(f"\n{'='*80}")
        print(f"  STAGE 2: FEATURE CALCULATION (COMPLETE HISTORY)")
        print(f"{'='*80}")
        
        # Sort chronologically
        data = self._sort_chronologically(self.raw_match_data.copy())
        
        print(f"\n  Processing {len(data):,} matches chronologically...")
        print(f"  This creates a complete feature file for all matches\n")
        
        all_features = []
        
        # Process by date to handle same-day matches correctly
        unique_dates = sorted(data['tourney_date'].unique())
        
        for date_idx, current_date in enumerate(unique_dates):
            matches_today = data[data['tourney_date'] == current_date]
            
            # STEP 1: Extract features for all matches today (before any updates)
            for idx, row in matches_today.iterrows():
                try:
                    feature_row = self._extract_match_features(row, current_date)
                    if feature_row:
                        all_features.append(feature_row)
                except Exception as e:
                    if len(all_features) < 10:
                        print(f"      Error processing match: {e}")
                    continue
                t_name = str(row.get('tourney_name', 'Unknown')).strip()
                if t_name not in self.tournament_store:
                    self.tournament_store[t_name] = {
                        "surface": str(row.get('surface', 'Hard')).capitalize(),
                        "tourney_level": row.get('tourney_level', 'O')
                    }
            
            # STEP 2: Update Elo ratings for all matches today
            for idx, row in matches_today.iterrows():
                self._update_elo_systems(row)
                self._update_snapshots(
                    row['winner_name'], current_date, 
                    row.get('winner_age'), row.get('winner_ht'), row.get('winner_hand')
                )
                self._update_snapshots(
                    row['loser_name'], current_date, 
                    row.get('loser_age'), row.get('loser_ht'), row.get('loser_hand')
                )
            
            # Progress reporting
            if (date_idx + 1) % 100 == 0:
                progress_pct = (date_idx + 1) / len(unique_dates) * 100
                print(f"    Progress: {date_idx + 1:,}/{len(unique_dates):,} dates "
                      f"({progress_pct:.1f}%) - {len(all_features):,} matches processed")
        
        features_df = pd.DataFrame(all_features)
        
        print(f"\n  ✓ Feature calculation complete!")
        print(f"    Total features extracted: {len(features_df):,} matches")
        print(f"    Feature columns: {len(features_df.columns)}")
        print(f"    Unique players encoded: {len(self.player_encoder):,}")
        print(f"    Date range: {features_df['tourney_date'].min()} to {features_df['tourney_date'].max()}")
        print(f"{'='*80}\n")
        
        return features_df
    
    def _sort_chronologically(self, data):
        """Sort data chronologically by date and round."""
        if self.has_round:
            data['round_num'] = data['round'].map(self.ROUND_ORDER).fillna(99)
            data = data.sort_values(['tourney_date', 'tourney_id', 'round_num'])
        else:
            data = data.sort_values('tourney_date')
        return data.reset_index(drop=True)
    
    def _extract_match_features(self, row, current_date):
        """Extract features for a single match."""
        winner_name = row['winner_name']
        loser_name = row['loser_name']
        surface = row['surface'].lower() if pd.notna(row['surface']) else 'hard'
        
        # Randomize player A/B assignment for training
        if random.random() > 0.5:
            player_a_name = winner_name
            player_b_name = loser_name
            player_a_won = True
            player_a_age = row.get('winner_age', 25)
            player_a_height = row.get('winner_ht', 180)
            player_a_hand = row.get('winner_hand', 'R')
            player_b_age = row.get('loser_age', 25)
            player_b_height = row.get('loser_ht', 180)
            player_b_hand = row.get('loser_hand', 'R')
        else:
            player_a_name = loser_name
            player_b_name = winner_name
            player_a_won = False
            player_a_age = row.get('loser_age', 25)
            player_a_height = row.get('loser_ht', 180)
            player_a_hand = row.get('loser_hand', 'R')
            player_b_age = row.get('winner_age', 25)
            player_b_height = row.get('winner_ht', 180)
            player_b_hand = row.get('winner_hand', 'R')
        
        # Get features for both players
        features_a = self._get_player_features(
            player_a_name, current_date, surface,
            player_a_age, player_a_height, player_a_hand
        )
        
        features_b = self._get_player_features(
            player_b_name, current_date, surface,
            player_b_age, player_b_height, player_b_hand
        )
        
        # Build feature row
        feature_row = self._build_feature_row(
            features_a, features_b, player_a_won, current_date,
            winner_name, loser_name, row
        )
        
        return feature_row
    
    def _get_player_features(self, player_name, match_date, surface, age, height, hand):
        """Get all features for a player based on history before match_date."""
        features = {}
        
        # Static features
        features['age'] = age if pd.notna(age) else 25
        features['height'] = height if pd.notna(height) else 180
        features['hand'] = self.HAND_CODES.get(hand, 0.5)
        
        # Get historical matches
        all_matches = self._get_matches_before_date(player_name, match_date)
        surface_matches = [m for m in all_matches if m['surface'] == surface]
        
        if not all_matches:
            return self._default_features(features)
        
        # Classic Elo features
        features['classic_elo_overall'] = self.current_classic_elo[player_name]['overall']
        features['classic_elo_surface'] = self.current_classic_elo[player_name][surface]
        
        # Elo history metrics
        classic_elo_history = [m['classic_elo'] for m in all_matches if 'classic_elo' in m]
        features['classic_peak_elo'] = max(classic_elo_history) if classic_elo_history else 1500
        
        if len(classic_elo_history) > 2:
            classic_elo_changes = [abs(classic_elo_history[i] - classic_elo_history[i-1]) 
                                  for i in range(1, min(len(classic_elo_history), 10))]
            features['classic_elo_volatility'] = np.std(classic_elo_changes) if classic_elo_changes else 0
        else:
            features['classic_elo_volatility'] = 0
        
        # Rest and fatigue
        last_match = all_matches[-1]
        features['rest_days'] = (match_date - last_match['date']).days
        features['has_long_layoff'] = 1 if features['rest_days'] > 60 else 0
        
        # Surface transition penalty
        last_surface = last_match['surface']
        if last_surface != surface and features['rest_days'] < 14:
            penalty_factor = 1 - (features['rest_days'] / 14) * 0.3
            features['surface_transition_penalty'] = 30 * penalty_factor
        else:
            features['surface_transition_penalty'] = 0
        
        # Match counts
        features['matches_last_7days'] = sum(
            1 for m in all_matches if (match_date - m['date']).days <= 7
        )
        features['matches_last_30days'] = sum(
            1 for m in all_matches if (match_date - m['date']).days <= 30
        )
        
        # Form features
        recent_matches = all_matches[-10:] if len(all_matches) >= 10 else all_matches
        wins_l10 = [1 if m['won'] else 0 for m in recent_matches]
        
        recent_5 = all_matches[-5:] if len(all_matches) >= 5 else all_matches
        wins_l5 = [1 if m['won'] else 0 for m in recent_5]
        
        features['win_pct_l5'] = np.mean(wins_l5) if wins_l5 else 0
        features['win_pct_l10'] = np.mean(wins_l10) if wins_l10 else 0
        
        # Momentum score
        if len(wins_l5) >= 3:
            weights = range(1, len(wins_l5) + 1)
            features['momentum_score'] = sum(w * win for w, win in zip(weights, wins_l5)) / sum(weights)
        else:
            features['momentum_score'] = 0
        
        # Upset wins
        upset_count = sum(1 for m in recent_matches if m.get('was_upset', False) and m['won'])
        features['upset_wins_l10'] = upset_count
        
        # Serve statistics
        features.update(self._extract_serve_stats(surface_matches, player_name, surface))
        
        return features
    
    def _get_matches_before_date(self, player_name, before_date):
        """Get all matches for a player before a given date."""
        matches = []
        for match in self.match_history:
            if match['player'] == player_name and match['date'] < before_date:
                matches.append(match)
        return matches
    
    def _default_features(self, base_features):
        """Return default features for first match."""
        base_features.update({
            'rest_days': 14, 'matches_last_7days': 0,
            'matches_last_30days': 0, 'win_pct_l5': 0, 'win_pct_l10': 0,
            'momentum_score': 0, 'upset_wins_l10': 0, 'ace_pct': 0.05,
            'df_pct': 0.03, 'first_won_pct': 0.65, 'second_won_pct': 0.50,
            'serve_speed': 170, 'shot_variety': 0.1,
            'surface_transition_penalty': 0, 'has_long_layoff': 0,
            'classic_elo_overall': 1500, 'classic_elo_surface': 1500,
            'classic_peak_elo': 1500, 'classic_elo_volatility': 0
        })
        return base_features
    
    def _extract_serve_stats(self, surface_matches, player_name, surface):
        """Extract serve statistics from historical matches."""
        if not surface_matches:
            return {
                'ace_pct': 0.05, 'df_pct': 0.03,
                'first_won_pct': 0.65, 'second_won_pct': 0.50,
                'serve_speed': 170, 'shot_variety': 0.1
            }
        
        ace_pcts, df_pcts, first_won_pcts, second_won_pcts = [], [], [], []
        recent_surface = surface_matches[-15:] if len(surface_matches) >= 15 else surface_matches
        
        for match in recent_surface:
            stats = match.get('stats', {})
            svpt = stats.get('svpt', 0)
            
            if svpt > 0:
                ace_pcts.append(stats.get('ace', 0) / svpt)
                df_pcts.append(stats.get('df', 0) / svpt)
                first_in = stats.get('1stIn', 0)
                if first_in > 0:
                    first_won_pcts.append(stats.get('1stWon', 0) / first_in)
                second_serves = svpt - first_in
                if second_serves > 0:
                    second_won_pcts.append(stats.get('2ndWon', 0) / second_serves)
        
        ace_pct = np.mean(ace_pcts) if ace_pcts else 0.05
        df_pct = np.mean(df_pcts) if df_pcts else 0.03
        first_won = np.mean(first_won_pcts) if first_won_pcts else 0.65
        second_won = np.mean(second_won_pcts) if second_won_pcts else 0.50
        
        return {
            'ace_pct': ace_pct,
            'df_pct': df_pct,
            'first_won_pct': first_won,
            'second_won_pct': second_won,
            'serve_speed': self.player_style.get_serve_speed_features(ace_pct, surface),
            'shot_variety': self.player_style.get_shot_variety_index(first_won_pcts)
        }
    
    def _build_feature_row(self, features_a, features_b, player_a_won, date, 
                           winner_name, loser_name, row):
        """Build complete feature row for the match."""
        # Compute adjusted Elo and expected probabilities
        elo_adjusted_a = features_a['classic_elo_surface'] - features_a['surface_transition_penalty']
        elo_adjusted_b = features_b['classic_elo_surface'] - features_b['surface_transition_penalty']
        
        classic_expected_a = self.classic_elo.expected_score(
            features_a['classic_elo_surface'], 
            features_b['classic_elo_surface']
        )
        
        # Extract and Encode Variables Using Centralized Mappings
        t_level = row.get('tourney_level', 'O')
        tourney_weight = self.LEVEL_WEIGHTS.get(t_level, 0)
        
        surface_raw = str(row.get('surface', 'Hard')).capitalize()
        surface_code = self.SURFACE_CODES.get(surface_raw, 0)

        round_raw = str(row.get('round', 'R32'))
        round_code = self.ROUND_CODES.get(round_raw, 2)
        
        # Figure out Player A and Player B names
        player_a_name = winner_name if player_a_won else loser_name
        player_b_name = loser_name if player_a_won else winner_name
        tourney_name = row.get('tourney_name', 'Unknown')
        
        # Apply Dynamic Categorical Encoders
        player_A_id = self._get_encoded_id(self.player_encoder, player_a_name)
        player_B_id = self._get_encoded_id(self.player_encoder, player_b_name)
        tourney_id = self._get_encoded_id(self.tourney_encoder, tourney_name)
        
        return {
            # Metadata
            'tourney_date': date,
            'outcome': 1 if player_a_won else 0,
            
            # String Data (Useful for inspection)
            'winner_name': winner_name,
            'loser_name': loser_name,
            'tourney_name': tourney_name,
            'surface': row.get('surface', 'Hard'),
            'round': row.get('round', 'R32'),
            
            # Encoded Categoricals for the Machine Learning Model
            'player_A_id': player_A_id,
            'player_B_id': player_B_id,
            'tourney_id': tourney_id,
            'surface_code': surface_code,
            'round_code': round_code,
            'tourney_level_weight': tourney_weight,
            
            # Elo Features
            'classic_elo_adjusted_A': elo_adjusted_a,
            'classic_elo_adjusted_B': elo_adjusted_b,
            
            'classic_elo_overall_A': features_a['classic_elo_overall'],
            'classic_elo_overall_B': features_b['classic_elo_overall'],
            'classic_elo_diff_overall': features_a['classic_elo_overall'] - features_b['classic_elo_overall'],
            
            'classic_elo_surface_A': features_a['classic_elo_surface'],
            'classic_elo_surface_B': features_b['classic_elo_surface'],
            'classic_elo_diff_surface': features_a['classic_elo_surface'] - features_b['classic_elo_surface'],
            
            'classic_expected_win_prob_A': classic_expected_a,
            'classic_expected_win_prob_B': 1 - classic_expected_a,
            
            'classic_elo_momentum_A': features_a['classic_elo_surface'] - features_a['classic_peak_elo'],
            'classic_elo_momentum_B': features_b['classic_elo_surface'] - features_b['classic_peak_elo'],
            'classic_elo_momentum_diff': (features_a['classic_elo_surface'] - features_a['classic_peak_elo']) - 
                                        (features_b['classic_elo_surface'] - features_b['classic_peak_elo']),
            
            'classic_elo_volatility_A': features_a['classic_elo_volatility'],
            'classic_elo_volatility_B': features_b['classic_elo_volatility'],
            'classic_elo_volatility_diff': features_a['classic_elo_volatility'] - features_b['classic_elo_volatility'],
            
            'classic_peak_elo_A': features_a['classic_peak_elo'],
            'classic_peak_elo_B': features_b['classic_peak_elo'],
            
            # Player profile
            'age_A': features_a['age'],
            'age_B': features_b['age'],
            'age_diff': features_a['age'] - features_b['age'],
            
            'height_A': features_a['height'],
            'height_B': features_b['height'],
            'height_diff': features_a['height'] - features_b['height'],
            
            'hand_A': features_a['hand'],
            'hand_B': features_b['hand'],
            'hand_mismatch': 1 if features_a['hand'] != features_b['hand'] else 0,
            
            # Fatigue features
            'rest_diff': features_a['rest_days'] - features_b['rest_days'],
            'has_long_layoff_A': features_a['has_long_layoff'],
            'has_long_layoff_B': features_b['has_long_layoff'],
            
            'matches_diff_7d': features_a['matches_last_7days'] - features_b['matches_last_7days'],
            'matches_diff_30d': features_a['matches_last_30days'] - features_b['matches_last_30days'],
            
            # Form features
            'win_pct_l5_A': features_a['win_pct_l5'],
            'win_pct_l5_B': features_b['win_pct_l5'],
            'form_diff_l5': features_a['win_pct_l5'] - features_b['win_pct_l5'],
            
            'win_pct_l10_A': features_a['win_pct_l10'],
            'win_pct_l10_B': features_b['win_pct_l10'],
            'form_diff_l10': features_a['win_pct_l10'] - features_b['win_pct_l10'],
            
            'momentum_A': features_a['momentum_score'],
            'momentum_B': features_b['momentum_score'],
            'momentum_diff': features_a['momentum_score'] - features_b['momentum_score'],
            
            'upset_wins_A': features_a['upset_wins_l10'],
            'upset_wins_B': features_b['upset_wins_l10'],
            'upset_capability_diff': features_a['upset_wins_l10'] - features_b['upset_wins_l10'],
            
            # Serve statistics
            'ace_pct_A': features_a['ace_pct'],
            'ace_pct_B': features_b['ace_pct'],
            'ace_pct_diff': features_a['ace_pct'] - features_b['ace_pct'],
            
            'df_pct_A': features_a['df_pct'],
            'df_pct_B': features_b['df_pct'],
            'df_pct_diff': features_a['df_pct'] - features_b['df_pct'],
            
            '1st_won_pct_A': features_a['first_won_pct'],
            '1st_won_pct_B': features_b['first_won_pct'],
            '1st_won_pct_diff': features_a['first_won_pct'] - features_b['first_won_pct'],
            
            '2nd_won_pct_A': features_a['second_won_pct'],
            '2nd_won_pct_B': features_b['second_won_pct'],
            '2nd_won_pct_diff': features_a['second_won_pct'] - features_b['second_won_pct'],
            
            'serve_speed_A': features_a['serve_speed'],
            'serve_speed_B': features_b['serve_speed'],
            'serve_speed_diff': features_a['serve_speed'] - features_b['serve_speed'],
            
            'shot_variety_A': features_a['shot_variety'],
            'shot_variety_B': features_b['shot_variety'],
            'shot_variety_diff': features_a['shot_variety'] - features_b['shot_variety'],
        }
    
    def _update_elo_systems(self, row):
        """Update both Elo systems and match history after a match."""
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row['surface'].lower() if pd.notna(row['surface']) else 'hard'
        match_date = row['tourney_date']
        
        # Update Classic Elo
        new_classic_w_surf, new_classic_l_surf = self.classic_elo.update_ratings(
            self.current_classic_elo[winner][surface],
            self.current_classic_elo[loser][surface],
            1
        )
        self.current_classic_elo[winner][surface] = new_classic_w_surf
        self.current_classic_elo[loser][surface] = new_classic_l_surf
        
        new_classic_w_overall, new_classic_l_overall = self.classic_elo.update_ratings(
            self.current_classic_elo[winner]['overall'],
            self.current_classic_elo[loser]['overall'],
            1
        )
        self.current_classic_elo[winner]['overall'] = new_classic_w_overall
        self.current_classic_elo[loser]['overall'] = new_classic_l_overall
        
        # Determine if this was an upset
        was_upset = self.current_enhanced_elo[winner][surface] < self.current_enhanced_elo[loser][surface] - 150
        
        # Extract serve stats if available
        winner_stats = {}
        loser_stats = {}
        if self.has_serve_stats:
            winner_stats = {
                'svpt': row.get('w_svpt', 0),
                'ace': row.get('w_ace', 0),
                'df': row.get('w_df', 0),
                '1stIn': row.get('w_1stIn', 0),
                '1stWon': row.get('w_1stWon', 0),
                '2ndWon': row.get('w_2ndWon', 0),
            }
            loser_stats = {
                'svpt': row.get('l_svpt', 0),
                'ace': row.get('l_ace', 0),
                'df': row.get('l_df', 0),
                '1stIn': row.get('l_1stIn', 0),
                '1stWon': row.get('l_1stWon', 0),
                '2ndWon': row.get('l_2ndWon', 0),
            }
        
        # Add to match history for both players
        self.match_history.append({
            'player': winner,
            'date': match_date,
            'surface': surface,
            'won': True,
            'classic_elo': new_classic_w_surf,
            'was_upset': was_upset,
            'stats': winner_stats
        })
        
        self.match_history.append({
            'player': loser,
            'date': match_date,
            'surface': surface,
            'won': False,
            'classic_elo': new_classic_l_surf,
            'was_upset': False,
            'stats': loser_stats
        })