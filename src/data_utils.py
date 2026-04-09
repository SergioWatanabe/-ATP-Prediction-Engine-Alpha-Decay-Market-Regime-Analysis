import pandas as pd
import requests
import io
import numpy as np
from sklearn.metrics import log_loss
import os
from pathlib import Path



def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        if df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    return df

def normalize_level(level):
    """Standardizes tournament levels across different eras/tours."""
    if pd.isna(level): return 'O'
    level = str(level).upper()
    if level in ['T1', 'G']: return 'G'
    if level in ['T2', 'P', 'PM', 'M']: return 'M'
    if level in ['T3', 'I']: return 'I'
    if level in ['T4', 'D']: return 'D'
    return 'O'

def load_and_prepare_data(tour='ATP', start_year=2010, end_year=2024):
    """
    Fetches raw data and performs strict cleaning to ensure model quality.
    """
    tour_lower = tour.lower()
    all_years_df = []
    
    print(f"  -> Downloading data from GitHub...")
    for year in range(start_year, end_year + 1):
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour_lower}/master/{tour_lower}_matches_{year}.csv"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            df = pd.read_csv(io.StringIO(res.content.decode('utf-8')))
            all_years_df.append(df)
        except Exception as e:
            print(f"     ! Skipping {year}: {e}")
    
    if not all_years_df:
        raise ValueError(f"No data found for {tour} in range {start_year}-{end_year}")
    
    df = pd.concat(all_years_df, ignore_index=True)
    
    # --- CLEANING LOGIC ---
    # 1. Date Conversion
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    
    # 2. Level Normalization
    df['tourney_level'] = df['tourney_level'].apply(normalize_level)
    
    # 3. Drop rows missing critical physical/match info
    # We drop these because features like 'age_diff' or 'height_diff' are key to your model
    initial_len = len(df)
    critical_cols = ['winner_ht', 'winner_hand', 'loser_hand', 'loser_ht', 'score', 'surface']
    df.dropna(subset=critical_cols, inplace=True)
    
    # 4. Filter out non-standard match outcomes (Walkovers, Retirements, Defaults)
    # These distort Elo and fatigue metrics
    df = df[~df['score'].str.contains('W/O|DEF|RET', na=False, regex=True)]
    
    # 5. Surface Filtering
    df = df[df['surface'].isin(['Hard', 'Clay', 'Grass', 'Carpet'])]
    
    # 6. Chronological Sort
    df.sort_values(by=['tourney_date'], inplace=True)
    
    print(f"  -> Cleaning complete. Kept {len(df):,} of {initial_len:,} matches.")
    return df.reset_index(drop=True)

def generate_match_id(df):
    """Standardized Match ID generation across the whole pipeline."""
    return (
        df['tourney_date'].dt.strftime('%Y%m%d') + '_' +
        df['winner_name'].str.replace(' ', '').str.slice(0, 5) + 'vs' +
        df['loser_name'].str.replace(' ', '').str.slice(0, 5) + '_' +
        df.index.astype(str)
    )



def get_ablation_groups_dynamic(all_columns):
    groups = {
        '1_Rating_Backbone': [], '2_Rating_Dynamics': [], '3_Fatigue_Rest': [],
        '4_Form_Recency': [], '5_Serve_Quality': [], '6_Player_Profile': [], '7_Tourney_Context': []
    }
    for col in all_columns:
        if 'baseline' in col or '_pb' in col or col in ['match_id', 'outcome', 'year']: continue
        if any(x in col for x in ['elo_momentum', 'elo_volatility', 'classic_elo_momentum']): groups['2_Rating_Dynamics'].append(col)
        elif any(x in col for x in ['win_pct', 'form_diff', 'momentum_score', 'upset_']): groups['4_Form_Recency'].append(col)
        elif any(x in col for x in ['elo', 'expected_win', 'prob', 'peak_elo']): groups['1_Rating_Backbone'].append(col)
        elif any(x in col for x in ['rest', 'layoff', 'matches_diff', 'transition']): groups['3_Fatigue_Rest'].append(col)
        elif any(x in col for x in ['ace', 'df_pct', '1st_won', '2nd_won', 'serve_speed']): groups['5_Serve_Quality'].append(col)
        elif any(x in col for x in ['age', 'height', 'hand']): groups['6_Player_Profile'].append(col)
        elif any(x in col for x in ['tourney_level', 'round', 'surface']): groups['7_Tourney_Context'].append(col)
    return groups

def ensure_match_id(df):
    """Ensures a unique match_id exists for merging predictions."""
    if 'match_id' not in df.columns:
        df['match_id'] = (
            df['tourney_date'].dt.strftime('%Y%m%d') + '_' +
            df['winner_name'].str.replace(' ', '').str.slice(0, 5) + 'vs' +
            df['loser_name'].str.replace(' ', '').str.slice(0, 5) + '_' +
            df.index.astype(str)
        )
    return df

def load_hyperparams(filepath):
    """Loads hyperparams from CSV into a dictionary: {year: {params}}"""
    if not filepath or not os.path.exists(filepath):
        return {}
    
    hp_df = pd.read_csv(filepath)
    hp_dict = {}
    for _, row in hp_df.iterrows():
        p_dict = row.dropna().to_dict()
        year = int(p_dict.pop('year'))
        # Clean 'opt_' prefix if it exists
        hp_dict[year] = {k.replace('opt_', ''): v for k, v in p_dict.items()}
    return hp_dict

def robust_csv_load(filepath):
    """Loads CSV with redundancy checks and NaN/Inf sanitization."""
    print(f"  [Load] Reading {filepath} with redundancy checks...")
    df = pd.read_csv(filepath, index_col=False)
    
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        print(f"  [Clean] Dropping {len(unnamed_cols)} ghost columns.")
        df = df.drop(columns=unnamed_cols)
        
    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    for col in pred_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        mask = df[col].isna() | np.isinf(df[col])
        if mask.any():
            print(f"  [Fix] Found {mask.sum()} NaNs/Infs in {col}. Filling with 0.5.")
            df.loc[mask, col] = 0.5
        df[col] = np.clip(df[col], 1e-15, 1 - 1e-15)
        
    df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce').fillna(0).astype(int)
    return df

def create_ensembles(input_file, tour_name, results_dir):
    """Generates Top 2, 3, 5 ensembles based on log loss."""
    print(f"\n{'='*80}\n  GENERATING ENSEMBLES (Top 2, Top 3, Top 5)\n{'='*80}")
    
    if not os.path.exists(input_file):
        print("  ERROR: Input file not found.")
        return None

    df = robust_csv_load(input_file)
    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    y_true = df['outcome'].values

    model_performance = []
    for col in pred_cols:
        y_pred = np.clip(df[col].values, 1e-15, 1 - 1e-15)
        model_performance.append((col, log_loss(y_true, y_pred)))

    ranked_models = sorted(model_performance, key=lambda x: x[1])

    for n in [2, 3, 5]:
        if len(ranked_models) < n: continue
        top_n_names = [x[0] for x in ranked_models[:n]]
        ensemble_name = f"Ensemble_Top{n}_pb"
        df[ensemble_name] = df[top_n_names].mean(axis=1)
        print(f"  Created {ensemble_name} (LL: {log_loss(y_true, np.clip(df[ensemble_name], 1e-15, 1-1e-15)):.5f})")

    output_file = Path(results_dir) / f"{tour_name}_Enriched_Ablation.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ SAVED UPDATED FILE: {output_file}\n{'='*80}")
    return output_file