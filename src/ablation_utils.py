import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.metrics import log_loss
from src.metrics_utils import match_logloss

# Constants from your original script
EPSILON = 1e-15
ROLLING_WINDOW = 300
TRIGGER_DELTA = 0.03
METADATA_COLS = [
    'tourney_date', 'outcome', 'winner_name', 'loser_name',
    'surface', 'tourney_name', 'round', 'match_id', 'model_probability', 'baseline_pb'
]

# ==============================================================================
#                      1. ABLATION GROUP DEFINITIONS
# ==============================================================================

def define_ablation_groups(all_columns):
    """
    1. Identifies "Personalized Elo" columns (The New Baseline exclusions).
    2. Maps Groups 1-7.
    3. Returns a dictionary where every group REMOVAL includes the Pers Elo columns.
    """
    pers_elo_cols = []
    base_groups = {
        '1_Rating_Backbone': [],
        '2_Rating_Dynamics': [],
        '3_Fatigue_Rest': [],
        '4_Form_Recency': [],
        '5_Serve_Quality': [],
        '6_Player_Profile': [],
        '7_Tourney_Context': []
    }

    for col in all_columns:
        if col in METADATA_COLS: continue

        # --- Identify Personalized Elo (Enhanced) ---
        is_pers_elo = False
        if ('elo_' in col or 'expected_win' in col) and 'classic_' not in col:
            pers_elo_cols.append(col)
            is_pers_elo = True

        # --- Identify Base Groups (Standard Logic) ---
        if any(x in col for x in ['elo_momentum', 'elo_volatility', 'classic_elo_momentum', 'classic_elo_volatility']):
            base_groups['2_Rating_Dynamics'].append(col)
        elif any(x in col for x in ['win_pct', 'form_diff', 'upset_', 'momentum_score', 'momentum_diff']):
            base_groups['4_Form_Recency'].append(col)
        elif any(x in col for x in ['elo', 'expected_win', 'prob', 'peak_elo']):
            base_groups['1_Rating_Backbone'].append(col)
        elif any(x in col for x in ['rest', 'layoff', 'matches_diff', 'transition']):
            base_groups['3_Fatigue_Rest'].append(col)
        elif any(x in col for x in ['ace', 'df_pct', '1st_won', '2nd_won', 'serve_speed', 'shot_variety']):
            base_groups['5_Serve_Quality'].append(col)
        elif any(x in col for x in ['age', 'height', 'hand']):
            base_groups['6_Player_Profile'].append(col)
        elif any(x in col for x in ['tourney_level', 'round', 'surface', 'tourney_level_weight']):
            base_groups['7_Tourney_Context'].append(col)

    final_scenarios = {}
    for group_name, cols in base_groups.items():
        combined_drop = list(set(cols + pers_elo_cols))
        final_scenarios[group_name] = combined_drop

    return final_scenarios, pers_elo_cols


# ==============================================================================
#                      2. LOW-LEVEL HELPERS
# ==============================================================================

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """All numeric non-metadata columns."""
    return [
        c for c in df.columns
        if c not in METADATA_COLS
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

def _build_lgb_params(fixed_params: dict | None) -> tuple[dict, int]:
    """Merge caller-supplied params with LightGBM defaults."""
    defaults: dict = {
        'objective':     'binary',
        'metric':        'binary_logloss',
        'verbosity':     -1,
        'boosting_type': 'gbdt',
        'random_state':  42,
        'num_leaves':    63,
        'learning_rate': 0.05,
        'n_estimators':  500,
    }
    merged = defaults.copy()
    if fixed_params:
        merged.update(fixed_params)

    for p in ['num_leaves', 'max_depth', 'n_estimators', 'min_child_samples']:
        if p in merged:
            merged[p] = int(merged[p])

    n_rounds = int(merged.pop('n_estimators', 500))
    return merged, n_rounds

def _save_model_metadata(model, experiment_name, year):
    """Save model to a .txt file (LightGBM text format)."""
    os.makedirs("models", exist_ok=True)
    filename = f"models/{experiment_name}_{year}.txt"
    model.save_model(filename)

def _train_and_predict(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       fixed_params: dict | None, 
                       experiment_name="experiment") -> np.ndarray | None:
    """Train LightGBM on train_df, return probabilities for test_df."""
    feature_cols      = _get_feature_cols(train_df)
    test_feature_cols = [c for c in feature_cols if c in test_df.columns]

    if not test_feature_cols or train_df.empty or test_df.empty:
        return None

    X_train = train_df[feature_cols].values
    y_train = train_df['outcome'].values
    X_test  = test_df[test_feature_cols].values

    params, n_rounds = _build_lgb_params(fixed_params)

    try:
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_rounds,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        current_year = test_df['tourney_date'].dt.year.iloc[0]
        _save_model_metadata(model, experiment_name, current_year)
        
        return model.predict(X_test, num_iteration=model.best_iteration)
    except Exception as e:
        print(f"      [_train_and_predict] Error: {e}")
        return None


# ==============================================================================
#                      3. TRIGGER EXPERIMENT LOGIC
# ==============================================================================

def _compute_baseline_ll(feat_df: pd.DataFrame, test_year: int, window_seasons: int, hyperparams_by_year: dict) -> float:
    """Compute the baseline log-loss for Window-N."""
    available_years = sorted(feat_df['tourney_date'].dt.year.unique())
    candidate_years = [y for y in available_years if y < test_year]

    if not candidate_years:
        return float(np.log(2))

    eval_years = candidate_years[-window_seasons:]
    season_lls: list[float] = []
    
    for yr in eval_years:
        train_df = feat_df[feat_df['tourney_date'].dt.year < yr].copy()
        test_df  = feat_df[feat_df['tourney_date'].dt.year == yr].copy()

        if train_df.empty or test_df.empty:
            continue

        preds = _train_and_predict(train_df, test_df, hyperparams_by_year.get(yr), experiment_name="baseline")
        if preds is None:
            continue

        ll = log_loss(test_df['outcome'].values, np.clip(preds, EPSILON, 1 - EPSILON))
        season_lls.append(ll)

    return float(np.mean(season_lls)) if season_lls else float(np.log(2))

def _run_single_window_experiment(feat_df: pd.DataFrame, window_seasons: int, hyperparams_by_year: dict,
                                  start_test_year: int, end_test_year: int, col_name: str) -> pd.Series:
    """Run the retrain-trigger experiment for one window size."""
    print(f"\n  {'━'*72}")
    print(f"  EXPERIMENT : {col_name}")
    print(f"  Window     : {window_seasons} seasons define the baseline LL")
    print(f"  Trigger    : rolling_LL({ROLLING_WINDOW}) > baseline + {TRIGGER_DELTA}")
    print(f"  Retrain on : ALL data up to trigger date")
    print(f"  {'━'*72}")

    predictions = np.full(len(feat_df), np.nan)
    rolling_buffer: list[float] = []

    for test_year in range(start_test_year, end_test_year + 1):
        print(f"\n    ── Year {test_year} ──")

        baseline_ll = _compute_baseline_ll(feat_df, test_year, window_seasons, hyperparams_by_year)
        threshold = baseline_ll + TRIGGER_DELTA
        print(f"    Baseline LL (last {window_seasons} seasons): {baseline_ll:.4f}  │  Threshold: {threshold:.4f}")

        year_mask = feat_df['tourney_date'].dt.year == test_year
        year_idx  = feat_df.index[year_mask].tolist()

        if not year_idx:
            print(f"    No matches found. Skipping.")
            continue

        train_df_initial = feat_df[feat_df['tourney_date'].dt.year < test_year].copy()
        test_df_year     = feat_df.loc[year_idx].copy()
        fixed_params     = hyperparams_by_year.get(test_year)

        if train_df_initial.empty:
            print(f"    No training data before {test_year}. Skipping.")
            continue

        initial_preds = _train_and_predict(train_df_initial, test_df_year, fixed_params, experiment_name=col_name)
        if initial_preds is None:
            continue

        current_year_preds = initial_preds.copy()
        retrain_count = 0

        for i, pos_idx in enumerate(year_idx):
            y_true   = float(feat_df.at[pos_idx, 'outcome'])
            y_pred   = float(current_year_preds[i])
            match_ll = match_logloss(y_true, y_pred)

            rolling_buffer.append(match_ll)
            if len(rolling_buffer) > ROLLING_WINDOW:
                rolling_buffer.pop(0)

            if len(rolling_buffer) == ROLLING_WINDOW:
                rolling_ll = float(np.mean(rolling_buffer))

                if rolling_ll > threshold:
                    retrain_count += 1
                    trigger_date = pd.Timestamp(feat_df.at[pos_idx, 'tourney_date'])
                    
                    print(f"\n    ⚡ TRIGGER #{retrain_count} at match {i+1}/{len(year_idx)} │ "
                          f"rolling_LL={rolling_ll:.4f} > {threshold:.4f}")

                    remaining_idx = year_idx[i + 1:]
                    if remaining_idx:
                        train_df_retrain = feat_df[feat_df['tourney_date'] <= trigger_date].copy()
                        test_df_remaining = feat_df.loc[remaining_idx].copy()
                        
                        new_preds = _train_and_predict(train_df_retrain, test_df_remaining, fixed_params, experiment_name=col_name)
                        if new_preds is not None:
                            current_year_preds[i + 1:] = new_preds
                            
                    rolling_buffer.clear()

        for i, pos_idx in enumerate(year_idx):
            predictions[pos_idx] = current_year_preds[i]

        print(f"\n    ✓ {test_year} complete │ Retrains: {retrain_count} │ Matches: {len(year_idx):,}")

    return pd.Series(predictions, index=feat_df.index, name=col_name)

def run_retrain_trigger_experiments(master_df: pd.DataFrame, complete_features_df: pd.DataFrame,
                                    hyperparams_by_year: dict, start_year: int, end_year: int,
                                    windows: list[int]) -> pd.DataFrame:
    """Wrapper that loops through windows and merges results into master_df."""
    feat_df = complete_features_df.copy()
    feat_df['tourney_date'] = pd.to_datetime(feat_df['tourney_date'])
    feat_df = feat_df.sort_values('tourney_date').reset_index(drop=True)

    for w in windows:
        col_name = f"Trigger_Window{w}_pb"
        pred_series = _run_single_window_experiment(feat_df, w, hyperparams_by_year, start_year, end_year, col_name)

        pred_df = feat_df[['match_id']].copy()
        pred_df[col_name] = pred_series.values
        pred_df = pred_df.dropna(subset=[col_name])

        if col_name in master_df.columns:
            master_df.drop(columns=[col_name], inplace=True)
            
        master_df = pd.merge(master_df, pred_df, on='match_id', how='left')

    return master_df