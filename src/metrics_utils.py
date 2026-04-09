import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
import pandas as pd


def match_logloss(y_true, y_pred):
    """Per-match log-loss (scalar)."""
    eps = 1e-15
    p = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI)."""
    breakpoints = np.nanpercentile(expected, np.arange(0, buckets + 1) / buckets * 100)
    breakpoints = sorted(list(set(breakpoints)))
    if len(breakpoints) < 2: return 0.0
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    return np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))

def calculate_ks(ref_data, curr_data):
    """Calculate KS Statistic."""
    try:
        stat, _ = stats.ks_2samp(ref_data, curr_data)
        return stat
    except:
        return 0.0
    
class CalibrationAnalyzer:
    """Comprehensive calibration diagnostics."""
    def __init__(self, n_bins=10000):
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0, 1, n_bins + 1)
        
    def reliability_table(self, y_true, y_pred, model_name='Model'):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        bin_indices = np.digitize(y_pred, self.bin_edges[1:-1])
        results = []
        for i in range(self.n_bins):
            mask = bin_indices == i
            count = np.sum(mask)
            if count > 0:
                mean_pred = np.mean(y_pred[mask])
                empirical_rate = np.mean(y_true[mask])
                results.append({
                    'experiment': model_name, 'bin': f'[{self.bin_edges[i]:.2f}, {self.bin_edges[i+1]:.2f})',
                    'count': count, 'mean_predicted': mean_pred, 'empirical_rate': empirical_rate,
                    'abs_calibration_error': np.abs(empirical_rate - mean_pred)
                })
        return pd.DataFrame(results)
    
    def calibration_slope_intercept(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.clip(np.array(y_pred), 1e-15, 1 - 1e-15)
        logit_pred = np.log(y_pred / (1 - y_pred))
        try:
            lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit_pred.reshape(-1, 1), y_true)
            return {'slope': lr.coef_[0][0], 'intercept': lr.intercept_[0]}
        except:
            return {'slope': np.nan, 'intercept': np.nan}
    
    def brier_decomposition(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        base_rate = np.mean(y_true)
        bin_indices = np.digitize(y_pred, self.bin_edges[1:-1])
        rel, res = 0.0, 0.0
        for i in range(self.n_bins):
            mask = bin_indices == i
            count = np.sum(mask)
            if count > 0:
                rel += count * (np.mean(y_pred[mask]) - np.mean(y_true[mask])) ** 2
                res += count * (np.mean(y_true[mask]) - base_rate) ** 2
        return {'reliability': rel / len(y_true), 'resolution': res / len(y_true), 
                'uncertainty': base_rate * (1 - base_rate), 'brier_score': brier_score_loss(y_true, y_pred)}
    
    def mid_probability_analysis(self, y_true, y_pred, lower=0.4, upper=0.6):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = (y_pred >= lower) & (y_pred <= upper)
        count = np.sum(mask)
        if count == 0: return {'abs_error': np.nan}
        return {'abs_error': np.abs(np.mean(y_true[mask]) - np.mean(y_pred[mask]))}

    def abstention_analysis(self, y_true, y_pred, model_name='Model'):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        results = []
        for l, u in [(0.48, 0.52), (0.45, 0.55), (0.40, 0.60)]:
            mask = (y_pred < l) | (y_pred > u)
            if np.sum(mask) > 0:
                results.append({
                    'experiment': model_name, 'policy': f'Abstain [{l}, {u}]',
                    'coverage': np.sum(mask)/len(y_true), 'log_loss': log_loss(y_true[mask], y_pred[mask]),
                    'brier_score': brier_score_loss(y_true[mask], y_pred[mask])
                })
        return pd.DataFrame(results)

def calculate_statistical_tests(y_true, pred_model, pred_elo):
    p_mod = np.clip(pred_model, 1e-15, 1 - 1e-15)
    p_elo = np.clip(pred_elo, 1e-15, 1 - 1e-15)
    
    # 1. Delta for mean difference (Model vs Elo)
    delta_elo = -(y_true * np.log(p_mod) + (1 - y_true) * np.log(1 - p_mod)) - -(y_true * np.log(p_elo) + (1 - y_true) * np.log(1 - p_elo))
    
    # 2. Delta for absolute p-value (Model vs 0.5 baseline)
    delta_abs = -(y_true * np.log(p_mod) + (1 - y_true) * np.log(1 - p_mod)) - (-np.log(0.5))
    
    try:
        # P-value is calculated against the 0.5 baseline (delta_abs)
        # Mean difference is calculated against Elo (delta_elo)
        p_val = 1.0 if np.allclose(delta_abs, 0) else stats.wilcoxon(delta_abs, alternative='two-sided')[1]
        return p_val, np.mean(delta_elo)
    except:
        return 1.0, np.mean(delta_elo)

def process_metrics_wide(df, year_label, tour):
    y_true = df['outcome'].values
    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    y_elo = df['classic_expected_win_prob_A'].values if 'classic_expected_win_prob_A' in df.columns else np.full_like(y_true, 0.5)
    
    calibrator = CalibrationAnalyzer()
    summary_rows, rel_tables, abs_tables = [], [], []
    
    for exp_col in pred_cols:
        exp_name = exp_col.replace('_pb', '')
        y_ml = df[exp_col].values
        
        p_val, mean_diff = calculate_statistical_tests(y_true, y_ml, y_elo)
        slope = calibrator.calibration_slope_intercept(y_true, y_ml)
        decomp = calibrator.brier_decomposition(y_true, y_ml)
        
        r_table = calibrator.reliability_table(y_true, y_ml, exp_name)
        r_table['year'] = year_label
        rel_tables.append(r_table)
        
        a_table = calibrator.abstention_analysis(y_true, y_ml, exp_name)
        a_table['year'] = year_label
        abs_tables.append(a_table)
        
        summary_rows.append({
            'year': year_label, 'experiment': exp_name, 'n_matches': len(df),
            'accuracy': accuracy_score(y_true, (y_ml > 0.5).astype(int)),
            'log_loss': log_loss(y_true, y_ml), 'brier_score': brier_score_loss(y_true, y_ml),
            'diff_vs_elo_log_loss': mean_diff, 'p_value_model': p_val,
            'calib_slope': slope['slope'], 'reliability': decomp['reliability'],
            'mid_prob_error': calibrator.mid_probability_analysis(y_true, y_ml)['abs_error']
        })
        
    return summary_rows, rel_tables, abs_tables

def calculate_subgroup_metrics(df, year_label, tour):
    y_true = df['outcome'].values
    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    y_elo = df['classic_expected_win_prob_A'].values if 'classic_expected_win_prob_A' in df.columns else np.full_like(y_true, 0.5)

    df = df.copy()
    elo_conf = np.where(y_elo > 0.5, y_elo, 1 - y_elo)
    df['Match_Tightness'] = np.where(elo_conf < 0.65, 'Tight (<65%)', 'Clear Fav (>65%)')
    
    results = []
    for group_type, col_name in {'Surface': 'surface', 'Level': 'tourney_level_weight', 'Round': 'round', 'Tightness': 'Match_Tightness'}.items():
        if col_name not in df.columns: continue
        for val in df[col_name].dropna().unique():
            mask = df[col_name] == val
            subset = df[mask]
            if len(subset) < 5: continue
            
            y_sub = subset['outcome'].values
            elo_sub = y_elo[mask]
            elo_ll_sub = log_loss(y_sub, elo_sub)
            elo_brier_sub = brier_score_loss(y_sub, elo_sub)
            
            for exp_col in pred_cols:
                y_ml_sub = subset[exp_col].values
                p_val, mean_diff = calculate_statistical_tests(y_sub, y_ml_sub, elo_sub)
                
                # FIX: Added back brier_score, p_value, and elo references
                results.append({
                    'year': year_label, 
                    'experiment': exp_col.replace('_pb', ''),
                    'subgroup_type': group_type, 
                    'subgroup_value': val, 
                    'n_matches': len(subset),
                    'log_loss': log_loss(y_sub, y_ml_sub), 
                    'diff_vs_elo_ll': mean_diff,
                    'p_value': p_val,
                    'brier_score': brier_score_loss(y_sub, y_ml_sub),
                    'elo_log_loss': elo_ll_sub,
                    'elo_brier_score': elo_brier_sub
                })
    return pd.DataFrame(results)