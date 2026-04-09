import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.metrics import log_loss
from src.data_utils import get_ablation_groups_dynamic
from src.metrics_utils import calculate_psi, calculate_ks # Import from your existing metrics

def analyze_feature_drift(df, feature_cols, year):
    """Calculates PSI and KS for all features comparing history vs current year."""
    df_ref = df[df['tourney_date'].dt.year < year]
    df_cur = df[df['tourney_date'].dt.year == year]
    
    results = []
    for feat in feature_cols:
        psi = calculate_psi(df_ref[feat].dropna().values, df_cur[feat].dropna().values)
        ks = calculate_ks(df_ref[feat].dropna().values, df_cur[feat].dropna().values)
        results.append({'feature': feat, 'psi': psi, 'ks': ks})
    return pd.DataFrame(results)


def calculate_psi(expected, actual, buckets=10):
    if len(expected) < 10 or len(actual) < 10: return 0.0
    try:
        breakpoints = sorted(list(set(np.nanpercentile(expected, np.linspace(0, 100, buckets + 1)))))
        if len(breakpoints) < 2: return 0.0
        expected_percents = np.where(np.histogram(expected, breakpoints)[0] / len(expected) == 0, 0.0001, np.histogram(expected, breakpoints)[0] / len(expected))
        actual_percents = np.where(np.histogram(actual, breakpoints)[0] / len(actual) == 0, 0.0001, np.histogram(actual, breakpoints)[0] / len(actual))
        return np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    except: return 0.0

def calculate_ks(ref_data, curr_data):
    try: return stats.ks_2samp(ref_data, curr_data)[0]
    except: return 0.0

def analyze_signal_failure(df, tour_name, output_dir):
    df['tourney_date'], df['year'] = pd.to_datetime(df['tourney_date']), pd.to_datetime(df['tourney_date']).dt.year
    if 'baseline_pb' not in df.columns: return

    age_cols = [c for c in df.columns if 'age' in c.lower() and ('_a' in c.lower() or '_b' in c.lower())]
    elo_cols = [c for c in df.columns if 'elo' in c.lower() and 'overall' in c.lower()]
    if not age_cols or not elo_cols: return

    data_points = []
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year]
        df_early = df_year[df_year['tourney_date'].dt.month <= 2]
        if df_early.empty or df_year.empty: continue
        
        y_pred = np.clip(df_year['baseline_pb'].values, 1e-15, 1 - 1e-15)
        ll_variance = np.var(-(df_year['outcome'].values * np.log(y_pred) + (1 - df_year['outcome'].values) * np.log(1 - y_pred)))
        data_points.append({'Year': year, 'Log_Loss_Variance': ll_variance, 'Avg_Age': np.mean(df_early[age_cols].values.flatten()), 'Avg_Elo': np.mean(df_early[elo_cols].values.flatten())})

    if len(data_points) < 3: return
    stats_df = pd.DataFrame(data_points)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (col, label) in enumerate([('Avg_Age', 'Average Age (Jan-Feb)'), ('Avg_Elo', 'Average Elo (Jan-Feb)')]):
        r, p_value = stats.pearsonr(stats_df[col], stats_df['Log_Loss_Variance'])
        sns.regplot(data=stats_df, x=col, y='Log_Loss_Variance', ax=axes[i], scatter_kws={'s': 100}, line_kws={'color': 'red'})
        for _, row in stats_df.iterrows(): axes[i].text(row[col], row['Log_Loss_Variance'], str(int(row['Year'])))
        axes[i].set_title(f"{label}\nR²={r**2:.3f}, p={p_value:.3f}", color='green' if p_value < 0.05 else 'black')
    
    plt.savefig(os.path.join(output_dir, f'{tour_name}_Signal_Failure_Plot.png'), dpi=300)
    plt.close()

def analyze_data_drift_correlation(df, tour_name, output_dir):
    df['year'] = pd.to_datetime(df['tourney_date']).dt.year
    feature_cols = [c for c in df.columns if c not in ['tourney_date', 'outcome', 'winner_name', 'loser_name', 'surface', 'tourney_name', 'round', 'match_id', 'ablation_group'] and not c.endswith('_pb') and df[c].dtype in ['float64', 'int64', 'float32']]
    
    yearly_metrics = []
    for year in sorted(df['year'].unique())[1:]:
        df_ref, df_cur = df[df['year'] < year], df[df['year'] == year]
        if df_ref.empty or df_cur.empty: continue
        
        psi_scores = [calculate_psi(df_ref[f].dropna().values, df_cur[f].dropna().values) for f in feature_cols if len(df_ref[f].dropna()) > 0 and len(df_cur[f].dropna()) > 0]
        y_prob = np.clip(df_cur['baseline_pb'].values, 1e-15, 1 - 1e-15)
        match_lls = -(df_cur['outcome'].values * np.log(y_prob) + (1 - df_cur['outcome'].values) * np.log(1 - y_prob))
        
        yearly_metrics.append({'Year': year, 'Feature_Drift_Mean_PSI': np.mean(psi_scores) if psi_scores else 0, 'Prediction_Drift_PSI': calculate_psi(df_ref['baseline_pb'].dropna().values, df_cur['baseline_pb'].dropna().values), 'Log_Loss_Variance': np.var(match_lls), 'Log_Loss_Mean': np.mean(match_lls)})

    if len(yearly_metrics) < 3: return
    drift_df = pd.DataFrame(yearly_metrics)
    drift_df.to_csv(os.path.join(output_dir, f'{tour_name}_Yearly_Drift_Metrics.csv'), index=False)

def analyze_weighted_drift(df, tour_name, output_dir):
    df['year'] = pd.to_datetime(df['tourney_date']).dt.year
    feature_groups = get_ablation_groups_dynamic(df.columns)
    test_years = sorted(df['year'].unique())[1:]
    
    results = []
    for year in test_years:
        df_ref, df_cur = df[df['year'] < year], df[df['year'] == year]
        if df_ref.empty or df_cur.empty: continue
        
        base_ll = log_loss(df_cur['outcome'].values, np.clip(df_cur['baseline_pb'].values, 1e-15, 1 - 1e-15))
        weights = {col.replace('Remove_', '').replace('_pb', ''): max(0, log_loss(df_cur['outcome'].values, np.clip(df_cur[col].values, 1e-15, 1 - 1e-15)) - base_ll) for col in [c for c in df.columns if c.startswith('Remove_') and c.endswith('_pb')]}
        
        total_psi = sum(np.mean([calculate_psi(df_ref[f].dropna().values, df_cur[f].dropna().values) for f in feats if f in df.columns]) * weights.get(grp, 0) for grp, feats in feature_groups.items() if feats) * 100
        ll_var = np.var(-(df_cur['outcome'].values * np.log(np.clip(df_cur['baseline_pb'].values, 1e-15, 1 - 1e-15)) + (1 - df_cur['outcome'].values) * np.log(1 - np.clip(df_cur['baseline_pb'].values, 1e-15, 1 - 1e-15))))
        
        results.append({'Year': year, 'Weighted_Drift_Score': total_psi, 'Log_Loss_Variance': ll_var})

    if len(results) > 2: pd.DataFrame(results).to_csv(os.path.join(output_dir, f"{tour_name}_Weighted_Drift.csv"), index=False)