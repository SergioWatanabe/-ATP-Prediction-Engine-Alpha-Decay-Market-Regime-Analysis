import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
import os

def plot_abstention_curve(df, pred_cols, tour_name, output_path):
    """Generates the 'Money on the Table' curve."""
    plt.figure(figsize=(12, 8))
    # [Include your logic for thresholds and plotting here]
    plt.savefig(output_path)
    plt.close()

def plot_drift_correlation(drift_df, tour_name, output_path):
    """Plots PSI vs LogLoss Variance."""
    sns.regplot(data=drift_df, x='Feature_Drift_Mean_PSI', y='Log_Loss_Variance')
    plt.savefig(output_path)
    plt.close()

   

def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def get_colors_by_impact(values): return ['#d62728' if v >= 0 else '#2ca02c' for v in values]

def generate_abstention_plot(df, tour_name, output_dir):
    pred_cols = ['baseline_pb'] + sorted([c for c in df.columns if c.endswith('_pb') and c != 'baseline_pb']) if 'baseline_pb' in df.columns else [c for c in df.columns if c.endswith('_pb')]
    plot_data = []
    
    for col in pred_cols:
        y_true, y_pred = df['outcome'].values, df[col].values
        for t in np.linspace(0.0, 0.49, 100):
            mask_keep = np.abs(y_pred - 0.5) >= t
            if np.sum(mask_keep) < 50: break
            plot_data.append({'Experiment': col.replace('_pb', '').replace('Remove_', 'No '), 'Abstained_Pct': (1 - (np.sum(mask_keep) / len(df))) * 100, 'Log_Loss': log_loss(y_true[mask_keep], y_pred[mask_keep])})
            
    plot_df = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    for exp in plot_df['Experiment'].unique():
        sub = plot_df[plot_df['Experiment'] == exp]
        plt.plot(sub['Abstained_Pct'], sub['Log_Loss'], linewidth=3 if exp == 'baseline' else 1.5, color='black' if exp == 'baseline' else None, label=exp)

    plt.title(f"{tour_name} Model Robustness: Log Loss vs Abstention Rate", fontsize=16)
    plt.xlabel("Percentage Abstained (%)")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{tour_name}_Abstention_Curve.png'), dpi=300)
    plt.close()

def generate_continuous_cohort_graphs(df, tour_name, output_dir, n_bins=12):
    df['Metric_Elo'] = (df['elo_overall_A'] + df['elo_overall_B']) / 2 if 'elo_overall_A' in df.columns else (df['classic_elo_overall_A'] + df['classic_elo_overall_B']) / 2 if 'classic_elo_overall_A' in df.columns else np.nan
    pred_cols = ['baseline_pb'] + sorted([c for c in df.columns if c.endswith('_pb') and c != 'baseline_pb'])
    
    for col_name, display_name in [('Metric_Elo', 'Elo Rating')]:
        if df[col_name].isnull().all(): continue
        try: df['bin'] = pd.qcut(df[col_name], n_bins, duplicates='drop')
        except: continue
        df['bin_center'] = df['bin'].apply(lambda x: x.mid)
        
        plot_data = []
        for bin_val, group in df.groupby('bin_center'):
            if len(group) < 50: continue
            for col in pred_cols: plot_data.append({'Metric_Value': bin_val, 'Log_Loss': log_loss(group['outcome'].values, group[col].values), 'Experiment': col.replace('_pb', '').replace('baseline', 'Baseline')})
        
        plot_df = pd.DataFrame(plot_data)
        plt.figure(figsize=(14, 8))
        for exp in plot_df['Experiment'].unique():
            sub = plot_df[plot_df['Experiment'] == exp]
            plt.plot(sub['Metric_Value'], sub['Log_Loss'], linewidth=4 if exp == 'Baseline' else 2, color='black' if exp == 'Baseline' else None, linestyle='--' if exp == 'Baseline' else '-', label=exp)
        plt.title(f"{tour_name}: Log Loss vs {display_name}"); plt.legend()
        plt.savefig(os.path.join(output_dir, f'{tour_name}_{display_name}_Curve.png'), dpi=300)
        plt.close()

def generate_temporal_graphs(df, tour_name, output_dir, bin_size=100):
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    df = df.sort_values(['tourney_date', 'match_id']).reset_index(drop=True)
    pred_col = 'baseline_pb' if 'baseline_pb' in df.columns else [c for c in df.columns if c.endswith('_pb')][0]
    
    df['Batch_ID'] = df.index // bin_size
    binned_data = [{'Date': group['tourney_date'].iloc[0], 'Log_Loss': log_loss(group['outcome'], np.clip(group[pred_col], 1e-15, 1 - 1e-15))} for _, group in df.groupby('Batch_ID')]
    
    df_binned = pd.DataFrame(binned_data)
    plt.figure(figsize=(14, 7))
    plt.plot(df_binned['Date'], df_binned['Log_Loss'], marker='o', alpha=0.7)
    plt.plot(df_binned['Date'], df_binned['Log_Loss'].rolling(5, min_periods=1).mean(), color='black', linewidth=2.5, linestyle='--')
    plt.title(f"{tour_name}: Temporal Stability ({bin_size} matches)")
    plt.savefig(os.path.join(output_dir, f"{tour_name}_Temporal_Stability.png"), dpi=300)
    plt.close()

def plot_yearly_logloss_breakdown(df, tour_name, output_dir):
    df['year'] = pd.to_datetime(df['tourney_date']).dt.year
    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    plot_data = [{'Year': y, 'Experiment': c.replace('_pb', '').replace('baseline', 'Baseline'), 'LogLoss': log_loss(df[df['year']==y]['outcome'].values, np.clip(df[df['year']==y][c].values, 1e-15, 1 - 1e-15))} for y in sorted(df['year'].unique()) for c in pred_cols]
    
    plot_df = pd.DataFrame(plot_data).sort_values(['Year', 'Experiment'])
    plt.figure(figsize=(14, 8))
    sns.barplot(data=plot_df, x='Year', y='LogLoss', hue='Experiment')
    plt.ylim(plot_df['LogLoss'].min() - 0.05, plot_df['LogLoss'].max() + 0.05)
    plt.title(f"{tour_name}: Yearly Breakdown")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(os.path.join(output_dir, f'{tour_name}_Yearly_Breakdown.png'), dpi=300)
    plt.close()

def run_best_model_viz(df, tour_name, output_dir):
    y_true = df['outcome'].values
    scores = {c: log_loss(y_true, np.clip(df[c].values, 1e-15, 1-1e-15)) for c in [x for x in df.columns if x.endswith('_pb')]}
    top_3 = [m[0] for m in sorted(scores.items(), key=lambda x: x[1])[:3]]
    
    df['year'] = pd.to_datetime(df['tourney_date']).dt.year
    plot_data = [{'Year': y, 'Model': m.replace('_pb', ''), 'LogLoss': log_loss(df[df['year']==y]['outcome'].values, np.clip(df[df['year']==y][m].values, 1e-15, 1-1e-15))} for y in sorted(df['year'].unique()) for m in top_3 if not df[df['year']==y].empty]
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=pd.DataFrame(plot_data), x='Year', y='LogLoss', hue='Model')
    plt.title(f"{tour_name}: Top 3 Models Yearly")
    plt.savefig(os.path.join(output_dir, f'{tour_name}_Top3_Yearly.png'), dpi=300)
    plt.close()

def generate_all_ablation_graphs(summary_df, subgroup_df, rel_df, abs_df, tour, output_dir):
    if summary_df is not None:
        df_all = summary_df[summary_df['year'] == 'ALL_TIME'].copy()
        if not df_all.empty and 'baseline' in df_all['experiment'].values:
            base_ll = df_all[df_all['experiment'] == 'baseline'].iloc[0]['log_loss']
            df_plot = df_all[df_all['experiment'] != 'baseline'].copy()
            df_plot['ll_delta'] = df_plot['log_loss'] - base_ll
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='ll_delta', y='experiment', data=df_plot.sort_values('ll_delta', ascending=False), palette=get_colors_by_impact(df_plot['ll_delta']), ax=ax)
            ax.axvline(0, color='black'); ax.set_title(f"{tour} All-Time Impact")
            save_plot(fig, os.path.join(output_dir, f"{tour}_Main_Impact.png"))