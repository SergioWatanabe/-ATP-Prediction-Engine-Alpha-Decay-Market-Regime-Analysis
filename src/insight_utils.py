import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import log_loss, brier_score_loss
from pathlib import Path
from src.metrics_utils import calculate_psi, calculate_ks

def save_plot(fig, results_dir, filename):
    filepath = Path(results_dir) / filename
    fig.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)

def get_colors_by_impact(values):
    return ['#d62728' if v >= 0 else '#2ca02c' for v in values]

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

# ==============================================================================
# SAVING TABLES (Wide Results)
# ==============================================================================
def save_wide_results(summary_list, rel_tables, abs_tables, subgroup_df, tour, results_dir):
    res = Path(results_dir)
    
    summary_df = pd.DataFrame(summary_list)
    if not summary_df.empty:
        summary_df['year'] = summary_df['year'].astype(str)
        summary_df.sort_values(['year', 'experiment'], inplace=True)
        summary_df.to_csv(res / f'{tour}_STEP6_Wide_1_Main_Metrics.csv', index=False)
    
    if not subgroup_df.empty:
        subgroup_df['year'] = subgroup_df['year'].astype(str)
        subgroup_df.sort_values(['year', 'subgroup_type', 'subgroup_value', 'experiment'], inplace=True)
        subgroup_df.to_csv(res / f'{tour}_STEP6_Wide_2_Subgroups.csv', index=False)
    
    if rel_tables:
        pd.concat(rel_tables).to_csv(res / f'{tour}_STEP6_Wide_3_Reliability.csv', index=False)
    if abs_tables:
        pd.concat(abs_tables).to_csv(res / f'{tour}_STEP6_Wide_4_Abstention.csv', index=False)

# ==============================================================================
# ALL YOUR ORIGINAL GRAPHING FUNCTIONS (RESTORED EXACTLY)
# ==============================================================================

def run_cohort_analysis(df, tour_name, results_dir):
    results = []
    df['match_avg_elo'] = (df.get('elo_overall_A', df.get('classic_elo_overall_A', 0)) + df.get('elo_overall_B', df.get('classic_elo_overall_B', 0))) / 2
    df['match_avg_exp'] = (df.get('age_A', 0) + df.get('age_B', 0)) / 2

    elo_p65 = df['match_avg_elo'].quantile(0.65)
    elo_p45 = df['match_avg_elo'].quantile(0.45)
    exp_p50 = df['match_avg_exp'].quantile(0.50) if not df['match_avg_exp'].isna().all() else 0

    cohorts = {'1_High_Elo': df['match_avg_elo'] > elo_p65, '2_Low_Elo': df['match_avg_elo'] < elo_p45, '3_Mid_Elo': (df['match_avg_elo'] >= elo_p45) & (df['match_avg_elo'] <= elo_p65)}
    if exp_p50 > 0:
        cohorts['4_Veteran_Group'] = df['match_avg_exp'] < exp_p50 
        cohorts['5_Rookie_Group'] = df['match_avg_exp'] >= exp_p50

    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    total_matches = len(df)
    
    for cohort_name, mask in cohorts.items():
        subset = df[mask]
        if len(subset) < 10: continue
        for exp_col in pred_cols:
            results.append({
                'Cohort': cohort_name, 'Experiment': exp_col.replace('_pb', ''), 'Matches': len(subset),
                'Pct_of_Dataset': (len(subset)/total_matches)*100,
                'LogLoss': log_loss(subset['outcome'], subset[exp_col]),
                'Brier_Score': brier_score_loss(subset['outcome'], subset[exp_col])
            })
            
    res_df = pd.DataFrame(results).sort_values(['Cohort', 'Experiment'])
    res_df.to_csv(Path(results_dir) / f"{tour_name}_STEP6_Wide_5_Cohort_Analysis.csv", index=False)
    return res_df

def generate_abstention_plot(input_file, tour_name, results_dir):
    df = pd.read_csv(input_file)
    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    if 'baseline_pb' in pred_cols: pred_cols.remove('baseline_pb'); pred_cols = ['baseline_pb'] + sorted(pred_cols)

    plot_data = []
    plt.figure(figsize=(12, 8)); sns.set_style("whitegrid")

    for col in pred_cols:
        label = col.replace('_pb', '').replace('Remove_', 'No ')
        pts = []
        for t in np.linspace(0.0, 0.49, 100):
            mask = np.abs(df[col] - 0.5) >= t
            if np.sum(mask) < 50: break
            ll = log_loss(df['outcome'][mask], df[col][mask])
            pts.append({'Experiment': label, 'Abstained_Pct': (1 - np.sum(mask)/len(df))*100, 'Log_Loss': ll})
            
        plot_data.extend(pts)
        pdf = pd.DataFrame(pts)
        
        if col == 'baseline_pb': plt.plot(pdf['Abstained_Pct'], pdf['Log_Loss'], color='black', linewidth=3, label='Baseline', zorder=10)
        else: plt.plot(pdf['Abstained_Pct'], pdf['Log_Loss'], linewidth=1.5, alpha=0.7, label=label)

    pd.DataFrame(plot_data).to_csv(Path(results_dir) / f"{tour_name}_Abstention_Curve_Data.csv", index=False)
    plt.title(f"{tour_name} Model Robustness: Log Loss vs Abstention Rate", fontsize=16, fontweight='bold')
    plt.xlabel("Percentage of Matches Abstained (%)"); plt.ylabel("Log Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(plt.gcf(), results_dir, f"{tour_name}_Abstention_Curve_LogLoss.png")

def generate_continuous_cohort_graphs(input_file, tour_name, results_dir, n_bins=12):
    df = pd.read_csv(input_file)
    df['Metric_Elo'] = (df.get('elo_overall_A', df.get('classic_elo_overall_A', 0)) + df.get('elo_overall_B', df.get('classic_elo_overall_B', 0))) / 2
    df['Metric_Exp'] = (df.get('age_A', 0) + df.get('age_B', 0)) / 2

    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    if 'baseline_pb' in pred_cols: pred_cols.remove('baseline_pb'); pred_cols = ['baseline_pb'] + sorted(pred_cols)

    for col_name, disp_name in [('Metric_Elo', 'Elo Rating'), ('Metric_Exp', 'Experience')]:
        if df[col_name].isnull().all(): continue
        try: df['bin'] = pd.qcut(df[col_name], n_bins, duplicates='drop')
        except: continue
        df['bin_center'] = df['bin'].apply(lambda x: x.mid)
        
        plot_data = []
        for bin_val, group in df.groupby('bin_center'):
            if len(group) < 50: continue
            for col in pred_cols:
                plot_data.append({'Metric_Value': bin_val, 'Log_Loss': log_loss(group['outcome'], group[col]), 'Experiment': col.replace('_pb', '').replace('Remove_', 'No ')})
        
        plot_df = pd.DataFrame(plot_data)
        plot_df.to_csv(Path(results_dir) / f"{tour_name}_Data_{disp_name.split(' ')[0]}_vs_LogLoss.csv", index=False)
        
        plt.figure(figsize=(14, 8)); sns.set_style("whitegrid")
        for exp in plot_df['Experiment'].unique():
            sub = plot_df[plot_df['Experiment'] == exp]
            if exp == 'baseline': plt.plot(sub['Metric_Value'], sub['Log_Loss'], color='black', linewidth=4, linestyle='--', label=exp, zorder=10)
            else: plt.plot(sub['Metric_Value'], sub['Log_Loss'], linewidth=2, alpha=0.8, label=exp)

        plt.title(f"{tour_name}: Log Loss vs {disp_name}"); plt.xlabel(disp_name); plt.ylabel("Log Loss")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        save_plot(plt.gcf(), results_dir, f"{tour_name}_Graph_{disp_name.split(' ')[0]}_vs_LogLoss.png")

def plot_main_impact(summary_df, tour, results_dir):
    df = summary_df[summary_df['year'] == 'ALL_TIME'].copy()
    if df.empty: return
    baseline = df[df['experiment'] == 'baseline'].iloc[0]
    df = df[df['experiment'] != 'baseline'].copy()
    
    df['ll_delta'] = df['log_loss'] - baseline['log_loss']
    df['brier_delta'] = df['brier_score'] - baseline['brier_score']
    df.sort_values('ll_delta', ascending=False, inplace=True)
    
    for col, title, fname in [('ll_delta', 'Log Loss Impact', '1_Main_LogLoss_Impact.png'), ('brier_delta', 'Brier Score Impact', '1_Main_Brier_Impact.png')]:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=col, y='experiment', data=df, palette=get_colors_by_impact(df[col]), ax=ax)
        ax.axvline(0, color='black'); ax.set_title(f"{tour}: {title} (vs Baseline)")
        save_plot(fig, results_dir, f"{tour}_Graph_{fname}")

def plot_best_vs_baseline_yearly(summary_df, tour, results_dir):
    all_time = summary_df[summary_df['year'] == 'ALL_TIME']
    if all_time.empty: return
    best_exp = all_time.loc[all_time['log_loss'].idxmin()]['experiment']
    
    yearly = summary_df[(summary_df['year'] != 'ALL_TIME') & (summary_df['experiment'].isin(['baseline', best_exp]))].copy()
    yearly['year'] = yearly['year'].astype(int).sort_values()
    
    for metric, label in [('log_loss', 'Log Loss'), ('brier_score', 'Brier Score')]:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=yearly, x='year', y=metric, hue='experiment', marker='o', ax=ax)
        ax.set_title(f"{tour}: Baseline vs Best ({best_exp}) - {label}"); ax.set_xticks(yearly['year'].unique())
        save_plot(fig, results_dir, f"{tour}_Graph_2_Yearly_{label.replace(' ','')}.png")

def plot_subgroups(subgroup_df, tour, results_dir):
    df = subgroup_df[subgroup_df['year'] == 'ALL_TIME'].copy() if 'ALL_TIME' in subgroup_df['year'].values else subgroup_df.groupby(['experiment', 'subgroup_type', 'subgroup_value'])[['log_loss', 'brier_score']].mean().reset_index()
    for sg_type in df['subgroup_type'].unique():
        pivot = df[df['subgroup_type'] == sg_type].pivot(index=['subgroup_value'], columns='experiment', values=['log_loss', 'brier_score'])
        if 'baseline' not in pivot['log_loss'].columns: continue
        
        for metric, label in [('log_loss', 'LogLoss'), ('brier_score', 'Brier')]:
            plot_data = []
            for exp in pivot[metric].columns:
                if exp == 'baseline': continue
                temp = (pivot[metric][exp] - pivot[metric]['baseline']).reset_index()
                temp.columns = ['subgroup_value', 'delta']; temp['experiment'] = exp
                plot_data.append(temp)
                
            if not plot_data: continue
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=pd.concat(plot_data), x='subgroup_value', y='delta', hue='experiment', ax=ax)
            ax.axhline(0, color='black'); ax.set_title(f"{tour} {sg_type}: {label} Impact")
            save_plot(fig, results_dir, f"{tour}_Graph_3_Subgroup_{sg_type}_{label}.png")

def plot_reliability_ece(rel_df, tour, results_dir):
    df = rel_df[rel_df['year'] == 'ALL_TIME'].copy() if 'ALL_TIME' in rel_df['year'].values else rel_df.copy()
    ece_data = [{'experiment': exp, 'ECE': np.sum((sub['count']/sub['count'].sum()) * sub['abs_calibration_error'])} 
                for exp, sub in df.groupby('experiment') if sub['count'].sum() > 0]
    
    fig, ax = plt.subplots(figsize=(10, 6)); ece_df = pd.DataFrame(ece_data).sort_values('ECE')
    sns.barplot(data=ece_df, x='ECE', y='experiment', palette=['black' if x=='baseline' else 'gray' for x in ece_df['experiment']], ax=ax)
    ax.set_title(f"{tour}: Expected Calibration Error (Lower is Better)")
    save_plot(fig, results_dir, f"{tour}_Graph_4_Reliability_ECE.png")

def plot_abstention_impact(abs_df, tour, results_dir):
    df = abs_df[abs_df['year'] == 'ALL_TIME'].copy() if 'ALL_TIME' in abs_df['year'].values else abs_df.copy()
    for policy in df['policy'].unique():
        sub = df[df['policy'] == policy]
        base = sub[sub['experiment'] == 'baseline']
        if base.empty: continue
        plot_df = sub[sub['experiment'] != 'baseline'].copy()
        plot_df['ll_delta'] = plot_df['log_loss'] - base['log_loss'].values[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=plot_df, x='ll_delta', y='experiment', palette=get_colors_by_impact(plot_df['ll_delta']), ax=ax)
        ax.axvline(0, color='black'); ax.set_title(f"{tour} Policy: {policy} - Log Loss Impact")
        save_plot(fig, results_dir, f"{tour}_Graph_5_Abstention_{policy.replace(' ','')}_LogLoss.png")

def plot_consistency(summary_df, tour, prefix, results_dir):
    df = summary_df[summary_df['year'] != 'ALL_TIME'].copy()
    if df.empty: return
    counts = df.loc[df.groupby('year')['log_loss'].idxmin()]['experiment'].value_counts().reset_index()
    counts.columns = ['experiment', 'years_best']
    for exp in df['experiment'].unique():
        if exp not in counts['experiment'].values: counts = pd.concat([counts, pd.DataFrame({'experiment': [exp], 'years_best': [0]})])
            
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=counts.sort_values('years_best', ascending=False), x='years_best', y='experiment', palette='viridis', ax=ax)
    ax.set_title(f"{tour} Consistency: Years as Top Performer ({prefix})")
    save_plot(fig, results_dir, f"{tour}_Graph_6_Consistency_{prefix}.png")

def analyze_signal_failure(input_file, tour_name, results_dir):
    df = pd.read_csv(input_file); df['tourney_date'] = pd.to_datetime(df['tourney_date']); df['year'] = df['tourney_date'].dt.year
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    elo_cols = [c for c in df.columns if 'elo' in c.lower() and 'overall' in c.lower()]
    
    pts = []
    for year in sorted(df['year'].unique()):
        df_y = df[df['year'] == year]; df_early = df_y[df_y['tourney_date'].dt.month <= 2]
        if df_early.empty: continue
        ll_var = np.var(-(df_y['outcome']*np.log(np.clip(df_y['baseline_pb'], 1e-15, 1-1e-15)) + (1-df_y['outcome'])*np.log(1-np.clip(df_y['baseline_pb'], 1e-15, 1-1e-15))))
        pts.append({'Year': year, 'Log_Loss_Variance': ll_var, 'Avg_Age_Early_Season': np.mean(df_early[age_cols].values), 'Avg_Elo_Early_Season': np.mean(df_early[elo_cols].values)})
    
    if len(pts) < 3: return
    stats_df = pd.DataFrame(pts)
    stats_df.to_csv(Path(results_dir) / f'{tour_name}_Signal_Failure_Correlation.csv', index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (col, label) in enumerate([('Avg_Age_Early_Season', 'Average Age (Jan-Feb)'), ('Avg_Elo_Early_Season', 'Average Elo (Jan-Feb)')]):
        r, p = stats.pearsonr(stats_df[col], stats_df['Log_Loss_Variance'])
        sns.regplot(data=stats_df, x=col, y='Log_Loss_Variance', ax=axes[i], line_kws={'color': 'red'})
        axes[i].set_title(f"{label} vs Instability\nR²={r**2:.3f}, p={p:.3f}", color='green' if p<0.05 else 'black')
    save_plot(fig, results_dir, f'{tour_name}_Signal_Failure_Plot.png')

def analyze_data_drift_correlation(input_file, tour_name, results_dir):
    df = pd.read_csv(input_file); df['tourney_date'] = pd.to_datetime(df['tourney_date']); df['year'] = df['tourney_date'].dt.year
    feat_cols = [c for c in df.columns if c not in ['tourney_date', 'outcome', 'match_id', 'year'] and not c.endswith('_pb') and df[c].dtype in ['float64', 'int64']]
    
    yearly_metrics = []
    for year in sorted(df['year'].unique())[1:]:
        df_ref, df_cur = df[df['year'] < year], df[df['year'] == year]
        psi = [calculate_psi(df_ref[f].dropna().values, df_cur[f].dropna().values) for f in feat_cols]
        ll_var = np.var(-(df_cur['outcome']*np.log(np.clip(df_cur['baseline_pb'], 1e-15, 1-1e-15)) + (1-df_cur['outcome'])*np.log(1-np.clip(df_cur['baseline_pb'], 1e-15, 1-1e-15))))
        yearly_metrics.append({'Year': year, 'Prediction_Drift_PSI': calculate_psi(df_ref['baseline_pb'].dropna(), df_cur['baseline_pb'].dropna()), 
                               'Feature_Drift_Mean_PSI': np.mean(psi), 'Feature_Drift_Max_PSI': np.max(psi),
                               'Log_Loss_Variance': ll_var, 'Log_Loss_Mean': log_loss(df_cur['outcome'], np.clip(df_cur['baseline_pb'], 1e-15, 1-1e-15))})
        
    drift_df = pd.DataFrame(yearly_metrics)
    drift_df.to_csv(Path(results_dir) / f"{tour_name}_Yearly_Data_Drift_Metrics.csv", index=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    corr_stats = []
    for i, (target, signal) in enumerate([('Log_Loss_Variance', 'Prediction_Drift_PSI'), ('Log_Loss_Mean', 'Prediction_Drift_PSI'), 
                                          ('Log_Loss_Variance', 'Feature_Drift_Mean_PSI'), ('Log_Loss_Mean', 'Feature_Drift_Mean_PSI')]):
        ax = axes.flatten()[i]
        r, p = stats.pearsonr(drift_df[signal], drift_df[target])
        corr_stats.append({'Target': target, 'Signal': signal, 'Pearson_R': r, 'P_Value': p})
        sns.regplot(data=drift_df, x=signal, y=target, ax=ax, line_kws={'color':'red'})
        ax.set_title(f"{signal} vs {target}\nR²={r**2:.3f}, p={p:.3f}", color='green' if p<0.05 else 'black')
    
    pd.DataFrame(corr_stats).to_csv(Path(results_dir) / f"{tour_name}_Drift_Correlation_Stats.csv", index=False)
    save_plot(fig, results_dir, f'{tour_name}_Drift_Correlation_Plot.png')

def generate_temporal_graphs(input_file, tour_name, results_dir, bin_size=100):
    df = pd.read_csv(input_file); df['tourney_date'] = pd.to_datetime(df['tourney_date']); df.sort_values('tourney_date', inplace=True)
    
    monthly_data = [{'Date': p.to_timestamp(), 'Log_Loss': log_loss(g['outcome'], np.clip(g['baseline_pb'], 1e-15, 1-1e-15)), 'Brier_Score': brier_score_loss(g['outcome'], np.clip(g['baseline_pb'], 1e-15, 1-1e-15))} 
                    for p, g in df.groupby(df['tourney_date'].dt.to_period('M')) if len(g) >= 10]
    if monthly_data:
        m_df = pd.DataFrame(monthly_data)
        m_df.to_csv(Path(results_dir) / f"{tour_name}_Temporal_Monthly_Data.csv", index=False)
        for metric, title in [('Log_Loss', 'Monthly Log Loss'), ('Brier_Score', 'Monthly Brier Score')]:
            fig, ax = plt.subplots(figsize=(14, 7)); ax.plot(m_df['Date'], m_df[metric], marker='o')
            ax.plot(m_df['Date'], m_df[metric].rolling(3, min_periods=1).mean(), color='black', linestyle='--')
            ax.set_title(f"{tour_name}: {title}"); save_plot(fig, results_dir, f"{tour_name}_Temporal_Monthly_{metric.split('_')[0]}.png")
        
    df['Batch_ID'] = df.index // bin_size
    binned_data = [{'Date': g['tourney_date'].iloc[0], 'Log_Loss': log_loss(g['outcome'], np.clip(g['baseline_pb'], 1e-15, 1-1e-15)), 'Brier_Score': brier_score_loss(g['outcome'], np.clip(g['baseline_pb'], 1e-15, 1-1e-15))} 
                   for _, g in df.groupby('Batch_ID')]
    if binned_data:
        b_df = pd.DataFrame(binned_data)
        b_df.to_csv(Path(results_dir) / f"{tour_name}_Temporal_Binned_Data.csv", index=False)
        for metric, title in [('Log_Loss', f'Log Loss (Every {bin_size} Games)'), ('Brier_Score', f'Brier Score (Every {bin_size} Games)')]:
            fig, ax = plt.subplots(figsize=(14, 7)); ax.plot(b_df['Date'], b_df[metric], marker='o')
            ax.plot(b_df['Date'], b_df[metric].rolling(5, min_periods=1).mean(), color='black', linestyle='--')
            ax.set_title(f"{tour_name}: {title}"); save_plot(fig, results_dir, f"{tour_name}_Temporal_Binned_{metric.split('_')[0]}.png")

def plot_yearly_logloss_breakdown(input_file, tour_name, results_dir):
    df = pd.read_csv(input_file); df['year'] = pd.to_datetime(df['tourney_date']).dt.year
    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    
    plot_data = [{'Year': y, 'Experiment': c.replace('_pb','').replace('Remove_',''), 'LogLoss': log_loss(g['outcome'], np.clip(g[c], 1e-15, 1-1e-15))}
                 for y, g in df.groupby('year') for c in pred_cols]
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv(Path(results_dir) / f"{tour_name}_Data_Yearly_LogLoss_Breakdown.csv", index=False)
    
    plt.figure(figsize=(14, 8)); sns.barplot(data=plot_df, x='Year', y='LogLoss', hue='Experiment', palette='tab20')
    plt.ylim(plot_df['LogLoss'].min() * 0.98, plot_df['LogLoss'].max() * 1.02)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left'); plt.title(f"{tour_name}: Yearly Breakdown")
    save_plot(plt.gcf(), results_dir, f'{tour_name}_Graph_Yearly_LogLoss_Breakdown.png')

def run_best_model_viz(input_file, tour_name, results_dir):
    df = pd.read_csv(input_file); df['year'] = pd.to_datetime(df['tourney_date']).dt.year
    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    
    # 1. Best Model identification
    scores = {c: log_loss(df['outcome'], np.clip(df[c], 1e-15, 1-1e-15)) for c in pred_cols}
    top_3 = [k for k, v in sorted(scores.items(), key=lambda item: item[1])[:3]]
    best_model = top_3[0]

    # Tournament Metrics
    df['match_log_loss'] = -(df['outcome']*np.log(np.clip(df[best_model], 1e-15, 1-1e-15)) + (1-df['outcome'])*np.log(1-np.clip(df[best_model], 1e-15, 1-1e-15)))
    t_stats = df.groupby('tourney_name').agg(Matches=('match_id', 'count'), LogLoss_Mean=('match_log_loss', 'mean')).reset_index()
    t_stats[t_stats['Matches'] >= 10].sort_values('LogLoss_Mean').to_csv(Path(results_dir)/f"{tour_name}_BestModel_Tournament_Metrics.csv", index=False)
    
    # Avg PSI Plot
    feat_cols = [c for c in df.columns if c not in ['tourney_date', 'outcome', 'match_id', 'year'] and not c.endswith('_pb') and df[c].dtype in ['float64', 'int64']]
    drift_data = [{'Year': y, 'Average_PSI': np.mean([calculate_psi(df[df['year']<y][f].dropna(), df[df['year']==y][f].dropna()) for f in feat_cols])} 
                  for y in sorted(df['year'].unique())[1:]]
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=pd.DataFrame(drift_data), x='Year', y='Average_PSI', color='skyblue', ax=ax)
    ax.axhline(0.1, color='red', linestyle='--'); ax.set_title(f"{tour_name}: Average Feature Drift (PSI)")
    save_plot(fig, results_dir, f'{tour_name}_Graph_Average_PSI_Drift.png')

    # Top 3 Models Plot
    top3_data = [{'Year': y, 'Model': m.replace('_pb',''), 'LogLoss': log_loss(g['outcome'], np.clip(g[m], 1e-15, 1-1e-15))} 
                 for y, g in df.groupby('year') for m in top_3]
    plt.figure(figsize=(12, 7)); sns.barplot(data=pd.DataFrame(top3_data), x='Year', y='LogLoss', hue='Model', palette='viridis')
    plt.ylim(pd.DataFrame(top3_data)['LogLoss'].min() * 0.98, pd.DataFrame(top3_data)['LogLoss'].max() * 1.02)
    plt.title(f"{tour_name}: Top 3 Models Yearly")
    save_plot(plt.gcf(), results_dir, f'{tour_name}_Graph_Top3_Models_Yearly.png')

def analyze_weighted_drift(input_file, tour_name, results_dir):
    df = pd.read_csv(input_file); df['year'] = pd.to_datetime(df['tourney_date']).dt.year
    remove_cols = [c for c in df.columns if c.startswith('Remove_') and c.endswith('_pb')]
    groups = get_ablation_groups_dynamic(df.columns)
    
    res_prev, res_curr = [], []
    for year in sorted(df['year'].unique())[1:]:
        df_cur = df[df['year'] == year]; df_prev_yr = df[df['year'] == year-1]
        if len(df_prev_yr) < 50: continue
        
        def calc_w(df_s):
            b_ll = log_loss(df_s['outcome'], np.clip(df_s['baseline_pb'], 1e-15, 1-1e-15))
            return {c.replace('Remove_','').replace('_pb',''): max(0, log_loss(df_s['outcome'], np.clip(df_s[c], 1e-15, 1-1e-15)) - b_ll) for c in remove_cols}
        
        w_prev, w_curr = calc_w(df_prev_yr), calc_w(df_cur)
        ll_var = np.var(-(df_cur['outcome']*np.log(np.clip(df_cur['baseline_pb'], 1e-15, 1-1e-15)) + (1-df_cur['outcome'])*np.log(1-np.clip(df_cur['baseline_pb'], 1e-15, 1-1e-15))))
        
        def calc_psi(w_dict): return sum(np.mean([calculate_psi(df[df['year']<year][f].dropna(), df_cur[f].dropna()) for f in feats if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]) * w 
                                         for grp, feats in groups.items() for k, w in w_dict.items() if grp in k and feats) * 100
        
        res_prev.append({'Year': year, 'Weighted_Drift_Score': calc_psi(w_prev), 'Log_Loss_Variance': ll_var, 'Scheme': 'Previous-Year'})
        res_curr.append({'Year': year, 'Weighted_Drift_Score': calc_psi(w_curr), 'Log_Loss_Variance': ll_var, 'Scheme': 'Current-Year'})
        
    df_comb = pd.concat([pd.DataFrame(res_prev), pd.DataFrame(res_curr)])
    df_comb.to_csv(Path(results_dir) / f"{tour_name}_Weighted_Drift_BothSchemes_Data.csv", index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    stats_list = []
    for ax, (scheme, subset) in zip(axes, df_comb.groupby('Scheme')):
        r, p = stats.pearsonr(subset['Weighted_Drift_Score'], subset['Log_Loss_Variance'])
        stats_list.append({'Scheme': scheme, 'Pearson_R': r, 'P_Value': p})
        sns.regplot(data=subset, x='Weighted_Drift_Score', y='Log_Loss_Variance', ax=ax, line_kws={'color':'red'})
        ax.set_title(f"{scheme}\nR={r:.3f}, p={p:.3f}", color='green' if p<0.05 else 'black')
        
    pd.DataFrame(stats_list).to_csv(Path(results_dir) / f"{tour_name}_Weighted_Drift_BothSchemes_Stats.csv", index=False)
    save_plot(fig, results_dir, f"{tour_name}_Weighted_Drift_Comparison.png")

def plot_hero_calibration_curve(input_file, tour_name, results_dir, n_bins=10):
    """
    Generates a presentation-ready calibration curve for the best performing model.
    Perfect for the GitHub README hero image.
    """
    print(f"\n  GENERATING HERO VISUAL: Calibration Curve")
    df = pd.read_csv(input_file)
    
    # 1. Identify Best Model (Lowest All-Time Log Loss)
    pred_cols = [c for c in df.columns if c.endswith('_pb')]
    y_true = df['outcome'].values
    
    best_model = None
    best_ll = float('inf')
    
    for col in pred_cols:
        y_pred = np.clip(df[col].values, 1e-15, 1 - 1e-15)
        ll = log_loss(y_true, y_pred)
        if ll < best_ll:
            best_ll = ll
            best_model = col

    # 2. Bin the Predictions
    # We use qcut to ensure equal number of matches in each bin (Deciles)
    df['prob_bin'] = pd.qcut(df[best_model], q=n_bins, duplicates='drop')
    
    # Calculate Mean Predicted Probability and Actual Win Rate per bin
    calibration_data = []
    for _, group in df.groupby('prob_bin'):
        mean_pred = group[best_model].mean()
        actual_win_rate = group['outcome'].mean()
        count = len(group)
        calibration_data.append({
            'Mean_Predicted_Probability': mean_pred,
            'Actual_Win_Rate': actual_win_rate,
            'Matches': count
        })
        
    calib_df = pd.DataFrame(calibration_data)
    
    # 3. Calculate Calibration Error (ECE) for the subtitle
    total_matches = calib_df['Matches'].sum()
    ece = np.sum((calib_df['Matches'] / total_matches) * np.abs(calib_df['Mean_Predicted_Probability'] - calib_df['Actual_Win_Rate']))
    
    # 4. Generate the Hero Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set_style("whitegrid")
    
    # The Perfect Calibration Line (y = x)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly Calibrated (y = x)', alpha=0.7)
    
    # The Model's Calibration Line
    ax.plot(calib_df['Mean_Predicted_Probability'], calib_df['Actual_Win_Rate'], 
            marker='o', markersize=10, linewidth=3, color='#1f77b4', 
            label=f'Best Model ({best_model.replace("_pb","")})')
            
    # Add Error Bounds / Confidence Intervals (Optional visual flair)
    # Using a simple binomial standard error: sqrt(p * (1-p) / n)
    stderr = np.sqrt(calib_df['Actual_Win_Rate'] * (1 - calib_df['Actual_Win_Rate']) / calib_df['Matches'])
    ax.fill_between(calib_df['Mean_Predicted_Probability'], 
                    calib_df['Actual_Win_Rate'] - 1.96 * stderr, 
                    calib_df['Actual_Win_Rate'] + 1.96 * stderr, 
                    color='#1f77b4', alpha=0.2, label='95% Confidence Interval')

    # Formatting for maximum visual impact
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal') # Make it a perfect square
    
    ax.set_title(f"{tour_name} Production ML Pipeline: Probability Calibration", 
                 fontsize=18, fontweight='bold', pad=20)
                 
    # Add Subtitle with stats
    ax.text(0.5, 1.02, f"Log Loss: {best_ll:.4f} | Expected Calibration Error (ECE): {ece:.4f}", 
            transform=ax.transAxes, ha='center', fontsize=12, color='gray')
            
    ax.set_xlabel("Model Predicted Win Probability", fontsize=14, fontweight='bold')
    ax.set_ylabel("Actual Empirical Win Rate", fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    
    # Save the Hero Image
    save_plot(fig, results_dir, f"{tour_name}_HERO_Calibration_Curve.png")