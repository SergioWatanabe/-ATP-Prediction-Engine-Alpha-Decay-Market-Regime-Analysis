import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevents plots from opening/freezing your screen while generating
from pathlib import Path
import warnings

from src.data_utils import robust_csv_load, create_ensembles
from src.metrics_utils import process_metrics_wide, calculate_subgroup_metrics
from src.insight_utils import (
    save_wide_results, run_cohort_analysis, generate_abstention_plot, 
    generate_continuous_cohort_graphs, plot_main_impact, plot_best_vs_baseline_yearly, 
    plot_subgroups, plot_reliability_ece, plot_abstention_impact, plot_consistency, 
    analyze_signal_failure, analyze_data_drift_correlation, generate_temporal_graphs, 
    plot_yearly_logloss_breakdown, run_best_model_viz, analyze_weighted_drift, plot_hero_calibration_curve
)

warnings.filterwarnings('ignore')

def main():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "Data"
    RESULTS_DIR = BASE_DIR / "Results"
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Change if you named the file differently
    INPUT_FILE = DATA_DIR / 'ablation_results_wide.csv'
    TOUR_NAME = 'ATP'

    print(f"\n{'='*80}\n  EXECUTIVE SUMMARY & INSIGHTS PIPELINE\n{'='*80}")
    
    if not INPUT_FILE.exists():
        print(f"ERROR: File {INPUT_FILE} not found.")
        return

    # 1. Ensembles
    enriched_file = create_ensembles(INPUT_FILE, TOUR_NAME, RESULTS_DIR) or INPUT_FILE
    df_full = robust_csv_load(enriched_file)
    df_full['tourney_date'] = pd.to_datetime(df_full['tourney_date'])

    # 2. Extract DataFrames
    years = sorted(df_full['tourney_date'].dt.year.unique())
    all_summaries, all_rel_tables, all_abs_tables, all_subgroups = [], [], [], []

    print("\nProcessing Metrics...")
    for year in years:
        df_year = df_full[df_full['tourney_date'].dt.year == year].copy()
        sums, rels, abss = process_metrics_wide(df_year, year, TOUR_NAME)
        all_summaries.extend(sums); all_rel_tables.extend(rels); all_abs_tables.extend(abss)
        all_subgroups.append(calculate_subgroup_metrics(df_year, year, TOUR_NAME))

    sums, rels, abss = process_metrics_wide(df_full, 'ALL_TIME', TOUR_NAME)
    all_summaries.extend(sums); all_rel_tables.extend(rels); all_abs_tables.extend(abss)
    all_subgroups.append(calculate_subgroup_metrics(df_full, 'ALL_TIME', TOUR_NAME))

    summary_df = pd.DataFrame(all_summaries)
    subgroup_df = pd.concat(all_subgroups, ignore_index=True)
    rel_df = pd.concat(all_rel_tables, ignore_index=True) if all_rel_tables else pd.DataFrame()
    abs_df = pd.concat(all_abs_tables, ignore_index=True) if all_abs_tables else pd.DataFrame()

    # Save Tables
    save_wide_results(all_summaries, all_rel_tables, all_abs_tables, subgroup_df, TOUR_NAME, RESULTS_DIR)

    # 32. Generate ALL Visual Insights
    print("\nGenerating Executive Dashboards and Plots...")
    plot_hero_calibration_curve(enriched_file, TOUR_NAME, RESULTS_DIR) # <--- ADDED THIS
    run_cohort_analysis(df_full, TOUR_NAME, RESULTS_DIR)
    generate_abstention_plot(enriched_file, TOUR_NAME, RESULTS_DIR)
    generate_continuous_cohort_graphs(enriched_file, TOUR_NAME, RESULTS_DIR)
    plot_main_impact(summary_df, TOUR_NAME, RESULTS_DIR)
    plot_best_vs_baseline_yearly(summary_df, TOUR_NAME, RESULTS_DIR)
    plot_subgroups(subgroup_df, TOUR_NAME, RESULTS_DIR)
    plot_reliability_ece(rel_df, TOUR_NAME, RESULTS_DIR)
    plot_abstention_impact(abs_df, TOUR_NAME, RESULTS_DIR)
    plot_consistency(summary_df, TOUR_NAME, "Main_Metrics", RESULTS_DIR)
    analyze_signal_failure(enriched_file, TOUR_NAME, RESULTS_DIR)
    analyze_data_drift_correlation(enriched_file, TOUR_NAME, RESULTS_DIR)
    generate_temporal_graphs(enriched_file, TOUR_NAME, RESULTS_DIR)
    plot_yearly_logloss_breakdown(enriched_file, TOUR_NAME, RESULTS_DIR)
    run_best_model_viz(enriched_file, TOUR_NAME, RESULTS_DIR)
    analyze_weighted_drift(enriched_file, TOUR_NAME, RESULTS_DIR)

    print(f"\n{'='*80}\n  PIPELINE COMPLETE. All 36 files saved to /Results/\n{'='*80}")

if __name__ == "__main__":
    main()