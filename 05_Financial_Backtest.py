from src.financial_utils import (
    load_js_data, download_all_odds, merge_datasets, assign_all_odds, plot_2d_panels, plot_3d_surface, plot_scenario_all_strategies, run_grid
)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    base_dir    = Path(__file__).resolve().parent
    data_dir    = base_dir / "Data"
    results_dir = base_dir / "Financial_Backtest_Results"
    results_dir.mkdir(exist_ok=True)
    

    input_file = data_dir / "ablation_results_wide.csv"
    

    js_df    = load_js_data(input_file)
    odds_df  = download_all_odds(YEARS)
    merged   = merge_datasets(js_df, odds_df, DATE_WINDOW)
    final_df = assign_all_odds(merged)
    final_df["year"] = final_df["Date"].dt.year

    scenarios = [
        ("Pinnacle", "odds_A_pin",  "odds_B_pin"),
        ("Average",  "odds_A_avg",  "odds_B_avg"),
        ("Best",     "odds_A_best", "odds_B_best"),
    ]

    for name, col_a, col_b in scenarios:
        df_scenario = final_df.dropna(subset=[col_a, col_b]).copy()
        df_scenario["odds_A"] = df_scenario[col_a]
        df_scenario["odds_B"] = df_scenario[col_b]

        print(f"\n\n{'*'*65}")
        print(f"{'*'*65}")

        pnl_mat, bets_mat = run_grid(df_scenario, EDGES, THRESHOLDS)

        plot_2d_panels(EDGES, THRESHOLDS, pnl_mat, bets_mat, results_dir, name)
        plot_3d_surface(EDGES, THRESHOLDS, pnl_mat, results_dir, name)

        # One clean file per scenario: every edge x threshold combination
        plot_scenario_all_strategies(
            df_scenario, EDGES, THRESHOLDS, YEARS, results_dir, name
        )

    plt.show()

YEARS       = list(range(2005, 2024))
DATE_WINDOW = 10
EDGES       = [.14,.17,.20]
THRESHOLDS  = np.arange(.1, 0.4, 0.1)

if __name__ == "__main__":
    main()