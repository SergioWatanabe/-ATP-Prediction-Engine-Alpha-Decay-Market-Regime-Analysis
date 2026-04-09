import pandas as pd
from pathlib import Path
from src.model_utils import walk_forward_validation
from src.data_utils import reduce_mem_usage

def run_training_pipeline():
    # 1. Setup Paths
    START_YEAR = 2005
    END_YEAR = 2024
    
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "Data"
    
    FEATURES_FILE = DATA_DIR / (f"features.csv")
    PREDS_FILE = DATA_DIR / "predictions.csv"
    HYPERPARAMS_FILE = DATA_DIR / "hyperparameters.csv"

    # 2. Load Data
    print(f"Loading features from {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    df = reduce_mem_usage(df)

    # 3. Run Walk-Forward Validation
    # Set N_TRIALS to 1 for a quick test, 50+ for production
    N_TRIALS = 50
    
    preds_df, params_df = walk_forward_validation(
        df,
        start_year=START_YEAR, 
        end_year=END_YEAR,
        n_trials=N_TRIALS
    )

    # 4. Save Results
    preds_df.to_csv(PREDS_FILE, index=False)
    params_df.to_csv(HYPERPARAMS_FILE, index=False)
    
    print(f"\n✅ Training Complete!")
    print(f"📂 Predictions saved to: {PREDS_FILE}")
    print(f"📂 Hyperparameters saved to: {HYPERPARAMS_FILE}")

if __name__ == "__main__":
    run_training_pipeline()