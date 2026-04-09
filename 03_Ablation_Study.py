## Ablation Study Script: The idea behind this script is to systematically remove groups of features 
# and evaluate the impact on model performance.
# This way is possible to take a look inside what is actually happening. 
# Conversely the retraining triggers work as way of minimizing the impact of concept drift, 
# by retraining the model at specific intervals and evaluating how that affects performance.

import pandas as pd
from pathlib import Path
from src.data_utils import generate_match_id
from src.model_utils import walk_forward_validation
from src.ablation_utils import define_ablation_groups, run_retrain_trigger_experiments

def main():
    # 1. Setup
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "Data"
    
    # Load Features
    print("Loading features...")
    df = pd.read_csv(DATA_DIR / "features.csv")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    # Generate unified match_id
    if 'match_id' not in df.columns:
        df['match_id'] = generate_match_id(df)
    
    # Load Hyperparams
    hp_df = pd.read_csv(DATA_DIR / "hyperparameters.csv")
    hp_dict = {int(r['year']): r.to_dict() for _, r in hp_df.iterrows()}

    # 2. Run Baseline Ablation
    groups, pers_elo = define_ablation_groups(df.columns)
    print("\nRunning Baseline...")
    
    # FIX IS HERE: Added ", _" to unpack the tuple
    master_df, _ = walk_forward_validation(
        df.drop(columns=pers_elo), 
        start_year=2005, 
        end_year=2024, 
        n_trials=0, 
        hyperparams_dict=hp_dict
    )
    master_df.rename(columns={'model_probability': 'baseline_pb'}, inplace=True)

    # 3. Run Group Removals
    for name, cols in groups.items():
        print(f"\nTesting Removal: {name}")
        
        # FIX IS HERE: Added ", _" to unpack the tuple
        res_df, _ = walk_forward_validation(
            df.drop(columns=cols), 
            start_year=2005, 
            end_year=2024, 
            n_trials=0, 
            hyperparams_dict=hp_dict
        )
        
        # Merge back into master
        master_df = master_df.merge(res_df[['match_id', 'model_probability']], on='match_id', how='left')
        master_df.rename(columns={'model_probability': f'Remove_{name}_pb'}, inplace=True)

    # 4. Run Trigger Experiments
    windows_to_test = [1, 4] # Testing 1-season and 4-season windows
    print("\nRunning Trigger Experiments...")
    master_df = run_retrain_trigger_experiments(
        master_df=master_df,
        complete_features_df=df,
        hyperparams_by_year=hp_dict,
        start_year=2005,
        end_year=2024,
        windows=windows_to_test
    )

    # 5. Save Results
    output_file = DATA_DIR / "ablation_results_wide.csv"
    master_df.to_csv(output_file, index=False)
    print(f"\n✅ Ablation Study Complete. Saved to: {output_file}")

if __name__ == "__main__":
    main()