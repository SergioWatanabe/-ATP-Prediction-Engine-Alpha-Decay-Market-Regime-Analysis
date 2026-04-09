from pathlib import Path
from src.data_utils import load_and_prepare_data
from src.feature_eng import FeatureCalculator

def main():
    # 1. Setup Paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "Data"
    API_DIR = BASE_DIR / "api"
    
    DATA_DIR.mkdir(exist_ok=True)
    API_DIR.mkdir(exist_ok=True)

    # 2. Configuration
    TOUR = 'ATP'
    START_YEAR = 2000
    END_YEAR = 2024
    FEATURES_FILE = DATA_DIR / f"features.csv"
    MAPPINGS_FILE = API_DIR / "deployment_mappings.json"

    print("\n" + "="*80)
    print(f"  TENNIS PREDICTION PIPELINE: {TOUR} {START_YEAR}-{END_YEAR}")
    print("="*80)

    # 3. Load and Clean Data
    raw_match_data = load_and_prepare_data(
        tour=TOUR,
        start_year=START_YEAR,
        end_year=END_YEAR
    )

    # 4. Initialize Feature Engine
    calculator = FeatureCalculator(raw_match_data)
    
    # 5. Export Mappings for API
    calculator.export_mappings(str(MAPPINGS_FILE))

    # 6. Calculate Features (Chronological Loop)
    complete_features_df = calculator.calculate_all_features()
    
    # 7. Save Results
    complete_features_df.to_csv(FEATURES_FILE, index=False)
    
    # 8. Export Snapshots for Simulator
    # Note: Passing the directory path as expected by your original function
    calculator.export_simulator_snapshots(export_dir="") 

    print(f"\n✅ SUCCESS: Features saved to {FEATURES_FILE}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()