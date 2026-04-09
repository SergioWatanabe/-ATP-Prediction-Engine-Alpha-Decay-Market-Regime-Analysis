from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import lightgbm as lgb
import json
from datetime import datetime
from thefuzz import process # For fuzzy string matching
import uvicorn
import os
import gradio as gr

# ==========================================
# 1. LOAD ARTIFACTS INTO RAM AT STARTUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_path(filename):
    return os.path.join(BASE_DIR, filename)
print("Loading Engine...")


model = lgb.Booster(model_file=get_path('lgbm_model.txt'))

with open(get_path('deployment_feature_order.json'), 'r') as f:
    FEATURE_ORDER = json.load(f)['feature_order']

with open(get_path('deployment_mappings.json'), 'r') as f:
    MAPPINGS = json.load(f)

with open(get_path('latest_stats.json'), 'r') as f:
    LATEST_STATS = json.load(f)

with open(get_path('peak_stats.json'), 'r') as f:
    PEAK_STATS = json.load(f)

with open(get_path('tournaments.json'), 'r') as f:
    TOURNAMENTS = json.load(f)

PLAYER_NAMES = list(LATEST_STATS.keys())
TOURNAMENT_NAMES = list(TOURNAMENTS.keys())


# ==========================================
# 2. REQUEST/RESPONSE SCHEMAS (PYDANTIC)
# ==========================================
class SimulationRequest(BaseModel):
    player_a: str
    player_b: str
    tournament: str

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def fuzzy_match(query: str, choices: list, threshold: int = 80):
    """Matches messy user inputs to perfect database keys."""
    match, score = process.extractOne(query, choices)
    if score < threshold:
        raise HTTPException(status_code=404, detail=f"Could not find a confident match for '{query}'. Closest was '{match}' ({score}%).")
    return match

def calculate_rest_days(last_match_date_str: str):
    """Calculates days since last match for 'latest' simulation."""
    if not last_match_date_str:
        return 14
    last_match = datetime.strptime(last_match_date_str, "%Y-%m-%d")
    return (datetime.now() - last_match).days

def build_model_features(pA, pB, pA_id, pB_id, tourney_info, t_id, is_peak=False):
    """Transforms two JSON player profiles into the exact 1D Array LightGBM expects."""
    
    # 1. Surface and Context
    surface = tourney_info['surface']
    surface_code = MAPPINGS['surface_codes'].get(surface, 0)
    level_weight = MAPPINGS['tourney_level_weights'].get(tourney_info['tourney_level'], 0)
    
    # 2. Rest Days (Choice 4B applied here)
    if is_peak:
        rest_A, rest_B = 14, 14 # Fully rested for fantasy matches
    else:
        rest_A = calculate_rest_days(pA['last_match_date'])
        rest_B = calculate_rest_days(pB['last_match_date'])

    # Extract the correct surface Elo
    elo_surf_A = pA['surface_elos'].get(surface, 1500)
    elo_surf_B = pB['surface_elos'].get(surface, 1500)

    # 3. Build the Raw Dictionary (Mirroring your FeatureCalculator logic exactly)
    raw_dict = {
        'player_A_id': pA_id,
        'player_B_id': pB_id,
        'tourney_id': t_id,
        'surface_code': surface_code,
        'round_code': 6, # Default to Final (6) for high-stakes simulations
        'tourney_level_weight': level_weight,
        
        # Elo Features
        'classic_elo_adjusted_A': elo_surf_A, # Ignoring transition penalty for simulations
        'classic_elo_adjusted_B': elo_surf_B,
        'classic_elo_overall_A': pA['classic_elo_overall'],
        'classic_elo_overall_B': pB['classic_elo_overall'],
        'classic_elo_diff_overall': pA['classic_elo_overall'] - pB['classic_elo_overall'],
        'classic_elo_surface_A': elo_surf_A,
        'classic_elo_surface_B': elo_surf_B,
        'classic_elo_diff_surface': elo_surf_A - elo_surf_B,
        
        # Form / Momentum
        'win_pct_l5_A': pA['form']['win_pct_l5'],
        'win_pct_l5_B': pB['form']['win_pct_l5'],
        'form_diff_l5': pA['form']['win_pct_l5'] - pB['form']['win_pct_l5'],
        'win_pct_l10_A': pA['form']['win_pct_l10'],
        'win_pct_l10_B': pB['form']['win_pct_l10'],
        'form_diff_l10': pA['form']['win_pct_l10'] - pB['form']['win_pct_l10'],
        'momentum_A': pA['form']['momentum_score'],
        'momentum_B': pB['form']['momentum_score'],
        'momentum_diff': pA['form']['momentum_score'] - pB['form']['momentum_score'],
        
        # Rest
        'rest_diff': rest_A - rest_B,
        'has_long_layoff_A': 1 if rest_A > 60 else 0,
        'has_long_layoff_B': 1 if rest_B > 60 else 0,
        
        # Physicals
        'age_A': pA['age'],
        'age_B': pB['age'],
        'age_diff': pA['age'] - pB['age'],
        'height_A': pA['height'],
        'height_B': pB['height'],
        'height_diff': pA['height'] - pB['height'],
        'hand_A': pA['hand'],
        'hand_B': pB['hand'],
        'hand_mismatch': 1 if pA['hand'] != pB['hand'] else 0,
        
        # Serve Stats
        'ace_pct_A': pA['serve_stats']['ace_pct'],
        'ace_pct_B': pB['serve_stats']['ace_pct'],
        'ace_pct_diff': pA['serve_stats']['ace_pct'] - pB['serve_stats']['ace_pct'],
        '1st_won_pct_A': pA['serve_stats']['first_won_pct'],
        '1st_won_pct_B': pB['serve_stats']['first_won_pct'],
        '1st_won_pct_diff': pA['serve_stats']['first_won_pct'] - pB['serve_stats']['first_won_pct']
    }
    
    # 4. Strict Column Ordering (Safeguard against ML Feature errors)
    ordered_row = {}
    for col in FEATURE_ORDER:
        ordered_row[col] = raw_dict.get(col, 0) # Fill missing with 0 safely
        
    return pd.DataFrame([ordered_row])

# ==========================================
# 4. ENDPOINTS
# ==========================================


def run_simulation(req: SimulationRequest, db: dict, is_peak: bool):
    # 1. Validate & Fuzzy Match Strings
    name_a = fuzzy_match(req.player_a, PLAYER_NAMES)
    name_b = fuzzy_match(req.player_b, PLAYER_NAMES)
    t_name = fuzzy_match(req.tournament, TOURNAMENT_NAMES)
    
    # 2. Extract Data
    pA_data = db[name_a]
    pB_data = db[name_b]
    t_data = TOURNAMENTS[t_name]
    
    # IDs for categorical features
    id_A = MAPPINGS['player_encoder'].get(name_a, -1)
    id_B = MAPPINGS['player_encoder'].get(name_b, -1)
    id_T = MAPPINGS['tourney_encoder'].get(t_name, -1)
    
    # 3. Format Array & Predict
    X_inference = build_model_features(pA_data, pB_data, id_A, id_B, t_data, id_T, is_peak)
    prob_A = float(model.predict(X_inference)[0])
    
    # 4. Tale of the Tape Response (Choice 3B)
    return {
        "matchup": f"{name_a} vs {name_b}",
        "scenario": "Peak Historical Simulation" if is_peak else "Latest Form Simulation",
        "context": {
            "tournament": t_name,
            "surface": t_data['surface']
        },
        "prediction": {
            "predicted_winner": name_a if prob_A > 0.5 else name_b,
            f"win_probability_{name_a.split()[-1]}": round(prob_A * 100, 1),
            f"win_probability_{name_b.split()[-1]}": round((1 - prob_A) * 100, 1)
        },
        "tale_of_the_tape": {
            name_a: {
                "snapshot_date": pA_data['last_match_date'],
                "overall_elo": round(pA_data['classic_elo_overall']),
                "surface_elo": round(pA_data['surface_elos'].get(t_data['surface'], 1500)),
                "age_at_time": round(pA_data['age'], 1),
                "recent_win_pct": round(pA_data['form']['win_pct_l10'] * 100, 1)
            },
            name_b: {
                "snapshot_date": pB_data['last_match_date'],
                "overall_elo": round(pB_data['classic_elo_overall']),
                "surface_elo": round(pB_data['surface_elos'].get(t_data['surface'], 1500)),
                "age_at_time": round(pB_data['age'], 1),
                "recent_win_pct": round(pB_data['form']['win_pct_l10'] * 100, 1)
            }
        }
    }



# Mount the UI so it shows up on the main page
def ui_predict(player_a, player_b, tournament, simulation_type):
    # Connect the UI to your existing ML logic
    req = SimulationRequest(player_a=player_a, player_b=player_b, tournament=tournament)
    is_peak = (simulation_type == "Peak Historical")
    db = PEAK_STATS if is_peak else LATEST_STATS
    
    try:
        # Run your model
        result = run_simulation(req, db, is_peak)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

# Build the visual interface
demo = gr.Interface(
    fn=ui_predict,
    inputs=[
        # Use Dropdown with sorted lists for easier navigation
        gr.Dropdown(
            choices=sorted(PLAYER_NAMES), 
            label="Player A", 
            info="Start typing or scroll to select"
        ),
        gr.Dropdown(
            choices=sorted(PLAYER_NAMES), 
            label="Player B", 
            info="Start typing or scroll to select"
        ),
        gr.Dropdown(
            choices=sorted(TOURNAMENT_NAMES), 
            label="Tournament", 
            info="Select the venue/surface"
        ),
        gr.Radio(
            ["Latest Form", "Peak Historical"], 
            value="Latest Form", 
            label="Simulation Type"
        )
    ],
    outputs=gr.Code(language="json", label="Match Prediction & Tale of the Tape"),
    title="🎾 ATP Tennis Simulator",
    description="Select two players and a tournament. The LightGBM Engine will calculate win probabilities based on surface elo, recent form, and physical stats."
)


# Safely mount the UI to a NEW variable, preserving your original 'app'
app = FastAPI(title="Tennis Simulator ML API", version="1.0.0")
app = gr.mount_gradio_app(app, demo, path="/")

@app.post("/simulate/peak")
def simulate_peak_matchup(req: SimulationRequest):
    return run_simulation(req, PEAK_STATS, is_peak=True)

@app.post("/simulate/latest")
def simulate_latest_matchup(req: SimulationRequest):
    return run_simulation(req, LATEST_STATS, is_peak=False)

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "endpoints": ["/docs", "/simulate/peak", "/simulate/latest"]
    }
if __name__ == "__main__":
    import os
    
    # Check if necessary artifacts exist before booting
    required_files = [
        'lgbm_model.txt', 'latest_stats.json', 'peak_stats.json', 
        'tournaments.json', 'deployment_feature_order.json', 'deployment_mappings.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(get_path(f))]
    
    if missing_files:
        print(f"❌ ERROR: Missing required artifacts: {', '.join(missing_files)}")
    else:
        print("--- Tennis Simulator Engine v1.0 ---")
        print("All artifacts loaded. Starting web interface...")
        
        # Run the NEW app_with_ui variable
        uvicorn.run(app, host="0.0.0.0", port=7860, reload=False, proxy_headers=True, forwarded_allow_ips="*")
