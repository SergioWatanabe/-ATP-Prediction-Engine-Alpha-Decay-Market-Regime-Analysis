import lightgbm as lgb
import optuna 
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import os

class ModelTrainer:
    """Unified Trainer for Standard (Optuna), Ablation, and Trigger experiments."""
    
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.best_params = None
        self.model = None
        self.feature_cols = None
    
    def train(self, train_df, test_df=None, fixed_params=None, experiment_name="model"):
        """
        Train model, return params, and generate predictions.
        If fixed_params is provided, skips Optuna.
        """
        # Identify feature columns (exclude metadata)
        metadata_cols = ['tourney_date', 'outcome', 'winner_name', 'loser_name', 
                        'surface', 'tourney_name', 'round', 'match_id', 'model_probability', 'baseline_pb']
        
        # Ensure we only keep numeric features and ignore metadata
        self.feature_cols = [col for col in train_df.columns if col not in metadata_cols and train_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        X_train = train_df[self.feature_cols]
        y_train = train_df['outcome']
        
        if fixed_params is not None:
            # MODE 1: Fixed Parameters (Ablation / Preloaded)
            self.best_params = fixed_params.copy()
            # Ensure required LightGBM defaults exist if missing from file
            defaults = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'verbosity': -1, 'boosting_type': 'gbdt', 'random_state': 42
            }
            for k, v in defaults.items():
                if k not in self.best_params:
                    self.best_params[k] = v
        else:
            # MODE 2: Hyperparameter Search (Optuna)
            cv_folds = self._create_cv_folds(train_df)
            self.best_params = self._optimize_hyperparameters(train_df, cv_folds)
        
        # --- FINAL TRAINING ---
        # Ensure integer params are actually integers (crucial when loading from CSV or Optuna)
        int_params = ['num_leaves', 'max_depth', 'n_estimators', 'min_child_samples']
        for p in int_params:
            if p in self.best_params:
                self.best_params[p] = int(self.best_params[p])

        train_data_full = lgb.Dataset(X_train, label=y_train)
        
        self.model = lgb.train(
            self.best_params,
            train_data_full,
            num_boost_round=self.best_params.get('n_estimators', 1000),
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Optional: Save model metadata for GitHub tracking
        os.makedirs("models", exist_ok=True)
        self.model.save_model(f"models/{experiment_name}.txt")
        
        results = {
            'model': self.model,
            'params': self.best_params,
            'feature_cols': self.feature_cols
        }
        
        # --- PREDICTION ---
        if test_df is not None:
            X_test = test_df[self.feature_cols]
            y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
            results['predictions'] = y_pred_proba
            
        return results

    def _create_cv_folds(self, train_df):
        """Create time-series cross-validation folds."""
        n_cv_folds = 3
        train_dates = train_df['tourney_date'].sort_values()
        fold_size = len(train_dates) // (n_cv_folds + 1)
        
        cv_folds = []
        for fold_idx in range(n_cv_folds):
            fold_train_end_idx = (fold_idx + 1) * fold_size
            fold_val_start_idx = fold_train_end_idx
            fold_val_end_idx = fold_train_end_idx + fold_size
            
            if fold_val_end_idx > len(train_dates):
                break
            
            fold_train_end_date = train_dates.iloc[fold_train_end_idx]
            fold_val_end_date = train_dates.iloc[min(fold_val_end_idx, len(train_dates) - 1)]
            
            fold_train = train_df[train_df['tourney_date'] < fold_train_end_date]
            fold_val = train_df[
                (train_df['tourney_date'] >= fold_train_end_date) &
                (train_df['tourney_date'] < fold_val_end_date)
            ]
            
            if len(fold_train) > 100 and len(fold_val) > 50:
                cv_folds.append((fold_train, fold_val))
        return cv_folds
    
    def _optimize_hyperparameters(self, train_df, cv_folds):
        """Optimize hyperparameters using Optuna (YOUR FULL ORIGINAL PARAM SPACE)."""
        optuna.logging.set_verbosity(optuna.logging.WARNING) # Keeps console clean
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'random_state': 42,
                'num_leaves': trial.suggest_int('num_leaves', 20, 220),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            cv_scores = []
            for fold_train, fold_val in cv_folds:
                X_fold_train = fold_train[self.feature_cols]
                y_fold_train = fold_train['outcome']
                X_fold_val = fold_val[self.feature_cols]
                y_fold_val = fold_val['outcome']
                
                fold_train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
                fold_val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=fold_train_data)
                
                model = lgb.train(
                    params,
                    fold_train_data,
                    valid_sets=[fold_val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
                cv_scores.append(log_loss(y_fold_val, y_pred))
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False, n_jobs = -1)
        
        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42
        })
        return best_params

# ==============================================================================
#                      WALK-FORWARD VALIDATION ORCHESTRATOR
# ==============================================================================

def walk_forward_validation(df, start_year, end_year, n_trials=50, hyperparams_dict=None, window_years=None):
    """
    Unified Walk-Forward Orchestrator. 
    Handles both standard training (with Optuna) and Ablation (fixed params).
    """
    all_preds = []
    all_params = []
    
    for year in range(start_year, end_year + 1):
        test_df = df[df['tourney_date'].dt.year == year].copy()
        
        # Apply window logic if specified
        train_mask = df['tourney_date'].dt.year < year
        if window_years is not None:
            train_mask &= (df['tourney_date'].dt.year >= (year - window_years))
        
        train_df = df[train_mask].copy()
        
        if train_df.empty or test_df.empty: 
            continue
        
        # Fetch fixed params if we are doing ablation
        fixed_p = None
        if hyperparams_dict and year in hyperparams_dict:
            fixed_p = hyperparams_dict[year]
            
        trainer = ModelTrainer(n_trials=n_trials)
        res = trainer.train(train_df, test_df, fixed_params=fixed_p, experiment_name=f"model_{year}")
        
        # Save Predictions
        test_df['model_probability'] = res['predictions']
        all_preds.append(test_df)
        
        # Save Params
        p_row = res['params'].copy()
        p_row['year'] = year
        all_params.append(p_row)
    
    if not all_preds:
        return pd.DataFrame(), pd.DataFrame()
        
    final_df = pd.concat(all_preds).reset_index(drop=True)
    
    # Generate Unique Match ID (Consistent across all scripts)
    if 'match_id' not in final_df.columns:
        final_df['match_id'] = (
            final_df['tourney_date'].dt.strftime('%Y%m%d') + '_' +
            final_df['winner_name'].str.replace(' ', '').str.slice(0, 5) + 'vs' +
            final_df['loser_name'].str.replace(' ', '').str.slice(0, 5) + '_' +
            final_df.index.astype(str)
        )
    
    return final_df, pd.DataFrame(all_params)