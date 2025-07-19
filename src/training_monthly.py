import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests


def load_feature_dataset(path: str = 'output/feature_monthly.csv') -> pd.DataFrame:
    """Load the feature dataset from CSV."""
    return pd.read_csv(path)


def train_test_split_time(df: pd.DataFrame, test_periods: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test sets, preserving temporal order."""
    # Remove rows where value is NaN before splitting
    df = df.dropna(subset=['value'])
    df = df.sort_values(['year', 'month'])
    if len(df) <= test_periods:
        return df.iloc[0:0], df  # all test if not enough data
    return df.iloc[:-test_periods], df.iloc[-test_periods:]


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE for forecast evaluation."""
    # Remove NaN values before evaluation
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    return {
        'mae': mean_absolute_error(y_true_clean, y_pred_clean),
        'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'mape': mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    }


def save_results(results: Dict, out_dir: str, model_name: str, product: str, country: str) -> None:
    """Save predictions, metrics, and model to disk."""
    os.makedirs(out_dir, exist_ok=True)
    # Save metrics
    metrics_path = os.path.join(out_dir, f'{model_name}_{product}_{country}_metrics.json')
    pd.Series(results['metrics']).to_json(metrics_path)
    # Save predictions
    preds_path = os.path.join(out_dir, f'{model_name}_{product}_{country}_predictions.csv')
    results['predictions'].to_csv(preds_path, index=False)
    # Save model if available
    if 'model' in results and results['model'] is not None:
        model_path = os.path.join(out_dir, f'{model_name}_{product}_{country}_model.pkl')
        joblib.dump(results['model'], model_path)


def run_sarimax(train, test, exog_cols):
    """Run SARIMAX model"""
    try:
        # Create date index for time series
        train_with_date = train.copy()
        test_with_date = test.copy()
        
        # Create date column from year and month
        train_with_date['date'] = pd.to_datetime(train_with_date['year'].astype(str) + '-' + 
                                               train_with_date['month'].astype(str) + '-01')
        test_with_date['date'] = pd.to_datetime(test_with_date['year'].astype(str) + '-' + 
                                              test_with_date['month'].astype(str) + '-01')
        
        # Set date as index
        train_with_date = train_with_date.set_index('date')
        test_with_date = test_with_date.set_index('date')
        
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        
        # Prepare exogenous variables
        train_exog = train_with_date[exog_cols] if exog_cols else None
        test_exog = test_with_date[exog_cols] if exog_cols else None
        
        model = SARIMAX(
            train_with_date['value'], 
            exog=train_exog, 
            order=order, 
            seasonal_order=seasonal_order, 
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        fit = model.fit(disp=False)
        
        # Generate predictions
        preds = fit.predict(start=test_with_date.index[0], end=test_with_date.index[-1], exog=test_exog)
        
        actuals = test_with_date['value'] if 'value' in test_with_date.columns else np.full(len(test_with_date), np.nan)
        predictions = pd.DataFrame({'date': test_with_date.index, 'actual': actuals, 'predicted': preds})
        
        return {
            'predictions': predictions,
            'model': fit
        }
    except Exception as e:
        print(f"SARIMAX failed: {e}")
        return None

def run_exp_smoothing(train, test):
    """Run Exponential Smoothing model"""
    try:
        # Create date index for time series
        train_with_date = train.copy()
        test_with_date = test.copy()
        
        # Create date column from year and month
        train_with_date['date'] = pd.to_datetime(train_with_date['year'].astype(str) + '-' + 
                                               train_with_date['month'].astype(str) + '-01')
        test_with_date['date'] = pd.to_datetime(test_with_date['year'].astype(str) + '-' + 
                                              test_with_date['month'].astype(str) + '-01')
        
        # Set date as index
        train_with_date = train_with_date.set_index('date')
        test_with_date = test_with_date.set_index('date')
        
        model = ExponentialSmoothing(
            train_with_date['value'], 
            trend='add', 
            seasonal='add', 
            seasonal_periods=12
        )
        fit = model.fit()
        preds = fit.forecast(steps=len(test_with_date))
        
        actuals = test_with_date['value'] if 'value' in test_with_date.columns else np.full(len(test_with_date), np.nan)
        predictions = pd.DataFrame({'date': test_with_date.index, 'actual': actuals, 'predicted': preds})
        
        return {
            'predictions': predictions,
            'model': fit
        }
    except Exception as e:
        print(f"Exponential Smoothing failed: {e}")
        return None

def run_prophet(train, test, exog_cols=None):
    """Run Prophet model"""
    try:
        # Create date column for Prophet
        train_with_date = train.copy()
        test_with_date = test.copy()
        
        # Create date column from year and month
        train_with_date['date'] = pd.to_datetime(train_with_date['year'].astype(str) + '-' + 
                                               train_with_date['month'].astype(str) + '-01')
        test_with_date['date'] = pd.to_datetime(test_with_date['year'].astype(str) + '-' + 
                                              test_with_date['month'].astype(str) + '-01')
        
        # Prepare data for Prophet
        prophet_train = train_with_date.rename(columns={'date': 'ds', 'value': 'y'})
        
        # Add exogenous variables as additional regressors if provided
        model = Prophet()
        if exog_cols:
            for col in exog_cols:
                if col in prophet_train.columns:
                    model.add_regressor(col)
        
        model.fit(prophet_train)
        
        # Prepare future dataframe
        future = test_with_date.rename(columns={'date': 'ds'})
        if exog_cols:
            for col in exog_cols:
                if col in future.columns:
                    future[col] = test_with_date[col].values
        
        # Generate predictions
        forecast = model.predict(future)
        preds = forecast['yhat'].values
        
        actuals = test_with_date['value'] if 'value' in test_with_date.columns else np.full(len(test_with_date), np.nan)
        predictions = pd.DataFrame({'date': test_with_date.index, 'actual': actuals, 'predicted': preds})
        
        return {
            'predictions': predictions,
            'model': model
        }
    except Exception as e:
        print(f"Prophet failed: {e}")
        return None

def run_ml_model(train, test, exog_cols, model_type='rf'):
    """Run ML model (Random Forest, XGBoost, or LightGBM)"""
    try:
        # Prepare features and target, removing NaN values
        train_clean = train.dropna(subset=['value'] + exog_cols)
        test_clean = test.dropna(subset=exog_cols)
        
        if len(train_clean) == 0:
            print(f"{model_type.upper()} failed: No valid training data after removing NaN values")
            return None
        
        X_train = train_clean[exog_cols]
        y_train = train_clean['value']
        X_test = test_clean[exog_cols]
        
        # Initialize model
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lgb':
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Generate predictions
        preds = model.predict(X_test)
        
        # Create predictions dataframe with proper indexing
        actuals = test_clean['value'] if 'value' in test_clean.columns else np.full(len(test_clean), np.nan)
        predictions = pd.DataFrame({
            'date': test_clean.index, 
            'actual': actuals, 
            'predicted': preds
        })
        
        return {
            'predictions': predictions,
            'model': model
        }
    except Exception as e:
        print(f"{model_type.upper()} failed: {e}")
        return None

def select_exog_features(train, test, exog_cols, model_type='rf'):
    # Remove columns with all NaN or constant values
    X = train[exog_cols].copy()
    y = train['value']
    X = X.loc[:, X.nunique() > 1]
    exog_cols = X.columns.tolist()
    if len(exog_cols) == 0:
        return []
    try:
        if model_type in ['prophet', 'xgb']:
            # Mutual information
            mi = mutual_info_regression(X.fillna(0), y)
            top_idx = np.argsort(mi)[-5:][::-1]
            selected = [exog_cols[i] for i in top_idx if mi[i] > 0]
            return selected[:5] if selected else exog_cols[:5]
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X.fillna(0), y)
            importances = rf.feature_importances_
            top_idx = np.argsort(importances)[-5:][::-1]
            selected = [exog_cols[i] for i in top_idx if importances[i] > 0]
            return selected[:5] if selected else exog_cols[:5]
        elif model_type == 'sarimax':
            # Granger causality test for each exog variable
            pvals = {}
            for col in exog_cols:
                try:
                    test_data = train[['value', col]].dropna()
                    if test_data.shape[0] > 10:
                        res = grangercausalitytests(test_data, maxlag=3, verbose=False)
                        min_pval = min([res[lag][0]['ssr_ftest'][1] for lag in res])
                        pvals[col] = min_pval
                except Exception:
                    continue
            selected = sorted(pvals, key=pvals.get)[:5]
            return selected if selected else exog_cols[:5]
        else:
            return exog_cols[:5]
    except Exception:
        return exog_cols[:5]

def ensemble_top_models(preds_list: List[pd.DataFrame]) -> pd.DataFrame:
    # Simple mean ensemble
    df = preds_list[0][['date', 'actual']].copy()
    pred_cols = []
    for i, preds in enumerate(preds_list):
        df[f'pred_{i}'] = preds['predicted'].values
        pred_cols.append(f'pred_{i}')
    df['ensemble_pred'] = df[pred_cols].mean(axis=1)
    return df

def process_product_country(product, country, group, exog_cols_all, exog_cols_sarimax, out_dir):
    group = group.reset_index(drop=True)
    train, test = train_test_split_time(group, test_periods=12)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    test.index = range(len(train), len(train) + len(test))
    # Feature selection per model
    selected_exog_rf = select_exog_features(train, test, exog_cols_all, model_type='rf')
    selected_exog_xgb = select_exog_features(train, test, exog_cols_all, model_type='xgb')
    selected_exog_lgb = select_exog_features(train, test, exog_cols_all, model_type='lgb')
    selected_exog_prophet = select_exog_features(train, test, exog_cols_all, model_type='prophet')
    selected_exog_sarimax = select_exog_features(train, test, exog_cols_sarimax, model_type='sarimax')
    results = {}
    # SARIMAX
    sarimax_out = run_sarimax(train, test, selected_exog_sarimax)
    if sarimax_out is not None:
        results['sarimax'] = sarimax_out
    
    # Exponential Smoothing (no exog)
    exp_smoothing_out = run_exp_smoothing(train, test)
    if exp_smoothing_out is not None:
        exp_smoothing_out['baseline'] = True  # Tag as baseline
        results['exp_smoothing'] = exp_smoothing_out
    
    # Prophet (with exogenous variables)
    prophet_out = run_prophet(train, test, selected_exog_prophet)
    if prophet_out is not None:
        results['prophet'] = prophet_out
    
    # Random Forest
    rf_out = run_ml_model(train, test, selected_exog_rf, model_type='rf')
    if rf_out is not None:
        results['rf'] = rf_out
    
    # XGBoost
    xgb_out = run_ml_model(train, test, selected_exog_xgb, model_type='xgb')
    if xgb_out is not None:
        results['xgb'] = xgb_out
    
    # LightGBM
    lgb_out = run_ml_model(train, test, selected_exog_lgb, model_type='lgb')
    if lgb_out is not None:
        results['lgb'] = lgb_out
    # Calculate metrics for each model
    model_exog_map = {
        'sarimax': selected_exog_sarimax,
        'exp_smoothing': [],
        'prophet': selected_exog_prophet,
        'rf': selected_exog_rf,
        'xgb': selected_exog_xgb,
        'lgb': selected_exog_lgb
    }
    for model_name, res in results.items():
        preds_df = res['predictions']
        metrics = evaluate_forecast(preds_df['actual'], preds_df['predicted'])
        # Add exogenous variable info
        exog_vars = model_exog_map.get(model_name, [])
        for i in range(5):
            metrics[f'exo_var_{i+1}'] = exog_vars[i] if i < len(exog_vars) else ''
        if model_name == 'exp_smoothing':
            metrics['baseline'] = True
        res['metrics'] = metrics
    # Check if any models succeeded
    if not results:
        print(f'No models succeeded for {product} - {country}')
        return [], []
    
    # Evaluate and pick top 3 by MAPE
    model_mape = {k: v['metrics'].get('mape', np.inf) for k, v in results.items()}
    top3 = sorted(model_mape, key=model_mape.get)[:3]
    print(f'Processing {product} - {country} | Top 3 models: {top3}')
    
    # Ensemble
    preds_list = [results[m]['predictions'] for m in top3]
    ensemble_preds = ensemble_top_models(preds_list)
    # Save individual model results
    for model_name, res in results.items():
        save_results(res, out_dir, model_name, product, country)
    # Save ensemble
    ensemble_path = os.path.join(out_dir, f'ensemble_{product}_{country}_predictions.csv')
    ensemble_preds.to_csv(ensemble_path, index=False)
    # Prepare data for aggregation
    all_forecasts = []
    all_metrics = []
    # Add individual model results
    for model_name, res in results.items():
        # Add metadata to predictions
        preds_df = res['predictions'].copy()
        preds_df['model'] = model_name
        preds_df['product'] = product
        preds_df['country'] = country
        all_forecasts.append(preds_df)
        # Add metadata to metrics
        metrics_series = pd.Series(res['metrics'])
        metrics_series['model'] = model_name
        metrics_series['product'] = product
        metrics_series['country'] = country
        all_metrics.append(metrics_series)
    # Add ensemble results
    ensemble_preds['model'] = 'ensemble'
    ensemble_preds['product'] = product
    ensemble_preds['country'] = country
    all_forecasts.append(ensemble_preds)
    # Add ensemble metrics
    if 'ensemble_pred' in ensemble_preds:
        ensemble_metrics = evaluate_forecast(ensemble_preds['actual'], ensemble_preds['ensemble_pred'])
    else:
        ensemble_metrics = {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}
    ensemble_metrics['model'] = 'ensemble'
    ensemble_metrics['product'] = product
    ensemble_metrics['country'] = country
    all_metrics.append(pd.Series(ensemble_metrics))
    print(f'Finished {product} - {country}')
    return all_forecasts, all_metrics

def main():
    """Run monthly forecasting for each product-country combination in parallel."""
    df = load_feature_dataset()
    out_dir = 'output/monthly_forecast_results'
    exog_cols_all = [col for col in df.columns if col.endswith('_norm') or 'lag' in col or 'rollmean' in col or col in ['quarter', 'is_year_end']]
    exog_cols_sarimax = [col for col in df.columns if col.endswith('_norm') or col in ['quarter', 'is_year_end']]
    grouped = list(df.groupby(['product', 'country']))
    
    # Run parallel processing and collect results
    results = Parallel(n_jobs=6, backend='loky')(
        delayed(process_product_country)(product, country, group, exog_cols_all, exog_cols_sarimax, out_dir)
        for (product, country), group in grouped
    )
    
    # Aggregate all forecasts and metrics from results
    all_forecasts = []
    all_metrics = []
    
    for forecasts, metrics in results:
        all_forecasts.extend(forecasts)
        all_metrics.extend(metrics)
    
    # Save aggregated results
    if all_forecasts:
        pd.concat(all_forecasts, ignore_index=True).to_csv(os.path.join(out_dir, 'all_forecasts.csv'), index=False)
        print(f"Saved aggregated forecasts with {len(all_forecasts)} model results")
    
    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(os.path.join(out_dir, 'all_metrics.csv'), index=False)
        print(f"Saved aggregated metrics with {len(all_metrics)} model results")

if __name__ == '__main__':
    main() 