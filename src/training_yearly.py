import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

def load_master_dataset(path: str = 'master_dataset.csv') -> pd.DataFrame:
    """Load the master dataset and aggregate to yearly level."""
    df = pd.read_csv(path)
    
    # Clean numeric columns first
    df = clean_numeric_columns(df)
    
    # Aggregate to yearly level
    yearly_data = []
    
    for (product, country), group in df.groupby(['product', 'country']):
        # Sum revenue (value) by year
        revenue_yearly = group.groupby('year')['value'].sum().reset_index()
        
        # Average exogenous variables by year (exclude lag and rolling features from monthly data)
        exog_cols = [col for col in group.columns if col not in ['product', 'country', 'year', 'month', 'value', 'Country_actual'] 
                    and not col.endswith('_lag') and not col.endswith('_rollmean')]
        
        # Only include numeric columns for averaging
        numeric_exog_cols = []
        for col in exog_cols:
            if pd.api.types.is_numeric_dtype(group[col]):
                numeric_exog_cols.append(col)
        
        if numeric_exog_cols:
            exog_yearly = group.groupby('year')[numeric_exog_cols].mean().reset_index()
        else:
            exog_yearly = pd.DataFrame({'year': group['year'].unique()})
        
        # Merge revenue and exogenous data
        yearly_group = pd.merge(revenue_yearly, exog_yearly, on='year', how='left')
        yearly_group['product'] = product
        yearly_group['country'] = country
        
        yearly_data.append(yearly_group)
    
    yearly_df = pd.concat(yearly_data, ignore_index=True)
    yearly_df = yearly_df.sort_values(['product', 'country', 'year']).reset_index(drop=True)
    
    # Add yearly lag and rolling features
    yearly_df = add_yearly_features(yearly_df)
    
    return yearly_df

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric columns by removing $ and , and converting to float."""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if col not in ['product', 'country', 'year', 'month', 'Country_actual']:
            try:
                # Remove $ and , and convert to float
                df_clean[col] = (
                    df_clean[col].astype(str)
                    .str.replace('$', '', regex=False)
                    .str.replace(',', '', regex=False)
                )
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            except Exception:
                # If conversion fails, keep as is
                continue
    
    return df_clean

def add_yearly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add yearly lag and rolling features."""
    # Add lag_1 (last year's revenue)
    df['value_lag_1'] = df.groupby(['product', 'country'])['value'].shift(1)
    
    # Add rolling 2-year average
    df['value_rollmean_2'] = df.groupby(['product', 'country'])['value'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean()
    )
    
    return df

def train_test_split_time(df: pd.DataFrame, test_periods: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets for time series forecasting."""
    # Remove rows where value is NaN before splitting
    df = df.dropna(subset=['value'])
    if len(df) <= test_periods:
        return df.iloc[0:0], df  # all test if not enough data
    train = df.iloc[:-test_periods].copy()
    test = df.iloc[-test_periods:].copy()
    return train, test


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate forecast performance."""
    # Remove NaN values before evaluation
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def save_results(results: Dict, out_dir: str, model_name: str, product: str, country: str) -> None:
    """Save model results to files."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(out_dir, f'{model_name}_{product}_{country}_metrics.json')
    pd.Series(results['metrics']).to_json(metrics_path)
    
    # Save predictions
    preds_path = os.path.join(out_dir, f'{model_name}_{product}_{country}_predictions.csv')
    results['predictions'].to_csv(preds_path, index=False)
    
    # Save model
    model_path = os.path.join(out_dir, f'{model_name}_{product}_{country}_model.pkl')
    pd.to_pickle(results['model'], model_path)

def run_sarimax(train, test, exog_cols):
    """Run SARIMAX model."""
    try:
        # Create date index for time series
        train_with_date = train.copy()
        test_with_date = test.copy()
        
        # Create date column from year
        train_with_date['date'] = pd.to_datetime(train_with_date['year'].astype(str) + '-01-01')
        test_with_date['date'] = pd.to_datetime(test_with_date['year'].astype(str) + '-01-01')
        
        # Set date as index
        train_with_date = train_with_date.set_index('date')
        test_with_date = test_with_date.set_index('date')
        
        # Prepare exogenous variables
        if exog_cols:
            exog_train = train_with_date[exog_cols].fillna(method='ffill')
            exog_test = test_with_date[exog_cols].fillna(method='ffill')
            model = SARIMAX(train_with_date['value'], exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        else:
            model = SARIMAX(train_with_date['value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        
        fitted_model = model.fit(disp=False)
        preds = fitted_model.forecast(steps=len(test_with_date), exog=exog_test if exog_cols else None)
        
        metrics = evaluate_forecast(test_with_date['value'].values, preds)
        results = {
            'metrics': metrics,
            'predictions': pd.DataFrame({'year': test_with_date['year'], 'actual': test_with_date['value'], 'predicted': preds}),
            'model': fitted_model
        }
    except Exception as e:
        print(f"SARIMAX failed: {e}")
        # Fallback to simple prediction
        preds = [train['value'].iloc[-1]] * len(test)
        metrics = evaluate_forecast(test['value'].values, preds)
        results = {
            'metrics': metrics,
            'predictions': pd.DataFrame({'year': test['year'], 'actual': test['value'], 'predicted': preds}),
            'model': None
        }
    
    return results

def run_exp_smoothing(train, test):
    """Run Exponential Smoothing model."""
    try:
        # Create date index for time series
        train_with_date = train.copy()
        test_with_date = test.copy()
        
        # Create date column from year
        train_with_date['date'] = pd.to_datetime(train_with_date['year'].astype(str) + '-01-01')
        test_with_date['date'] = pd.to_datetime(test_with_date['year'].astype(str) + '-01-01')
        
        # Set date as index
        train_with_date = train_with_date.set_index('date')
        test_with_date = test_with_date.set_index('date')
        
        model = ExponentialSmoothing(train_with_date['value'], trend='add', seasonal='add', seasonal_periods=4)
        fitted_model = model.fit()
        preds = fitted_model.forecast(steps=len(test_with_date))
        
        metrics = evaluate_forecast(test_with_date['value'].values, preds)
        results = {
            'metrics': metrics,
            'predictions': pd.DataFrame({'year': test_with_date['year'], 'actual': test_with_date['value'], 'predicted': preds}),
            'model': fitted_model
        }
    except Exception as e:
        print(f"Exponential Smoothing failed: {e}")
        # Fallback to simple prediction
        preds = [train['value'].iloc[-1]] * len(test)
        metrics = evaluate_forecast(test['value'].values, preds)
        results = {
            'metrics': metrics,
            'predictions': pd.DataFrame({'year': test['year'], 'actual': test['value'], 'predicted': preds}),
            'model': None
        }
    
    return results

def run_prophet(train, test):
    """Run Prophet model."""
    try:
        prophet_train = train[['year', 'value']].copy()
        prophet_train['ds'] = pd.to_datetime(prophet_train['year'], format='%Y')
        prophet_train = prophet_train.rename(columns={'value': 'y'})
        
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(prophet_train[['ds', 'y']])
        
        future = pd.DataFrame({'ds': pd.to_datetime(test['year'], format='%Y')})
        forecast = m.predict(future)
        preds = forecast['yhat'].values
        
        metrics = evaluate_forecast(test['value'].values, preds)
        results = {
            'metrics': metrics,
            'predictions': pd.DataFrame({'year': test['year'], 'actual': test['value'], 'predicted': preds}),
            'model': m
        }
    except Exception as e:
        print(f"Prophet failed: {e}")
        # Fallback to simple prediction
        preds = [train['value'].iloc[-1]] * len(test)
        metrics = evaluate_forecast(test['value'].values, preds)
        results = {
            'metrics': metrics,
            'predictions': pd.DataFrame({'year': test['year'], 'actual': test['value'], 'predicted': preds}),
            'model': None
        }
    
    return results

def run_ml_model(train, test, exog_cols, model_type='rf'):
    """Run ML model (Random Forest, XGBoost, or LightGBM)."""
    try:
        # Prepare features and target, removing NaN values
        train_clean = train.dropna(subset=['value'] + exog_cols)
        test_clean = test.dropna(subset=exog_cols)
        
        if len(train_clean) == 0:
            print(f"{model_type.upper()} failed: No valid training data after removing NaN values")
            return None
        
        X_train, y_train = train_clean[exog_cols], train_clean['value']
        X_test, y_test = test_clean[exog_cols], test_clean['value']
        
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lgb':
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError('Unknown model_type')
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        metrics = evaluate_forecast(y_test, preds)
        results = {
            'metrics': metrics,
            'predictions': pd.DataFrame({'year': test_clean['year'], 'actual': y_test, 'predicted': preds}),
            'model': model
        }
    except Exception as e:
        print(f"{model_type.upper()} failed: {e}")
        # Fallback to simple prediction
        preds = [train['value'].iloc[-1]] * len(test)
        metrics = evaluate_forecast(test['value'].values, preds)
        results = {
            'metrics': metrics,
            'predictions': pd.DataFrame({'year': test['year'], 'actual': test['value'], 'predicted': preds}),
            'model': None
        }
    
    return results

def select_exog_features(train, test, exog_cols, model_type='rf'):
    """Select exogenous features based on model type."""
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
                    if test_data.shape[0] > 5:  # Need at least 5 observations
                        res = grangercausalitytests(test_data, maxlag=2, verbose=False)
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
    """Create ensemble from top models."""
    # Simple mean ensemble
    df = preds_list[0][['year', 'actual']].copy()
    pred_cols = []
    for i, preds in enumerate(preds_list):
        df[f'pred_{i}'] = preds['predicted'].values
        pred_cols.append(f'pred_{i}')
    df['ensemble_pred'] = df[pred_cols].mean(axis=1)
    return df

def process_product_country(product, country, group, exog_cols_all, exog_cols_sarimax, out_dir):
    """Process a single product-country combination."""
    group = group.reset_index(drop=True)
    train, test = train_test_split_time(group, test_periods=1)  # 1 year for testing
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
    res = run_sarimax(train, test, selected_exog_sarimax)
    for i in range(5):
        res['metrics'][f'exo_var_{i+1}'] = selected_exog_sarimax[i] if i < len(selected_exog_sarimax) else ''
    results['sarimax'] = res
    
    # Exponential Smoothing (no exog)
    res = run_exp_smoothing(train, test)
    for i in range(5):
        res['metrics'][f'exo_var_{i+1}'] = ''
    res['baseline'] = True  # Tag as baseline
    results['exp_smoothing'] = res
    
    # Prophet
    res = run_prophet(train, test)
    for i in range(5):
        res['metrics'][f'exo_var_{i+1}'] = selected_exog_prophet[i] if i < len(selected_exog_prophet) else ''
    results['prophet'] = res
    
    # Random Forest
    res = run_ml_model(train, test, selected_exog_rf, model_type='rf')
    for i in range(5):
        res['metrics'][f'exo_var_{i+1}'] = selected_exog_rf[i] if i < len(selected_exog_rf) else ''
    results['rf'] = res
    
    # XGBoost
    res = run_ml_model(train, test, selected_exog_xgb, model_type='xgb')
    for i in range(5):
        res['metrics'][f'exo_var_{i+1}'] = selected_exog_xgb[i] if i < len(selected_exog_xgb) else ''
    results['xgb'] = res
    
    # LightGBM
    res = run_ml_model(train, test, selected_exog_lgb, model_type='lgb')
    for i in range(5):
        res['metrics'][f'exo_var_{i+1}'] = selected_exog_lgb[i] if i < len(selected_exog_lgb) else ''
    results['lgb'] = res
    
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
    # Add ensemble metrics using evaluate_forecast
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
    """Run yearly forecasting for each product-country combination in parallel."""
    print("Loading yearly feature data...")
    df = pd.read_csv('output/feature_yearly.csv')
    print(f"Yearly feature dataset shape: {df.shape}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Products: {df['product'].unique()}")
    print(f"Countries: {df['country'].unique()}")
    out_dir = 'output/yearly_forecast_results'
    exog_cols_all = [col for col in df.columns if col not in ['product', 'country', 'year', 'value', 'Country_actual']]
    exog_cols_sarimax = [col for col in exog_cols_all if not col.endswith('_lag') and not col.endswith('_rollmean')]
    print(f"All exogenous columns: {len(exog_cols_all)}")
    print(f"SARIMAX exogenous columns: {len(exog_cols_sarimax)}")
    grouped = list(df.groupby(['product', 'country']))
    print(f"Processing {len(grouped)} product-country combinations...")
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