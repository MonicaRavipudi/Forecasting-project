import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
import json
import os
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

def load_metrics_and_get_top_models():
    """Load metrics and get top 3 models for each product-country combination"""
    metrics_df = pd.read_csv('output/monthly_forecast_results/all_metrics.csv')
    
    # Filter out ensemble models and get top 3 by MAPE for each product-country
    individual_models = metrics_df[metrics_df['model'] != 'ensemble'].copy()
    top_models = []
    
    for (product, country), group in individual_models.groupby(['product', 'country']):
        # Sort by MAPE (lower is better) and get top 3
        top_3 = group.nsmallest(3, 'mape')[['model', 'mape', 'exo_var_1', 'exo_var_2', 'exo_var_3', 'exo_var_4', 'exo_var_5']]
        for _, row in top_3.iterrows():
            top_models.append({
                'product': product,
                'country': country,
                'model': row['model'],
                'mape': row['mape'],
                'exo_vars': [row['exo_var_1'], row['exo_var_2'], row['exo_var_3'], row['exo_var_4'], row['exo_var_5']]
            })
    
    return top_models

def create_lag_rolling_features(df, forecast_values=None):
    """Create lag and rolling features, using forecast_values if provided for future dates"""
    df = df.copy()
    
    # Remove existing lag and rolling features to avoid conflicts
    lag_roll_cols = [col for col in df.columns if 'lag' in col or 'rollmean' in col]
    df = df.drop(columns=lag_roll_cols)
    
    # If forecast_values provided, use them to fill missing values
    if forecast_values is not None:
        df.loc[df['value'].isna(), 'value'] = forecast_values
    
    # Create lag features
    df['value_lag_3'] = df['value'].shift(3)
    df['value_lag_6'] = df['value'].shift(6)
    df['value_lag_12'] = df['value'].shift(12)
    
    # Create rolling features
    df['value_rollmean_3'] = df['value'].rolling(window=3, min_periods=1).mean()
    df['value_rollmean_6'] = df['value'].rolling(window=6, min_periods=1).mean()
    df['value_rollmean_12'] = df['value'].rolling(window=12, min_periods=1).mean()
    
    return df

def run_exp_smoothing_forecast(train_data, forecast_periods=12):
    """Run exponential smoothing to generate initial forecasts for filling lag/rolling features"""
    try:
        # Prepare data for exponential smoothing
        train_series = train_data['value'].dropna()
        if len(train_series) < 2:
            return None
        
        # Fit exponential smoothing model
        model = ExponentialSmoothing(
            train_series, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=12
        )
        fitted_model = model.fit()
        
        # Generate forecasts
        forecasts = fitted_model.forecast(steps=forecast_periods)
        return forecasts.values
        
    except Exception as e:
        print(f"Exponential smoothing failed: {e}")
        return None

def run_sarimax(train, test, exog_cols):
    """Run SARIMAX model"""
    try:
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        
        # Prepare exogenous variables
        train_exog = train[exog_cols] if exog_cols else None
        test_exog = test[exog_cols] if exog_cols else None
        
        model = SARIMAX(
            train['value'], 
            exog=train_exog, 
            order=order, 
            seasonal_order=seasonal_order, 
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        fit = model.fit(disp=False)
        
        # Generate predictions
        preds = fit.predict(start=test.index[0], end=test.index[-1], exog=test_exog)
        
        return {
            'predictions': preds,
            'model': fit
        }
    except Exception as e:
        print(f"SARIMAX failed: {e}")
        return None

def run_exp_smoothing_model(train, test):
    """Run Exponential Smoothing model"""
    try:
        model = ExponentialSmoothing(
            train['value'], 
            trend='add', 
            seasonal='add', 
            seasonal_periods=12
        )
        fit = model.fit()
        preds = fit.forecast(steps=len(test))
        
        return {
            'predictions': preds,
            'model': fit
        }
    except Exception as e:
        print(f"Exponential Smoothing failed: {e}")
        return None

def run_prophet_model(train, test, exog_cols):
    """Run Prophet model"""
    try:
        # Prepare data for Prophet
        prophet_train = train.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
        
        # Add exogenous variables as additional regressors
        model = Prophet()
        valid_exog_cols = []
        
        for col in exog_cols:
            if col in prophet_train.columns:
                # Check if the column has any NaN values
                if prophet_train[col].isna().any():
                    print(f"  Skipping {col} for Prophet due to NaN values")
                    continue
                # Fill any remaining NaN values in exogenous variables for training
                prophet_train[col] = prophet_train[col].fillna(method='ffill').fillna(method='bfill')
                model.add_regressor(col)
                valid_exog_cols.append(col)
        
        if not valid_exog_cols:
            print("  No valid exogenous variables for Prophet, using only time series")
        
        model.fit(prophet_train)
        
        # Prepare future dataframe
        future = test.reset_index().rename(columns={'date': 'ds'})
        for col in valid_exog_cols:
            if col in future.columns:
                # Fill NaN values in exogenous variables for prediction
                future[col] = future[col].fillna(method='ffill').fillna(method='bfill')
        
        # Generate predictions
        forecast = model.predict(future)
        preds = forecast['yhat'].values
        
        return {
            'predictions': preds,
            'model': model
        }
    except Exception as e:
        print(f"Prophet failed: {e}")
        return None

def run_ml_model(train, test, exog_cols, model_type='rf'):
    """Run ML model (Random Forest, XGBoost, or LightGBM)"""
    try:
        # Prepare features and handle NaN values
        X_train = train[exog_cols].fillna(method='ffill').fillna(method='bfill')
        y_train = train['value']
        X_test = test[exog_cols].fillna(method='ffill').fillna(method='bfill')
        
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
        
        return {
            'predictions': preds,
            'model': model
        }
    except Exception as e:
        print(f"{model_type.upper()} failed: {e}")
        return None

def run_model_by_type(train, test, exog_cols, model_type):
    """Run model based on type"""
    if model_type == 'sarimax':
        return run_sarimax(train, test, exog_cols)
    elif model_type == 'exp_smoothing':
        return run_exp_smoothing_model(train, test)
    elif model_type == 'prophet':
        return run_prophet_model(train, test, exog_cols)
    elif model_type in ['rf', 'xgb', 'lgb']:
        return run_ml_model(train, test, exog_cols, model_type)
    else:
        print(f"Unknown model type: {model_type}")
        return None

def create_ensemble_forecast(predictions_list, weights=None):
    """Create ensemble forecast from multiple model predictions"""
    if not predictions_list:
        return None
    
    # If no weights provided, use equal weights
    if weights is None:
        weights = [1/len(predictions_list)] * len(predictions_list)
    
    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)
    
    # Calculate weighted average
    ensemble_pred = np.zeros(len(predictions_list[0]))
    for pred, weight in zip(predictions_list, weights):
        ensemble_pred += weight * pred
    
    return ensemble_pred

def main():
    print("Starting Future Monthly Forecasting...")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('output/feature_monthly.csv')
    
    # Remove existing lag and rolling features to avoid conflicts
    lag_roll_cols = [col for col in df.columns if 'lag' in col or 'rollmean' in col]
    df = df.drop(columns=lag_roll_cols)
    
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.set_index('date')
    
    # Load top models
    print("Loading top models...")
    top_models = load_metrics_and_get_top_models()
    
    # Create output directory
    os.makedirs('output/future_forecast_results', exist_ok=True)
    
    all_forecasts = []
    all_metrics = []
    
    # Process each product-country combination
    for product in df['product'].unique():
        for country in df[df['product'] == product]['country'].unique():
            if pd.isna(country):
                continue
                
            print(f"\nProcessing {product} - {country}")
            
            # Filter data for this product-country
            product_country_data = df[(df['product'] == product) & (df['country'] == country)].copy()
            
            # Split data: use all data up to 2019 for training, 2020 for forecasting
            train_data = product_country_data[product_country_data.index.year < 2020].copy()
            future_data = product_country_data[product_country_data.index.year == 2020].copy()
            
            if len(train_data) == 0 or len(future_data) == 0:
                print(f"No data available for {product} - {country}")
                continue
            
            # Get top 3 models for this product-country
            product_country_models = [m for m in top_models if m['product'] == product and m['country'] == country]
            
            if not product_country_models:
                print(f"No models found for {product} - {country}")
                continue
            
            # Step 1: Run exponential smoothing to get initial forecasts for filling lag/rolling features
            print("  Step 1: Generating initial forecasts with Exponential Smoothing...")
            initial_forecasts = run_exp_smoothing_forecast(train_data, len(future_data))
            
            if initial_forecasts is None:
                print(f"  Failed to generate initial forecasts for {product} - {country}")
                continue
            
            # Step 2: Create lag and rolling features using initial forecasts
            print("  Step 2: Creating lag and rolling features...")
            
            # First, create lag features for training data using actual values
            train_with_features = create_lag_rolling_features(train_data)
            
            # Then, create lag features for future data using initial forecasts
            future_with_forecasts = future_data.copy()
            future_with_forecasts['value'] = initial_forecasts
            future_with_features = create_lag_rolling_features(future_with_forecasts)
            
            # Combine the datasets
            combined_data = pd.concat([train_with_features, future_with_features])
            
            # Step 3: Run top 3 models and collect predictions
            print("  Step 3: Running top 3 models...")
            model_predictions = []
            model_weights = []
            successful_models = []  # Track successful models for ensemble
            
            for model_info in product_country_models[:3]:  # Top 3 models
                model_type = model_info['model']
                mape = model_info['mape']
                exo_vars = [var for var in model_info['exo_vars'] if pd.notna(var) and var != '']
                
                print(f"    Running {model_type}...")
                
                # Get exogenous variables
                available_exo_vars = [var for var in exo_vars if var in train_with_features.columns]
                
                # Run model
                result = run_model_by_type(train_with_features, future_with_features, available_exo_vars, model_type)
                
                if result is not None:
                    model_predictions.append(result['predictions'])
                    # Use inverse MAPE as weight (lower MAPE = higher weight)
                    model_weights.append(1 / (mape + 1e-6))  # Add small constant to avoid division by zero
                    successful_models.append(model_type)  # Track successful models
                    
                    # Generate predictions for training data as well
                    train_result = run_model_by_type(train_with_features, train_with_features, available_exo_vars, model_type)
                    
                    # Save individual model results for future data
                    future_forecast_df = pd.DataFrame({
                        'date': future_with_features.index,
                        'product': product,
                        'country': country,
                        'model': model_type,
                        'predicted': result['predictions'],
                        'actual': np.nan,  # No actuals for future
                        'in_ensemble': True  # All successful models are in ensemble
                    })
                    all_forecasts.append(future_forecast_df)
                    
                    # Save individual model results for training data
                    if train_result is not None:
                        train_forecast_df = pd.DataFrame({
                            'date': train_with_features.index,
                            'product': product,
                            'country': country,
                            'model': model_type,
                            'predicted': train_result['predictions'],
                            'actual': train_with_features['value'].values,
                            'in_ensemble': True  # All successful models are in ensemble
                        })
                        all_forecasts.append(train_forecast_df)
            
            # Step 4: Create ensemble forecast
            print("  Step 4: Creating ensemble forecast...")
            if model_predictions:
                ensemble_pred = create_ensemble_forecast(model_predictions, model_weights)
                
                # Save ensemble forecast for future data
                ensemble_forecast_df = pd.DataFrame({
                    'date': future_with_features.index,
                    'product': product,
                    'country': country,
                    'model': 'ensemble',
                    'predicted': ensemble_pred,
                    'actual': np.nan,  # No actuals for future
                    'in_ensemble': True  # All successful models are in ensemble
                })
                all_forecasts.append(ensemble_forecast_df)
                
                # Generate ensemble for training data
                train_model_predictions = []
                for model_info in product_country_models[:3]:
                    model_type = model_info['model']
                    mape = model_info['mape']
                    exo_vars = [var for var in model_info['exo_vars'] if pd.notna(var) and var != '']
                    available_exo_vars = [var for var in exo_vars if var in train_with_features.columns]
                    
                    train_result = run_model_by_type(train_with_features, train_with_features, available_exo_vars, model_type)
                    if train_result is not None:
                        train_model_predictions.append(train_result['predictions'])
                        model_weights.append(1 / (mape + 1e-6))
                
                if train_model_predictions:
                    train_ensemble_pred = create_ensemble_forecast(train_model_predictions, model_weights)
                    train_ensemble_df = pd.DataFrame({
                        'date': train_with_features.index,
                        'product': product,
                        'country': country,
                        'model': 'ensemble',
                        'predicted': train_ensemble_pred,
                        'actual': train_with_features['value'].values,
                        'in_ensemble': True  # All successful models are in ensemble
                    })
                    all_forecasts.append(train_ensemble_df)
                
                print(f"  Successfully generated ensemble forecast for {product} - {country}")
            else:
                print(f"  No successful models for {product} - {country}")
    
    # Combine all forecasts
    if all_forecasts:
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        
        # Save results
        combined_forecasts.to_csv('output/future_forecast_results/all_forecasts_2020.csv', index=False)
        
        # Create summary
        summary = combined_forecasts.groupby(['product', 'country', 'model'])['predicted'].agg(['mean', 'std', 'min', 'max']).reset_index()
        summary.to_csv('output/future_forecast_results/forecast_summary_2020.csv', index=False)
        
        print(f"\nForecasting completed!")
        print(f"Results saved to:")
        print(f"  - output/future_forecast_results/all_forecasts_2020.csv")
        print(f"  - output/future_forecast_results/forecast_summary_2020.csv")
        
        # Print summary
        print(f"\nForecast Summary:")
        print(summary.to_string(index=False))
    else:
        print("No forecasts were generated!")

if __name__ == "__main__":
    main() 