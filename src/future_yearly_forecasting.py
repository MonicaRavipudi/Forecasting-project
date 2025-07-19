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
    metrics_df = pd.read_csv('output/yearly_forecast_results/all_metrics.csv')
    
    # Filter out ensemble models and get top 3 by MAPE
    individual_metrics = metrics_df[metrics_df['model'] != 'ensemble'].copy()
    
    top_models = {}
    for (product, country), group in individual_metrics.groupby(['product', 'country']):
        # Sort by MAPE and get top 3
        top_3 = group.nsmallest(3, 'mape')[['model', 'mape']].to_dict('records')
        top_models[f"{product}-{country}"] = top_3
    
    return top_models

def run_sarimax(train, test, exog_cols):
    """Run SARIMAX model"""
    try:
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 4)  # Yearly data, so seasonal period is 4
        
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
        
        actuals = test['value'] if 'value' in test.columns else np.full(len(test), np.nan)
        predictions = pd.DataFrame({'year': test['year'].values, 'actual': actuals, 'predicted': preds})
        
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
        model = ExponentialSmoothing(
            train['value'], 
            trend='add', 
            seasonal='add', 
            seasonal_periods=4  # Yearly data
        )
        fit = model.fit()
        preds = fit.forecast(steps=len(test))
        
        actuals = test['value'] if 'value' in test.columns else np.full(len(test), np.nan)
        predictions = pd.DataFrame({'year': test['year'].values, 'actual': actuals, 'predicted': preds})
        
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
        # Prepare data for Prophet
        prophet_train = train.reset_index().rename(columns={'year': 'ds', 'value': 'y'})
        
        # Add exogenous variables as additional regressors if provided
        model = Prophet()
        valid_exog_cols = []
        
        if exog_cols:
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
        
        if not valid_exog_cols and exog_cols:
            print("  No valid exogenous variables for Prophet, using only time series")
        
        model.fit(prophet_train)
        
        # Prepare future dataframe
        future = test.reset_index().rename(columns={'year': 'ds'})
        if valid_exog_cols:
            for col in valid_exog_cols:
                if col in future.columns:
                    # Fill NaN values in exogenous variables for prediction
                    future[col] = future[col].fillna(method='ffill').fillna(method='bfill')
        
        # Generate predictions
        forecast = model.predict(future)
        preds = forecast['yhat'].values
        
        actuals = test['value'] if 'value' in test.columns else np.full(len(test), np.nan)
        predictions = pd.DataFrame({'year': test['year'].values, 'actual': actuals, 'predicted': preds})
        
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
        
        actuals = test['value'] if 'value' in test.columns else np.full(len(test), np.nan)
        predictions = pd.DataFrame({'year': test['year'].values, 'actual': actuals, 'predicted': preds})
        
        return {
            'predictions': predictions,
            'model': model
        }
    except Exception as e:
        print(f"{model_type.upper()} failed: {e}")
        return None

def create_lag_features(df, forecast_values=None):
    """Create lag features, using forecast_values if provided for future dates"""
    df = df.copy()
    
    # Remove existing lag features to avoid conflicts
    lag_cols = [col for col in df.columns if 'lag' in col]
    df = df.drop(columns=lag_cols)
    
    # If forecast_values provided, use them to fill missing values
    if forecast_values is not None:
        df.loc[df['value'].isna(), 'value'] = forecast_values
    
    # Create lag features
    df['value_lag_1'] = df['value'].shift(1)
    df['value_lag_2'] = df['value'].shift(2)
    df['value_lag_3'] = df['value'].shift(3)
    
    # Create rolling mean features
    df['value_rollmean_2'] = df['value'].rolling(window=2, min_periods=1).mean()
    df['value_rollmean_3'] = df['value'].rolling(window=3, min_periods=1).mean()
    
    return df

def ensemble_predictions(predictions_list):
    """Create ensemble from multiple model predictions"""
    if not predictions_list:
        return None
    
    # Use the first prediction as base
    ensemble = predictions_list[0]['result'].copy()
    
    # Calculate mean of all predictions
    all_preds = np.column_stack([pred['result']['predictions']['predicted'].values for pred in predictions_list])
    ensemble['predictions']['ensemble_predicted'] = np.mean(all_preds, axis=1)
    
    return ensemble

def main():
    """Main function to generate yearly forecasts for 2020-2024"""
    print("Starting yearly future forecasting for 2020-2024...")
    
    # Load data
    df = pd.read_csv('output/feature_yearly.csv')
    
    # Remove existing lag features to avoid conflicts
    lag_cols = [col for col in df.columns if 'lag' in col]
    df = df.drop(columns=lag_cols)
    
    # Load top models for each product-country
    top_models = load_metrics_and_get_top_models()
    
    # Create output directory
    os.makedirs('output/future_yearly_forecast_results', exist_ok=True)
    
    # Store exponential smoothing forecasts for lag feature creation
    exp_smoothing_forecasts = {}
    
    all_forecasts = []
    forecast_summary = []
    
    # Process each product-country combination
    for product_country, models in top_models.items():
        product, country = product_country.split('-')
        print(f"\nProcessing {product} - {country}")
        print(f"Top 3 models: {[m['model'] for m in models]}")
        
        # Get data for this product-country
        group_data = df[(df['product'] == product) & (df['country'] == country)].copy()
        group_data = group_data.sort_values('year').reset_index(drop=True)
        
        # Split data: use all data up to 2019 for training, 2020-2024 for forecasting
        train_data = group_data[group_data['year'] < 2020].copy()
        future_data = group_data[group_data['year'] >= 2020].copy()
        
        if len(train_data) == 0:
            print(f"No training data available for {product} - {country}")
            continue
        
        # Step 1: Run Exponential Smoothing first to get initial forecasts
        print(f"  Running Exponential Smoothing for {product} - {country}")
        exp_result = run_exp_smoothing(train_data, future_data)
        
        if exp_result is not None:
            # Store exponential smoothing forecasts
            exp_forecasts = {}
            for idx, row in exp_result['predictions'].iterrows():
                year = row['year']
                pred = row['predicted']
                exp_forecasts[year] = pred
            
            exp_smoothing_forecasts[product_country] = exp_forecasts
            print(f"  Exponential Smoothing completed for {product} - {country}")
        else:
            print(f"  Exponential Smoothing failed for {product} - {country}")
            continue
        
        # Step 2: Create lag features using initial forecasts
        print(f"  Creating lag features for {product} - {country}")
        
        # First, create lag features for training data using actual values
        train_with_features = create_lag_features(train_data)
        
        # Then, create lag features for future data using initial forecasts
        future_with_forecasts = future_data.copy()
        future_with_forecasts['value'] = list(exp_forecasts.values())
        future_with_features = create_lag_features(future_with_forecasts)
        
        # Combine the datasets
        combined_data = pd.concat([train_with_features, future_with_features])
        
        # Split back into train and future
        updated_train_data = combined_data[combined_data['year'] < 2020].copy()
        updated_future_data = combined_data[combined_data['year'] >= 2020].copy()
        
        # Step 3: Run top 3 models with updated features
        model_results = []
        successful_models = []  # Track successful models for ensemble
        
        for model_info in models:
            model_name = model_info['model']
            print(f"  Running {model_name} for {product} - {country}")
            
            # Define exogenous variables based on model type
            if model_name == 'sarimax':
                exog_cols = [col for col in updated_train_data.columns 
                           if col.endswith('_norm') or col in ['physicians_1k_inhabitants', 'population_female']]
            elif model_name in ['rf', 'xgb', 'lgb']:
                exog_cols = [col for col in updated_train_data.columns 
                           if col.endswith('_norm') or col in ['physicians_1k_inhabitants', 'population_female', 'value_lag_1']]
            else:
                exog_cols = [col for col in updated_train_data.columns 
                           if col.endswith('_norm') or col in ['physicians_1k_inhabitants', 'population_female', 'value_lag_1']]
            
            # Remove any columns that don't exist in the data
            exog_cols = [col for col in exog_cols if col in updated_train_data.columns]
            
            # Run the model
            if model_name == 'sarimax':
                result = run_sarimax(updated_train_data, updated_future_data, exog_cols)
            elif model_name == 'exp_smoothing':
                result = run_exp_smoothing(updated_train_data, updated_future_data)
            elif model_name == 'prophet':
                result = run_prophet(updated_train_data, updated_future_data, exog_cols)
            elif model_name == 'rf':
                result = run_ml_model(updated_train_data, updated_future_data, exog_cols, 'rf')
            elif model_name == 'xgb':
                result = run_ml_model(updated_train_data, updated_future_data, exog_cols, 'xgb')
            elif model_name == 'lgb':
                result = run_ml_model(updated_train_data, updated_future_data, exog_cols, 'lgb')
            else:
                print(f"  Unknown model: {model_name}")
                continue
            
            if result is not None:
                model_results.append({
                    'model': model_name,
                    'result': result
                })
                successful_models.append(model_name)  # Track successful models
                print(f"  {model_name} completed successfully")
            else:
                print(f"  {model_name} failed")
        
        # Step 4: Create ensemble and collect all predictions
        if len(model_results) > 0:
            print(f"  Creating ensemble for {product} - {country}")
            ensemble_result = ensemble_predictions(model_results)
            
            if ensemble_result is not None:
                # Collect predictions from all models including ensemble
                for model_result in model_results:
                    model_name = model_result['model']
                    # Get predictions for training years
                    train_preds = None
                    if model_name == 'sarimax':
                        train_preds = run_sarimax(updated_train_data, updated_train_data, exog_cols)
                    elif model_name == 'exp_smoothing':
                        # Use fittedvalues for training predictions
                        try:
                            model = ExponentialSmoothing(
                                updated_train_data['value'], 
                                trend='add', 
                                seasonal='add', 
                                seasonal_periods=4
                            )
                            fit = model.fit()
                            fitted = fit.fittedvalues
                            train_predictions = pd.DataFrame({
                                'year': updated_train_data['year'].values,
                                'actual': updated_train_data['value'].values,
                                'predicted': fitted.values,
                                'product': product,
                                'country': country,
                                'model': model_name,
                                'in_ensemble': True  # All successful models are in ensemble
                            })
                            train_predictions = train_predictions[['year', 'actual', 'predicted', 'product', 'country', 'model', 'in_ensemble']]
                            train_predictions = train_predictions.loc[:,~train_predictions.columns.duplicated()]
                            all_forecasts.append(train_predictions.reset_index(drop=True))
                        except Exception as e:
                            print(f"Exponential Smoothing (train) failed: {e}")
                        train_preds = None
                    elif model_name == 'prophet':
                        train_preds = run_prophet(updated_train_data, updated_train_data, exog_cols)
                    elif model_name == 'rf':
                        train_preds = run_ml_model(updated_train_data, updated_train_data, exog_cols, 'rf')
                    elif model_name == 'xgb':
                        train_preds = run_ml_model(updated_train_data, updated_train_data, exog_cols, 'xgb')
                    elif model_name == 'lgb':
                        train_preds = run_ml_model(updated_train_data, updated_train_data, exog_cols, 'lgb')
                    
                    # Collect training predictions for non-ExpSmoothing models
                    if train_preds is not None:
                        train_predictions = train_preds['predictions'].copy()
                        train_predictions['product'] = product
                        train_predictions['country'] = country
                        train_predictions['model'] = model_name
                        train_predictions['in_ensemble'] = True  # All successful models are in ensemble
                        train_predictions = train_predictions[['year', 'actual', 'predicted', 'product', 'country', 'model', 'in_ensemble']]
                        train_predictions = train_predictions.loc[:,~train_predictions.columns.duplicated()]
                        all_forecasts.append(train_predictions.reset_index(drop=True))
                    
                    # Collect forecast predictions
                    predictions = model_result['result']['predictions'].copy()
                    predictions['product'] = product
                    predictions['country'] = country
                    predictions['model'] = model_name
                    predictions['in_ensemble'] = True  # All successful models are in ensemble
                    predictions = predictions[['year', 'actual', 'predicted', 'product', 'country', 'model', 'in_ensemble']]
                    predictions = predictions.loc[:,~predictions.columns.duplicated()]
                    all_forecasts.append(predictions.reset_index(drop=True))
                
                # Add ensemble predictions (only for forecast years)
                ensemble_predictions_df = ensemble_result['predictions'].copy()
                ensemble_predictions_df['product'] = product
                ensemble_predictions_df['country'] = country
                ensemble_predictions_df['model'] = 'ensemble'
                ensemble_predictions_df['in_ensemble'] = True  # Ensemble is always in ensemble
                ensemble_predictions_df = ensemble_predictions_df.rename(columns={'ensemble_predicted': 'predicted'})
                ensemble_predictions_df['actual'] = np.nan
                ensemble_predictions_df = ensemble_predictions_df[['year', 'actual', 'predicted', 'product', 'country', 'model', 'in_ensemble']]
                ensemble_predictions_df = ensemble_predictions_df.loc[:,~ensemble_predictions_df.columns.duplicated()]
                all_forecasts.append(ensemble_predictions_df.reset_index(drop=True))
                
                # Create summary
                for year in [2020, 2021, 2022, 2023, 2024]:
                    year_data = ensemble_result['predictions'][ensemble_result['predictions']['year'] == year]
                    if not year_data.empty:
                        forecast_summary.append({
                            'product': product,
                            'country': country,
                            'year': year,
                            'forecast': year_data['ensemble_predicted'].iloc[0],
                            'models_used': len(model_results)
                        })
                
                print(f"  Ensemble completed for {product} - {country}")
            else:
                print(f"  Ensemble failed for {product} - {country}")
        else:
            print(f"  No successful models for {product} - {country}")
    
    # Save results
    if all_forecasts:
        all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        all_forecasts_df = all_forecasts_df[['year', 'actual', 'predicted', 'product', 'country', 'model', 'in_ensemble']]
        all_forecasts_df.to_csv('output/future_yearly_forecast_results/all_forecasts_2020_2024.csv', index=False)
        print(f"\nSaved all forecasts to output/future_yearly_forecast_results/all_forecasts_2020_2024.csv")
    
    if forecast_summary:
        summary_df = pd.DataFrame(forecast_summary)
        summary_df.to_csv('output/future_yearly_forecast_results/forecast_summary_2020_2024.csv', index=False)
        print(f"Saved forecast summary to output/future_yearly_forecast_results/forecast_summary_2020_2024.csv")
        
        # Print summary
        print("\nForecast Summary:")
        print(summary_df.to_string(index=False))
    
    print("\nYearly future forecasting for 2020-2024 complete!")

if __name__ == "__main__":
    main() 