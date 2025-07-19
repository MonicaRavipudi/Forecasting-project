import pandas as pd
import numpy as np
import warnings
import json
import os
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import shap

warnings.filterwarnings('ignore')

def load_metrics_and_get_top_models():
    """Load metrics and get top 3 models for each product-country combination"""
    metrics_df = pd.read_csv('output/yearly_forecast_results/all_metrics.csv')
    
    # Filter out ensemble models and get top 3 by MAPE for each product-country
    individual_metrics = metrics_df[metrics_df['model'] != 'ensemble'].copy()
    
    top_models = []
    for (product, country), group in individual_metrics.groupby(['product', 'country']):
        # Sort by MAPE and get top 3
        top_3 = group.nsmallest(3, 'mape')
        
        for _, row in top_3.iterrows():
            top_models.append({
                'product': product,
                'country': country,
                'model': row['model'],
                'mape': row['mape'],
                'exo_vars': [row['exo_var_1'], row['exo_var_2'], row['exo_var_3'], 
                           row['exo_var_4'], row['exo_var_5']]
            })
    
    return top_models

def load_trained_models():
    """Load the trained models from pickle files"""
    models = {}
    
    # Get all pickle files in yearly_forecast_results
    for filename in os.listdir('output/yearly_forecast_results'):
        if filename.endswith('.pkl') and not filename.startswith('ensemble'):
            # Parse filename: model_Product_Country_model.pkl
            parts = filename.replace('.pkl', '').split('_')
            if len(parts) >= 4:
                model_type = parts[0]
                # Reconstruct product and country names (they may contain spaces)
                if len(parts) >= 4:
                    # Handle cases where country name has spaces (e.g., "United States")
                    if len(parts) == 4:
                        product = parts[1]
                        country = parts[2]
                    else:
                        # Handle multi-word country names
                        product = parts[1]
                        country = '_'.join(parts[2:-1])  # All parts except first and last
                
                key = f"{product}_{country}_{model_type}"
                
                try:
                    with open(f'output/yearly_forecast_results/{filename}', 'rb') as f:
                        models[key] = pickle.load(f)
                except Exception as e:
                    print(f"Could not load {filename}: {e}")
    
    return models

def run_sarimax_scenario(train_data, test_data, exog_cols, scenario_data):
    """Run SARIMAX model with scenario data"""
    try:
        # Prepare data
        train_y = train_data['value'].values
        train_exog = train_data[exog_cols].values if exog_cols else None
        test_exog = scenario_data[exog_cols].values if exog_cols else None
        
        # Fit model
        model = SARIMAX(train_y, exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        fitted_model = model.fit(disp=False)
        
        # Forecast
        forecast = fitted_model.forecast(steps=len(scenario_data), exog=test_exog)
        return np.array(forecast)  # Convert to numpy array for consistent indexing
    except Exception as e:
        print(f"SARIMAX scenario failed: {e}")
        return None

def run_exp_smoothing_scenario(train_data, test_data):
    """Run Exponential Smoothing model (no exogenous variables)"""
    try:
        # Fit model
        model = ExponentialSmoothing(
            train_data['value'], 
            trend='add', 
            seasonal='add', 
            seasonal_periods=4
        )
        fitted_model = model.fit()
        
        # Forecast (exponential smoothing doesn't use exogenous variables)
        forecast = fitted_model.forecast(steps=len(test_data))
        return np.array(forecast)  # Convert to numpy array for consistent indexing
    except Exception as e:
        print(f"Exponential Smoothing scenario failed: {e}")
        return None

def run_prophet_scenario(train_data, test_data, exog_cols):
    """Run Prophet model with scenario data"""
    try:
        # Prepare data for Prophet
        train_df = train_data[['year', 'value'] + exog_cols].copy()
        train_df['ds'] = pd.to_datetime(train_df['year'].astype(str) + '-01-01')
        train_df['y'] = train_df['value']
        
        test_df = test_data[['year'] + exog_cols].copy()
        test_df['ds'] = pd.to_datetime(test_df['year'].astype(str) + '-01-01')
        
        # Create and fit model
        model = Prophet()
        
        # Add regressors
        for col in exog_cols:
            if col in train_df.columns and not train_df[col].isna().all():
                model.add_regressor(col)
        
        model.fit(train_df)
        
        # Make forecast
        forecast = model.predict(test_df)
        return forecast['yhat'].values
    except Exception as e:
        print(f"Prophet scenario failed: {e}")
        return None

def run_ml_model_scenario(train_data, test_data, exog_cols, model_type):
    """Run ML model with scenario data"""
    try:
        # Prepare data
        X_train = train_data[exog_cols].values
        y_train = train_data['value'].values
        X_test = test_data[exog_cols].values
        
        # Create and fit model
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lgb':
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        else:
            return None
        
        model.fit(X_train, y_train)
        
        # Predict
        forecast = model.predict(X_test)
        return forecast
    except Exception as e:
        print(f"{model_type.upper()} scenario failed: {e}")
        return None

def generate_shap_values(model, X_train, X_test):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        return shap_values, explainer
    except Exception as e:
        print(f"SHAP value generation failed: {e}")
        return None, None

def generate_comprehensive_scenario_forecasts(train_data, test_data, model_type, exog_cols, test_years):
    """Generate comprehensive scenario forecasts for each model"""
    scenario_results = []
    
    # Define percentage-based scenarios
    percentage_scenarios = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    
    # Define point-based scenarios (for percentage variables)
    point_scenarios = [-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    # Percentage-based variables
    percentage_vars = ['physicians_1k_inhabitants', 'population_female', 'population_total', 
                      'population_urban', 'total_population_internetusage']
    
    # Point-based variables (percentage values)
    point_vars = ['health_exp_percapita', 'internet_usage', 'health_exp_pcnt_gdp']
    
    # Additional variables used by models
    additional_vars = ['physicians_1k_inhabitants_norm', 'population_female_norm', 
                      'health_exp_percapita_norm', 'internet_usage_norm', 'health_exp_pcnt_gdp_norm',
                      'population_total_norm', 'population_urban_norm', 'total_population_internetusage_norm']
    
    # Test percentage-based scenarios
    for var in percentage_vars:
        if var in test_data.columns:
            for scenario in percentage_scenarios:
                # Create scenario data
                scenario_data = test_data.copy()
                
                # Apply percentage change
                scenario_data[var] = scenario_data[var] * (1 + scenario / 100)
                
                # Make prediction based on model type
                if model_type == 'sarimax':
                    forecast = run_sarimax_scenario(train_data, scenario_data, exog_cols, scenario_data)
                elif model_type == 'exp_smoothing':
                    forecast = run_exp_smoothing_scenario(train_data, scenario_data)
                elif model_type == 'prophet':
                    forecast = run_prophet_scenario(train_data, scenario_data, exog_cols)
                elif model_type in ['rf', 'xgb', 'lgb']:
                    forecast = run_ml_model_scenario(train_data, scenario_data, exog_cols, model_type)
                else:
                    forecast = None
                
                # Store results for each year
                if forecast is not None:
                    for i, year in enumerate(test_years):
                        scenario_results.append({
                            'testing_column': var,
                            'scenario': f"{scenario}%",
                            'forecast': float(forecast[i]),
                            'year': year
                        })
                else:
                    for year in test_years:
                        scenario_results.append({
                            'testing_column': var,
                            'scenario': f"{scenario}%",
                            'forecast': np.nan,
                            'year': year
                        })
    
    # Test point-based scenarios
    for var in point_vars:
        if var in test_data.columns:
            for scenario in point_scenarios:
                # Create scenario data
                scenario_data = test_data.copy()
                
                # Apply point change
                scenario_data[var] = scenario_data[var] + scenario
                
                # Make prediction based on model type
                if model_type == 'sarimax':
                    forecast = run_sarimax_scenario(train_data, scenario_data, exog_cols, scenario_data)
                elif model_type == 'exp_smoothing':
                    forecast = run_exp_smoothing_scenario(train_data, scenario_data)
                elif model_type == 'prophet':
                    forecast = run_prophet_scenario(train_data, scenario_data, exog_cols)
                elif model_type in ['rf', 'xgb', 'lgb']:
                    forecast = run_ml_model_scenario(train_data, scenario_data, exog_cols, model_type)
                else:
                    forecast = None
                
                # Store results for each year
                if forecast is not None:
                    for i, year in enumerate(test_years):
                        scenario_results.append({
                            'testing_column': var,
                            'scenario': f"{scenario:+} points",
                            'forecast': float(forecast[i]),
                            'year': year
                        })
                else:
                    for year in test_years:
                        scenario_results.append({
                            'testing_column': var,
                            'scenario': f"{scenario:+} points",
                            'forecast': np.nan,
                            'year': year
                        })
    
    # Test additional normalized variables (small changes since they're normalized)
    for var in additional_vars:
        if var in test_data.columns:
            # Use smaller changes for normalized variables
            norm_scenarios = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
            for scenario in norm_scenarios:
                # Create scenario data
                scenario_data = test_data.copy()
                
                # Apply point change to normalized variable
                scenario_data[var] = scenario_data[var] + scenario
                
                # Make prediction based on model type
                if model_type == 'sarimax':
                    forecast = run_sarimax_scenario(train_data, scenario_data, exog_cols, scenario_data)
                elif model_type == 'exp_smoothing':
                    forecast = run_exp_smoothing_scenario(train_data, scenario_data)
                elif model_type == 'prophet':
                    forecast = run_prophet_scenario(train_data, scenario_data, exog_cols)
                elif model_type in ['rf', 'xgb', 'lgb']:
                    forecast = run_ml_model_scenario(train_data, scenario_data, exog_cols, model_type)
                else:
                    forecast = None
                
                # Store results for each year
                if forecast is not None:
                    for i, year in enumerate(test_years):
                        scenario_results.append({
                            'testing_column': var,
                            'scenario': f"{scenario:+} points",
                            'forecast': float(forecast[i]),
                            'year': year
                        })
                else:
                    for year in test_years:
                        scenario_results.append({
                            'testing_column': var,
                            'scenario': f"{scenario:+} points",
                            'forecast': np.nan,
                            'year': year
                        })
    
    return scenario_results

def get_norm_stats(train_data, actual_var):
    mean = train_data[actual_var].mean()
    std = train_data[actual_var].std()
    return mean, std

def actual_to_norm(val, mean, std):
    return (val - mean) / std

def run_ml_model(train_data, test_data, exog_cols, model_type):
    X_train = train_data[exog_cols].values
    y_train = train_data['value'].values
    X_test = test_data[exog_cols].values
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    elif model_type == 'lgb':
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    else:
        return None, None
    model.fit(X_train, y_train)
    base_forecast = model.predict(X_test)
    shap_values, _ = generate_shap_values(model, X_train, X_test)
    return model, base_forecast, shap_values

def main():
    print("Starting Yearly Scenario Analysis...")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('output/feature_yearly.csv')
    
    # Load top models
    print("Loading top models...")
    top_models = load_metrics_and_get_top_models()
    
    # Load trained models
    print("Loading trained models...")
    trained_models = load_trained_models()
    
    # Create output directory
    os.makedirs('output/yearly_scenario_results', exist_ok=True)
    
    all_scenario_results = []
    scenario_actual_vars = [
        'physicians_1k_inhabitants', 'population_female', 'health_exp_percapita',
        'internet_usage', 'health_exp_pcnt_gdp', 'population_total',
        'population_urban', 'total_population_internetusage'
    ]
    scenario_types = {
        'percentage': ['physicians_1k_inhabitants', 'population_female', 'population_total', 'population_urban', 'total_population_internetusage'],
        'point': ['health_exp_percapita', 'internet_usage', 'health_exp_pcnt_gdp']
    }
    
    # Process each product-country combination
    for product in df['product'].unique():
        for country in df[df['product'] == product]['country'].unique():
            if pd.isna(country):
                continue
                
            print(f"\nProcessing {product} - {country}")
            
            # Filter data for this product-country
            product_country_data = df[(df['product'] == product) & (df['country'] == country)].copy()
            
            # Split data: use all data up to 2019 for training, 2020-2024 for testing
            train_data = product_country_data[product_country_data['year'] < 2020].copy()
            test_data = product_country_data[product_country_data['year'] >= 2020].copy()
            
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"No data available for {product} - {country}")
                continue
            
            # Get top 3 models for this product-country
            product_country_models = [m for m in top_models if m['product'] == product and m['country'] == country]
            
            if not product_country_models:
                print(f"No models found for {product} - {country}")
                continue
            
            # Process each model
            for model_info in product_country_models[:3]:  # Top 3 models
                model_type = model_info['model']
                mape = model_info['mape']
                exo_vars = [var for var in model_info['exo_vars'] if pd.notna(var) and var != '']
                
                print(f"  Processing {model_type} model...")
                
                # Get available exogenous variables
                available_exo_vars = [var for var in exo_vars if var in train_data.columns and var.endswith('_norm')]
                
                if not available_exo_vars and model_type != 'exp_smoothing':
                    print(f"    No available exogenous variables for {model_type}")
                    continue
                
                # SHAP for ML models
                top_influential_vars = set()
                base_forecast = None
                if model_type in ['rf', 'xgb', 'lgb']:
                    model, base_forecast, shap_values = run_ml_model(train_data, test_data, available_exo_vars, model_type)
                    if shap_values is not None:
                        # Mean absolute SHAP value per feature
                        mean_abs_shap = np.abs(shap_values).mean(axis=0)
                        feature_importance = pd.Series(mean_abs_shap, index=available_exo_vars)
                        top_influential_vars = set(feature_importance.nlargest(5).index) # Changed to 5
                        # Map _norm features to actual variable
                        norm_to_actual = {f'{v}_norm': v for v in scenario_actual_vars}
                        for norm_var in top_influential_vars:
                            if norm_var not in norm_to_actual:
                                continue
                            actual_var = norm_to_actual[norm_var]
                            if actual_var in scenario_types['percentage']:
                                scenarios = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
                                scenario_label = lambda s: f"{s}%"
                                apply_scenario = lambda x, s: x * (1 + s / 100)
                            else:
                                scenarios = [-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]
                                scenario_label = lambda s: f"{s:+} points"
                                apply_scenario = lambda x, s: x + s
                            mean, std = get_norm_stats(train_data, actual_var)
                            for scenario in scenarios:
                                scenario_test = test_data.copy()
                                for i, year in enumerate(scenario_test['year'].values):
                                    orig_actual = scenario_test.iloc[i][actual_var]
                                    new_actual = apply_scenario(orig_actual, scenario)
                                    new_norm = actual_to_norm(new_actual, mean, std)
                                    scenario_test.at[scenario_test.index[i], norm_var] = new_norm
                                scenario_pred = model.predict(scenario_test[available_exo_vars].values)
                                for i, year in enumerate(scenario_test['year'].values):
                                    base_val = float(base_forecast[i])
                                    scenario_val = float(scenario_pred[i])
                                    all_scenario_results.append({
                                        'product': product,
                                        'country': country,
                                        'model': model_type,
                                        'mape': mape,
                                        'year': year,
                                        'testing_column': actual_var,
                                        'scenario': scenario_label(scenario),
                                        'forecast': base_val,
                                        'scenario_forecast': scenario_val,
                                        'influence_var': True
                                    })
                else:
                    # For non-ML models, get base forecast for test years
                    if model_type == 'sarimax':
                        base_forecast = run_sarimax_scenario(train_data, test_data, available_exo_vars, test_data)
                    elif model_type == 'exp_smoothing':
                        base_forecast = run_exp_smoothing_scenario(train_data, test_data)
                    elif model_type == 'prophet':
                        base_forecast = run_prophet_scenario(train_data, test_data, available_exo_vars)
                
                # Generate comprehensive scenario forecasts
                print(f"    Generating comprehensive scenario forecasts...")
                scenario_results = generate_comprehensive_scenario_forecasts(
                    train_data, test_data, model_type, available_exo_vars, test_data['year'].values
                )
                
                # Store scenario results with new columns
                for idx, scenario_result in enumerate(scenario_results):
                    year = scenario_result['year']
                    var = scenario_result['testing_column']
                    scenario_val = scenario_result['scenario']
                    # Find the index for this year in test_data
                    try:
                        year_idx = list(test_data['year'].values).index(year)
                        base_val = float(base_forecast[year_idx]) if base_forecast is not None else np.nan
                    except Exception:
                        base_val = np.nan
                    influence_var = (var in top_influential_vars) if model_type in ['rf', 'xgb', 'lgb'] else False
                    scenario_result.update({
                        'product': product,
                        'country': country,
                        'model': model_type,
                        'mape': mape,
                        'forecast': base_val,
                        'scenario_forecast': scenario_result['forecast'],
                        'influence_var': influence_var
                    })
                    all_scenario_results.append(scenario_result)
                
                print(f"    Completed {model_type} analysis")
    
    # Save results
    if all_scenario_results:
        scenario_df = pd.DataFrame(all_scenario_results)
        scenario_df.to_csv('output/yearly_scenario_results/comprehensive_scenarios.csv', index=False)
        print(f"Comprehensive scenario forecasts saved to: output/yearly_scenario_results/comprehensive_scenarios.csv")
        
        # Create scenario comparison summary
        scenario_summary = scenario_df.groupby(['product', 'country', 'model', 'testing_column', 'scenario'])['scenario_forecast'].agg(['mean', 'std', 'min', 'max']).reset_index()
        scenario_summary.to_csv('output/yearly_scenario_results/scenario_summary.csv', index=False)
        print(f"Scenario summary saved to: output/yearly_scenario_results/scenario_summary.csv")
        
        # Create impact analysis
        impact_analysis = scenario_df.groupby(['product', 'country', 'model', 'testing_column'])['scenario_forecast'].agg(['mean', 'std', 'min', 'max']).reset_index()
        impact_analysis.to_csv('output/yearly_scenario_results/impact_analysis.csv', index=False)
        print(f"Impact analysis saved to: output/yearly_scenario_results/impact_analysis.csv")
    
    print(f"\nScenario analysis completed!")
    
    # Print summary statistics
    if all_scenario_results:
        print(f"\nComprehensive Scenario Analysis Summary:")
        print(f"Total scenario predictions: {len(all_scenario_results)}")
        print(f"Testing columns: {scenario_df['testing_column'].nunique()}")
        print(f"Scenarios tested: {scenario_df['scenario'].nunique()}")
        print(f"Years forecasted: {scenario_df['year'].nunique()}")
        print(f"Percentage-based variables: {len([col for col in scenario_df['testing_column'].unique() if 'physicians' in col or 'population' in col or 'internetusage' in col])}")
        print(f"Point-based variables: {len([col for col in scenario_df['testing_column'].unique() if 'health_exp' in col or 'internet_usage' in col or 'urban' in col])}")

if __name__ == "__main__":
    main() 