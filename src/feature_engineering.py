import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List

def load_master_dataset(path: str) -> pd.DataFrame:
    """Load the master dataset from CSV."""
    return pd.read_csv(path)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add quarter and is_year_end columns based on month and year."""
    if 'month' in df.columns:
        df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
        df['is_year_end'] = (df['month'] == 12).astype(int)
    return df

def add_lag_features(df: pd.DataFrame, group_cols: List[str], target_col: str, lags: List[int]) -> pd.DataFrame:
    """Add lag features for the target column for each group."""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, group_cols: List[str], target_col: str, windows: List[int]) -> pd.DataFrame:
    """Add rolling mean features for the target column for each group."""
    for window in windows:
        df[f'{target_col}_rollmean_{window}'] = df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window, min_periods=1).mean())
    return df

def feature_engineering_monthly():
    df = load_master_dataset('output/master_monthly.csv')
    df = add_time_features(df)
    group_cols = ['country', 'product']
    # Lag and rolling features for revenue (value)
    df = add_lag_features(df, group_cols, 'value', lags=[3, 6, 12])
    df = add_rolling_features(df, group_cols, 'value', windows=[3, 6, 12])
    # Exogenous columns to normalize only
    exo_cols = [
        'physicians_1k_inhabitants', 'population_female', 'health_exp_percapita',
        'internet_usage', 'population_urban', 'population_total',
        'population_65plus', 'total_population_internetusage'
    ]
    exo_cols = [col for col in exo_cols if col in df.columns]
    # No cleaning or normalization here; already handled in data_preparation.py
    # Only drop rows with NaN/inf in lag/rolling features where value is present
    exog_cols = [col for col in df.columns if col.endswith('_norm') or 'lag' in col or 'rollmean' in col or col in ['quarter', 'is_year_end']]
    # Mask for rows where value is present
    mask_value_present = df['value'].notnull()
    # For those rows, drop if any lag/roll/exog is NaN/inf
    mask_bad = (df[exog_cols].isnull().any(axis=1) | np.isinf(df[exog_cols]).any(axis=1)) & mask_value_present
    df = df[~mask_bad].reset_index(drop=True)
    df.to_csv('output/feature_monthly.csv', index=False)
    print('Monthly feature engineering complete. Saved as output/feature_monthly.csv')

def feature_engineering_yearly():
    df = load_master_dataset('output/master_yearly.csv')
    group_cols = ['country', 'product']
    # Add lag_1 for revenue (value)
    df['value_lag_1'] = df.groupby(group_cols)['value'].shift(1)
    # Exogenous columns: all except revenue, product, country, year
    exo_cols = [col for col in df.columns if col not in ['product', 'country', 'year', 'value', 'value_lag_1', 'Country_actual']]
    # Only drop rows with NaN/inf in lag_1 where value is present
    mask_value_present = df['value'].notnull()
    mask_bad = (df['value_lag_1'].isnull() | np.isinf(df['value_lag_1'])) & mask_value_present
    df = df[~mask_bad].reset_index(drop=True)
    df.to_csv('output/feature_yearly.csv', index=False)
    print('Yearly feature engineering complete. Saved as output/feature_yearly.csv')

def main():
    feature_engineering_monthly()
    feature_engineering_yearly()

if __name__ == '__main__':
    main() 