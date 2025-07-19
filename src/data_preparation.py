import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.preprocessing import StandardScaler


def to_snake_case(s: str) -> str:
    """Convert a string to snake_case."""
    import re
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'[^a-zA-Z0-9]+', '_', s)
    return s.lower().strip('_')


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column names to snake_case."""
    df.columns = [to_snake_case(col) for col in df.columns]
    return df


def load_revenue_data(path: str) -> pd.DataFrame:
    """Load and clean the revenue dataset."""
    df = pd.read_csv(path)
    df = standardize_columns(df)
    # Rename the original country column
    df = df.rename(columns={'country': 'Country_actual'})
    # Map Country_A-E to real country names in a new 'country' column
    country_map = {
        'Country_A': 'India',
        'Country_B': 'Germany',
        'Country_C': 'Canada',
        'Country_D': 'Mexico',
        'Country_E': 'United States',
    }
    df['country'] = df['Country_actual'].map(country_map)
    return df


def load_physicians_data(path: str) -> pd.DataFrame:
    """Load and clean the physicians dataset as specified."""
    df = pd.read_csv(path)
    df = standardize_columns(df)
    # Debug: print columns and head
    print("\nPhysicians DataFrame columns after standardization:", df.columns.tolist())
    print(df.head())
    # Rename columns
    df = df.rename(columns={
        'reference_area': 'country',
        'time_period': 'year',
        'obs_value': 'physicians_1k_inhabitants'
    })
    # Filter for unit_of_measure == 'Per 1 000 inhabitants'
    df = df[df['unit_of_measure'] == 'Per 1 000 inhabitants']
    # Keep only relevant columns
    df = df[['country', 'year', 'physicians_1k_inhabitants']]
    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]
    # Debug: print columns and head after renaming and filtering
    print("\nPhysicians DataFrame after renaming and filtering:", df.columns.tolist())
    print(df.head())
    # Robustly extract year as int (handles string or mixed types)
    df['year'] = pd.to_numeric(df['year'].astype(str).str[:4], errors='coerce')
    df = df[df['year'].between(2011, 2024)]
    df['year'] = df['year'].astype(int)
    return df


def load_femalepop_data(path: str) -> pd.DataFrame:
    """Load and clean the female population dataset as specified."""
    df = pd.read_csv(path)
    df = standardize_columns(df)
    df = df.rename(columns={'country_name': 'country'})
    # Only use columns that are 4-digit years between 2011 and 2024
    year_cols = [col for col in df.columns if col.isdigit() and 2011 <= int(col) <= 2024]
    id_vars = ['country']
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='year', value_name='population_female')
    df_long['year'] = df_long['year'].astype(int)
    # Filter for years 2011-2024 (redundant but safe)
    df_long = df_long[df_long['year'].between(2011, 2024)]
    return df_long[['country', 'year', 'population_female']]


def load_worldbank_data(path: str, indicator_name: str) -> pd.DataFrame:
    """Load and clean World Bank data files (internet_usage, health_exp_pcnt_gdp, etc.)."""
    df = pd.read_csv(path, skiprows=4)  # Skip first 4 rows as data starts from row 5
    df = standardize_columns(df)
    
    # Rename country name to country
    df = df.rename(columns={'country_name': 'country'})
    
    # Only use columns that are 4-digit years between 2011 and 2024
    year_cols = [col for col in df.columns if col.isdigit() and 2011 <= int(col) <= 2024]
    id_vars = ['country']
    
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='year', value_name=indicator_name)
    df_long['year'] = df_long['year'].astype(int)
    
    # Filter for years 2011-2024
    df_long = df_long[df_long['year'].between(2011, 2024)]
    
    return df_long[['country', 'year', indicator_name]]


def load_internet_usage_data(path: str) -> pd.DataFrame:
    """Load and clean internet usage data."""
    return load_worldbank_data(path, 'internet_usage')


def load_health_exp_pcnt_gdp_data(path: str) -> pd.DataFrame:
    """Load and clean health expenditure as percentage of GDP data."""
    return load_worldbank_data(path, 'health_exp_pcnt_gdp')


def load_health_exp_percapita_data(path: str) -> pd.DataFrame:
    """Load and clean health expenditure per capita data."""
    return load_worldbank_data(path, 'health_exp_percapita')


def load_population_total_data(path: str) -> pd.DataFrame:
    """Load and clean total population data."""
    return load_worldbank_data(path, 'population_total')


def load_population_65plus_data(path: str) -> pd.DataFrame:
    """Load and clean population 65+ data."""
    return load_worldbank_data(path, 'population_65plus')


def load_population_urban_data(path: str) -> pd.DataFrame:
    """Load and clean urban population data."""
    return load_worldbank_data(path, 'population_urban')


def show_missing_data_report(df: pd.DataFrame, name: str) -> None:
    """Show missing/null data summary for a DataFrame using pandas."""
    print(f"\nMissing/null data summary for {name}:")
    print(df.info())
    print(df.isnull().sum())
    print(df.describe(include='all'))


def clean_exogenous_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean exogenous columns (remove $, commas, convert to float) for both monthly and yearly."""
    exog_cols = [col for col in df.columns if col not in ['product', 'country', 'year', 'month', 'value', 'Country_actual']]
    for col in exog_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def prepare_all_data(
    revenue_path: str = 'data/CASE1A_MonthlyData.csv',
    physicians_path: str = 'data/OECD_Physicians_data.csv',
    femalepop_path: str = 'data/WorldBank_FemalePopulation.csv',
    internet_usage_path: str = 'data/internet_usage.csv',
    health_exp_pcnt_gdp_path: str = 'data/health_exp_pcnt_gdp.csv',
    health_exp_percapita_path: str = 'data/health_exp_percapita.csv',
    population_total_path: str = 'data/population_total.csv',
    population_65plus_path: str = 'data/population_65plus.csv',
    population_urban_path: str = 'data/population_urban.csv',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load, clean, and merge all datasets. Show missing data reports for each.
    Returns the cleaned individual DataFrames and the merged DataFrame.
    """
    revenue = load_revenue_data(revenue_path)
    show_missing_data_report(revenue, 'Revenue')
    
    physicians = load_physicians_data(physicians_path)
    show_missing_data_report(physicians, 'Physicians')
    
    femalepop = load_femalepop_data(femalepop_path)
    show_missing_data_report(femalepop, 'Population Female')
    
    internet_usage = load_internet_usage_data(internet_usage_path)
    show_missing_data_report(internet_usage, 'Internet Usage')
    
    health_exp_pcnt_gdp = load_health_exp_pcnt_gdp_data(health_exp_pcnt_gdp_path)
    show_missing_data_report(health_exp_pcnt_gdp, 'Health Expenditure % GDP')
    
    health_exp_percapita = load_health_exp_percapita_data(health_exp_percapita_path)
    show_missing_data_report(health_exp_percapita, 'Health Expenditure Per Capita')
    
    population_total = load_population_total_data(population_total_path)
    show_missing_data_report(population_total, 'Total Population')
    
    population_65plus = load_population_65plus_data(population_65plus_path)
    show_missing_data_report(population_65plus, 'Population 65+')
    
    population_urban = load_population_urban_data(population_urban_path)
    show_missing_data_report(population_urban, 'Urban Population')
    
    # Only keep exogenous data for countries in revenue file
    revenue_countries = revenue['country'].dropna().unique().tolist()
    revenue_products = revenue['product'].dropna().unique().tolist()
    
    # Build a full product-country-year-month grid for all revenue products/countries, all years 2011-2024, all months 1-12
    exo_years = list(range(2011, 2025))
    exo_months = list(range(1, 13))
    grid = pd.MultiIndex.from_product(
        [revenue_products, revenue_countries, exo_years, exo_months],
        names=['product', 'country', 'year', 'month']
    ).to_frame(index=False)
    
    # Merge all exogenous sources into a country-year-month grid
    exo_grid = pd.MultiIndex.from_product(
        [revenue_countries, exo_years, exo_months],
        names=['country', 'year', 'month']
    ).to_frame(index=False)
    exo_merged = exo_grid.copy()
    
    # Merge all exogenous datasets
    exogenous_datasets = [
        physicians, femalepop, internet_usage, health_exp_pcnt_gdp, 
        health_exp_percapita, population_total, population_65plus, population_urban
    ]
    
    for df in exogenous_datasets:
        exo_merged = pd.merge(exo_merged, df, on=['country', 'year'], how='left')
    
    # Clean exogenous columns
    exo_merged = clean_exogenous_columns(exo_merged)
    
    # Create total_population_internetusage variable
    if 'internet_usage' in exo_merged.columns and 'population_total' in exo_merged.columns:
        exo_merged['total_population_internetusage'] = (
            exo_merged['internet_usage'] / 100.0 * exo_merged['population_total']
        )
    
    # Fill missing values in exogenous columns using forward fill (by country), then backward fill
    exog_cols = [col for col in exo_merged.columns if col not in ['country', 'year', 'month']]
    exo_merged[exog_cols] = exo_merged.groupby('country')[exog_cols].ffill().bfill()
    
    # Normalize exogenous columns (excluding internet_usage)
    scaler = StandardScaler()
    for col in exog_cols:
        if col in exo_merged.columns and col != 'internet_usage':
            exo_merged[f'{col}_norm'] = scaler.fit_transform(exo_merged[[col]].values)
    
    # Merge exogenous data into the full product-country grid
    merged = pd.merge(grid, exo_merged, on=['country', 'year', 'month'], how='left')
    
    # Merge revenue data (which is monthly) into merged
    revenue_monthly = revenue.copy()
    revenue_monthly['month'] = revenue_monthly['month'].astype(int)
    merged = pd.merge(merged, revenue_monthly[['product', 'country', 'year', 'month', 'value', 'Country_actual']],
                     on=['product', 'country', 'year', 'month'], how='left')
    
    # Reorder columns for clarity
    col_order = ['product', 'country', 'Country_actual', 'year', 'month', 'value'] + [c for c in merged.columns if c not in ['product', 'country', 'Country_actual', 'year', 'month', 'value']]
    merged = merged[[c for c in col_order if c in merged.columns]]
    
    show_missing_data_report(merged, 'Merged Data')
    
    return revenue, physicians, femalepop, merged, exo_merged


if __name__ == '__main__':
    _, _, _, merged, _ = prepare_all_data()
    # Save monthly master file
    merged.to_csv('output/master_monthly.csv', index=False)
    print('\nSaved master_monthly.csv')
    
    # Aggregate to yearly: sum revenue, mean exogenous, per product-country-year
    group_cols = ['product', 'country', 'year']
    # Sum revenue (value)
    revenue_yearly = merged.groupby(group_cols)['value'].sum().reset_index()
    # Mean exogenous variables (exclude lags/rolls, month, Country_actual)
    exog_cols = [col for col in merged.columns if col not in ['product', 'country', 'year', 'month', 'value', 'Country_actual']]
    if exog_cols:
        exog_yearly = merged.groupby(group_cols)[exog_cols].mean().reset_index()
        yearly = pd.merge(revenue_yearly, exog_yearly, on=group_cols, how='left')
    else:
        yearly = revenue_yearly.copy()
    # Add Country_actual for reference (first available in group)
    yearly = pd.merge(yearly, merged[['product', 'country', 'year', 'Country_actual']].drop_duplicates(), on=group_cols, how='left')
    yearly.to_csv('output/master_yearly.csv', index=False)
    print('Saved master_yearly.csv')
    print('\nTop 10 rows of the master_monthly.csv:')
    print(merged.head(10))
    print('\nTop 10 rows of the master_yearly.csv:')
    print(yearly.head(10)) 