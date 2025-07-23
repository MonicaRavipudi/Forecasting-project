# Revenue Forecasting Project

A comprehensive time series forecasting system for pharmaceutical revenue prediction using multiple machine learning and statistical models.

## 📋 Project Overview

This project implements a robust forecasting pipeline for predicting pharmaceutical revenue across multiple products and countries. It combines traditional statistical methods (SARIMAX, Exponential Smoothing) with modern machine learning approaches (Random Forest, XGBoost, LightGBM, Prophet) to generate accurate revenue forecasts.


## 🚀 Features

### **Multi-Model Forecasting**
- **SARIMAX**: Seasonal ARIMA with exogenous variables
- **Exponential Smoothing**: Holt-Winters seasonal method
- **Prophet**: Facebook's time series forecasting tool
- **Random Forest**: Ensemble tree-based method
- **XGBoost**: Gradient boosting implementation
- **LightGBM**: Light gradient boosting machine

### **Comprehensive Data Integration**
- Revenue data from multiple products and countries
- World Bank indicators (population, health expenditure, internet usage)
- OECD physician data
- Automatic data cleaning and normalization

### **Feature Engineering**
- Lag features (1, 2, 3 periods)
- Rolling mean features (2, 3 periods)
- Exogenous variable normalization
- Combined features (e.g., total population × internet usage)

### **Ensemble Forecasting**
- Automatic selection of top 3 performing models
- Weighted ensemble based on inverse MAPE
- Comprehensive model evaluation metrics

## 📊 Model Performance

### **Evaluation Metrics**
- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination

### **Model Comparison**
The system evaluates all models and automatically selects the best performers for ensemble creation.

### **Scenario Analysis**
- Sensitivity analysis across multiple variables
- Percentage-based scenarios (-25% to +25%)
- Point-based scenarios for percentage variables
- Impact analysis for strategic decision-making


## 🎯 Key Features

### **Data Leakage Prevention**
- Proper train-test splits (2011-2019 for training, 2020-2024 for forecasting)
- Separate lag feature creation for training vs. forecasting data
- Exponential smoothing used only for forecasting lag features

### **Robust Error Handling**
- Graceful handling of model failures
- NaN value management
- Automatic fallback mechanisms

### **Scalability**
- Supports multiple products and countries
- Modular design for easy extension
- Efficient data processing pipeline

## 🔍 Scenario Analysis

The scenario analysis tests the impact of changes in exogenous variables:

### **Percentage Changes** (-25% to +25%)
- Physicians per 1000 inhabitants
- Population variables
- Internet usage

### **Point Changes** (-0.25 to +0.25)
- Health expenditure variables
- Percentage-based indicators

### **Normalized Variables** (-0.5 to +0.5)
- All normalized exogenous variables

## 📊 Data Sources

| Source | Description | Variables |
|--------|-------------|-----------|
| **Revenue Data** | Monthly revenue by product-country | Revenue values, product, country, year, month |
| **World Bank** | Economic and demographic indicators | Population, health expenditure, internet usage |
| **OECD** | Healthcare statistics | Physicians per 1000 inhabitants |
| **Combined Features** | Derived variables | Total population × internet usage |


## 📝 Notes

- **Data Privacy**: All data files are excluded from version control
- **Model Persistence**: Trained models are saved as pickle files
- **Reproducibility**: Random seeds are set for consistent results
- **Performance**: Prophet models may take longer due to MCMC sampling


**Last Updated**: July 2025
**Version**: 1.0.0 
