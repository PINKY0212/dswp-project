# Short-Term Electricity Demand Forecasting using Statistical and Deep Learning Models

A comprehensive comparative study of SARIMA, LSTM, hybrid SARIMA+LSTM, and ensemble approaches for short-term electricity demand forecasting, with a special focus on peak-load stability and model explainability using SHAP.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Motivation](#motivation)  
3. [Problem Statement & Goals](#problem-statement--goals)  
4. [Research Questions](#research-questions)  
5. [Data](#data)  
   - [Source](#source)  
   - [Pre-processing](#pre-processing)  
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
7. [Methods](#methods)  
   - [Model Pipelines](#model-pipelines)  
   - [Hybrid & Ensemble Focus](#hybrid--ensemble-focus)  
   - [Explainability (XAI)](#explainability-xai)  
8. [Evaluation Strategy](#evaluation-strategy)  
9. [Repository Structure](#repository-structure)  
10. [Getting Started](#getting-started)  
11. [Results & Outputs](#results--outputs)  
12. [References](#references)  

---

## Project Overview

Short-term electricity demand forecasting is a core task for transmission system and grid operators. Modern electricity systems are influenced by complex, non-linear interactions between demand, weather, generation mix, and consumer behavior. This project compares classical statistical models and deep learning architectures to understand:

- How well they predict short-term electricity demand (1 hour ahead),
- How stable they are during peak-load conditions,
- How interpretable and operationally useful their predictions are.

The project includes:

- **SARIMA** as a statistical baseline,  
- **LSTM** as a deep learning model,  
- **Hybrid SARIMA+LSTM** architecture (residual correction approach),  
- **Ensemble** of SARIMA, LSTM, and Hybrid forecasts,  
- **Explainability analysis** using SHAP (SHapley Additive exPlanations).

---

## Motivation

Electricity generation and consumption are no longer simple, linear functions of time and basic weather variables. Rapid weather changes, dynamic consumer behavior, and increasing shares of renewables result in:

- Strong non-linearities in load patterns,  
- Abrupt demand spikes during peak-load conditions,  
- Higher risk for grid instability if forecasts are poor.

Statistical models like SARIMA are trusted and interpretable but may struggle with sudden regime changes. Deep learning models like LSTMs can learn complex non-linear patterns, but they often lack transparency and can be harder to validate in high-stakes environments.

This project investigates whether combining these approaches and adding explainability can provide more accurate and trustworthy forecasts for real-time grid management.

---

## Problem Statement & Goals

There is no universally accepted "best" model for short-term load forecasting that simultaneously optimizes:

- **Prediction accuracy**,  
- **Robustness under peak-load conditions**,  
- **Interpretability and trustworthiness**,  
- **Computational cost for real-time operation**.

### Core Goals

1. Systematically compare SARIMA, LSTM, hybrid SARIMA+LSTM, and ensemble models for short-term electricity demand forecasting.
2. Evaluate how these models behave under peak-load conditions and whether hybrid/ensemble approaches improve stability.
3. Use explainability tools (SHAP) to make deep learning and hybrid models more transparent and operationally usable.

---

## Research Questions

**RQ1:**  
How do SARIMA and LSTM differ in prediction accuracy, interpretability, and computational cost for short-term (1-hour ahead) electricity demand forecasting?

**RQ2 (Hybrid & Ensemble):**  
Can hybrid SARIMA+LSTM models or ensemble combinations reduce error variance and improve forecast stability during peak-load periods?

**RQ3 (XAI):**  
Do explainability tools (SHAP) provide meaningful interpretability for LSTM and hybrid model predictions, enhancing trust and practical usefulness for grid operators?

---

## Data

### Source

- **Dataset:** Open Power System Data (OPSD) – Time Series (2020-10-06)  
- **Geographic Focus:** Germany (DE) electricity load data
- **Key Variables:**
  - Electricity load (actual demand from ENTSO-E Transparency)
  - Generation from solar, wind, hydro, conventional sources
  - Weather data (temperature, wind speed, solar radiation, cloud cover)
  - Temporal information (UTC timestamps, weekends, holidays)
  - Control region data (50hertz, Amprion, TenneT, TransnetBW)

**Data characteristics:**

- **Frequency:** 60-minute (hourly) aggregated data  
- **Duration:** 2015-01-08 to 2018-09-30 (training: 2015-2017, test: 2018)  
- **Format:** CSV with standardized metadata  
- **Granularity:** National and control region level

### Pre-processing

The preprocessing pipeline (`scripts/preprocessing.ipynb`) includes:

1. **Time handling**
   - Parse UTC timestamps and handle timezone conversions
   - Aggregate to 60-minute resolution
   - Handle missing timestamps and irregular intervals

2. **Data cleaning & imputation**
   - Fill gaps in load and weather series using forward-fill and interpolation
   - Remove or flag extreme outliers using statistical methods
   - Handle control region coverage analysis

3. **Feature engineering**
   - Lagged load features (1-hour, 24-hour, 168-hour lags)
   - Weather features (temperature, wind speed, solar radiation, cloud cover)
   - Calendar features (hour of day, day of week, month, weekend/weekday flag, holidays)
   - Rolling statistics (moving averages, standard deviations)
   - Target variable: `target_load` (Germany national load)

4. **Normalization / scaling**
   - RobustScaler for weather and feature variables
   - Separate scaling for target variable

5. **Train/Test Split**
   - Training: 2015-01-08 to 2017-12-31 (26,136 samples)
   - Test: 2018-01-01 to 2018-09-30 (6,552 samples)
   - Internal validation splits for model tuning

**Output:** Processed dataset saved to `scripts/processed_data_60min/processed_data_60min.csv` (83 features total)

---

## Exploratory Data Analysis (EDA)

Comprehensive EDA (`scripts/comprehensive_eda.ipynb`) covers:

- **Temporal trends**
  - Daily, weekly, monthly patterns in load
  - Long-term trends and seasonal peaks
  - Coverage timeline analysis

- **Seasonality**
  - Annual and weekly seasonality using STL decomposition
  - ACF/PACF analysis for ARIMA model selection

- **Correlation analysis**
  - Relationships between load and temperature, wind, solar radiation
  - Lagged relationships and autocorrelation patterns
  - Multicollinearity assessment

- **Weather effects**
  - Load responses to heat waves, cold spells
  - Temperature-load relationships
  - Solar and wind generation impacts

- **Peak load analysis**
  - Peak hour identification (10 AM typically)
  - Peak vs non-peak load distributions
  - Error sensitivity analysis

- **Calendar effects**
  - Weekday vs weekend behavior
  - Holiday impacts
  - Monthly and seasonal patterns

**Outputs:** All EDA visualizations saved to `saved_model_outputs/eda_outputs/`

---

## Methods

### Model Pipelines

The project implements four main forecasting pipelines:

#### 1. SARIMA (Statistical Baseline)
- **Location:** `scripts/models/sarima_model.ipynb`
- **Model:** ARIMA(1,1,1)×(1,1,1)₂₄ with weekly seasonality
- **Approach:** Univariate time series forecasting
- **Features:** Uses only historical load data
- **Outputs:** 
  - Forecasts: `saved_model_outputs/sarima_outputs/sarima_forecasts.csv`
  - Diagnostics: 13 visualization files
  - Metrics: JSON summary

#### 2. LSTM (Deep Learning)
- **Location:** `scripts/models/lstm_model.ipynb`
- **Architecture:** Multi-layer LSTM with dropout regularization
- **Features:** 
  - Historical load (24-hour lookback window)
  - Weather variables (temperature, wind, solar)
  - Calendar features (hour, day of week, month)
  - 83 engineered features total
- **Training:** Early stopping, learning rate reduction, model checkpointing
- **Outputs:**
  - Model: `saved_model_outputs/lstm_outputs/best_model.h5`
  - Forecasts: `saved_model_outputs/lstm_outputs/lstm_predictions.csv`
  - SHAP analysis: 16 visualization files
  - Metrics: JSON summary

#### 3. Hybrid SARIMA+LSTM (Residual Correction)
- **Location:** `scripts/models/hybrid_model.ipynb`
- **Approach:**
  1. Train SARIMA on full training data
  2. Generate SARIMA forecasts and compute residuals
  3. Train LSTM to predict residuals using all features
  4. Final forecast: `Forecast_hybrid = Forecast_SARIMA + Forecast_LSTM_residual`
- **Architecture:** 
  - SARIMA: ARIMA(1,1,1)×(1,1,1)₂₄
  - Residual LSTM: 2-layer LSTM (64→32 units) with dropout
- **Features:** All 83 features + SARIMA prediction as additional feature
- **Outputs:**
  - Model: `saved_model_outputs/hybrid_outputs/best_residual_model.h5`
  - Forecasts: `saved_model_outputs/hybrid_outputs/hybrid_predictions.csv`
  - SHAP analysis: 8 visualization files + comparison CSVs
  - Metrics: JSON summary

#### 4. Ensemble (Multiple Strategies)
- **Location:** `scripts/models/ensemble.ipynb`
- **Base Models:** SARIMA, LSTM, Hybrid
- **Ensemble Methods:**
  - **Simple Average:** Mean of all base models
  - **Median:** Median of all base models
  - **Stacked Ridge:** Ridge regression meta-learner
  - **Weighted Ridge:** Performance-weighted combination
  - **NNLS (Non-Negative Least Squares):** Constrained optimization
  - **Residual Corrector:** LSTM-based error correction
  - **Switch Models:** Conditional model selection based on peak detection
- **Peak Detection:** Precision-Recall curve analysis for peak load identification
- **Outputs:**
  - Comparison tables: CSV files with all model metrics
  - SHAP analysis: Feature importance for ensemble components
  - Visualizations: Performance comparisons, error distributions

### Hybrid & Ensemble Focus

A key focus of the project is **peak-load conditions**:

- **Defining peak conditions**
  - Threshold: 90th percentile of load in test set (~69,295 MW)
  - Mark all timestamps with load above threshold as **peak hours**
  - Critical peaks: Hour=10 AND Load≥p90

- **Peak-focused metrics**
  - MAE_all, RMSE_all for all hours
  - MAE_peak, RMSE_peak only for peak hours
  - Error variance and stability metrics
  - Peak detection precision-recall analysis

- **Research question:**  
  Does the **hybrid SARIMA+LSTM** (or ensemble) significantly reduce peak-hour errors compared to pure SARIMA or pure LSTM?

### Explainability (XAI)

To make LSTM and hybrid models more interpretable:

- **SHAP (SHapley Additive exPlanations)**
  - GradientExplainer for LSTM and hybrid models
  - Feature importance analysis
  - Waterfall plots for individual predictions
  - Summary plots and bar charts

- **Contextual comparisons:**
  - **Peak vs normal hours** - Feature importance differences
  - **Weekdays vs weekends** - Behavioral pattern analysis
  - **Extreme heat vs mild days** - Weather-driven differences
  - **Mis-predicted vs accurately predicted samples** - Error analysis

- **Outputs:**
  - SHAP values CSV files
  - Comparison visualizations (peak vs non-peak, weekday vs weekend, etc.)
  - Waterfall plots for specific scenarios
  - Feature importance rankings

This analysis supports grid operators in trusting and debugging model outputs.

---

## Evaluation Strategy

### Backtesting

- **SARIMA**
  - Fixed train/test split (2015-2017 / 2018)
  - Residual diagnostics: ACF/PACF, Ljung–Box tests
  - Model selection via AIC/BIC

- **LSTM / Hybrid / Ensemble**
  - Fixed train/validation/test splits
  - Hyperparameter tuning using validation set
  - Early stopping to prevent overfitting
  - Learning curve analysis

### Metrics

- **Point forecast metrics**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - sMAPE (symmetric Mean Absolute Percentage Error)
  - WAPE (Weighted Absolute Percentage Error)
  - MASE (Mean Absolute Scaled Error)
  - R² (Coefficient of Determination)
  - Directional Accuracy

- **Peak-focused metrics**
  - MAE_all, RMSE_all (all hours)
  - MAE_peak, RMSE_peak (only peak hours)
  - Error variance and standard deviation
  - Peak detection precision-recall AUC

- **Computational performance**
  - Training time (documented in notebooks)
  - Model file sizes
  - Inference capabilities

---

## Repository Structure

```
.
├── opsd-time_series-2020-10-06/     # Raw OPSD data files
│   ├── time_series_60min_singleindex.csv
│   ├── time_series_30min_singleindex.csv
│   ├── time_series_15min_singleindex.csv
│   └── README.md
│
├── scripts/                          # Main analysis notebooks
│   ├── preprocessing.ipynb           # Data preprocessing pipeline
│   ├── comprehensive_eda.ipynb       # Exploratory data analysis
│   ├── models/
│   │   ├── sarima_model.ipynb        # SARIMA baseline model
│   │   ├── lstm_model.ipynb          # LSTM deep learning model
│   │   ├── hybrid_model.ipynb        # Hybrid SARIMA+LSTM model
│   │   └── ensemble.ipynb            # Ensemble methods
│   └── processed_data_60min/        # Processed datasets
│       ├── processed_data_60min.csv
│       ├── training_data_with_class_weights_60min.csv
│       └── test_data_60min.csv
│
├── saved_model_outputs/              # All model outputs
│   ├── preprocessing_outputs/        # Preprocessing visualizations
│   ├── eda_outputs/                  # EDA plots and summaries
│   ├── sarima_outputs/               # SARIMA forecasts and diagnostics
│   ├── lstm_outputs/                 # LSTM predictions and SHAP analysis
│   ├── hybrid_outputs/               # Hybrid predictions and SHAP analysis
│   └── ensemble_outputs/             # Ensemble comparisons and metrics
│
├── pages/                            # Website/dashboard HTML pages
│   ├── problem.html                  # Problem statement
│   ├── dataset.html                  # Data description
│   ├── preprocessing.html            # Preprocessing overview
│   ├── eda.html                      # EDA results
│   ├── features.html                 # Feature engineering
│   ├── models.html                   # Model overview
│   ├── model_sarima.html             # SARIMA details
│   ├── model_lstm.html               # LSTM details
│   ├── model_hybrid.html            # Hybrid details
│   ├── model_ensemble.html          # Ensemble details
│   ├── explainability.html          # SHAP/XAI results
│   └── references.html               # References
│
├── assets/                           # Website assets
│   ├── style.css                     # Styling
│   ├── app.js                        # JavaScript functionality
│
├── ShortTermElectricityDemandForecasting.html  # Main dashboard
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required packages (install via pip/conda):
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  tensorflow (or tensorflow-cpu)
  statsmodels
  pmdarima (optional, for auto-ARIMA)
  shap
  scipy
  holidays
  ```

### Running the Analysis

1. **Data Preparation**
   - Ensure OPSD data is in `opsd-time_series-2020-10-06/` directory
   - Run `scripts/preprocessing.ipynb` to generate processed dataset

2. **Exploratory Data Analysis**
   - Run `scripts/comprehensive_eda.ipynb` to generate EDA visualizations

3. **Model Training & Evaluation**
   - Run models in order:
     1. `scripts/models/sarima_model.ipynb`
     2. `scripts/models/lstm_model.ipynb`
     3. `scripts/models/hybrid_model.ipynb`
     4. `scripts/models/ensemble.ipynb`
   - Each notebook saves outputs to `saved_model_outputs/`

4. **View Results**
   - Open `ShortTermElectricityDemandForecasting.html` in a web browser
   - Or navigate individual pages in `pages/` directory

### Output Files

All model outputs are saved in `saved_model_outputs/` with the following structure:
- **Predictions:** CSV files with date, actual, and predicted values
- **Metrics:** JSON files with comprehensive performance metrics
- **Visualizations:** PNG files (300 DPI) for all plots and analyses
- **Models:** H5 files for trained LSTM and hybrid models
- **SHAP Values:** CSV files with feature importance scores

---

## Results & Outputs

### Key Findings

1. **Model Performance Comparison**
   - All models evaluated on 2018 test set (6,552 hourly samples)
   - Comprehensive metrics saved in comparison tables
   - Peak vs non-peak performance analysis

2. **Peak Load Stability**
   - Error variance analysis for peak conditions
   - Precision-recall curves for peak detection
   - Critical peak (Hour=10, Load≥p90) performance

3. **Explainability Insights**
   - SHAP feature importance rankings
   - Context-specific feature contributions
   - Waterfall plots for individual predictions
   - Regime-specific analysis (peak/non-peak, weekday/weekend, etc.)

4. **Visualizations**
   - Time series forecasts vs actuals
   - Error distributions
   - Performance comparison charts
   - SHAP summary and comparison plots
   - Training history plots

All results are available in the `saved_model_outputs/` directory and visualized in the HTML dashboard.

---

## References

1. Akhtar, S., et al. (2023). *Short-Term Load Forecasting Models: A Review of Challenges, Progress, and the Road Ahead.* Energies.

2. Tzelepi, M., et al. (2023). *Deep Learning for Energy Time-Series Analysis and Forecasting.*

3. Aksöz, A., et al. (2024). *Analysis of SARIMA Models for Forecasting Electricity Demand.*

4. Qureshi, M., et al. (2024). *Deep learning-based forecasting of electricity consumption.*

5. Open Power System Data. *Time Series* (Version 2020-10-06). Available at:  
   https://doi.org/10.25832/time_series/2020-10-06

6. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* NIPS.

---

## License

This project is for educational and research purposes. Data from Open Power System Data (OPSD) is licensed under CC0 1.0.

---

## Contact

For questions or contributions, please refer to the project repository.
