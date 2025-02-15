# Equity Risk Premium Forecasting

This repository contains the code and analysis for forecasting the equity risk premium using predictor variables from Welch and Goyal (2008). The project evaluates the predictive power of various financial indicators through in-sample and out-of-sample regression analysis. It also explores advanced forecasting techniques, including "kitchen sink" models and forecast combination methods.

## Project Overview

The objective is to determine whether financial predictor variables can forecast the equity risk premium and to assess the predictive performance of individual and combined models. The methodology includes:
1. In-sample analysis of predictor variables.
2. Out-of-sample forecasting using an expanding window.
3. Evaluation of a "kitchen sink" regression model.
4. Combination of forecasts to enhance predictive performance.

## Project Structure

- **`01_in_sample_analysis.py`**: Performs in-sample regression analysis for each predictor variable. Includes:
  - Testing the null hypothesis $H_0: \beta_i = 0$ against $H_A: \beta_i > 0$.
  - Calculating heteroscedasticity-robust t-statistics and p-values.
  - Reporting adjusted $R^2$.

- **`02_out_of_sample_analysis.py`**: Conducts out-of-sample forecasting with an expanding window. Features:
  - Dividing data into training, holdout, and test sets.
  - Calculating out-of-sample $R^2$ to compare forecasts against a historical average benchmark.

- **`03_kitchen_sink_model.py`**: Implements a "kitchen sink" regression that includes all predictor variables to assess their combined forecasting power.

- **`04_forecast_combination.py`**: Combines individual forecasts using three methods:
  - Mean combination.
  - Median combination.
  - Discounted Mean Squared Prediction Error (DMSPE) with different discount factors.
  - Evaluates the performance of combined forecasts relative to individual predictors.

## Dataset Description

The dataset, `predictor_data.csv`, contains quarterly financial indicators based on Welch and Goyal (2008). Predictors include:
1. Log dividend-price ratio (logDP)
2. Log dividend yield (logDY)
3. Log earnings-price ratio (logEP)
4. Log dividend-payout ratio (logDE)
5. Stock Variance (svar)
6. Book-to-market ratio (b/m)
7. Net equity expansion (ntis)
8. Treasury bill rate (tbl)
9. Long-term yield (lty)
10. Long-term returns (ltr)
11. Term spread (tms)
12. Default yield spread (dfy)
13. Default return spread (dfr)
14. Lagged inflation (lagINFL)

## Methodology

### 1. In-Sample Analysis
- Regression for each predictor variable:
  $\bar{r}_{t+1} = \frac{1}{t} \sum_{j=1}^{t} r_j$
- Tested the predictive power of each variable.

### 2. Out-of-Sample Forecasting
- Generated forecasts with an expanding window.
- Compared against the historical average benchmark:
  <p align="center">
  $\bar{r}_{t+1} = \frac{1}{t} \sum_{j=1}^{t} r_j$
</p>

- Evaluated predictive performance using out-of-sample $R^2$.

### 3. "Kitchen Sink" Regression
- Assessed the combined effect of all predictors in a single model.

### 4. Forecast Combination
- Combined forecasts using mean, median, and DMSPE methods:
  <p align="center">
  $\hat{r}_{c,t+1}$ = $\sum_{i=1}^N \omega_{i,t} \hat{r}_{i,t+1}$
  </p>
- Evaluated the improvement in predictive accuracy from combined forecasts.

## Results

- Detailed results, including $R^2$ values, t-statistics, and forecast performance metrics, are provided in the output of each script.

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>

2. **Install Dependencies**: Ensure Python and required libraries are installed:
    ```bash
    pip install pandas numpy statsmodels matplotlib
    
## 3. Execute the Scripts

### A. Run In-Sample Analysis:

    python 01_in_sample_analysis.py

### B.Run out-of-sample analysis:

    python 02_out_of_sample_analysis.py

### C. Execute the "kitchen sink" model:

    python 03_kitchen_sink_model.py

### D. Combine forecasts:

    python 04_forecast_combination.py
