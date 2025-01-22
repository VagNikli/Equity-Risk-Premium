import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the dataset (replace "predictor_data.csv" with your dataset file)
data = pd.read_csv("predictor_data.csv")

# Define the dependent variable and the 14 predictor variables
dependent_variable = "r"
predictor_variables = [
    "logDP", "logDY", "logEP", "logDE", "svar", "b/m",
    "ntis", "tbl", "lty", "ltr", "tms", "dfy", "dfr", "lagINFL"
]
N = len(predictor_variables)
# Define the window sizes and initialize results lists
T = len(data)
m = 80  # In-sample window size (quarters, corresponds to 20 years)
p = 40  # Holdout out-of-sample window size (quarters, corresponds to 10 years)

# Initialize a list to store R-squared values for each predictor variable
r_squared_values = []
theta = 0.9 

# Loop through each predictor variable
fai_predictor_t = {} #### TT
forecasts_matrix = []

for predictor in predictor_variables:
    # Initialize a list to store out-of-sample forecasts for this variable
    out_of_sample_forecasts_variable = []
    # fai_predictor_t[predictor] = []
    
    # Loop through the data with the updated window size
    for t in range(m + p, T):
        # Define the in-sample data for this iteration
        in_sample_data = data.iloc[1 : t - p, :]

        # Extract the relevant variables for this iteration
        X_in = in_sample_data[predictor]
        X_in = sm.add_constant(X_in)  # Add a constant for the intercept
        y_in = in_sample_data[dependent_variable]

        # Fit the regression model using the in-sample data
        model = sm.OLS(y_in, X_in)
        results = model.fit()

        # Extract the coefficients for this variable
        alpha_t = results.params[0]
        beta_t = results.params[1]

        # Calculate the out-of-sample forecast using the formula you provided
        x_t = data.loc[t, predictor]
        forecast = alpha_t + beta_t * x_t

        # 
        fai_i_t = 0
        s = m 
        for x in out_of_sample_forecasts_variable: 
            fai_i_t += theta**(t-1-s) * (data.loc[s+1, dependent_variable] - x)**2
            s += 1
        if t not in fai_predictor_t: 
            fai_predictor_t[t] = [fai_i_t]
        else: 
            fai_predictor_t[t].append(fai_i_t)  ## equation (9)
        # Append the forecast to the list for this variable
        out_of_sample_forecasts_variable.append(forecast)
    forecasts_matrix.append(out_of_sample_forecasts_variable)

# to get the matrix of weight_i_t
weight = {}
for fai_t in fai_predictor_t: 
    total = 0
    for i in range(len(predictor_variables)):
        if fai_predictor_t[fai_t][i]:
            total += 1/fai_predictor_t[fai_t][i]
    for i in range(len(predictor_variables)):
        val = fai_predictor_t[fai_t][i] 
        if fai_t not in weight: 
            weight[fai_t] = [] 
        if val:
            weight[fai_t].append((1/val)/total) ## w_i_t 
        else: 
            weight[fai_t].append(0)  
            
combined_forecasts = [] 
for w_t in weight:
    tmp = 0
    for i in range(len(predictor_variables)):
        tmp += weight[w_t][i] * forecasts_matrix[i][w_t - m - p]
    combined_forecasts.append(tmp)

mean_combined_forecasts = [] 
for w_t in range(m+p, len(data)):
    tmp = 0
    for i in range(N):
        tmp += (1/N) * forecasts_matrix[i][w_t - m - p]
    mean_combined_forecasts.append(tmp)

median_combined_forecasts = [] 
for w_t in range(m+p, len(data)):
    tmp = []
    for i in range(N):
        tmp.append(forecasts_matrix[i][w_t - m - p])
    median_combined_forecasts.append(np.median(tmp))
print(median_combined_forecasts)

forecast_diff = np.array(data[dependent_variable][m + p:]) - np.array(combined_forecasts)
forecast_diff_mean_combined = np.array(data[dependent_variable][m + p:]) - np.array(mean_combined_forecasts)
forecast_diff_median_combined = np.array(data[dependent_variable][m + p:]) - np.array(median_combined_forecasts)
benchmark_returns = np.array(data[dependent_variable][m + p:])
r_bar = np.mean(benchmark_returns)
sum_squared_forecast_diff = np.sum(forecast_diff**2)
sum_squared_benchmark_diff = np.sum((benchmark_returns - r_bar)**2)
out_of_sample_r_squared = 1 - (sum_squared_forecast_diff / sum_squared_benchmark_diff)
out_of_sample_r_squared_mean_combined = 1 - (np.sum(forecast_diff_mean_combined**2) / sum_squared_benchmark_diff)
out_of_sample_r_squared_median_combined = 1 - (np.sum(forecast_diff_median_combined**2) / sum_squared_benchmark_diff)

# Append the R-squared value to the list
# r_squared_values.append((predictor, out_of_sample_r_squared))

# Print out-of-sample R-squared values for each predictor variable
print("Theta = :", theta, ", Mean Comebined forecast: R_OS^2 =", "{:.4f}".format(out_of_sample_r_squared_mean_combined))
print("Theta = :", theta, ", Median Comebined forecast: R_OS^2 =", "{:.4f}".format(out_of_sample_r_squared_median_combined))
print("Theta = :", theta, ", Comebined forecast: R_OS^2 =", "{:.4f}".format(out_of_sample_r_squared))

data_ = list(data[dependent_variable][120:])
plt.plot(combined_forecasts) 
plt.plot(forecasts_matrix[0])
plt.plot(data_)
plt.legend(['combined_forecasts', 'forecast','data'])
plt.xlabel('time')
plt.ylabel('data & forecast')
plt.show() 