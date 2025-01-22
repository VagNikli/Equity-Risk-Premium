import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the dataset (replace "predictor_data.csv" with your dataset file)
data = pd.read_csv("predictor_data.csv")

# Define the dependent variable and the 14 predictor variables
dependent_variable = "r"
predictor_variables = [
    "logDP", "logDY", "logEP", "logDE", "svar", "b/m",
    "ntis", "tbl", "lty", "ltr", "tms", "dfy", "dfr", "lagINFL"
]

# Define the window sizes and initialize results lists
T = len(data)
m = 80  # In-sample window size (quarters, corresponds to 20 years)
p = 40  # Holdout out-of-sample window size (quarters, corresponds to 10 years)

# Initialize a list to store R-squared values for each predictor variable
r_squared_values = []

# Loop through each predictor variable
for predictor in predictor_variables:
    # Initialize a list to store out-of-sample forecasts for this variable
    sum_squared_forecast_diff = 0
    sum_squared_benchmark_diff = 0
    # Loop through the data with the updated window size
    for t in range(m + p, T):
        # Define the in-sample data for this iteration
        in_sample_data = data.iloc[0: t, :]

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

        # Calculate historical average of returns up to period t
        historical_returns = data[dependent_variable][:t]
        r_bar = historical_returns.mean()

        forecast_diff = data[dependent_variable][t] - forecast
        sum_squared_forecast_diff += forecast_diff ** 2

        historical_diff = data[dependent_variable][t] - r_bar
        sum_squared_benchmark_diff += historical_diff ** 2

    out_of_sample_r_squared = 1 - (sum_squared_forecast_diff / sum_squared_benchmark_diff)

    # Append the R-squared value to the list
    r_squared_values.append((predictor, out_of_sample_r_squared))

# Create a DataFrame from the list of R-squared values
df_r_squared = pd.DataFrame(r_squared_values, columns=["Predictor Variable", "R_OS^2"])

# Print the R-squared table
print("----R_OS^2 Table----")
print(df_r_squared.to_string(index=False))
