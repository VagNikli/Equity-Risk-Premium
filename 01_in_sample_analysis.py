import pandas as pd
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv("predictor_data.csv")

# Define the dependent variable and the 14 predictor variables
dependent_variable = "r"
predictor_variables = [
    "logDP", "logDY", "logEP", "logDE", "svar", "b/m",
    "ntis", "tbl", "lty", "ltr", "tms", "dfy", "dfr", "lagINFL"
]

# Define the predictor variables to be negated
negate_predictors = ["ntis", "tbl", "lty", "lagINFL"]

# Loop through each predictor variable and run the regression
results = []
for predictor in predictor_variables:
    # Check if the predictor should be negated
    if predictor in negate_predictors:
        data[predictor] = -data[predictor]

    # Add a constant term (alpha) to the model
    X = sm.add_constant(data[predictor])
    y = data[dependent_variable]

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Get the results
    alpha = model.params["const"]
    beta = model.params[predictor]

    # Calculate heteroscedasticity-robust standard errors and t-statistic
    robust_se = model.HC0_se[predictor]
    t_stat = beta / robust_se

    # Calculate the one-sided p-value for HA: βi > 0
    p_value = model.pvalues[predictor] / 2  # Divide by 2 for one-sided test

    adj_r_squared = model.rsquared_adj

    results.append((predictor, alpha, beta, t_stat, p_value, adj_r_squared))

# Create a DataFrame to display the results
results_df = pd.DataFrame(results,
                          columns=["Predictor Variable", "Alpha", "Beta", "T-Stat", "One-sided P-Value", "Adjusted R-squared"])

# Display the results
print(results_df['T-Stat'])
print(results_df['One-sided P-Value'])
print(results_df['Adjusted R-squared'])

