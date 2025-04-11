
import yfinance as yf
import matplotlib.pyplot as plt

# (a) META and the S&P 500 seem to both have increased steadily, while Snapchat seems to have not appreciated in value at all.

tickers = ['SNAP', 'META', '^GSPC']
start_date = '2022-02-01'
end_date = '2025-01-31'
data = yf.download(tickers, start=start_date, end=end_date)
data = data['Close']

# Plot
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(data[ticker], label=ticker)
plt.title('Closing Prices of SNAP, META, and S&P 500')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

#(b)
#Mean Returns:
#snap.rtn    0.001027
#fb.rtn     -0.001448
#sp.rtn      0.000385

# Standard Deviations:
#snap.rtn    0.031084
#fb.rtn      0.055648
#sp.rtn      0.010957
 

import numpy as np
returns = np.log(data / data.shift(1))

returns.columns = ['snap.rtn', 'fb.rtn', 'sp.rtn']
returns = returns.dropna()

mean_returns = returns.mean()
std_returns = returns.std()

print("Mean Returns:\n", mean_returns)
print("\nStandard Deviations:\n", std_returns)

# code for part (c)
import statsmodels.api as sm

tickers = ['SNAP', '^GSPC']
start_date = "2022-02-01"
end_date = "2025-01-31"
data = yf.download(tickers, start=start_date, end=end_date)['Close']

log_returns = np.log(data / data.shift(1))
log_returns.columns = ['snap.rtn', 'sp.rtn']
log_returns = log_returns.dropna()

#(CAPM Model)
X = sm.add_constant(log_returns['sp.rtn'])  # Add intercept (alpha)
y = log_returns['snap.rtn']
model = sm.OLS(y, X).fit()
print(model.summary())

#                       OLS Regression Results
#==============================================================================
#Dep. Variable:               snap.rtn   R-squared:                       0.167
#Model:                            OLS   Adj. R-squared:                  0.166
#Method:                 Least Squares   F-statistic:                     149.9
#Date:                Thu, 13 Feb 2025   Prob (F-statistic):           1.48e-31
#Time:                        20:51:49   Log-Likelihood:                 1172.8
#No. Observations:                 751   AIC:                            -2342.
#Df Residuals:                     749   BIC:                            -2332.
#Df Model:                           1
#Covariance Type:            nonrobust
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const         -0.0022      0.002     -1.210      0.227      -0.006       0.001
#sp.rtn         2.0741      0.169     12.245      0.000       1.742       2.407
#==============================================================================
#Omnibus:                      660.782   Durbin-Watson:                   2.120
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):            73039.608
#Skew:                          -3.372   Prob(JB):                         0.00
#Kurtosis:                      50.840   Cond. No.                         91.3
#==============================================================================

# (c) - sp.rtn of 2.0741 means that for every 1% increase in the S&P 500, Snapchat is expected to return 2.07% on average. This represents Snapchat's Beta, which is greater than 1,
# which would classify Snapchat as a high risk, high reward stock.

# (d) the p-value is > 0.05, which means the intercept is not significant at the 5% level.
# The 95% confidence interval contains zero (Lower bound of -0.006 to 0.001) which furthers the idea that the intercept is not significant
# For the slope, the bound is from 1.742 to 2.407 which makes the slope highly significant in the regression

# Code for part (e)

tickers = ['META', '^GSPC']
start_date = "2022-02-01"
end_date = "2025-01-31"
data = yf.download(tickers, start=start_date, end=end_date)['Close']

log_returns = np.log(data / data.shift(1))
log_returns.columns = ['fb.rtn', 'sp.rtn']
log_returns = log_returns.dropna()

#(CAPM Model)
X = sm.add_constant(log_returns['sp.rtn'])
y = log_returns['fb.rtn']
model = sm.OLS(y, X).fit()
print(model.summary())

#                           OLS Regression Results
#==============================================================================
#Dep. Variable:                 fb.rtn   R-squared:                       0.364
#Model:                            OLS   Adj. R-squared:                  0.364
#Method:                 Least Squares   F-statistic:                     429.5
#Date:                Thu, 13 Feb 2025   Prob (F-statistic):           9.33e-76
#Time:                        21:06:09   Log-Likelihood:                 1711.8
#No. Observations:                 751   AIC:                            -3420.
#Df Residuals:                     749   BIC:                            -3410.
#Df Model:                           1
#Covariance Type:            nonrobust
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const          0.0004      0.001      0.405      0.685      -0.001       0.002
#sp.rtn         1.7125      0.083     20.723      0.000       1.550       1.875
#==============================================================================
#Omnibus:                      538.436   Durbin-Watson:                   1.889
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):            59289.704
#Skew:                          -2.374   Prob(JB):                         0.00
#Kurtosis:                      46.269   Cond. No.                         91.3
#==============================================================================

# (e) - Meta seems to have a higher R-squared, meaning it moves more with the S&P 500.
# Snapchat has a higher beta (2.0741 > 1.7125), making Meta a more conservative and possibly stable stock option
# Snapchat has a negative intercept while Meta has a slighly positive intercept
# I would personally choose Meta as my stock choice as it seems to move more closely with the overall market and has a lower beta, representing more stability and less overall risk.

# Code for part (f)
tickers = ['SNAP', 'META', '^GSPC']
start_date = "2022-02-01"
end_date = "2025-01-31"
data = yf.download(tickers, start=start_date, end=end_date)['Close']

log_returns = np.log(data / data.shift(1))
log_returns.columns = ['snap.rtn', 'fb.rtn', 'sp.rtn']
log_returns = log_returns.dropna()
#Snap model
X_snap = sm.add_constant(log_returns['sp.rtn'])  # Adding the intercept
y_snap = log_returns['snap.rtn']
model_snap = sm.OLS(y_snap, X_snap).fit()
#FB model
X_fb = sm.add_constant(log_returns['sp.rtn'])  # Adding the intercept
y_fb = log_returns['fb.rtn']
model_fb = sm.OLS(y_fb, X_fb).fit()

#Residuals
residuals_snap = model_snap.resid
residuals_fb = model_fb.resid


fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Plot Snap
ax[0].scatter(log_returns['sp.rtn'], residuals_snap, color='blue', alpha=0.5)
ax[0].axhline(y=0, color='red', linestyle='--')
ax[0].set_title("Residuals of Snapchat Regression (Returns on S&P 500)")
ax[0].set_xlabel("S&P 500 Returns")
ax[0].set_ylabel("Residuals")

# Plot FB
ax[1].scatter(log_returns['sp.rtn'], residuals_fb, color='green', alpha=0.5)
ax[1].axhline(y=0, color='red', linestyle='--')
ax[1].set_title("Residuals of Facebook Regression (Returns on S&P 500)")
ax[1].set_xlabel("S&P 500 Returns")
ax[1].set_ylabel("Residuals")

plt.tight_layout()
plt.show()

#(g) - I did a Shapiro-Wilk test on the residuals to determine the answer. (I used AI to brainstorm ideas and came across the Shapiro-Wilk test as I did not feel confident in my abilities to discern normal distribution from the graph produced in part (f))
#(g results) - 
#Shapiro-Wilk Test for Snapchat residuals: Stat=0.7080035387152475, p-value=3.850708838320876e-34
#Shapiro-Wilk Test for Facebook residuals: Stat=0.6150758939623027, p-value=7.986793624265968e-38
# Both p-values came out extremely small, meaning that the residuals do not follow a normal distribution for both stocks.

import scipy.stats as stats

# Shapiro-Wilk test for Snapchat residuals
shapiro_stat_snap, shapiro_p_value_snap = stats.shapiro(residuals_snap)
print(f"Shapiro-Wilk Test for Snapchat residuals: Stat={shapiro_stat_snap}, p-value={shapiro_p_value_snap}")

# Shapiro-Wilk test for Facebook residuals
shapiro_stat_fb, shapiro_p_value_fb = stats.shapiro(residuals_fb)
print(f"Shapiro-Wilk Test for Facebook residuals: Stat={shapiro_stat_fb}, p-value={shapiro_p_value_fb}")