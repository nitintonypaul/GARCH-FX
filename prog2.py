import numpy as np
from arch import arch_model
import yfinance as yf
import matplotlib.pyplot as plt

# Constants
ANNUALIZER = np.sqrt(252)
FORECASTS = 1000
THETA = 0.01 # Can be calibrate for regime switching
DELTA = 1 # Can be modelled using entropy-change and sigmoid
ticker = 'AAPL'

AAPL = yf.Ticker(ticker).history(period="180d")["Close"]
AAPLreturns = np.log(AAPL / AAPL.shift(1)).dropna()
AAPLreturns *= 100
model = arch_model(AAPLreturns, vol='Garch', p=1, q=1, dist='normal')
results = model.fit(disp='off')

OMEGA, ALPHA, BETA = results.params["omega"], results.params["alpha[1]"], results.params["beta[1]"]
VOLATILITY = results.conditional_volatility.iloc[-1] 

# NORMAL GARCH
GARCHforecast = []
GARCHforecast.append(VOLATILITY / 100)
previousVariance = VOLATILITY ** 2
for i in range(FORECASTS):
    newVariance = OMEGA + (ALPHA + BETA) * previousVariance
    previousVariance = newVariance
    GARCHforecast.append(np.sqrt(previousVariance) / 100)

# SAMPLED GARCH
# Possible regime shift
GARCHsampled = []
previousSampledVariance = VOLATILITY ** 2
GARCHsampled.append(VOLATILITY / 100)
for i in range(FORECASTS):

    # Passing Variance as mode
    # DELTA is decaying constant
    shape_k = (DELTA * previousSampledVariance) / THETA + 1

    # Sampling from gamma distribution
    sample = np.random.gamma(shape=shape_k, scale=THETA, size=100)

    # GARCH forecast
    newSampledVariance = OMEGA + (ALPHA + BETA) * np.mean(sample)
    previousSampledVariance = newSampledVariance
    
    GARCHsampled.append(np.sqrt(previousSampledVariance) / 100)

for NG, SG in zip(GARCHforecast, GARCHsampled):
    print(f"NG: {NG:.6f}    GD-G: {SG:.6f}")

print("ANNUALIZED GARCH:", np.sqrt(previousVariance) / 100 * ANNUALIZER)
print("ANNUALIZED GARCH SAMPLED:", np.sqrt(previousSampledVariance) / 100 * ANNUALIZER)

plt.plot(GARCHforecast, label="GARCH")
plt.plot(GARCHsampled, label="GD-GARCH")
plt.legend()
plt.grid(True)
plt.show()