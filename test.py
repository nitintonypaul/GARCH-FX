import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tabulate import tabulate
import time

from libs.utils import garchforecast, getGARCHdata, realVol, hestonforecast
from libs.garchfx import fxforecast

# Constants
FORECASTS = 1000
TICKER = input("Enter Ticker: ")
regime = True if input("Enable Regime Shifting (Y/N): ").lower() == "y" else False
sigma = float(input("Enter Vol of Vol. (Heston): "))

# Defining variables of this GARCH implementation
# SCALE controls the volatility wiggliness, proportional to volatility of volatility
# SCALE → 0 => Volatility models closer to GARCH volatility, i.e. the long term variance 
# SCALE → 1 (or infinity) => Heavy volatility spikes are observed
# SCALE = 1e-3 to 5e-3 seems ideal (1e-2 for rough)
THETA = 5e-3

# DELTA is the regime shift variable. 
# D = 1 normal market mean. 
# D > 1 => highly volatile regime. (1.5)
# D < 1 => unusually calm regime. (0.5)
DELTA = 1

# Fetching data using yfinance
# logReturns are scaled by 100 for better optimizer performance. It is scaled back by 100 whenever necessary
data = yf.Ticker(TICKER).history(period="2000d")["Close"]
realizedVolatility = 100 * realVol(np.log(data / data.shift(1)).dropna(), FORECASTS)
volatilityData = data[-1000:]
logReturns = np.log(volatilityData / volatilityData.shift(1)).dropna()
logReturns *= 100

# Obtaining GARCH conditional volatility and parameters
VOLATILITY, PARAMS = getGARCHdata(logReturns)

# Heston forecast
# Computing GARCH parameters and passing into Heston for fair comparison
GARCH_PERSISTENCE = PARAMS[0]+PARAMS[1]
speed = -np.log(GARCH_PERSISTENCE) * 252
variance = PARAMS[2]/(1-GARCH_PERSISTENCE)
hestonForecast = hestonforecast(speed, variance, VOLATILITY**2, sigma)

# manual GARCH to forecast in percentages
GARCHforecast = garchforecast(VOLATILITY, nahead=FORECASTS, params=PARAMS)

# GARCH-FX stochastic forecasting extension in percentages
FXforecast = fxforecast(VOLATILITY, nahead=FORECASTS, params=PARAMS, theta=THETA, reg=regime)

# Root Mean Squared Error
rmseFX = np.sqrt(np.mean((FXforecast - realizedVolatility) ** 2))
rmseGARCH = np.sqrt(np.mean((GARCHforecast - realizedVolatility) ** 2))
rmseHESTON = np.sqrt(np.mean((hestonForecast - realizedVolatility) ** 2))

# Mean absolute error
maeFX = np.mean(np.abs(FXforecast - realizedVolatility))
maeGARCH = np.mean(np.abs(GARCHforecast - realizedVolatility))
maeHESTON = np.mean(np.abs(hestonForecast - realizedVolatility))

# Printing parameters
print("\nGARCH-FX Parameters:")
print(tabulate(
    [["Omega", PARAMS[2]],
     ["Alpha", PARAMS[0]],
     ["Beta", PARAMS[1]]],
     headers=[],
     tablefmt="plain"
))

# Summary table
summary = [
    ["GARCH", f"{rmseGARCH:.5f}", f"{maeGARCH:.5f}", f"{np.mean(GARCHforecast):.3f}%"],
    ["Heston", f"{rmseHESTON:.5f}", f"{maeHESTON:.5f}", f"{np.mean(hestonForecast):.3f}%"],
    ["GARCH-FX", f"{rmseFX:.5f}", f"{maeFX:.5f}", f"{np.mean(FXforecast):.3f}%"],
    ["Realized Vol.", f"-", f"-", f"{np.mean(realizedVolatility):.3f}%"]
]

# Displaying summary
print("\n" + tabulate(summary, headers=["Model", "RMSE", "MAE", "Mean Vol."], tablefmt="plain") + "\n")

# Plotting Values
plt.plot(realizedVolatility, label="Realized Volatility", alpha=0.8)
plt.plot(GARCHforecast, label="GARCH", alpha=0.8)
plt.plot(FXforecast, label=f"GARCH-FX (θ = {THETA})")
plt.plot(hestonForecast, label=f"Heston (σ = {sigma})")
plt.title(f"{TICKER} Daily Volatility Forecasts: Comparison of GARCH, GARCH-FX and Heston Models")
plt.xlabel("Forecasted Days")
plt.ylabel("Daily Volatility (%)")
plt.legend()
plt.grid(True)
plt.show()
plt.close()