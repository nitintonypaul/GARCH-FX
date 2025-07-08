# Necessary modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tabulate import tabulate

# Custom modules
from libs.utils import getGARCHdata, garchforecast
from libs.garchfx import fxforecast

# Parser object 
parser = argparse.ArgumentParser()

# Adding and parsing arguments
parser.add_argument('--ticker', type=str, help="Ticker Symbol (Stock)", required=True)
parser.add_argument('--train', type=int, help="Training days (1000+ recommended for stable variance)", required=True)
parser.add_argument('--horizon', type=int, help="Forecasting Horizon (1000+ for best visual results)", required=True)
parser.add_argument('--theta', type=float, help="Theta (scale) Variable for Forecasting", required=False, default=1e-3)
parser.add_argument('--garchcomp', type=bool, help="Compare with standard GARCH forecast (true or false)", required=False, default=True)
parser.add_argument('--reg', type=bool, help="Include Regime Shifting (true or false)", required=False, default=False)
parser.add_argument('--customreg', type=bool, help="Enter Custom Regimes (true or false)", required=False, default=False)
args = parser.parse_args()

# Obtaining stocks Arguments from the user
ticker = args.ticker 
train = args.train 
horizon = args.horizon 
THETA = args.theta 
garchcomp = args.garchcomp
regimesh = args.reg
custom = args.customreg

# Defining probability array and regimes
probabilityArray = None
regimes = None

# Custom regime probabilities and regime input
if custom:

    # Demo regimes
    # probability array = [[0.97, 0.029, 0.001], [0.015, 0.95, 0.035], [0.00, 0.04, 0.96]]
    # regimes = [0.8, 1, 1.2]
    probabilityArray = eval(input("Enter probability array (list of lists format): "))
    regimes = eval(input("Enter regime multipliers (list format):"))

# Obtaining GARCH modelling data
data = yf.Ticker(ticker).history(period=f"{train}d")["Close"]
logReturns = np.log(data / data.shift(1)).dropna()
logReturns *= 100

# Getting conditional volatility until shocks are not available
# Also fetching the parameters
VOLATILITY, PARAMS = getGARCHdata(logReturns)

# Forecasting using GARCH-FX with or without regime shifting
FXforecast = fxforecast(VOLATILITY, nahead=horizon, params=PARAMS, theta=THETA, reg=regimesh, regimeStates=probabilityArray, regimes=regimes)

# Obtaining GARCH forecast if garchcomp is True
# default is True for demonstration purposes
if garchcomp:
    GARCHforecast = garchforecast(VOLATILITY, nahead=horizon, params=PARAMS)
    plt.plot(GARCHforecast, label="GARCH forecast", alpha=0.8, linewidth=3)

# Forecasting summary
summary = [
    ["Mean Forecasted Volatility", f"{np.mean(FXforecast):.5f}"],
    [f"Volatility at Horizon ({horizon} days)", f"{FXforecast[-1]:.5f}"],
    ["Min / Max Volatility", f"{min(FXforecast):.5f} / {max(FXforecast):.5f}"],
    ["Standard Deviation", f"{np.std(FXforecast, ddof=1):.5f}"],
    ["Roughness Index (std / mean)", f"{(np.std(FXforecast, ddof=1) / np.mean(FXforecast)):.3f}"]
]

print("\n" + tabulate(summary, headers=[], tablefmt="plain") + "\n")

# Plotting values
plt.plot(FXforecast, label=f"GARCH-FX (θ = {THETA})")
plt.title(f"{ticker} Daily Volatility Forecasts")
plt.xlabel("Forecasted Days")
plt.ylabel("Daily Volatility (%)")
plt.legend()
plt.grid(True)
plt.show()