import numpy as np
from arch import arch_model
import yfinance as yf

ANNUALIZER = np.sqrt(252)
def simulate_heston(v0, kappa=2, theta=0.04, sigma=0.3, rho=-0.7, T=10/252, N=10):
    dt = T / N
    v = np.zeros(N)
    v[0] = v0

    for t in range(1, N):
        z1 = np.random.normal()
        z2 = np.random.normal()
        dw2 = (rho * z1 + np.sqrt(1 - rho ** 2) * z2) * np.sqrt(dt)

        v[t] = np.abs(v[t - 1] + kappa * (theta - v[t - 1]) * dt + sigma * np.sqrt(v[t - 1]) * dw2)

    return v

AAPL = yf.Ticker('AAPL').history(period="180d")["Close"]
AAPLreturns = np.log(AAPL / AAPL.shift(1)).dropna()
AAPLvol = np.std(AAPLreturns, ddof=1) * ANNUALIZER
print(AAPLvol)
AAPLreturns *= 100

# Heston
print("HESTON")
hestonArray = np.sqrt(simulate_heston(AAPLvol)) / ANNUALIZER
for i in hestonArray:
    print(i)
print("ANNUALIZED HESTON:", hestonArray[-1] * ANNUALIZER)

model = arch_model(AAPLreturns, vol='Garch', p=1, q=1, dist='normal')
results = model.fit(disp='off')

OMEGA, BETA = results.params["omega"], results.params["beta[1]"]
# print("PARAMETERS: ", OMEGA, BETA)

print(" ")

VOLATILITY = results.conditional_volatility.iloc[-1]
print("GARCH FORECAST")
print(VOLATILITY/100)

prev = VOLATILITY
for i in range(10):
    newVOL = OMEGA + BETA*prev
    print(newVOL/100)
    prev = newVOL

print("ANNUALIZED GARCH:", prev * ANNUALIZER)

print(" ")


# SAMPLED GARCH
# Vol explodes at 0.5, stable at 0.1-0.15, decays at 0.01
# Possible regime shift?

vol = VOLATILITY
THETA = 0.1
print("GARCH SAMPLED")
print(vol/100)
for i in range(10):
    shape_k = vol / THETA + 1
    sample = np.random.gamma(shape=shape_k, scale=THETA, size=100)
    newVol = OMEGA + BETA * np.mean(sample)
    print(newVol/100)
    vol = newVol
print("ANNUALIZED GARCH SAMPLED:", vol * ANNUALIZER)