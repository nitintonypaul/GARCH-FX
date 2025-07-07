import numpy as np
from arch import arch_model

# Global seed for fair comparison
GLOBAL_SEED = None

# Normal GARCH forecasting
# Takes final conditional volatility, forecast horizon and GARCH parameters
def garchforecast(volatility, nahead, params):
    
    GARCHforecasts = []
    GARCHforecasts.append(volatility / 100)
    previousVariance = volatility ** 2

    OMEGA, ALPHA, BETA = params[0], params[1], params[2]
    for i in range(nahead):

        # GARCH equation
        newVariance = OMEGA + (ALPHA + BETA) * previousVariance
        previousVariance = newVariance

        # Appending GARCH volatility
        GARCHforecasts.append(np.sqrt(previousVariance) / 100)
    
    return np.array(GARCHforecasts)

# Function to obtain final conditional volatility and parameters
def getGARCHdata(returns):

    model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
    results = model.fit(disp='off')

    # Obtaining parameters and final volatility by using residuals to prepare manual forecasting
    OMEGA, ALPHA, BETA = results.params["omega"], results.params["alpha[1]"], results.params["beta[1]"]
    VOLATILITY = np.sqrt(results.forecast(horizon=1).variance.values[-1, :][0])

    return VOLATILITY, [ALPHA, BETA, OMEGA]

# Function to get realized volatility
def realVol(returns, steps):

    # Assigning a small np array
    volarr = np.zeros(1)

    # Find each volatility
    for i in range(1000, 1000+steps):
        partial = returns[:i][-180:]
        volarr = np.append(volarr, np.std(partial))
    
    # Deleting the first element (0) and returning
    return np.delete(volarr, 0)

# Heston model function to model volatility
def heston(kappa, theta, v0):
    
    # Heston parameters
    sigma = 0.6
    T = 2.75          
    N = 1001       
    dt = T / N

    # Simulate
    np.random.seed(GLOBAL_SEED)
    vt = np.zeros(N)
    vt[0] = v0

    for t in range(1, N):
        z = np.random.normal()
        vt[t] = vt[t-1] + kappa * (theta - vt[t-1]) * dt + sigma * np.sqrt(max(vt[t-1], 0)) * np.sqrt(dt) * z
        vt[t] = max(vt[t], 0)

    return np.sqrt(vt)