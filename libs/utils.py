import numpy as np
from arch import arch_model

# Global seed for fair comparison
GLOBAL_SEED = None

# Normal GARCH forecasting
# Takes final conditional volatility, forecast horizon and GARCH parameters
def garchforecast(volatility, nahead, params):
    
    GARCHforecasts = []
    GARCHforecasts.append(volatility)
    previousVariance = volatility ** 2

    ALPHA, BETA, OMEGA = params[0], params[1], params[2]
    for i in range(nahead-1):

        # GARCH equation
        newVariance = OMEGA + (ALPHA + BETA) * previousVariance
        previousVariance = newVariance

        # Appending GARCH volatility
        GARCHforecasts.append(np.sqrt(previousVariance))
    
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

# Heston model function to forecast volatility
def hestonforecast(kappa, theta, v0, sigma):

    # Time
    T = 2.75
    N = 1000

    # Use a modern random number generator
    rng = np.random.default_rng(GLOBAL_SEED)
    
    dt = T / N
    
    # Critical value for switching between distributions
    psi_c = 1.5 
    
    # Initialize variance array
    vt = np.zeros(N)
    vt[0] = v0

    for t in range(1, N):
        
        # Calculate moments for the next step's distribution
        m = theta + (vt[t-1] - theta) * np.exp(-kappa * dt)
        s2 = (vt[t-1] * sigma**2 * np.exp(-kappa * dt) / kappa * (1 - np.exp(-kappa * dt)) +
              theta * sigma**2 / (2 * kappa) * (1 - np.exp(-kappa * dt))**2)
        
        psi = s2 / m**2

        # Check if we should use the "Quadratic" or "Exponential" part of the scheme
        if psi <= psi_c:
            
            # "Exponential" approximation (more stable for low variance of variance)
            inv_psi = 1 / psi
            b2 = 2 * inv_psi - 1 + np.sqrt(2 * inv_psi * (2 * inv_psi - 1))
            a = m / (1 + b2)
            z = rng.standard_normal()
            vt[t] = a * (np.sqrt(b2) + z)**2
        else:
            
            # "Quadratic" approximation (for high variance of variance)
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m
            u = rng.uniform()
            
            if u <= p:
                vt[t] = 0.0
            else:
                vt[t] = np.log((1 - p) / (1 - u)) / beta
                
    return np.sqrt(vt)