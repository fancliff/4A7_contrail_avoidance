import numpy as np
import matplotlib.pyplot as plt

def simulate_temperature_change(RF, lambda_param, C, years):
    """
    Simulate temperature change from radiative forcing (RF).
    
    :param RF: Radiative forcing (W/m²)
    :param lambda_param: Climate feedback parameter (W/m²/°C)
    :param C: Effective heat capacity (W·yr/m²/°C)
    :param years: Number of years to simulate
    :return: Array of temperature change over time
    """
    # Equilibrium temperature and timescale
    T_equil = RF / lambda_param
    tau = C / lambda_param

    # Initialize temperature array
    T = np.zeros(years + 1)

    # Iterate temperature change year by year
    for t in range(1, years + 1):
        dT_dt = (RF - lambda_param * T[t-1]) / C
        T[t] = T[t-1] + dT_dt  # Increment temperature

    return T

# Parameters
RF = 3.7  # Radiative forcing for CO₂ doubling (W/m²)
lambda_param = 1.0  # Climate feedback parameter (W/m²/°C)
C = 4.0  # Effective heat capacity (W·yr/m²/°C)
years = 500  # Simulation period

# Simulate temperature change
T = simulate_temperature_change(RF, lambda_param, C, years)

# Plot results
plt.plot(np.arange(0, years + 1), T, label='Temperature Change')
plt.axhline(y=RF / lambda_param, color='r', linestyle='--', label='Equilibrium Temp (T_equil)')
plt.xlabel('Years')
plt.ylabel('Temperature Change (°C)')
plt.title('Temperature Change from Radiative Forcing')
plt.legend()
plt.grid()
plt.show()