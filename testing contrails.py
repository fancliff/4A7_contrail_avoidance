import numpy as np
import matplotlib.pyplot as plt

def simulate_contrail_temperature_change(RF_0, tau, lambda_param, C, total_time_hours, time_step=1):
    """
    Simulate temperature change from contrail radiative forcing with exponential decay.
    
    :param RF_0: Initial radiative forcing (W/m²)
    :param tau: E-folding time of RF (hours)
    :param lambda_param: Climate feedback parameter (W/m²/°C)
    :param C: Effective heat capacity (W·yr/m²/°C)
    :param total_time_hours: Total simulation time (hours)
    :param time_step: Timestep for simulation (hours)
    :return: Arrays for time, RF, and temperature change over time
    """
    # Convert effective heat capacity to hourly scale (W·hr/m²/°C)
    C_hour = C * 24 * 365

    # Time array
    time = np.arange(0, total_time_hours + time_step, time_step)
    
    # Initialize RF and temperature arrays
    RF = RF_0 * np.exp(-time / tau)  # RF decays exponentially
    T = np.zeros_like(time)
    
    # Simulate temperature change over time
    for t in range(1, len(time)):
        dT_dt = (RF[t-1] - lambda_param * T[t-1]) / C_hour
        T[t] = T[t-1] + dT_dt * time_step  # Increment temperature by timestep

    return time, RF, T

# Parameters
RF_0 = 10  # Initial radiative forcing due to contrails (W/m²)
tau = 2  # E-folding time of contrails (hours)
lambda_param = 1.0  # Climate feedback parameter (W/m²/°C)
C = 4.0  # Effective heat capacity (W·yr/m²/°C)
total_time_hours = 1024  # Simulate over 48 hours
time_step = 0.5  # 30-minute timestep

# Run simulation
time, RF, T = simulate_contrail_temperature_change(RF_0, tau, lambda_param, C, total_time_hours, time_step)

# Plot results
plt.figure(figsize=(10, 5))

# Plot RF
plt.subplot(1, 2, 1)
plt.plot(time, RF, label='Radiative Forcing (RF)', color='blue')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Radiative Forcing Decay')
plt.xlabel('Time (hours)')
plt.ylabel('RF (W/m²)')
plt.grid()
plt.legend()

# Plot Temperature
plt.subplot(1, 2, 2)
plt.plot(time, T, label='Temperature Change (T)', color='red')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Temperature Response')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature Change (°C)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()