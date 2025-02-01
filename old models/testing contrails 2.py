import numpy as np
import matplotlib.pyplot as plt

# Constants
C = 1.0  # Heat capacity, arbitrary units for simplicity
lambda_val = 0.1  # Climate feedback parameter (1 / climate sensitivity)
RF_0 = 10  # Initial radiative forcing, arbitrary units (W/m^2)
tau = 2  # Time constant for radiative forcing decay in hours

# Time range (in hours)
time = np.linspace(0, 100, 1000)  # From 0 to 100 hours

# Effective decay timescale for temperature
T_eff = C / lambda_val  # Effective timescale for temperature decay

# True solution for temperature change (based on full equation)
RF_t = RF_0 * np.exp(-time / tau)  # Exponential decay of RF
T_true = (RF_t / lambda_val)  # Temperature response from RF

# Approximate solution (instantaneous rise followed by exponential decay)
T_max = RF_0 / lambda_val  # Instantaneous maximum temperature rise
T_approx = T_max * np.exp(-time / T_eff)  # Decay from T_max

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot True Solution
plt.plot(time, T_true, label="True Solution (Radiative Forcing Decay)", color='b', linestyle='-', linewidth=2)

# Plot Approximate Solution
plt.plot(time, T_approx, label="Approximate Solution (Instantaneous Rise + Decay)", color='r', linestyle='--', linewidth=2)

# Labels and title
plt.xlabel("Time (hours)")
plt.ylabel("Temperature Change (°C or arbitrary units)")
plt.title("Temperature Change Due to Contrails (True vs Approximate)")
plt.legend(loc='upper right')

# Show grid and plot
plt.grid(True)
plt.tight_layout()
plt.show()
