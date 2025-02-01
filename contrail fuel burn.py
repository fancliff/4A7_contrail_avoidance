import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def avoidance_eff(fuel_penalty):
    """
    Interpolates avoidance effectiveness based on data points from the paper:
    Evaluating Fuel-Climate Tradeoffs in Contrail Avoidance, Jad A. Elmourad
    Input: fuel_penalty (float) - Fuel penalty as a fraction.
    Output: avoidance_effectiveness (float) - Avoidance effectiveness as a fraction.
    """
    avoid_data = [0, 0.548, 0.73, 0.817, 0.85, 0.91, 0.935, 0.952, 0.963, 0.972, 0.977, 0.996, 1]
    fuel_pen_data = [0.0, 0.0009, 0.0017, 0.0023, 0.0026, 0.0032, 0.0035, 0.0037, 0.0039, 0.0041, 0.0042, 0.0048, 0.0051]
    if fuel_penalty <= min(fuel_pen_data):
        return min(avoid_data)
    elif fuel_penalty >= max(fuel_pen_data):
        return max(avoid_data)
    else:
        interp_func = interp1d(fuel_pen_data, avoid_data, kind="cubic")
        avoidance_effectiveness = interp_func(fuel_penalty)
        return avoidance_effectiveness

fuel_penalties = np.linspace(0, 0.02, 1000)
interpolated_avoidance = [avoidance_eff(pen) for pen in fuel_penalties]

plt.plot(fuel_penalties*100, interpolated_avoidance, label="Interpolated", color="orange")
plt.xlabel("Fuel Penalty (%)")
plt.ylabel("Avoidance Effectiveness (fraction)")
#plt.legend()
plt.grid()
plt.show()