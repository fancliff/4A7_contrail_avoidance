import numpy as np
import matplotlib.pyplot as plt
from numba import jit

class CO2_climate_impact:
    def __init__(self,
                fuel,
                penalty,
                CO2_params,
                climate_params,
                horizons):
        self.fuel = fuel
        self.penalty = penalty
        self.CO2_params = CO2_params
        self.climate_params = climate_params
        self.horizons = horizons
        self.years = np.arange(0, max(horizons)+1)  # Array of years from 0 to max_years
    
    def  calc_CO2_per_unit_energy(self):
        self.CO2_per_unit_energy = self.fuel['EI_CO2'] / self.fuel['LCV'] / self.fuel['eta']
    
    def calc_CO2_impulse_response(self):
        a0 = self.CO2_params['a0']
        a = self.CO2_params['a_i']
        tau = self.CO2_params['tau_i']
        IRF = a0 + sum(ai * np.exp(-self.years / ti) for ai, ti in zip(a, tau))
        
        radiative_efficiency = self.CO2_params['radiative_efficiency']
        self.RF = IRF*radiative_efficiency
    
    def calc_temperature_change(self):
        C = self.climate_params['C']
        lambda_param = self.climate_params['lambda']
        RF = self.RF
        self.deltaT = np.zeros_like(self.years)
        for t in range(1, len(self.years)):
            dT_dt = (RF[t] - lambda_param * self.deltaT[t-1]) / C
            self.deltaT[t] = self.deltaT[t-1] + dT_dt
    
    def integrate_t_data(data, horizon):
        return np.sum(data[:horizon])
    
    def run_scenario(self):
        self.calc_CO2_per_unit_energy()
        self.calc_CO2_impulse_response()
        self.calc_temperature_change()
        self.integrated_RF = {horizon: self.integrate_t_data(self.RF, horizon) for horizon in self.horizons}
        self.integrated_deltaT = {horizon: self.integrate_t_data(self.deltaT, horizon) for horizon in self.horizons}





class contrail_climate_impact:
    def __init__(self,
                fuel,
                baseline_contrail_RF,
                baseline_fuel,
                contrail_params,
                climate_params,
                horizons
                ):
        self.fuel = fuel
        self.RF0 = baseline_contrail_RF # needs to be per unit baseline fuel burned
        self.baseline_fuel = baseline_fuel
        self.horizons = horizons
        self.years = np.arange(0, max(horizons)+1)
        self.contrail_params = contrail_params
        self.climate_params = climate_params

    def calculate_G(fuel):
        return fuel['EI_H2O'] / (fuel['LCV'] * (1 - fuel['eta']))
    
    def contrail_RF_per_unit_fuel(self):
        self.RF0 *= self.calculate_G(self.fuel) / self.calculate_G(self.baseline_fuel)
    
    def contrail_RF_per_unit_energy(self):
        self.RF0 *= self.baseline_fuel['LCV'] / self.fuel['LCV']
        self.RF0 *= self.baseline_fuel['eta'] / self.fuel['eta']
    
    def contrail_RF_contrail_avoidance(self):
        self.RF0 *= (1-self.contrail_params['avoidance_success_rate'])
    
    def contrail_RF_erf(self):
        self.RF0 *= self.contrail_params['erf']
    
    def calc_RF(self):
        self.RF0 = self.RF0
        self.contrail_RF_per_unit_fuel()
        self.contrail_RF_per_unit_energy()
        self.contrail_RF_contrail_avoidance()
        self.contrail_RF_erf()
    
    # Assume RF is constant for all time due to very short lifespan of contrails
    def calc_temperature_change(self):
        C = self.climate_params['C']
        lambda_param = self.climate_params['lambda']
        RF = self.RF
        self.deltaT = np.zeros_like(self.years)
        for t in range(1, len(self.years)):
            dT_dt = (RF - lambda_param * self.deltaT[t-1]) / C
            self.deltaT[t] = self.deltaT[t-1] + dT_dt
    
    def integrate_t_data(data, horizon):
        return np.sum(data[:horizon])
    
    def run_scenario(self):
        self.calc_RF()
        self.calc_temperature_change()
        self.integrated_RF = {horizon: self.integrate_t_data(self.RF, horizon) for horizon in self.horizons}
        self.integrated_deltaT = {horizon: self.integrate_t_data(self.deltaT, horizon) for horizon in self.horizons}


