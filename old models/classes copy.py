import numpy as np
import matplotlib.pyplot as plt

class ContrailClimateImpact:
    def __init__(self, params):
        """
        Initialize the ContrailClimateImpact class.
        :param params: Dictionary of model parameters:
                        - time_horizons
                        - baseline_RF
                        - thermal_efficiency_increase
                        - climate_sensitivity
        """
        self.time_horizons = params['time_horizons']
        # Can be changed to a dictionary with fuel type as key
        # If using different RF values for different fuels 
        # Due to changes in EI_N(CO2)
        # Code will need to be updated so that G is not adjusted and recalculated etc
        self.baseline_RF = params['baseline_RF']
        self.thermal_efficiency_increase = params['thermal_efficiency_increase']
        self.climate_sensitivity = params['climate_sensitivity']
        
        # Initialize measurements dictionary
        self.measurements = {
            'RF' : 0,
            'temp_change' : 0,
            'integrated_RF' : 0,
            'integrated_temp_change' : 0,
        }

    def calculate_G(self, fuel_data):
        """
        Calculate the contrail mixing line gradient (G).
        :param fuel_data: Dictionary containing EI_H2O, LCV, eta for the fuel.
        :return: Gradient (G) value
        """
        return fuel_data['EI_H2O'] / (fuel_data['LCV'] * (1 - fuel_data['eta']))

    def contrail_RF_per_unit_fuel(self, fuel_data, baseline_data):
        """
        Calculate RF per unit fuel burned
        Accounting for changes in G due to fuel properties:
            - EI_H2O
            - LCV
            - eta
        :return: Adjusted RF value (mW/m^2) / unit fuel burned
        """
        # Calculate G for the non-baseline fuel
        delta_G = self.calculate_G(fuel_data)/self.calculate_G(baseline_data)
        self.measurements['RF'] *= delta_G
    
    def contrail_RF_per_unit_energy(self, fuel_data, baseline_data):
        """
        Adjust RF for changes in fuel burn due to fuel properties:
            - LCV
            - eta
        :return: Adjusted RF value (mW/m^2) / unit propulsive energy
        """
        # Calculate fuel burn increase for the non-baseline fuel
        delta_fuel = baseline_data['LCV']/fuel_data['LCV']
        delta_fuel *= baseline_data['eta']/fuel_data['eta']
        self.measurements['RF'] *= delta_fuel
    
    def contrail_RF_contrail_avoidance(self, penalty):
        """
        Adjust RF for contrail avoidance
        :return: Adjusted RF value (mW/m^2) / unit propulsive energy
        """
        penalty_factor = 1 + penalty / 100
        self.measurements['RF'] *= penalty_factor
    
    def get_RF(self, fuel_data, baseline_data, penalty):
        if fuel_data['baseline']:
            self.measurements['RF'] = self.baseline_RF
        else:
            self.contrail_RF_per_unit_fuel(self, fuel_data, baseline_data)
            self.contrail_RF_per_unit_energy(self, fuel_data, baseline_data)
            self.contrail_RF_contrail_avoidance(self, penalty)

    def temperature_change_from_RF(self):
        temp_change = self.measurements['RF'] * self.climate_sensitivity
        self.measurements['temp_change'] = temp_change

    def integrate_climate_impact(self, RF_values, years):
        impacts = {}
        for horizon in years:
            impacts[horizon] = np.sum(RF_values[:horizon])
        return impacts

    def get_measurements(self, fuel_data, baseline_data, penalty):
        self.get_RF(self, fuel_data, jetA1_data, penalty)
        self.temperature_change_from_RF(self)


    def run_scenario(self, fuel_penalty, jetA1_data, LNG_fuel_data, jetA1_high_eff_data):
        """
        Run a scenario for contrail radiative forcing and temperature change.
        :param fuel_penalty: Fuel penalty as a percentage
        :param jetA1_data: Dictionary for Jet-A1 fuel data
        :param LNG_fuel_data: Dictionary for LNG fuel data
        :param jetA1_high_eff_data: Dictionary for Jet-A1 fuel data with increased efficiency
        :return: Dictionary of results
        """
        scenarios = {}
        
        # Baseline scenario (Jet-A1 with baseline data)
        baseline_RF = self.contrail_RF_per_unit_fuel(jetA1_data, jetA1_data)
        baseline_temp_change = self.temperature_change_from_RF(baseline_RF / 1000)
        scenarios['Baseline'] = {
            'RF': baseline_RF,
            'Temperature Change': baseline_temp_change
        }

        # Alternative fuel (LNG)
        alt_RF = self.contrail_RF_per_unit_fuel(LNG_fuel_data, jetA1_data)
        alt_temp_change = self.temperature_change_from_RF(alt_RF / 1000)
        scenarios['Alternative Fuel (LNG)'] = {
            'RF': alt_RF,
            'Temperature Change': alt_temp_change
        }

        # Higher efficiency scenario (increased efficiency for Jet-A1)
        efficiency_adjusted_RF = self.contrail_RF_per_unit_fuel(jetA1_high_eff_data, jetA1_data)
        efficiency_temp_change = self.temperature_change_from_RF(efficiency_adjusted_RF / 1000)
        scenarios['Higher Efficiency'] = {
            'RF': efficiency_adjusted_RF,
            'Temperature Change': efficiency_temp_change
        }

        # Integrated impact for all scenarios
        RF_values = np.array([scenarios[key]['RF'] for key in scenarios])
        integrated_RF = self.integrate_climate_impact(RF_values, self.time_horizons)
        scenarios['Integrated RF'] = integrated_RF
        
        return scenarios


    def visualize_results(self, results, fuel_penalty, variables=None, plot_type='bar'):
        if variables is None:
            variables = ['Baseline', 'Alternative Fuel', 'Higher Efficiency']
        
        selected_results = {key: results[key] for key in variables}
        RF_values = [data['RF'] for data in selected_results.values()]
        temp_changes = [data['Temperature Change'] for data in selected_results.values()]
        
        fig, ax1 = plt.subplots()

        if plot_type == 'bar':
            x = np.arange(len(variables))
            width = 0.35
            ax1.bar(x - width / 2, RF_values, width, label='Radiative Forcing (mW/m^2)')
            ax1.bar(x + width / 2, temp_changes, width, label='Temperature Change (°C)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(variables)
        elif plot_type == 'line':
            ax1.plot(variables, RF_values, label='Radiative Forcing (mW/m^2)', marker='o')
            ax1.plot(variables, temp_changes, label='Temperature Change (°C)', marker='s')

        ax1.set_title(f"Climate Impact for Fuel Penalty {fuel_penalty}%")
        ax1.set_ylabel("Value")
        ax1.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Model parameters dictionary
    params = {
        'time_horizons': [20, 100, 500],
        'baseline_RF': 30,
        'climate_sensitivity': 0.8
    }
    model = ContrailClimateImpact(params)

    # Jet-A1 fuel data
    jetA1_data = {
        'baseline': True,
        'fuel_type': 'Jet-A1',
        'EI_H2O': 1.25,  # kg H2O per kg fuel
        'LCV': 43,  # MJ/kg
        'eta': 0.39  # Efficiency
    }

    # LNG fuel data
    LNG_fuel_data = {
        'baseline': False,
        'fuel_type': 'LNG',
        'EI_H2O': 2.0,  # kg H2O per kg fuel
        'LCV': 50,  # MJ/kg
        'eta': 0.35  # Efficiency
    }
    
    #Increased efficiency JetA1 data
    eff_inc = 0.1
    jetA1_high_eff_data = {
        'baseline': False,
        'fuel_type': 'Jet-A1',
        'EI_H2O': 1.25,  # kg H2O per kg fuel
        'LCV': 43,  # MJ/kg
        'eta': jetA1_data['eta'] * (1+eff_inc) # Efficiency
    }

    # Run scenarios for fuel penalties
    fuel_penalties = [0.1, 0.5, 1, 2]
    for penalty in fuel_penalties:
        results = model.run_scenario(
            fuel_penalty=penalty,
            jetA1_data=jetA1_data,
            LNG_fuel_data=LNG_fuel_data
        )
        model.visualize_results(results, fuel_penalty=penalty)
