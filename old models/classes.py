import numpy as np
import matplotlib.pyplot as plt

class ContrailClimateImpact:
    def __init__(self, time_horizons, baseline_RF_per_fuel, LNG_G_factor, thermal_efficiency_increase, climate_sensitivity):
        self.time_horizons = time_horizons
        self.baseline_RF_per_fuel = baseline_RF_per_fuel
        self.LNG_G_factor = LNG_G_factor
        self.thermal_efficiency_increase = thermal_efficiency_increase
        self.climate_sensitivity = climate_sensitivity

    def contrail_RF_per_unit_fuel(self, fuel_type, penalty):
        """
        Estimate the radiative forcing (RF) per unit fuel burned.
        :param fuel_type: 'Jet-A' or 'LNG'
        :param penalty: Fuel penalty as a percentage
        :return: Adjusted RF value (mW/m^2)
        """
        base_RF = self.baseline_RF_per_fuel
        if fuel_type == 'LNG':
            base_RF *= self.LNG_G_factor
        penalty_factor = 1 + penalty / 100
        return base_RF * penalty_factor

    def temperature_change_from_RF(self, RF):
        """
        Estimate temperature change from RF.
        :param RF: Radiative forcing in W/m^2
        :param sensitivity: Climate sensitivity in °C/(W/m^2)
        :return: Temperature change in °C
        """
        return RF * self.climate_sensitivity

    def integrate_climate_impact(self, RF_values, years):
        """
        Integrate RF or temperature change over time horizons.
        :param RF_values: List or array of RF values (in W/m^2)
        :param years: List of time horizons to integrate over
        :return: Integrated impacts for each time horizon
        """
        impacts = {}
        for horizon in years:
            impacts[horizon] = np.sum(RF_values[:horizon])
        return impacts

    def run_scenario(self, fuel_penalty):
        scenarios = {}
        baseline_RF = self.contrail_RF_per_unit_fuel('Jet-A', penalty=fuel_penalty)
        baseline_temp_change = self.temperature_change_from_RF(baseline_RF / 1000)
        scenarios['Baseline'] = {
            'RF': baseline_RF,
            'Temperature Change': baseline_temp_change
        }
        LNG_RF = self.contrail_RF_per_unit_fuel('LNG', penalty=fuel_penalty)
        LNG_temp_change = self.temperature_change_from_RF(LNG_RF / 1000)
        scenarios['LNG'] = {
            'RF': LNG_RF,
            'Temperature Change': LNG_temp_change
        }
        efficiency_adjusted_RF = baseline_RF * (1 - self.thermal_efficiency_increase)
        efficiency_temp_change = self.temperature_change_from_RF(efficiency_adjusted_RF / 1000)
        scenarios['Higher Efficiency'] = {
            'RF': efficiency_adjusted_RF,
            'Temperature Change': efficiency_temp_change
        }
        RF_values = np.array([scenarios[key]['RF'] for key in scenarios])
        integrated_RF = self.integrate_climate_impact(RF_values, self.time_horizons)
        scenarios['Integrated RF'] = integrated_RF
        return scenarios

    def visualize_results(self, results, fuel_penalty, variables=None, plot_type='bar'):
        """
        Visualize the results for selected variables.
        :param results: Scenario results dictionary
        :param fuel_penalty: Fuel penalty as a percentage
        :param variables: List of variables to plot (e.g., ['Baseline', 'LNG', 'Higher Efficiency'])
        :param plot_type: Type of plot ('bar' or 'line')
        """
        if variables is None:
            variables = ['Baseline', 'LNG', 'Higher Efficiency']
        
        selected_results = {key: results[key] for key in variables}
        RF_values = [data['RF'] for data in selected_results.values()]
        temp_changes = [data['Temperature Change'] for data in selected_results.values()]
        
        # Plotting
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

    def compare_scenarios(self, all_results, variables=None):
        """
        Compare multiple scenarios (e.g., fuel penalties) for selected variables.
        :param all_results: Dictionary of results for each fuel penalty
        :param variables: List of variables to compare (e.g., ['Baseline', 'LNG'])
        """
        if variables is None:
            variables = ['Baseline', 'LNG', 'Higher Efficiency']
        
        fig, ax = plt.subplots()
        for penalty, results in all_results.items():
            selected_results = {key: results[key] for key in variables}
            RF_values = [data['RF'] for data in selected_results.values()]
            ax.plot(variables, RF_values, marker='o', label=f"Penalty {penalty}")

        ax.set_title("Radiative Forcing Across Scenarios")
        ax.set_ylabel("RF (mW/m^2)")
        ax.set_xlabel("Scenarios")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Initialize the model
    model = ContrailClimateImpact(
        time_horizons=[20, 100, 500],
        baseline_RF_per_fuel=30,
        LNG_G_factor=0.8,
        thermal_efficiency_increase=0.15,
        climate_sensitivity=0.8
    )

    # Run scenarios for multiple fuel penalties
    fuel_penalties = [0.1, 0.5, 1, 2]
    all_results = {}
    for penalty in fuel_penalties:
        results = model.run_scenario(fuel_penalty=penalty)
        all_results[f'Penalty {penalty}%'] = results

        # Visualize results for each penalty
        model.visualize_results(results, fuel_penalty=penalty, variables=['Baseline', 'LNG', 'Higher Efficiency'], plot_type='bar')

    # Compare scenarios across penalties
    model.compare_scenarios(all_results, variables=['Baseline', 'LNG'])
