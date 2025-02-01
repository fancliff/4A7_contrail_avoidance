import numpy as np
import matplotlib.pyplot as plt

# COMBINED CLIMATE IMPACT MODEL
# - Combines CO2 and contrail climate impact models
# - Calculates radiative forcing (RF) and temperature change due to contrail avoidance
# - As well as time integrated RF and temperature change for different time horizons
# - Ignores baseline contrail emissions and baseline CO2 emissions
# - Only considers additional CO2 emissions due to contrail avoidance 
# - And the reduced RF of contrails due to contrail avoidance

# ASSUMPTIONS:
# - CO2 emissions per unit energy are constant
# - CO2 impulse response function is constant
# - Contrail RF is constant
# - Contrail ERF is constant
# - Climate parameters are constant
# - Global aviation energy use is constant
# - Avoidance success rate and penalty are constant


# LIMITATIONS:
# - DOES NOT ACCOUNT FOR WELL TO PUMP FUEL EMISSIONS SO FUEL CO2 IMPACT OF CONTRAIL AVOIDANCE IS UNDERESTIMATED


class combined_climate_impact:
    def __init__(self,
                fuel,
                baseline_fuel,
                CO2_params,
                contrail_params,
                climate_params,
                avoidance_params,
                global_aviation_energy_use,
                horizons,
                ):
        self.fuel = fuel
        self.baseline_fuel = baseline_fuel
        self.CO2_params = CO2_params
        self.contrail_params = contrail_params
        self.climate_params = climate_params
        self.avoidance_params = avoidance_params
        self.global_aviation_energy_use = global_aviation_energy_use
        self.horizons = horizons
        self.years = np.arange(0, max(horizons)+1)
    
    def calc_CO2_per_unit_energy(self):
        # Calculate CO2 emissions per unit energy (MJ)
        return self.fuel['EI_CO2'] / (self.fuel['LCV'] * self.fuel['eta'])
    
    def calc_CO2_impulse_response(self):
        # Calculate impulse response function for CO2
        # As a fraction of CO2 remaining after n years
        a0 = self.CO2_params['a0']
        a = self.CO2_params['a_i']
        tau = self.CO2_params['tau_i']
        self.IRF_CO2 = a0 + sum(ai * np.exp(-self.years / ti) for ai, ti in zip(a, tau))
    
    def calc_RF_CO2(self):
        # Initialize RF_CO2 array to store the cumulative RF over 500 years
        self.RF_CO2 = np.zeros(len(self.years))
        radiative_efficiency = self.CO2_params['radiative_efficiency']
        self.calc_CO2_impulse_response()
        # Loop through each year and add the contribution of that year's emission
        for year in self.years:
            # Shifted impulse response for emissions in each year
            CO2_emitted = self.calc_CO2_per_unit_energy() * self.global_aviation_energy_use # kg CO2
            # Extra emissions due to contrail avoidance
            # If contrail avoidance is 0 then CO2_contrail_avoidance = 0
            CO2_contrail_avoidance = CO2_emitted * (self.avoidance_params['penalty']) # kg CO2
            
            #can include total emissions here but chosen to just include extra emissions due to contrail avoidance
            #CO2_contrail_avoidance = CO2_emitted * (1 + self.avoidance_params['penalty']) # kg CO2
            
            CO2_IRF = self.IRF_CO2 * CO2_contrail_avoidance # kg CO2
            shifted_IRF = np.roll(CO2_IRF, year)
            shifted_IRF[:year] = 0  # No effect before emission year

            # Add contribution to RF_CO2
            self.RF_CO2 += shifted_IRF * radiative_efficiency
    
    def calculate_G(self,fuel):
        return fuel['EI_H2O'] / (fuel['LCV'] * (1 - fuel['eta']))
    
    def fuel_adjust_contrail_RF(self):
        self.RF_contrail *= self.calculate_G(self.fuel) / self.calculate_G(self.baseline_fuel)
    
    # not used - not sure if correct
    # and not necessary as assuming contrail RF scales with G from above function
    #def energy_adjust_contrail_RF(self):
        #self.RF_contrail *= self.baseline_fuel['LCV'] / self.fuel['LCV']
        #self.RF_contrail *= self.baseline_fuel['eta'] / self.fuel['eta']
    
    def contrail_RF_contrail_avoidance(self):
        self.RF_contrail *= (1-self.avoidance_params['avoidance_success_rate'])
    
    def contrail_RF_erf(self):
        self.RF_contrail *= self.contrail_params['erf']
    
    def calc_RF_contrail(self):
        self.RF_contrail = np.ones(len(self.years))*self.contrail_params['baseline_RF']
        self.fuel_adjust_contrail_RF()
        
        # I don't think energy adjustment is necessary
        # And function is incorrect anyway
        #self.energy_adjust_contrail_RF()
        
        self.contrail_RF_contrail_avoidance()
        
        # Subtract baseline contrail RF from calculated contrail RF s.t.
        # Contrail RF is negative due to contrail avoidance
        # If no contrail avoidance then RF_contrail = 0
        self.RF_contrail -= np.ones(len(self.years))*self.contrail_params['baseline_RF'] # W/m^2
        
        # Scale by ERF factor for contrails
        self.contrail_RF_erf()
    
    def calc_RF_total(self):
        self.RF_total = self.RF_CO2 + self.RF_contrail
    
    def calc_temperature_change(self):
        C = self.climate_params['C']
        lambda_param = self.climate_params['lambda']
        RF = self.RF_total
        self.deltaT = np.zeros(len(self.years))
        for t in range(1, len(self.years)):
            dT_dt = (RF[t] - lambda_param * self.deltaT[t-1]) / C
            self.deltaT[t] = self.deltaT[t-1] + dT_dt
    
    def horizons_values(self,data,horizon):
        return np.sum(data[:horizon])
    
    def integrate_t_data(self,data):
        integrated_data = np.zeros(len(self.years))
        for year in self.years:
            integrated_data[year] = np.sum(data[:year])
        return integrated_data
    
    def run_scenario(self):
        self.calc_RF_CO2()
        self.calc_RF_contrail()
        self.calc_RF_total()
        self.calc_temperature_change()
        self.RF_horizons = {horizon: self.horizons_values(self.RF_total, horizon) for horizon in self.horizons}
        self.deltaT_horizons = {horizon: self.horizons_values(self.deltaT, horizon) for horizon in self.horizons}
        self.integrated_RF = self.integrate_t_data(self.RF_total)
        self.integrated_deltaT = self.integrate_t_data(self.deltaT)
    
    def visualise_model_results(self):
        plt.plot(self.years, self.RF_contrail, label='Contrail RF')
        plt.plot(self.years, self.RF_CO2, label='CO2 RF (due to contrail avoidance)')
        plt.plot(self.years, self.RF_total, label='Total RF')
        plt.xlabel('Years')
        plt.ylabel('Radiative Forcing (W/m^2)')
        plt.legend()
        plt.show()
        
        plt.plot(self.years, self.deltaT, label='Temperature Change')
        plt.xlabel('Years')
        plt.ylabel('Temperature change (K)')
        plt.legend()
        plt.show()
        
        plt.plot(self.years, self.integrated_RF, label='Integrated RF')
        plt.xlabel('Years')
        plt.ylabel('Integrated RF (W.yrs/m^2)')
        plt.legend()
        plt.show()
        
        plt.plot(self.years, self.integrated_deltaT, label='Integrated Temperature Change')
        plt.xlabel('Years')
        plt.ylabel('Integrated Temperature Change (K.yrs)')
        plt.legend()
        plt.show()
        
        print(f'Radiative forcing horizons: {self.RF_horizons}')
        print(f'Temperature change horizons: {self.deltaT_horizons}')


######## Visualisation ########

# Plot results for multiple scenarios
def visualise_results_for_multiple_scenarios(models, plot_labels):
    
    # Radiative Forcing comparison
    plt.figure(figsize=(10, 6))
    for model, label in zip(models, plot_labels):
        plt.plot(model.years,
                model.RF_total, 
                label=label
                )
    plt.plot(model.years, np.zeros(len(model.years)), 'k--')
    plt.xlabel('Years')
    plt.ylabel('Radiative Forcing (W/m^2)')
    plt.title(f'Radiative Forcing - fuel type: {models[0].fuel["fuel_type"]}, eta: {models[0].fuel["eta"]:0.3f}')
    plt.legend()
    plt.grid()
    plt.show()

    # Temperature change comparison
    plt.figure(figsize=(10, 6))
    for model, label in zip (models, plot_labels):
        plt.plot(model.years,
                model.deltaT, 
                label=label
                )
    plt.plot(model.years, np.zeros(len(model.years)), 'k--')
    plt.xlabel('Years')
    plt.ylabel('Temperature Change (K)')
    plt.title(f'Temperature Change - fuel type: {models[0].fuel["fuel_type"]}, eta: {models[0].fuel["eta"]:0.3f}')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Integrated RF comparison
    plt.figure(figsize=(10, 6))
    for model, label in zip(models, plot_labels):
        plt.plot(model.years,
                model.integrated_RF, 
                label=label
                )
    plt.plot(model.years, np.zeros(len(model.years)), 'k--')
    plt.xlabel('Years')
    plt.ylabel('Integrated Radiative Forcing (W.yrs/m^2)')
    plt.title(f'Integrated Radiative Forcing - fuel type: {models[0].fuel["fuel_type"]}, eta: {models[0].fuel["eta"]:0.3f}')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Integrated temperature change comparison
    plt.figure(figsize=(10, 6))
    for model, label in zip(models, plot_labels):
        plt.plot(model.years,
                model.integrated_deltaT, 
                label=label
                )
    plt.plot(model.years, np.zeros(len(model.years)), 'k--')
    plt.xlabel('Years')
    plt.ylabel('Integrated Temperature Change (K.yrs)')
    plt.title(f'Integrated Temperature Change - fuel type: {models[0].fuel["fuel_type"]}, eta: {models[0].fuel["eta"]:0.3f}')
    plt.legend()
    plt.grid()
    plt.show()


# Print final integrated results and horizons for each scenario
def print_results(models, plot_labels):
    # Print results for each scenario
    print(f'Fuel type: {models[0].fuel["fuel_type"]}, eta: {models[0].fuel["eta"]:3f}')
    print()
    for model, label in zip(models, plot_labels):
        print(label)
        print(f'Integrated RF: {model.integrated_RF[-1]}, Integrated Temperature Change: {model.integrated_deltaT[-1]}')
        print(f'RF horizons: {model.RF_horizons}')
        print(f'Temperature Change horizons: {model.deltaT_horizons}')
        print()


######## Testing scenarios ########

def test_avoidance_success_rate(fuel, baseline_fuel, avoidance_scenarios, print_results=False):

    # Store models for each scenario
    models = []
    for avoidance_params in avoidance_scenarios:
        model = combined_climate_impact(
            fuel=fuel,
            baseline_fuel=baseline_fuel,
            CO2_params=CO2_params,
            contrail_params=contrail_params,
            climate_params=climate_params,
            avoidance_params=avoidance_params,
            global_aviation_energy_use=global_aviation_energy_use,
            horizons=horizons
        )
        model.run_scenario()
        models.append(model)

    # Visualize results for different scenarios
    labels = [f'Success Rate: {scenario["avoidance_success_rate"]*100:.1f}%, Penalty: {scenario["penalty"]*100:.1f}%' for scenario in avoidance_scenarios]
    visualise_results_for_multiple_scenarios(models, labels)
    
    if print_results:
        print_results(models, labels)


def test_avoidance_penalty(fuel, baseline_fuel, avoidance_scenarios, print_results=False):

    # Store models for each scenario
    models = []
    for avoidance_params in avoidance_scenarios:
        model = combined_climate_impact(
            fuel=fuel,
            baseline_fuel=baseline_fuel,
            CO2_params=CO2_params,
            contrail_params=contrail_params,
            climate_params=climate_params,
            avoidance_params=avoidance_params,
            global_aviation_energy_use=global_aviation_energy_use,
            horizons=horizons
        )
        model.run_scenario()
        models.append(model)

    # Visualize results for different scenarios
    labels = [f'Success Rate: {scenario["avoidance_success_rate"]*100:.1f}%, Penalty: {scenario["penalty"]*100:.1f}%' for scenario in avoidance_scenarios]
    visualise_results_for_multiple_scenarios(models, labels)
    
    if print_results:
        print_results(models, labels)



######## Main ########

if __name__ == '__main__':

######## Baseline fuel data ########

    # Baseline Jet-A1 fuel data
    jetA1_data = {
        'fuel_type': 'Jet-A1',
        'EI_CO2': 3.14,  # kg CO2 per kg fuel (combustion only)
        'EI_H2O': 1.28,  # kg H2O per kg fuel
        'LCV': 43.2,  # MJ/kg
        'eta': 0.37  # Overall Efficiency
    }

    # if well to pump emissions are included for JetA1 then EI_CO2 += 0.86
    jetA1_data['EI_CO2'] += 0.86
    
    #Increased efficiency JetA1 data
    eff_inc = 0.1
    jetA1_high_eff_data = jetA1_data.copy()
    jetA1_high_eff_data['eta'] *= (1 + eff_inc)

    # LNG fuel data
    LNG_fuel_data = {
        'fuel_type': 'LNG',
        'EI_CO2': 2.75,  # kg CO2 per kg fuel (combustion only)
        'EI_H2O': 2.25,  # kg H2O per kg fuel
        'LCV': 48.6,  # MJ/kg
        'eta': 0.37  # Thermal Efficiency
    }
    
    # if well to pump emissions for LNG are included then:
    # EI_CO2 += 5.103 (GWP20) or 4.059 (GWP100)
    # high due to methane emissions
    LNG_fuel_data['EI_CO2'] += 5.103
    
######## C02 data ########
    
    radiative_eff = 1.37e-5  # W/m^2 per ppb
    m_atm = 5.1480e18  # kg
    m_co2 = 1 # kg
    MW_atm = 28.97  # g/mol
    MW_co2 = 44.01  # g/mol
    ppb_per_kgCO2 = m_co2 / m_atm * MW_atm / MW_co2 * 1e9
    radiative_eff_per_kgCO2 = radiative_eff * ppb_per_kgCO2 # W/m^2 per kg CO2
    # print(f'radiative efficiency / kgCO2: {radiative_eff_per_kgCO2}')
    # 1.7517783536870108e-15
    
    CO2_params = {
        'a0': 0.2173,
        'a_i': [0.2240, 0.2824, 0.2763],
        'tau_i': [394.4, 36.54, 4.304],
        'radiative_efficiency': radiative_eff_per_kgCO2 # W/m^2 per kg CO2
    }
    
######## Contrail data ########
    
    global_avg_contrail_rf = 0.1114 # W/m^2 (lecture notes)
    global_aviation_fuel_use = 2.344e11 # kg
    LCV = jetA1_data['LCV'] # MJ/kg
    global_aviation_energy_use = global_aviation_fuel_use * LCV # MJ
    contrail_rf_per_unit_energy = global_avg_contrail_rf / global_aviation_energy_use # W/m^2 per MJ
    
    contrail_params = {
        'baseline_RF': global_avg_contrail_rf,
        'erf': 0.42
    }
    
######## Climate data ########
    
    climate_params = {
        'C': 16.7, # W yr/m^2 per K (from lecture notes)
        'lambda': 0.75 # W/m^2 per K
    }
    
    horizons = [20, 100, 500]

######## Testing of scenarios ########

    # Test avoidance success rate

    success_rates = [
        {'avoidance_success_rate': 0.1, 'penalty': 0.01},
        {'avoidance_success_rate': 0.3, 'penalty': 0.01},
        {'avoidance_success_rate': 0.5, 'penalty': 0.01},
        {'avoidance_success_rate': 0.7, 'penalty': 0.01},
        {'avoidance_success_rate': 0.9, 'penalty': 0.01},
    ]

    test_avoidance_success_rate(
        fuel = jetA1_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=success_rates,
        print_results=False
        )

    test_avoidance_success_rate(
        fuel = jetA1_high_eff_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=success_rates,
        print_results=False
        )


    test_avoidance_success_rate(
        fuel = LNG_fuel_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=success_rates,
        print_results=False
        )


    # Test avoidance penalty

    penalties = [
        {'avoidance_success_rate': 0.5, 'penalty': 0.001},
        {'avoidance_success_rate': 0.5, 'penalty': 0.002},
        {'avoidance_success_rate': 0.5, 'penalty': 0.005},
        {'avoidance_success_rate': 0.5, 'penalty': 0.01},
        {'avoidance_success_rate': 0.5, 'penalty': 0.02},
        {'avoidance_success_rate': 0.5, 'penalty': 0.03},
        {'avoidance_success_rate': 0.5, 'penalty': 0.04},
        {'avoidance_success_rate': 0.5, 'penalty': 0.05},
    ]

    test_avoidance_penalty(
        fuel = jetA1_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=penalties,
        print_results=False
        )

    test_avoidance_penalty(
        fuel = jetA1_high_eff_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=success_rates,
        print_results=False
        )

    test_avoidance_penalty(
        fuel = LNG_fuel_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=penalties,
        print_results=False
    )




# Below not used as evaluation of different fuel and efficiency 
# should be done by changing inputs to above functions

'''

def test_different_fuel(fuel, baseline_fuel, avoidance_params):

    # Store models for each scenario
    models = []
    for fuel_data in [fuel, baseline_fuel]:
        model = combined_climate_impact(
            fuel=fuel_data,
            baseline_fuel=baseline_fuel,
            CO2_params=CO2_params,
            contrail_params=contrail_params,
            climate_params=climate_params,
            avoidance_params=avoidance_params,
            global_aviation_energy_use=global_aviation_energy_use,
            horizons=horizons
        )
        model.run_scenario()
        models.append(model)

    # Visualize results for different scenarios
    labels = [f'Fuel Type: {fuel_data["fuel_type"]}' for fuel_data in [fuel, baseline_fuel]]
    visualise_results_for_multiple_scenarios(models, labels)


def test_higher_efficiency(baseline_fuel_data, avoidance_params, efficiency_scenarios):

    # Store models for each scenario
    models = []
    for eff_inc in efficiency_scenarios:
        high_eff_data = baseline_fuel_data.copy()
        high_eff_data['eta'] *= (1 + eff_inc)
        
        model = combined_climate_impact(
            fuel=high_eff_data,
            baseline_fuel=baseline_fuel_data,
            CO2_params=CO2_params,
            contrail_params=contrail_params,
            climate_params=climate_params,
            avoidance_params=avoidance_params,
            global_aviation_energy_use=global_aviation_energy_use,
            horizons=horizons
        )
        model.run_scenario()
        models.append(model)

    # Visualize results for different scenarios
    labels = [f'Efficiency Increase: {eff_inc*100:.1f}%' for eff_inc in efficiency_scenarios]
    visualise_results_for_multiple_scenarios(models, labels)

'''