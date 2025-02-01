import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# COMBINED CLIMATE IMPACT MODEL
# - Combines CO2 and contrail climate impact models
# - Calculates radiative forcing (RF) and temperature change due to contrail avoidance
# - As well as time integrated RF and temperature change for different time horizons
# - Ignores baseline contrail emissions and baseline CO2 emissions
# - Only considers additional CO2 emissions due to contrail avoidance 
# - And the reduced RF of contrails due to contrail avoidance

# ASSUMPTIONS:
# - CO2 emissions per unit energy are constant - no economies of scale or technological improvements
# - Contrail RF is constant
# - Contrail ERF is constant
# - Climate parameters are constant
# - Global aviation energy use is constant
# - Avoidance success rate and penalty are constant
# - Assumes radiative efficiency of CO2 and CH4 are constant
# - Assumes CO2 and CH4 impulse response functions are constant


# LIMITATIONS:


class combined_climate_impact:
    def __init__(self,
                fuel,
                baseline_fuel,
                CO2_params,
                CH4_params,
                contrail_params,
                climate_params,
                avoidance_params,
                global_aviation_energy_use,
                horizons,
                ):
        self.fuel = fuel
        self.baseline_fuel = baseline_fuel
        self.CO2_params = CO2_params
        self.CH4_params = CH4_params
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
        radiative_efficiency_CO2 = self.CO2_params['radiative_efficiency']
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
            self.RF_CO2 += shifted_IRF * radiative_efficiency_CO2
    
    def calc_CH4_per_unit_energy(self):
        # Calculate CH4 emissions per unit energy (MJ)
        return self.fuel['EI_CH4'] / (self.fuel['LCV'] * self.fuel['eta'])
    
    def calc_CH4_impulse_response(self):
        # Calculate impulse response function for CH4
        # As a fraction of CH4 remaining after n years
        a0 = self.CH4_params['a0']
        a = self.CH4_params['a_i']
        tau = self.CH4_params['tau_i']
        self.IRF_CH4 = a0 + sum(ai * np.exp(-self.years / ti) for ai, ti in zip(a, tau))
    
    def calc_RF_CH4(self):
        # Initialize RF_CO2 array to store the cumulative RF over 500 years
        self.RF_CH4 = np.zeros(len(self.years))
        radiative_efficiency_CH4 = self.CH4_params['radiative_efficiency']
        self.calc_CH4_impulse_response()
        # Loop through each year and add the contribution of that year's emission
        for year in self.years:
            # Shifted impulse response for emissions in each year
            CH4_emitted = self.calc_CH4_per_unit_energy() * self.global_aviation_energy_use # kg CH4
            # Extra emissions due to contrail avoidance
            # If contrail avoidance is 0 then CH4_contrail_avoidance = 0
            CH4_contrail_avoidance = CH4_emitted * (self.avoidance_params['penalty']) # kg CH4
            
            #can include total emissions here but chosen to just include extra emissions due to contrail avoidance
            
            CH4_IRF = self.IRF_CH4 * CH4_contrail_avoidance # kg CH4
            shifted_IRF = np.roll(CH4_IRF, year)
            shifted_IRF[:year] = 0  # No effect before emission year

            # Add contribution to RF_CO2
            self.RF_CH4 += shifted_IRF * radiative_efficiency_CH4
    
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
        self.RF_total = self.RF_CO2 + self.RF_CH4 + self.RF_contrail
    
    def calc_temperature_change(self):
        C = self.climate_params['C']
        inv_lambda_param = self.climate_params['inv_lambda']
        RF = self.RF_total
        self.deltaT = np.zeros(len(self.years))
        for t in range(1, len(self.years)):
            dT_dt = (RF[t] - inv_lambda_param * self.deltaT[t-1]) / C
            self.deltaT[t] = self.deltaT[t-1] + dT_dt
    
    def integrate_t_data(self,data):
        integrated_data = np.zeros(len(self.years))
        for year in self.years:
            integrated_data[year] = np.sum(data[:year])
        return integrated_data
    
    def run_scenario(self):
        self.calc_RF_CO2()
        self.calc_RF_CH4()
        self.calc_RF_contrail()
        self.calc_RF_total()
        self.calc_temperature_change()
        self.RF_horizons = {horizon: self.RF_total[horizon] for horizon in self.horizons}
        self.deltaT_horizons = {horizon: self.deltaT[horizon] for horizon in self.horizons}
        self.integrated_RF = self.integrate_t_data(self.RF_total)
        self.integrated_deltaT = self.integrate_t_data(self.deltaT)
    
    def visualise_model_results(self):
        plt.plot(self.years, self.RF_contrail, label='Reduction in Contrail RF')
        plt.plot(self.years, self.RF_CO2, label='CO2 RF (due to contrail avoidance)')
        plt.plot(self.years, self.RF_CH4, label='CH4 RF (due to contrail avoidance)')
        plt.plot(self.years, self.RF_total, label='Total RF')
        plt.xlabel('Years')
        plt.ylabel('Radiative Forcing (W/m^2)')
        plt.legend(loc='best')
        plt.show()
        
        plt.plot(self.years, self.deltaT, label='Temperature Change')
        plt.xlabel('Years')
        plt.ylabel('Temperature change (K)')
        plt.legend(loc='best')
        plt.show()
        
        plt.plot(self.years, self.integrated_RF, label='Integrated RF')
        plt.xlabel('Years')
        plt.ylabel('Integrated RF (W.yrs/m^2)')
        plt.legend(loc='best')
        plt.show()
        
        plt.plot(self.years, self.integrated_deltaT, label='Integrated Temperature Change')
        plt.xlabel('Years')
        plt.ylabel('Integrated Temperature Change (K.yrs)')
        plt.legend(loc='best')
        plt.show()
        
        print(f'Radiative forcing horizons: {self.RF_horizons}')
        print(f'Temperature change horizons: {self.deltaT_horizons}')


######## Visualisation ########

# Plot results for multiple scenarios
def visualise_results_for_multiple_scenarios(models, plot_labels):
    # Create subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()  # Flatten to iterate over axes

    # Radiative Forcing comparison
    for model, label in zip(models, plot_labels):
        axes[0].plot(model.years, model.RF_total, label=label)
    axes[0].plot(model.years, [0] * len(model.years), 'k-', linewidth=1)
    axes[0].set_xlim(0, model.years[-1])
    axes[0].set_xlabel('Years')
    axes[0].set_ylabel('RF ($W/m^2$)')
    axes[0].set_title(f'Radiative Forcing')

    # Temperature change comparison
    for model, label in zip(models, plot_labels):
        axes[1].plot(model.years, model.deltaT, label=label)
    axes[1].plot(model.years, [0] * len(model.years), 'k-', linewidth=1)
    axes[1].set_xlim(0, model.years[-1])
    axes[1].set_xlabel('Years')
    axes[1].set_ylabel('$\Delta T$ ($K$)')
    axes[1].set_title(f'Temperature Change - AGTP')

    # Integrated RF comparison
    for model, label in zip(models, plot_labels):
        axes[2].plot(model.years, model.integrated_RF, label=label)
    axes[2].plot(model.years, [0] * len(model.years), 'k-', linewidth=1)
    axes[2].set_xlim(0, model.years[-1])
    axes[2].set_xlabel('Years')
    axes[2].set_ylabel('AGWP ($W.yrs/m^2$)')
    axes[2].set_title(f'Integrated RF - AGWP')

    # Integrated temperature change comparison
    for model, label in zip(models, plot_labels):
        axes[3].plot(model.years, model.integrated_deltaT, label=label)
    axes[3].plot(model.years, [0] * len(model.years), 'k-', linewidth=1)
    axes[3].set_xlim(0, model.years[-1])
    axes[3].set_xlabel('Years')
    axes[3].set_ylabel('iAGTP ($K.yrs$)')
    axes[3].set_title(f'Integrated $\Delta T$ - iAGTP')

    # Adjust layout and add minor ticks
    for ax in axes:
        ax.grid(which='major', axis='both', linestyle='-')
        ax.grid(which='minor', axis='both', linestyle=':')
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(20))
        # Get default major ticks
        major_ticks = ax.yaxis.get_majorticklocs()

        # Calculate minor tick interval (5 per major interval)
        if len(major_ticks) > 1:  # Ensure there are at least two major ticks
            major_interval = major_ticks[1] - major_ticks[0]
            minor_interval = major_interval / 5
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(minor_interval))

    # Add one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=2)

    # Tight layout
    fig.tight_layout(rect=[0.0, 0.09, 1, 0.95])  # Leave space at bottom for legend
    fig.suptitle(f'Fuel type: {models[0].fuel["fuel_type"]}, $\eta$: {models[0].fuel["eta"]:0.3f}')
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

# Example format for results:
def create_table(models, plot_labels):
    # Gather data from models using print_results logic
    results = []
    for model in models:
        RF_horizons = model.RF_horizons
        deltaT_horizons = model.deltaT_horizons

        # Append the data row for this model
        results.append({
            'avoidance': model.avoidance_params['avoidance_success_rate'],
            'penalty': model.avoidance_params['penalty'],
            'RF_20': RF_horizons[20],
            'RF_100': RF_horizons[100],
            'RF_500': RF_horizons[500],
            'deltaT_20': deltaT_horizons[20],
            'deltaT_100': deltaT_horizons[100],
            'deltaT_500': deltaT_horizons[500],
        })

    # Generate the table format
    print()
    print(f"{'Avoidance':<10} {'Penalty':<10} {'RF_20':<10} {'RF_100':<10} {'RF_500':<10} {'deltaT_20':<10} {'deltaT_100':<10} {'deltaT_500':<10}")
    for result in results:
        print(f"{result['avoidance']:<10} {result['penalty']:<10} {result['RF_20']:<10.3f} {result['RF_100']:<10.3f} {result['RF_500']:<10.3f} {result['deltaT_20']:<10.3f} {result['deltaT_100']:<10.3f} {result['deltaT_500']:<10.3f}")


######## Testing scenarios ########

def test_avoidance_success_rate(fuel, baseline_fuel, avoidance_scenarios, printing=False):

    # Store models for each scenario
    models = []
    for avoidance_params in avoidance_scenarios:
        model = combined_climate_impact(
            fuel=fuel,
            baseline_fuel=baseline_fuel,
            CO2_params=CO2_params,
            CH4_params=CH4_params,
            contrail_params=contrail_params,
            climate_params=climate_params,
            avoidance_params=avoidance_params,
            global_aviation_energy_use=global_aviation_energy_use,
            horizons=horizons
        )
        model.run_scenario()
        models.append(model)

    # Visualize results for different scenarios
    labels = [f'Avoidance: {scenario["avoidance_success_rate"]*100:.1f}%, Penalty: {scenario["penalty"]*100:.1f}%' for scenario in avoidance_scenarios]
    visualise_results_for_multiple_scenarios(models, labels)
    
    if printing:
        #print_results(models, labels)
        create_table(models, labels)


def test_avoidance_penalty(fuel, baseline_fuel, avoidance_scenarios, printing=False):

    # Store models for each scenario
    models = []
    for avoidance_params in avoidance_scenarios:
        model = combined_climate_impact(
            fuel=fuel,
            baseline_fuel=baseline_fuel,
            CO2_params=CO2_params,
            CH4_params=CH4_params,
            contrail_params=contrail_params,
            climate_params=climate_params,
            avoidance_params=avoidance_params,
            global_aviation_energy_use=global_aviation_energy_use,
            horizons=horizons
        )
        model.run_scenario()
        models.append(model)

    # Visualize results for different scenarios
    labels = [f'Avoidance: {scenario["avoidance_success_rate"]*100:.1f}%, Penalty: {scenario["penalty"]*100:.1f}%' for scenario in avoidance_scenarios]
    visualise_results_for_multiple_scenarios(models, labels)
    
    if printing:
        #print_results(models, labels)
        create_table(models, labels)



######## Main ########

if __name__ == '__main__':

######## Baseline fuel data ########

    # Baseline Jet-A1 fuel data
    jetA1_data = {
        'fuel_type': 'Jet-A1',
        'EI_CO2': 3.14,  # kg CO2 per kg fuel (combustion only)
        'EI_CH4': 0.0,  # kg CH4 per kg fuel
        'EI_H2O': 1.28,  # kg H2O per kg fuel
        'LCV': 43.2,  # MJ/kg
        'eta': 0.37  # Overall Efficiency
    }

    # if well to pump emissions are included for JetA1 then EI_CO2 += 0.86
    jetA1_data['EI_CO2'] += 0.86
    
    #Increased efficiency JetA1 data
    eff_inc = 0.2
    jetA1_high_eff_data = jetA1_data.copy()
    jetA1_high_eff_data['eta'] *= (1 + eff_inc)

    # LNG fuel data
    LNG_fuel_data = {
        'fuel_type': 'LNG',
        'EI_CO2': 2.75,  # kg CO2 per kg fuel (combustion only)
        'EI_CH4': 0.0,  # kg CH4 per kg fuel
        'EI_H2O': 2.25,  # kg H2O per kg fuel
        'LCV': 48.6,  # MJ/kg
        'eta': 0.37  # Thermal Efficiency
    }
    
    LNG_fuel_data['EI_CO2'] +=  1.36566
    LNG_fuel_data['EI_CH4'] += 0.045198
    
######## C02 data ########
    
    radiative_eff_CO2 = 1.33e-5  # W/m^2 per ppb
    m_atm = 5.1e18  # kg
    m_co2 = 1 # kg
    MW_atm = 28.97  # g/mol
    MW_co2 = 44.01  # g/mol
    ppb_per_kgCO2 = m_co2 / m_atm * MW_atm / MW_co2 * 1e9
    radiative_eff_per_kgCO2 = radiative_eff_CO2 * ppb_per_kgCO2 # W/m^2 per kg CO2
    # print(f'radiative efficiency / kgCO2: {radiative_eff_per_kgCO2}')
    # 1.7517783536870108e-15
    
    CO2_params = {
        'a0': 0.2173,
        'a_i': [0.2240, 0.2824, 0.2763],
        'tau_i': [394.4, 36.54, 4.304],
        'radiative_efficiency': radiative_eff_per_kgCO2 # W/m^2 per kg CO2
    }

######## CH4 data ########

    radiative_eff_CH4 = 3.88e-4  # W/m^2 per ppb
    m_ch4 = 1 # kg
    MW_ch4 = 16.04  # g/mol
    ppb_per_kgCH4 = m_ch4 / m_atm * MW_atm / MW_ch4 * 1e9
    radiative_eff_per_kgCH4 = radiative_eff_CH4 * ppb_per_kgCH4 # W/m^2 per kg CH4
    # print(f'radiative efficiency / kgCH4: {radiative_eff_per_kgCH4}')
    # 1.2735389091374127e-13
    
    CH4_params = {
        'a0': 0.0,
        'a_i': [1],
        'tau_i': [11.8], # years
        'radiative_efficiency': radiative_eff_per_kgCH4 # W/m^2 per kg CH4
    }

######## Contrail data ########
    
    global_avg_contrail_rf = 0.1114 # W/m^2 (lecture notes)
    global_aviation_fuel_use = 2.344e11 # kg
    LCV = jetA1_data['LCV'] # MJ/kg
    global_aviation_energy_use = global_aviation_fuel_use * LCV # MJ
    print(f'global aviation energy use: {global_aviation_energy_use:.2e} MJ')
    contrail_rf_per_unit_energy = global_avg_contrail_rf / global_aviation_energy_use # W/m^2 per MJ
    
    contrail_params = {
        'baseline_RF': global_avg_contrail_rf,
        'erf': 0.42
    }
    
######## Climate data ########
    
    climate_params = {
        'C': 17, # W yr/m^2 per K (from lecture notes)
        'inv_lambda': 0.8 # W/m^2 per K
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
        printing=False
        )

    test_avoidance_success_rate(
        fuel = jetA1_high_eff_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=success_rates,
        printing=False
        )


    test_avoidance_success_rate(
        fuel = LNG_fuel_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=success_rates,
        printing=False
        )


    # Test avoidance penalty
    # select baseline avoidance success rate
    # e.g 10% of flights form 80% of contrail RF # find exact figures from paper
    # avoid contrails on these flights with 60% success rate
    # low success rate due to inaccuracy of predicting contrail formation
    # total contrail RF reduction = 0.8 * 0.6 = 0.48 ~ 50%
    penalties = [
        {'avoidance_success_rate': 0.5, 'penalty': 0.001},
        {'avoidance_success_rate': 0.5, 'penalty': 0.002},
        {'avoidance_success_rate': 0.5, 'penalty': 0.005},
        {'avoidance_success_rate': 0.5, 'penalty': 0.01},
        {'avoidance_success_rate': 0.5, 'penalty': 0.02},
    ]

    test_avoidance_penalty(
        fuel = jetA1_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=penalties,
        printing=True
        )

    test_avoidance_penalty(
        fuel = jetA1_high_eff_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=penalties,
        printing=True
        )

    test_avoidance_penalty(
        fuel = LNG_fuel_data,
        baseline_fuel = jetA1_data,
        avoidance_scenarios=penalties,
        printing=True
    )