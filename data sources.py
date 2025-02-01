radiative efficiency of C02 is 1.33e-5 W/m^2 per ppb of CO2 
radiative efficiency of CH4 is 3.88e-4 W/m^2 per ppb of CH4
# https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter07_SM.pdf

mass of atmosphere is 5.1e18 kg # https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
MW of atmosphere is 28.97 g/mol # https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
MW of CO2 is 44.01 g/mol # no source needed
CO2 decay parameters are a0 = 0.217, a_i = [0.259, 0.338, 0.186, 0.118], tau_i = [394.4, 36.54, 4.304, 0.525] 
    # https://www.researchgate.net/publication/235431147_Carbon_dioxide_and_climate_impulse_response_functions_for_the_computation_of_greenhouse_gas_metrics_A_multi-model_analysis
CH4 decay parameters are a_1 = 1, tau_1 = 11.8 yrs
    # https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter07_SM.pdf


global average contrail RF is 0.1114 W/m^2 # https://www.sciencedirect.com/science/article/pii/S1352231020305689
contrail erf is 0.42 # https://www.sciencedirect.com/science/article/pii/S1352231020305689
global aviation fuel use is 2.344e11 kg/year
    # https://www.iata.org/en/iata-repository/publications/economic-reports/global-outlook-for-air-transport-june-2024-report/
    # https://www.statista.com/chart/32715/fuel-usage-and-fuel-spend-by-the-global-airline-industry/

earth system heat capacity is 17, # W yr/m^2 per K https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007JD008746
earth lambda is 0.8 # K per (W/m^2) https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007JD008746

fuel data is as follows:
Jet-A1: 
    EI_CO2 = 3.14, https://web.stanford.edu/group/haiwanglab/HyChem/fuels/A1_spec.html
        + 0.86 for well to pump emissions, https://pubs.rsc.org/en/content/articlehtml/2020/se/c9se00788a
        # estimate below not used as better source found
        # + 0.5 for well to pump emissions, https://offsetguide.org/understanding-carbon-offsets/air-travel-climate/climate-impacts-from-aviation/co2-emissions/
    EI_H2O = 1.28, https://web.stanford.edu/group/haiwanglab/HyChem/fuels/A1_spec.html
    LCV = 43.2, https://web.stanford.edu/group/haiwanglab/HyChem/fuels/A1_spec.html
    eta = 0.37, https://www.grida.no/climate/ipcc/aviation/097.htm

LNG: 
    EI_CO2 = 2.75, # assume pure methane
        + 1.36566 for well to pump emissions, https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.1934
    EI_CH4 = 0.045198 for well to pump emissions, https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.1934
    EI_H2O = 2.25, # assume pure methane
    LCV = 48.6, https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.1934
    eta = 0.37 # assumed same as for jet A1