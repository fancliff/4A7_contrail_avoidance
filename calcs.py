radiative_eff = 1.37e-5  # W/m^2 per ppb
m_atm = 5.1480e18  # kg
m_co2 = 1 # kg
MW_atm = 28.97  # g/mol
MW_co2 = 44.01  # g/mol

ppb_per_kgCO2 = m_co2 / m_atm * MW_atm / MW_co2 * 1e9
print(ppb_per_kgCO2)
radiative_eff_per_kgCO2 = radiative_eff * ppb_per_kgCO2
print(radiative_eff_per_kgCO2)