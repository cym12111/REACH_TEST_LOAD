#!/usr/bin/env python
# coding: utf-8

# Importing modules and defining directories.

# In[28]:


from dash import Dash, html, dcc
import pandas as pd
import numpy as np
from copy import deepcopy
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import df_tools # must be in local directory
import itertools

dirname = os.getcwd()

expID = '20230320' # experimental ID

rio_pth = os.path.join(expID, expID+'_rio.csv')
mspec_pth = os.path.join(expID, expID+'_mass_spec.csv')
tc08_pth = os.path.join(expID, expID+'_tc08.csv')
tenma_pth = os.path.join(expID, expID+'_tenma.txt')


# Reading the raw files into pandas Dataframes.

# In[29]:


rio_raw = pd.read_csv(rio_pth) # skiprows may be necessary e.g. skiprows=0
mspec_raw = pd.read_csv(mspec_pth, header=0, skiprows=22, sep=',')
mspec_raw.columns = mspec_raw.columns.str.replace(' ', '')
mspec_max_amu = float(mspec_raw.columns[-1]) # retrieve maximum AMU from mass spectrometer data
tc08_raw = pd.read_csv(tc08_pth, skiprows=1, names=['unix', 'T_tphe_5.0m (C)', 'T_tphe_3.75m (C)', 'T_tphe_2.5m (C)', 'T_iv_gas (C)',	'T_tw (C)', 'T_exhaust (C)'])

try:
    tenma_raw = pd.read_csv(tenma_pth, delimiter="\t", names=['t', 'V', 'I'])
    tenma_exists = True
except FileNotFoundError:
    print('No TENMA data found. Proceeding with analysis anyway.')
    tenma_exists = False
print('Mass spectrometer maximum atomic mass unit: %3.2f' % (mspec_max_amu))


# Processing the mass spectrometer data

# In[30]:


mspec_delay = 200 # mass spec delay (seconds)

# getting mass spec timestamps in the desired format
start_date_col, start_time_col = 1, 3

with open(expID+'/'+expID+'_mass_spec.csv') as file:
    file.readline() # skips one line
    file.readline() # skips one line
    header = file.readline()
    initial_date, initial_time = header.split(",")[start_date_col], header.split(",")[start_time_col]

# Convert the initial date and time values into a datetime object
start_datetime = pd.to_datetime(initial_date + ' ' + initial_time, dayfirst=True)

mspec_raw['diff_ms'] = mspec_raw['ms'].diff()

mspec_raw['end_time'] = start_datetime + pd.to_timedelta(mspec_raw['ms']/1000, unit='s')
mspec_raw['start_time'] = mspec_raw['end_time'] - pd.to_timedelta(mspec_raw['diff_ms']/1000, unit='s')
mspec_raw.at[0, 'start_time'] = start_datetime
delta = pd.Timedelta(mspec_delay, unit='s')

mspec = pd.DataFrame()

mspec['t0'] = mspec_raw['start_time'] + delta
mspec['t1'] = mspec_raw['end_time'] + delta
mspec['p_total'] = mspec_raw.iloc[:, 3:-3].sum(1)
mspec['H2%'] = mspec_raw['2.00']/mspec['p_total']*100
mspec['NH3%'] = mspec_raw['17.00']/mspec['p_total']*100
mspec['H2O%'] = mspec_raw['18.00']/mspec['p_total']*100
mspec['NO%'] = mspec_raw['30.00']/mspec['p_total']*100
mspec['O2%'] = mspec_raw['32.00']/mspec['p_total']*100
mspec['N2%'] = mspec_raw['28.00']/mspec['p_total']*100
mspec['Ar%'] = mspec_raw['40.00']/mspec['p_total']*100
if mspec_max_amu >= 44: mspec['N2O%'] = mspec_raw['44.00']/mspec['p_total']*100
if mspec_max_amu >= 46: mspec['NO2%'] = mspec_raw['46.00']/mspec['p_total']*100
mspec['Outstanding %'] = 100-mspec.iloc[:,3:-1].sum(1)

mspec = mspec.set_index('t0')


# Processing TENMA data.

# In[31]:


if tenma_exists:
    tenma = pd.DataFrame()
    tenma['t0'] = pd.to_datetime(tenma_raw['t'], dayfirst=True)
    tenma['t1'] = tenma['t0']+(tenma['t0'][1]-tenma['t0'][0])
    tenma['V_tenma (V)'] = tenma_raw['V'].str.replace('V','')
    tenma['V_tenma (V)'] = tenma['V_tenma (V)'].astype(float)
    tenma['I_tenma (A)'] = tenma_raw['I'].str.replace('A','')
    tenma['I_tenma (A)'] = tenma['I_tenma (A)'].astype(float)
    tenma['P_tenma (W)'] = tenma['I_tenma (A)']*tenma['V_tenma (V)']

    tenma = tenma.set_index('t0')


# Processing TC08 data.

# In[32]:


tc08 = pd.DataFrame()
tc08['t0'] = pd.to_datetime(tc08_raw['unix']*1e9)
tc08['t1'] = tc08['t0']+(tc08['t0'][1]-tc08['t0'][0])
tc08 = tc08.join(tc08_raw.iloc[:,1:])

tc08 = tc08.set_index('t0')


# Processing the Rio data.

# In[33]:


# Rio post processing
n_ch = 4
ch_power = 135
I_offset = 0.038 # A
Va_offset = 0.2 # SLM
NH3_2_H2 = 1.5
NH3_2_N2 = 0.5
cal_H2 = 10805.08 # J/SLM
cal_NH3 = 13636.75 # J/SLM
rho_H2 = 0.08988 # g/L
cp_H2 = 14.304 # J/gK
rho_air = 1.29 # g/L
cp_air = 1 # J/gK
rho_N2 = 1.1606 # g/L
cp_N2 = 1.04 # J/gK
T_amb = 20 # K


rio = pd.DataFrame() # initialising final file

rio['t0'] = pd.to_datetime(rio_raw['TimeStart']*1e6)
rio['t1'] = pd.to_datetime(rio_raw['TimeEnd']*1e6)
rio['T_tw_ave (C)'] = rio_raw[['Heater_1_TC_degC','Heater_2_TC_degC','Heater_3_TC_degC','Heater_4_TC_degC']].mean(1)
rio['T_sofc (C)'] = rio_raw['SOFC_TC_degC']
rio['T_comb (C)'] = rio_raw['Comb_TC_degC']

rio['Va_Ar (SLM)'] = np.where(rio_raw['Air_Input']=='Ar', rio_raw['Air_Flow_Meter_units_undetermined'], np.nan)+Va_offset
rio['Va_air (SLM)'] = np.where(rio_raw['Air_Input']=='Air', rio_raw['Air_Flow_Meter_units_undetermined'], np.nan)+Va_offset
rio['Vf_Ar (SLM)'] = np.where(rio_raw['Fuel_Input']=='Ar', rio_raw['Fuel_Flow_Meter_units_undetermined'], np.nan)
rio['Vf_NH3 (SLM)'] = np.where(rio_raw['Fuel_Input']=='NH3', rio_raw['Fuel_Flow_Meter_units_undetermined'], np.nan)
rio['Vf_eqH2 (SLM)'] = NH3_2_H2*rio['Vf_NH3 (SLM)']
rio['Vf_H2 (SLM)'] = np.where(rio_raw['Fuel_Input']=='H2', rio_raw['Fuel_Flow_Meter_units_undetermined'], np.nan)
rio['R_eq'] = np.where(rio_raw['Fuel_Input']=='NH3', rio['Vf_NH3 (SLM)']/(0.32*rio['Va_air (SLM)']), rio['Vf_H2 (SLM)']/(0.42*rio['Va_air (SLM)']))
rio['P_in_NH3'] = cal_NH3/60*rio['Vf_NH3 (SLM)']
rio['P_in_eqH2'] = cal_H2/60*rio['Vf_eqH2 (SLM)']
rio['P_in_H2'] = cal_H2/60*rio['Vf_H2 (SLM)']

rio['P_ch (W)'] = n_ch*ch_power*rio_raw['Heater_PID_value']

rio['V_out (V)'] = rio_raw['SOFC_Voltage_V']
rio['I_out (A)'] = rio_raw['SOFC_Current_A'] - I_offset
rio['P_out (W)'] = rio[['V_out (V)', 'I_out (A)']].product(1)

rio['Q_comb'] = np.where(np.isnan(rio['Vf_H2 (SLM)']),
                         (rio['Vf_eqH2 (SLM)']*rho_H2*cp_H2
                         + NH3_2_N2*rio['Vf_NH3 (SLM)']*rho_N2*cp_N2
                         + rio['Va_air (SLM)']*rho_air*cp_air)/60
                         *(rio['T_comb (C)']-rio['T_sofc (C)']),
                         (rio['Vf_H2 (SLM)']*rho_H2*cp_H2
                         + rio['Va_air (SLM)']*rho_air*cp_air)/60
                         *(rio['T_comb (C)']-rio['T_sofc (C)'])
                         )


rio['Eff_sofc'] = np.where(np.isnan(rio['Vf_H2 (SLM)']), rio['P_out (W)']/rio['P_in_eqH2'], rio['P_out (W)']/rio['P_in_H2'])*100
rio['Eff_sofc'] = np.where(rio['Eff_sofc']>100, np.nan, rio['Eff_sofc']) # removing nonesense values
rio['Eff_sofc'] = np.where(rio['Eff_sofc']<0, np.nan, rio['Eff_sofc']) # removing nonesense values
rio['Eff_sys'] = np.where(np.isnan(rio['Vf_H2 (SLM)']), rio['P_out (W)']/rio['P_in_NH3'], rio['P_out (W)']/rio['P_in_H2'])*100
rio['Eff_sys'] = np.where(rio['Eff_sys']>100, np.nan, rio['Eff_sys']) # removing nonesense values
rio['Eff_sys'] = np.where(rio['Eff_sys']<0, np.nan, rio['Eff_sys']) # removing nonesense values

rio = rio.set_index('t0')

tc08_rio = tc08.reindex(rio.index, method='nearest')

rio['Q_sofc'] = np.where(np.isnan(rio['Vf_H2 (SLM)']),
                         (rio['Vf_eqH2 (SLM)']*rho_H2*cp_H2
                         + NH3_2_N2*rio['Vf_NH3 (SLM)']*rho_N2*cp_N2
                         + rio['Va_air (SLM)']*rho_air*cp_air)/60
                         *(rio['T_comb (C)']-tc08_rio['T_tphe_5.0m (C)']),
                         (rio['Vf_H2 (SLM)']*rho_H2*cp_H2
                         + rio['Va_air (SLM)']*rho_air*cp_air)/60
                         *(rio['T_sofc (C)']-tc08_rio['T_tphe_5.0m (C)'])
                         )

rio['Q_exhaust'] = np.where(np.isnan(rio['Vf_H2 (SLM)']),
                         (rio['Vf_eqH2 (SLM)']*rho_H2*cp_H2
                         + NH3_2_N2*rio['Vf_NH3 (SLM)']*rho_N2*cp_N2
                         + rio['Va_air (SLM)']*rho_air*cp_air)/60
                         *(tc08_rio['T_exhaust (C)']-T_amb),
                         (rio['Vf_H2 (SLM)']*rho_H2*cp_H2
                         + rio['Va_air (SLM)']*rho_air*cp_air)/60
                         *(tc08_rio['T_exhaust (C)']-T_amb)
                         ) # need to update to made the thermal properties match the exhaust gas mixture.

rio['Q_loss'] = np.where(np.isnan(rio['Vf_H2 (SLM)']),
                         rio['P_in_eqH2']-(rio['Q_sofc']+rio['Q_comb']+rio['P_out (W)']+rio['Q_exhaust']),
                         rio['P_in_H2']-(rio['Q_sofc']+rio['Q_comb']+rio['P_out (W)']+rio['Q_exhaust'])
                         )
                


# Plotting data.

# In[34]:


fig = make_subplots(rows=6, cols=1, shared_xaxes=True)

# Temperature Plot
# Rio Temperatures
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['T_sofc (C)'],
     name = 'Temperature: Fuel Cell',
), row=1, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['T_comb (C)'],
     name = 'Temperature: Combustor',
), row=1, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['T_tw_ave (C)'],
     name = 'Temperature: Thermowell OV Ave.',
), row=1, col=1)
# TC08 Temperatures
fig.append_trace(go.Scatter(
     x=tc08.index,
     y=tc08['T_tphe_5.0m (C)'],
     name = 'Temperature: Heat Exchanger @ 5.0m',
), row=1, col=1)
fig.append_trace(go.Scatter(
     x=tc08.index,
     y=tc08['T_tphe_3.75m (C)'],
     name = 'Temperature: Heat Exchanger @ 3.75m',
), row=1, col=1)
fig.append_trace(go.Scatter(
     x=tc08.index,
     y=tc08['T_tphe_2.5m (C)'],
     name = 'Temperature: Heat Exchanger @ 2.5m',
), row=1, col=1)
fig.append_trace(go.Scatter(
     x=tc08.index,
     y=tc08['T_iv_gas (C)'],
     name = 'Temperature: Inner Vessel Gas',
), row=1, col=1)
fig.append_trace(go.Scatter(
     x=tc08.index,
     y=tc08['T_tw (C)'],
     name = 'Temperature: Thermowell IV Surface',
), row=1, col=1)
fig.append_trace(go.Scatter(
     x=tc08.index,
     y=tc08['T_exhaust (C)'],
     name = 'Temperature: Exhaust Gas',
), row=1, col=1)
fig.layout['yaxis1'].update(title_text='Temperature (C)')

# Electrical Plot
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['P_out (W)'],
     name = 'DC Output: Rio',
), row=2, col=1)
if tenma_exists:
     fig.append_trace(go.Scatter(
          x=tenma.index,
          y=tenma['P_tenma (W)'],
          name = 'DC Output: TENMA',
     ), row=2, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['V_out (V)'],
     name = '*Rio Voltage (V)',
), row=2, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['I_out (A)'],
     name = '*Rio Current (A)',
), row=2, col=1)
if tenma_exists:
     fig.append_trace(go.Scatter(
          x=tenma.index,
          y=tenma['V_tenma (V)'],
          name = '*TENMA Voltage (V)',
     ), row=2, col=1)
     fig.append_trace(go.Scatter(
          x=tenma.index,
          y=tenma['I_tenma (A)'],
          name = '*TENMA Current (A)',
     ), row=2, col=1)
fig.layout['yaxis2'].update(title_text='DC Output (W)')

# Thermal Power Plot
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['P_ch (W)'],
     name = 'Thermal Power: Cartridge Heaters',
), row=3, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['P_in_NH3'],
     name = 'Thermal Power: Calorific Rate of NH3',
), row=3, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['P_in_eqH2'],
     name = 'Thermal Power: Calorific Rate of H2 from Cracking',
), row=3, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['P_in_H2'],
     name = 'Thermal Power: Calorific Rate of H2',
), row=3, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Q_sofc'],
     name = 'Thermal Power: Fuel Cell',
), row=3, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Q_comb'],
     name = 'Thermal Power: Combustor',
), row=3, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Q_exhaust'],
     name = 'Thermal Power: Exhaust',
), row=3, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Q_loss'],
     name = 'Thermal Power: Losses',
), row=3, col=1)
fig.layout['yaxis3'].update(title_text='Power (W)')


# Air Flow Rate Plot
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Va_Ar (SLM)'],
     name = 'Air Flow Rate: Argon',
), row=4, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Va_air (SLM)'],
     name = 'Air Flow Rate: Air',
), row=4, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Vf_Ar (SLM)'],
     name = 'Fuel Flow Rate: Argon',
), row=4, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Vf_NH3 (SLM)'],
     name = 'Fuel Flow Rate: Ammonia',
), row=4, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Vf_eqH2 (SLM)'],
     name = 'Fuel Flow Rate: Assumed Hydrogen from Cracking',
), row=4, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Vf_H2 (SLM)'],
     name = 'Fuel Flow Rate: Hydrogen',
), row=4, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['R_eq'],
     name = '*Equivalence Ratio',
), row=4, col=1)
fig.layout['yaxis4'].update(title_text='Flow Rate (SLM)')


# Efficiency Plot
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Eff_sofc'],
     name = 'Efficiency: Fuel Cell',
), row=5, col=1)
fig.append_trace(go.Scatter(
     x=rio.index,
     y=rio['Eff_sys'],
     name = 'Efficiency: System',
), row=5, col=1)
fig.layout['yaxis5'].update(title_text='Efficiency (%)')



# Mass Spectrometry Plot
fig.append_trace(go.Scatter(
     x=mspec.index,
     y=mspec['H2%'],
     name = 'Volume Fraction: H2',
), row=6, col=1)
fig.append_trace(go.Scatter(
     x=mspec.index,
     y=mspec['NH3%'],
     name = 'Volume Fraction: NH3',
), row=6, col=1)
fig.append_trace(go.Scatter(
     x=mspec.index,
     y=mspec['H2O%'],
     name = 'Volume Fraction: H2O',
), row=6, col=1)
fig.append_trace(go.Scatter(
     x=mspec.index,
     y=mspec['NO%'],
     name = 'Volume Fraction: NO',
), row=6, col=1)
fig.append_trace(go.Scatter(
     x=mspec.index,
     y=mspec['O2%'],
     name = 'Volume Fraction: O2',
), row=6, col=1)
fig.append_trace(go.Scatter(
     x=mspec.index,
     y=mspec['N2%'],
     name = 'Volume Fraction: N2',
), row=6, col=1)
fig.append_trace(go.Scatter(
     x=mspec.index,
     y=mspec['Ar%'],
     name = 'Volume Fraction: Ar',
), row=6, col=1)
if mspec_max_amu >= 44:
     fig.append_trace(go.Scatter(
          x=mspec.index,
          y=mspec['N2O%'],
          name = 'Volume Fraction: N2O',
     ), row=6, col=1)
if mspec_max_amu >= 46:
     fig.append_trace(go.Scatter(
          x=mspec.index,
          y=mspec['NO2%'],
          name = 'Volume Fraction: NO2',
     ), row=6, col=1)
fig.append_trace(go.Scatter(
     x=mspec.index,
     y=mspec['Outstanding %'],
     name = 'Volume Fraction: Leftover',
), row=6, col=1)
fig.layout['yaxis6'].update(title_text='Exhaust Volume Fraction (%)')

# Overall Fomatting
fig = fig.update_layout(
     height=1000,
     xaxis1_showticklabels=True,
     xaxis2_showticklabels=True,
     xaxis3_showticklabels=True,
     xaxis4_showticklabels=True,
     xaxis5_showticklabels=True,
     xaxis6_showticklabels=True,
     font_family="Roboto",
     paper_bgcolor="#f8f5f1",
     plot_bgcolor="#ede7da",
     hovermode="x unified"
     )

# Title Formatting
fig = fig.update_layout(
     title_font_family="Moderat",
     title_font_size=30,
     title_text= "REACH Data Viewer: Stacked Timeseries for %s Dataset" % (expID)
     )

# Hoverlabel Formatting
fig = fig.update_layout(
     hoverlabel_namelength=-1,
     #hoverdistance=100
     hoverlabel_bgcolor="#ffffff",
     hoverlabel_bordercolor="#c4b491"
     )


# Legend Formatting
fig = fig.update_layout(
     #legend_bgcolor="#a69559",
     legend_title_text="Processed Traces",
     legend_title_font_size=20,
     #legend_font_color="#FFFFFF",
     legend_title_font_family="Moderat"
     )

#fig.write_html("%s_timeseries_viewer.html" % (expID))


# Dash App.

# In[ ]:


app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='REACH Data Web-Viewer'),

    html.Div(children='''
        Simple.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=False)


# Slicing Dataframe to isolate constant conditions within a specified time range. To obtain Power vs Fuel Rate, we need to isolate equivalence ratios. To obtain Current vs Voltage curves, we wish to isolate fuel rates and equivalence ratios.

# In[ ]:


# t0 = '2023-03-20 15:40:00'
# t1 = '2023-03-20 19:00:00'

# rio = rio[t0:t1]

# # Equivalence Ratio Mask
# R_eq_nom_vals = np.linspace(0.2, 0.8, 7)
# R_eq_mask = df_tools.nominal_vals_mask(df=rio, varname='R_eq', nominal_vals=R_eq_nom_vals, tolerance=0.05)

# # Fuel Rate Mask
# Vf_name = 'Vf_NH3 (SLM)' # adjust depending on fuel input
# Vf_nom_vals = np.linspace(0, 1, 11)
# Vf_mask = df_tools.nominal_vals_mask(df=rio, varname=Vf_name, nominal_vals=Vf_nom_vals, tolerance=0.01)

# # Combining Masks
# IV_mask = df_tools.combine_masks(R_eq_mask, Vf_mask)

# sliced_fig = make_subplots(rows=3,cols=1)
# for col in R_eq_mask.columns:
#     rio_slice = rio.loc[R_eq_mask[col]]
#     sliced_fig.append_trace(go.Scatter(
#         x = rio_slice.index,
#         y = rio_slice['R_eq'],
#         name = col,
#         mode = 'markers'
#         ), row=1, col=1)
# for col in Vf_mask.columns:
#     rio_slice = rio.loc[Vf_mask[col]]
#     sliced_fig.append_trace(go.Scatter(
#         x = rio_slice.index,
#         y = rio_slice[Vf_name],
#         name = col,
#         mode = 'markers'
#         ), row=2, col=1)
# for col in IV_mask.columns:
#     rio_slice = rio.loc[IV_mask[col]]
#     sliced_fig.append_trace(go.Scatter(
#         x=rio_slice.index,
#         y=rio_slice['P_out (W)'],
#         name = col,
#         mode = 'markers'
#         ), row=3, col=1)

# sliced_fig


# Performance mapping.

# In[ ]:


# col_pal = px.colors.qualitative.Plotly # obtaining the default color cycle of Plotly
# col_pal_iterator = itertools.cycle(col_pal)
# fit_count_threshold = 10 # at least 10 datapoints for fitted relationship

# map_fig = make_subplots(rows=1, cols=2)
# for col in R_eq_mask.columns:
#      new_color = next(col_pal_iterator)
#      rio_slice = rio.loc[R_eq_mask[col]]
#      xvals=rio_slice["Vf_NH3 (SLM)"]
#      yvals=rio_slice['P_out (W)']
#      map_fig.append_trace(go.Scatter(
#           x=xvals,
#           y=yvals,
#           name = col,
#           mode="markers",
#           marker_color = new_color,
#      ), row=1, col=1)
#      # Curve fitting for P, V_F plot
#      try:
#           xfit, yfit = df_tools.exponential_fit(xvals,yvals)
#           map_fig.append_trace(go.Scatter(
#                x=xfit,
#                y=yfit,
#                name = col+' linear fit',
#                mode="lines",
#                line_color = new_color,
#           ), row=1, col=1)
#      except:
#           print('Exponential fit for %s failed.' % (col))
#           pass

# for col in IV_mask.columns:
#      new_color = next(col_pal_iterator)
#      rio_slice = rio.loc[IV_mask[col]]
#      map_fig.append_trace(go.Scatter(
#           x=xvals,
#           y=yvals,
#           name = col,
#           mode="markers",
#           marker_color=new_color,
#      ), row=1, col=2)
#      # Curve fitting for V, I plot
#      if rio_slice.shape[0]>fit_count_threshold:
#           xvals=rio_slice['I_out (A)']
#           yvals=rio_slice["V_out (V)"]
#           try:
#                xfit, yfit = df_tools.line_fit(xvals,yvals)
#                map_fig.append_trace(go.Scatter(
#                     x=xfit,
#                     y=yfit,
#                     name = col+' linear fit',
#                     mode="lines",
#                     line_color=new_color,
#                ), row=1, col=2)
#           except:
#                print('Line fit for %s failed.' % (col))
#                pass


# map_fig.layout['xaxis1'].update(title_text='NH3 Flow Rate (SLM)')
# map_fig.layout['yaxis1'].update(title_text='Power (W)')
# map_fig.layout['xaxis2'].update(title_text='Current (A)')
# map_fig.layout['yaxis2'].update(title_text='Voltage (V)')


# map_fig


# 3D Graph Attempt

# In[ ]:


# fig3d = make_subplots(rows=2, cols=2, specs=[[{'type':'scene'}, {'type':'scene'}], [{'type':'xy'}, {'type':'xy'}]])

# # Uniform parameters
# T_max = int(rio['T_sofc (C)'].max())
# T_min = 650

# marker_format = dict(
#     size=3,
#     color=rio['T_sofc (C)'],
#     colorscale='Magma',
#     cmax=T_max,
#     cmin=T_min,
#     colorbar=dict(
#         x=0,
#         y=0.8,
#         title_text='Temperature (C)',
#         title_font_family='Moderat',
#         tickfont_family='Roboto',
#         len=0.5
#         )
# )

# scene_format = dict(
#         xaxis = dict(title_text='x',
#                     title_font_family='Moderat',
#                     title_font_size=10,
#                     tickfont_family='Roboto',
#                     tickfont_size=10,
#                     ),
#         yaxis = dict(title_text='y',
#                     title_font_family='Moderat',
#                     title_font_size=10,
#                     tickfont_size=10,
#                     ),
#         zaxis = dict(title_text='z',
#                     title_font_family='Moderat',
#                     title_font_size=10,
#                     tickfont_size=10,
#                     ),
#         aspectratio = dict(x=2, y=2, z=1),
#         camera = dict(
#              up=dict(x=0, y=0, z=1),
#              center=dict(x=0, y=0, z=0),
#              eye=dict(x=2, y=2, z=0.1)
#              ),
# )

# marker_format1 = deepcopy(marker_format)
# marker_format1['colorbar']['x'] = -0.05
# fig3d.append_trace(go.Scatter3d(
#     x=rio['Vf_NH3 (SLM)'],
#     y=rio['Va_air (SLM)'],
#     z=rio['P_out (W)'],
#     mode='markers',
#     name='Power Out',
#     marker=marker_format1,
#     hovertemplate='SOFC Temp.: %{marker.color:.2f} C<br />NH3 Flow Rate: %{x:.2f} SLM<br />Air Flow Rate: %{y:.2f} SLM<br />Power Out: %{z:.2f} W'
# ), row=1, col=1)
# scene_format1 = deepcopy(scene_format)
# scene_format1['xaxis']['title_text']='Ammonia Flow Rate (SLM)'
# scene_format1['yaxis']['title_text']='Air Flow Rate (SLM)'
# scene_format1['zaxis']['title_text']='Power Out (W)'

# marker_format2 = deepcopy(marker_format)
# marker_format2['colorbar']['x'] = 0.45
# fig3d.append_trace(go.Scatter3d(
#     x=rio['Vf_NH3 (SLM)'],
#     y=rio['Va_air (SLM)'],
#     z=rio['Eff_sys'],
#     mode='markers',
#     name='System Efficiency',
#     marker=marker_format2,
#     hovertemplate='SOFC Temp.: %{marker.color:.2f} C<br />NH3 Flow Rate: %{x:.2f} SLM<br />Air Flow Rate: %{y:.2f} SLM<br />System Efficiency: %{z:.2f} W'
# ), row=1, col=2)
# scene_format2 = deepcopy(scene_format)
# scene_format2['xaxis']['title_text']='Ammonia Flow Rate (SLM)'
# scene_format2['yaxis']['title_text']='Air Flow Rate (SLM)'
# scene_format2['zaxis']['title_text']='System Efficiency (%)'

# fig3d.layout.update(
#     scene1 = scene_format1,
#     scene2 = scene_format2,
# )



# fig3d.layout.update(
#      height=1000,
#      width=1800,
#      title= dict(font_family="Moderat",
#              font_size=30,
#              text= "REACH Data Viewer: Performance Maps %s Dataset" % (expID)
#              )
# )

# fig3d


# In[ ]:


# t_0 = '2023-03-20 15:40:00'
# t_1 = '2023-03-20 19:00:00'
# rio[t_0:t_1]

