grids = ['Balearic','Irish','Iceland']
models = ['model 1','model 2','model 3','model 4']

'''Model 1...'''
synth_data_model_1 = {}
'''adapt the parameter estimation to the particulat grids'''
for grid in grids:
  data = ...
  data = data_cleaning(data)
  if grid = 'Balearic':
    diff_drift = ...
  elif grid = 'Irish':
    bw_drift = ...
  elif grid = 'Iceland':
    bw_drift = ...
  c_1 = ...
  c_2 = ...
  Delta_P = ...
  epsilon = ...
  omega_synth_model_1 = Euler_Maruyama(...)
  synth_data_model_1(grid) = omega_synth_model_1
