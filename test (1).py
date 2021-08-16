import os
import sys
import time
import h5py
import warnings
import xarray as xr
import numpy  as np
from pathlib import Path

# Paths
CWD = os.getcwd()
# sys.path.append(os.path.join(CWD, "../../../"))

# Import library specific modules
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import utils_weights as utils_weights

# Inspect and load data 
file = os.path.join(CWD,'fluidmechanics_data.mat')
variables = ['p']
with h5py.File(file, 'r') as f:
	data_arrays = dict()
	for k, v in f.items():
		data_arrays[k] = np.array(v)
dt = data_arrays['dt'][0,0]
block_dimension = 64 * dt
x1 = data_arrays['r'].T; x1 = x1[:,0]
x2 = data_arrays['x'].T; x2 = x2[0,:]
X = data_arrays[variables[0]].T
t = dt * np.arange(0,X.shape[0]); t = t.T

print('t.shape  = ', t.shape)
print('x1.shape = ', x1.shape)
print('x2.shape = ', x2.shape)
print('X.shape  = ', X.shape)
# # Let's create some 2D syntetic data

# # -- define spatial and time coordinates
# x1 = np.linspace(0,10,100) 
# x2 = np.linspace(0, 5, 50) 
# xx1, xx2 = np.meshgrid(x1, x2)
# t = np.linspace(0, 200, 1000)

# # -- define 2D syntetic data
# s_component = np.sin(xx1 * xx2) + np.cos(xx1)**2 + np.sin(0.1*xx2)
# t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
# p = np.empty((t_component.shape[0],)+s_component.shape)
# for i, t_c in enumerate(t_component):
#     p[i] = s_component * t_c


# Let's define the required parameters into a dictionary
params = dict()

# -- required parameters
params['time_step'   ] = 1              # data time-sampling
params['n_snapshots' ] = t.shape[0]     # number of time snapshots (we consider all data)
params['n_space_dims'] = 2              # number of spatial dimensions 
params['n_variables' ] = 1 		# number of variables
params['n_DFT'       ] = 100          	# length of FFT blocks (100 time-snapshots)
params['nt'          ] = 50              # number of time steps
params['xdim'        ] = 5              # number of dimensions
params['nv'          ] = 5              # number of variables
params['dt'          ] = 5              # timestep
params['n_FFT'       ] = 5              # Number of Fast Fourier transforms
params['n_overlap'   ] = 1              # number of overlap
params['mean'        ] = 'blockwise'    # mean
params['n_blocks'    ] = 5              # number of blocks

# -- optional parameters
params['overlap'          ] = 0           # dimension block overlap region
params['mean_type'        ] = 'blockwise' # type of mean to subtract to the data
params['normalize_weights'] = False       # normalization of weights by data variance
params['normalize_dta'   ] = False       # normalize data by data variance
params['n_modes_save'     ] = 3           # modes to be saved
params['conf_level'       ] = 0.95        # calculate confidence level
params['reuse_blocks'     ] = True        # whether to reuse blocks if present
params['savefft'          ] = False       # save FFT blocks to reuse them in the future (saves time)
params['savedir'          ] = os.path.join('results', 'simple_test') # folder where to save results


# Perform SPOD analysis using low storage module
SPOD_analysis = SPOD_streaming(data=X, params=params, data_handler=False, variables=variables)
spod = SPOD_analysis.fit()

# Show results
T_approx = 12.5 # approximate period
freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)

spod.plot_eigs(filename = 'eigs')

freq = spod.freq
spod.plot_eigs_vs_frequency(freq=freq, filename = 'eigs_vs_frequency')

spod.plot_eigs_vs_period   (freq=freq, xticks=[1, 0.5, 0.2, 0.1, 0.05, 0.02], filename = 'eigs_vs_period')

spod.plot_2D_modes_at_frequency(
      freq_required=freq_found, 
    freq=freq, 
    x1=x1, 
    x2=x2, 
    modes_idx=[0,1], 
    vars_idx=[0])

spod.plot_2D_data(x1=x1, x2=x2, vars_idx=[0], time_idx=[0,100,200], filename = '2D_data')







# # and run the analysis
# spod_ls.fit()

# # spod.plot_2D_data(time_idx = [1,2], filename = 'test')
# # Let's plot the data
# spod_ls.plot_2D_data(time_idx=[1,2], filename = '2D_data')
# spod_ls.plot_data_tracers(coords_list=[(5,2.5)], time_limits=[0,t.shape[0]], filename = 'data_tracers')
# spod_ls.generate_2D_data_video(sampling=10, time_limits=[0,t.shape[0]], filename = '2D_data_video')


# # X = spod.get_data(0, 200)

# # print(type(X))
# # # print(X)
# # print(X.shape)
# # from pyspod.postprocessing import plot_2D_data

# # from matplotlib import pyplot as plt

# # plt.savefig('test-1_results')

# #  # Let's plot the data
# # # spod.plot_2D_data(time_idx=[1,2])
# # # spod.plot_data_tracers(coords_list=[(5,2.5)], time_limits=[0,t.shape[0]])
