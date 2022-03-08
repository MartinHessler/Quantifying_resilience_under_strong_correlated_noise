import numpy as np
import matplotlib
import matplotlib.pylab as plt
from scipy.ndimage.filters import gaussian_filter

sigma = 0.1
kernelwidth = 2.5
phi = 0.92

simulation_length = 160 # choose 160
within_year_points = 50 # high sampling 50; low sampling 1

samples = simulation_length* within_year_points

init_sigma_count = 0

start_time = 0
dt = 1. / 50.
time = np.linspace(start_time, simulation_length, simulation_length * within_year_points)

data = np.load('ecological_model_planktivores_sig' + str(sigma) + '_phi' + str(phi) + 'ppyear' + str(within_year_points) + '.npy')


plt.plot(time,data)

STGF = gaussian_filter(data, kernelwidth, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
deseasonalized_data = data - STGF

plt.plot(time, data, alpha = 1)
plt.plot(time, STGF, alpha = 0.7)
plt.plot(time, deseasonalized_data, alpha = 1)
plt.savefig('deseasonalized_data_sigma' + str(sigma) + '.png')
plt.show()

np.save('deseasonalized' + str(kernelwidth) + '_corr_eco_model_planktivores_sig_' + str(sigma) + 'phi' + str(phi) + 'ppyear' + str(within_year_points) + '.npy', deseasonalized_data)
