import numpy as np
import matplotlib
import matplotlib.pylab as plt

sigma = 4.5

simulation_length = 160 # choose 160
within_year_points = 50 # high sampling 50; low sampling 1

samples = simulation_length * within_year_points

init_sigma_count = 0

start_time = 0
dt = 1. / 50.
time = np.linspace(start_time, simulation_length, samples)
original_time_copy = np.copy(time)
data = np.load('ecological_model_planktivores_sig' + str(sigma) + 'ppyear' + str(within_year_points) + '.npy')
original_data_copy = np.copy(data)

plt.plot(time,data)

data_to_subtract = data[0:-50]
data_to_be_deseasonalized = data[50:]

deseasonalized_data = data_to_be_deseasonalized - data_to_subtract
plt.plot(time[50:], deseasonalized_data, alpha = 0.5)
plt.savefig('deseasonalized_data_sigma' + str(sigma) + '.png')
plt.show()

np.save('deseasonalized_eco_model_planktivores_sig_' + str(sigma) + 'ppyear' + str(within_year_points) + '.npy', deseasonalized_data)