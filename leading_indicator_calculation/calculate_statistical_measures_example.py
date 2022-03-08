import numpy as np
import pylab
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def mean(data):
	return sum(data)/data.size

def standard_deviation(data):
	global window_size
	return np.sqrt((1./(window_size-1))*sum((data-mean(data))**2))

def standard_deviation_n_denominator(data):
	global window_size
	return np.sqrt((1./(window_size))*sum((data-mean(data))**2))

def nominator(data, time_lag):
	rolleddata=np.roll(data,shift=-time_lag)
	return((1./(window_size-1-time_lag))*sum((data[:data.size-time_lag]-mean(data[:data.size-time_lag]))*(rolleddata[:data.size-time_lag]-mean(rolleddata[:data.size-time_lag]))))

def autocorrelation(data, time_lag = 1):
	return (nominator(data, time_lag)/standard_deviation(data)**2)

def skewness(data):
	global window_size
	return (1./window_size)*sum((data-mean(data))**3)/standard_deviation_n_denominator(data)

def kurtosis(data):
	global window_size
	return (1./window_size) * sum((data-mean(data))**4)/standard_deviation_n_denominator(data)**2


sigma_array = np.array([0.1, 2.2, 4.5])
sigma_array_size = sigma_array.size

simulation_length = 160 # choose 160
end_time = simulation_length # in years
within_year_points = 50 # high sampling 50; low sampling 1

samples = end_time * within_year_points

start_time = 0
dt = 1. / 50.
time = np.linspace(start_time, simulation_length, simulation_length * within_year_points)

qE_increment = 0.013 # per year (choose 0.013)
qE_init = 1.
qE_param = np.arange(qE_init, qE_init + simulation_length * qE_increment, qE_increment / within_year_points)

print(qE_param.size)

window_size = 750
window_shift = 30

loop_range = np.arange(0, simulation_length * within_year_points - window_size, window_shift)
loop_range_size = loop_range.size
show_indicators = True

AR1_array = np.zeros((sigma_array_size, loop_range_size))
std_array = np.zeros((sigma_array_size, loop_range_size))
skew_array = np.zeros((sigma_array_size, loop_range_size))
kurt_array = np.zeros((sigma_array_size, loop_range_size))

i = 0
for sigma in sigma_array:
	data = np.load('ecological_model_planktivores_sig' + str(sigma) + 'ppyear' + str(within_year_points) + '.npy')
	for k in range(loop_range_size):
		data_window = np.roll(data, shift = - loop_range[k])[:window_size]
		time_window = np.roll(time, shift = - loop_range[k])[:window_size]
		AR1_array[i, k] = autocorrelation(data_window)
		std_array[i, k] = standard_deviation(data_window)
		skew_array[i, k] = skewness(data_window)
		kurt_array[i, k] = kurtosis(data_window)
		print(r'progress: ' + str(k+1) + r'/' + str(loop_range_size) + r'   |||   ' + r'$\sigma$: ' + str(sigma))
	if show_indicators:
		print('Show indicators.')
		plt.plot(time[loop_range], AR1_array[i,:], label = 'AR1')
		plt.plot(time[loop_range], std_array[i,:], label = 'std')
		plt.plot(time[loop_range], skew_array[i,:], label = 'skew')
		plt.plot(time[loop_range], kurt_array[i,:], label = 'kurt')
		plt.xlabel('indicator')
		plt.xlabel('control parameter r')
		plt.legend()
		plt.savefig('indicators_sigma' + str(sigma) + '.png')
		plt.show()
	i += 1

print('Save indicators.')
np.save('AR1_array_all_sigma.npy', AR1_array)
np.save('std_array_all_sigma.npy', std_array)
np.save('skew_array_all_sigma.npy', skew_array)
np.save('kurt_array_all_sigma.npy', kurt_array)
print('Calculations completed.')