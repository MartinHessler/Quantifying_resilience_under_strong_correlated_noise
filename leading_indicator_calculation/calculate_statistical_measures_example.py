import numpy as np
import pylab
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from statsmodels.tsa.stattools import acf
import scipy.stats as sts


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
		AR1_array[i, k] = acf(data_window, adjusted = False, nlags = 1)[1]
		std_array[i, k] = np.std(data_window, ddof = 0)
		skew_array[i, k] = sts.skew(data_window, bias = True)
		kurt_array[i, k] = sts.kurtosis(data_window, fisher = False, bias = True)
		print(r'progress: ' + str(k+1) + r'/' + str(loop_range_size) + r'   |||   ' + r'$\sigma$: ' + str(sigma))
	if show_indicators:
		figure_label = np.array(['A', 'B', 'C'])
		original_data = np.load('ecological_model_planktivores_sig' + str(sigma) + 'ppyear50.npy')
		for j in range(original_data.size):
			if original_data[j] >= 21:
				critical_transition = j
				break
		for j in range(loop_range_size):
			if window_size - 1 + loop_range[j] >= critical_transition:
				critical_transition = j + 1
				break
		print('Show indicators.')
		fig, ax = plt.subplots(1,1)
		plt.text(0.025, 0.85, figure_label[i], transform=ax.transAxes, fontsize = 40)
		plt.plot(time[window_size - 1 + loop_range[:critical_transition]], AR1_array[i,:loop_range[:critical_transition].size], label = 'AR1')
		plt.plot(time[window_size - 1 + loop_range[:critical_transition]], std_array[i,:loop_range[:critical_transition].size], label = r'std $\hat{\sigma}$')
		plt.plot(time[window_size - 1 + loop_range[:critical_transition]], skew_array[i,:loop_range[:critical_transition].size], label = r'skewness $\gamma$')
		plt.plot(time[window_size - 1 + loop_range[:critical_transition]], kurt_array[i,:loop_range[:critical_transition].size], label = r'kurtosis $\omega$')
		plt.xlim(0,time[window_size - 1 + loop_range[critical_transition - 1]])
		plt.xticks([0,75], fontsize = 22)
		plt.axvline(94.63182897862232, ls = '-', lw = 2, color = 'orange')
		plt.xlabel(r'time $t$', fontsize = 25)
		if i == 0:
			plt.yticks([0,6,12], fontsize = 22)
			plt.legend(loc = 'upper center', fontsize = 20)
		elif i == 1:
			plt.yticks([0,3,6], fontsize = 22)
		else:
			plt.yticks([0,2.5,5], fontsize = 22)
		plt.subplots_adjust(left  = 0.093,
			right = 0.94,
			bottom = 0.18,
			top = 0.955,
			wspace = 0.2,
			hspace = 0.2)
		plt.savefig('indicators_without_deseason_white_sigma' + str(sigma) + '.png')
		plt.show()
	i += 1


print('Save indicators.')
np.save('AR1_array_all_sigma.npy', AR1_array)
np.save('std_array_all_sigma.npy', std_array)
np.save('skew_array_all_sigma.npy', skew_array)
np.save('kurt_array_all_sigma.npy', kurt_array)
print('Calculations completed.')