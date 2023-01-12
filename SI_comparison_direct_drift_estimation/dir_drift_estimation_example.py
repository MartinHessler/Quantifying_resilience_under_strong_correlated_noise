import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.stats import iqr
import time
from scipy import optimize
import scipy.stats as cpy

sigma_array = np.array([0.1, 2.2, 4.5])

burn_in_period = 250 # choose 250
simulation_length = 160 # choose 160
end_time = burn_in_period + simulation_length # in years
within_year_points = 50

samples = end_time * within_year_points

init_sigma_count = 0

start_time = 0
dt = 1. / 50.
time = np.linspace(start_time, simulation_length, simulation_length * within_year_points)

def prepare_data(data_window, bin_num):
	bin_array = np.linspace(np.min(data_window), np.max(data_window), bin_num + 1)
	bin_centers = bin_array[:-1] + (bin_array[1] - bin_array[0]) / 2.
	bin_labels = np.digitize(data_window, bin_array, right = False)
	num_bin_members = np.zeros(bin_num) 
	bin_incr_mean = np.zeros(bin_num)
	bin_incr_mean_squared = np.zeros(bin_num)
	for j in range(1,bin_num+1):
		bin_indizes = np.where(bin_labels == j)[0]
		num_bin_members[j - 1] = bin_indizes.size
		zero_members = False
		if num_bin_members[j - 1] == 0:
			num_bin_members[j - 1] = 1
			zero_members = True
		if printbool:
			print('bin_indizes size: ', bin_indizes.size)
		if not any(bin_indizes == window_size - 1):
			incr = data_window[bin_indizes+1] - data_window[bin_indizes]
		elif any(bin_indizes == window_size - 1):
			incr = np.zeros(bin_indizes.size)
			for m in range(bin_indizes.size):
				if bin_indizes[m] != window_size - 1:
					incr[m] = data_window[bin_indizes[m]+1] - data_window[bin_indizes[m]]
				else:
					incr[m] = 0
		bin_incr_mean[j - 1] = 1./dt * np.sum(incr) / num_bin_members[j - 1]
		bin_incr_mean_squared[j - 1] = 1./dt *  np.sum(incr**2) / num_bin_members[j - 1]
		if zero_members:
			num_bin_members[j - 1] = 0
	return bin_array, bin_centers, bin_labels, num_bin_members, bin_incr_mean, bin_incr_mean_squared

def direct_drift_estimation(x, time, window_size, window_shift, bin_num = 250):
	dt = time[1] - time[0]
	loop_range = np.arange(0, x.size - window_size + 1, window_shift)
	steps = loop_range.size
	drift_estimates = np.zeros(( steps, window_size-1)) 
	diffusion_estimates = np.zeros((steps, window_size-1))
	sorted_data_window = np.zeros((2,window_size))
	slope_arrayI = np.zeros(steps)
	slope_arrayII = np.zeros(steps)
	for i in range(steps):
		print('steps: ' + str(i+1) + '/' + str(steps))
		data_window = np.roll(x, shift = -int(loop_range[i]))[:window_size]
		bin_array, bin_centers, bin_labels, num_bin_members, bin_incr_mean, bin_incr_mean_squared = prepare_data(data_window, bin_num)
		p = np.polyfit(bin_centers, bin_incr_mean, deg = 3)
		fixed_point_estimate = np.mean(data_window)
		slope_arrayII[i] = 3* p[0] * fixed_point_estimate**2 + 2 * p[1] * fixed_point_estimate + p[2]
		p = np.polyfit(bin_centers, bin_incr_mean, deg = 1)
		slope_arrayI[i] = p[0] 
	return slope_arrayI, slope_arrayII, loop_range

window_size = 750
window_shift = 30

for sigma in sigma_array:
	data = np.load('ecological_model_planktivores_sig' + str(sigma) + 'ppyear50.npy') # Example: white noise without deseasonalization.
	direct_slopesI, direct_slopesII, loop_range = direct_drift_estimation(data, time, window_size, window_shift)
	np.save('direct_slopes_linear' +'white_noise_sigma' + str(sigma) + '.npy', direct_slopesI)
	np.save('direct_slopes_polynomial' +'white_noise_sigma' + str(sigma) + '.npy', direct_slopesII)