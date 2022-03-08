import numpy as np
import pylab
import time
import math
import scipy.stats as cpy
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def M1(x, theta):
	b0 = theta[0]
	alpha = theta[1]
	return b0 + np.tan(alpha) * x

def M2(x, theta):
	b0 = theta[0]
	return np.ones(x.size) * b0

def sample_prior_params(n_param, n_prior_samples, x, data):
	print('data_shape: ', data.shape)
	intercept_loc = data[0]
	print(intercept_loc)
	slope_range = 1.5 * ((np.max(data) - np.min(data)) / (x[-1] - x[0]))
	print(slope_range)
	sampled_array = np.zeros((n_prior_samples, n_param))
	sampled_array[:,0] = np.random.normal(loc = intercept_loc, scale = 1, size = n_prior_samples) # -1.35; -12.5; -17.4; -11.2; -1.37; 0.53
	sampled_array[:,1] = np.random.uniform(low = 0, high = slope_range, size = n_prior_samples)#  0.001; 0.01; 0.01 h; 0.01; 0.01; 0.001
	sampled_array[:,2] = np.random.uniform(low = np.log(0.5), high = np.log(5.), size = n_prior_samples)
	return sampled_array

def log_likelihood(x, y, theta, model):
	result = np.sum(-0.5*np.log(2 * np.pi* np.exp(theta[-1])**2) - (y - model(x,theta))**2/(2.* np.exp(theta[-1])**2))
	if np.isfinite(result):
		return result
	else:
		return - np.inf

def harmonic_average(x):
	N = x.size
	denominator = np.sum(1. / x)
	return N / denominator

sigma_array = np.array([0.1,2.2,4.5])
phi = 0.92
shift_of_seasonality = 50

data = np.load('deseasoned_polynom_slope_trends_sigma0.1_2.2_4.5_ws750.npy')[:,:,0]
simulation_length = 160 # choose 160
end_time = simulation_length # in years
within_year_points = 50 # high sampling 50; low sampling 1

samples = end_time * within_year_points

start_time = 0
dt = 1. / 50.
time = np.linspace(start_time, simulation_length, simulation_length * within_year_points)
time = time[shift_of_seasonality:]

qE_increment = 0.013 # per year (choose 0.013)
qE_init = 1.
qE_param = np.arange(qE_init, qE_init + simulation_length * qE_increment, qE_increment / within_year_points)
qE_param = qE_param[shift_of_seasonality:]
window_size = 750
window_shift = 30

loop_range = np.arange(0, simulation_length * within_year_points - window_size - shift_of_seasonality, window_shift, dtype = int)


for i in range(time.size):
	    if qE_param[i] >= 1.78:
	        attractor_switch = i
	        break


for j in range(loop_range.size): #### adapt point_of_no_return to make it the index of loop_range
	if qE_param[window_size - 1 + loop_range[j]] >= 2.23:
		point_of_no_return = j
		break

n_param = 3
print_progress = False
save_results = True
control_plots_prior = False
print_log_LH = False
plot_prior_samples = False
BF_output = True
MC_BF_statistics = True
control_plot_point_of_no_return = False

n_prior_samples = 10**7
MC_runs = 1

upper_BF_control_limit = 10**(1)
lower_BF_control_limit = 10**(-1)
BF12_array = np.zeros(MC_runs)
BF21_array = np.zeros(MC_runs)
log_evidence_M1_array = np.zeros(MC_runs)
log_evidence_M2_array = np.zeros(MC_runs)
average_BF12 = 0
average_BF21 = 0


for i in range(sigma_array.size):
	time_series = np.load('deseasonalized_eco_model_planktivores_sig_' + str(sigma_array[i]) + 'ppyear' + str(within_year_points) + '.npy')

	if control_plot_point_of_no_return:
		plt.plot(time[window_size - 1 + loop_range[:point_of_no_return]], data[0,:loop_range[:point_of_no_return].size])
		plt.plot(time[window_size - 1 + loop_range[:point_of_no_return]], data[1,:loop_range[:point_of_no_return].size])
		plt.plot(time[window_size - 1 + loop_range[:point_of_no_return]], data[2,:loop_range[:point_of_no_return].size])
		plt.show()
	for k in range(MC_runs):
		print('______MC run: ' + str(k + 1) + '/' + str(MC_runs) + '______')
		prior_sampling_array = sample_prior_params(n_param, n_prior_samples, time[window_size - 1 + loop_range[:point_of_no_return]], data[i,:loop_range[:point_of_no_return].size])
		if plot_prior_samples:
			plt.plot(prior_sampling_array[:,0])
			plt.plot(prior_sampling_array[:,1])
			plt.plot(prior_sampling_array[:,2])
			plt.show()

		if control_plots_prior:
			plt.plot(time[window_size - 1 + loop_range[:point_of_no_return]], data[i,:loop_range[:point_of_no_return].size], marker = 'o', color = 'r')
			for j in range(n_prior_samples):
				plt.plot(time[window_size - 1 + loop_range[:point_of_no_return]], M1(time[window_size - 1 + loop_range[:point_of_no_return]], prior_sampling_array[j,:]), color = 'grey')
			plt.show()

			plt.plot(time[window_size - 1 + loop_range[:point_of_no_return]], data[i,:loop_range[:point_of_no_return].size], marker = 'o', color = 'r')
			for j in range(n_prior_samples):
				plt.plot(time[window_size - 1 + loop_range[:point_of_no_return]], M2(time[window_size - 1 + loop_range[:point_of_no_return]], prior_sampling_array[j,:]), color = 'grey')
			plt.show()

		log_evidence_model1 = 0
		log_evidence_model2 = 0
		log_LH_model1_array = np.zeros(n_prior_samples)
		log_LH_model2_array = np.zeros(n_prior_samples)
		for j in range(n_prior_samples):
			if print_progress:
				print('progress: ' + str(j + 1) + '/' + str(n_prior_samples) + '   ||   MC run: ' + str(k+1) + '/' + str(MC_runs))
			log_LH_model1_array[j] = log_likelihood(time[window_size - 1 + loop_range[:point_of_no_return]], data[i,:loop_range[:point_of_no_return].size], prior_sampling_array[j,:], M1)
			log_LH_model2_array[j] = log_likelihood(time[window_size - 1 + loop_range[:point_of_no_return]], data[i,:loop_range[:point_of_no_return].size], prior_sampling_array[j,:], M2)
			if print_log_LH:
				print(log_LH_model1_array[j])
				print(log_LH_model2_array[j])
		print(log_LH_model1_array)
		print(log_LH_model2_array)
		print(float(n_prior_samples))
		log_evidence_model1 = np.log(np.sum(np.exp(log_LH_model1_array)) / float(n_prior_samples))
		log_evidence_model2 = np.log(np.sum(np.exp(log_LH_model2_array)) / float(n_prior_samples))

		log_BF12 = log_evidence_model1 - log_evidence_model2
		log_BF21 = log_evidence_model2 - log_evidence_model1
		BF12 = np.exp(log_BF12)
		BF21 = np.exp(log_BF21)

		BF12_array[k] = BF12
		BF21_array[k] = BF21

		log_evidence_M1_array[k] = log_evidence_model1
		log_evidence_M2_array[k] = log_evidence_model2

		if BF_output:
			print('________________________________________')
			print('______Bayes factor analysis output______')
			print('________________________________________')
			print('log_evidence_model1: ' + str(log_evidence_model1))
			print('log_evidence_model2: ' + str(log_evidence_model2))
			print('log_BF12: ' + str(log_BF12))
			print('BF12: ' + str(BF12))
			print('log_BF21: ' + str(log_BF21))
			print('BF21: ' + str(BF21))
			print('________________________________________')
			print('________________________________________')

	if MC_BF_statistics:
		print('All BF21 factors are greater than ' + str(upper_BF_control_limit) + ' : ' + str(np.all(BF21_array > upper_BF_control_limit)))
		print('All BF12 factors are smaller than ' + str(lower_BF_control_limit) + ' : ' + str(np.all(BF12_array < lower_BF_control_limit)))
		average_BF12 = np.sum(BF12_array) / float(MC_runs)
		average_BF21 = np.sum(BF21_array) / float(MC_runs)
		print('average_BF12: ' + str(average_BF12))
		print('average_BF21: ' + str(average_BF21))
		print('averaged evidences BF12: ' + str(np.sum(np.exp(log_evidence_M1_array))/np.sum(np.exp(log_evidence_M2_array))))
		print('averaged evidences BF21: ' + str(np.sum(np.exp(log_evidence_M2_array))/np.sum(np.exp(log_evidence_M1_array))))
		print('harmonic average BF12: ' + str(harmonic_average(BF12_array)))
		print('harmonic average BF21: ' + str(harmonic_average(BF21_array)))

	if save_results:
		np.save('log_evidence_M1_slope_indicator_sigma' + str(sigma_array[i]) + '.npy', log_evidence_M1_array)
		np.save('log_evidence_M2_slope_indicator_sigma' + str(sigma_array[i]) + '.npy', log_evidence_M2_array)
		np.save('BF12_slope_indicator_sigma' + str(sigma_array[i]) + '.npy', BF12_array)
		np.save('BF21_slope_indicator_sigma' + str(sigma_array[i]) + '.npy', BF21_array)