import numpy as np
import pylab
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

####  Deterministischer kontinuierlicher Anteil ("Monitoring Interval") ####

def dA_dt(A, param):
	q_E = param# effort
	return -q_E*A

def dF_dt(F, A, sigma):
	D_F = 0.1 # foraging arena
	F_R = 100 # refuge reservoir
	c_FA = 0.3 # consumption rate of planktivore by adult piscivores
	return D_F * (F_R - F) - c_FA * F * A

def dJ_dt(J, A, F):
	c_JA = 0.001 # consumption rate of juvenile by adult piscivores
	c_JF = 0.5 # consumption rate of juvenile piscivores by planktivores
	nu = 1 # rate at which juvenile piscivores become vulnerable against planktivores
	h = 8 # rate at which juvenile planktivores enter the refuge
	return - c_JA * J * A  - (c_JF * nu * F * J)/(h + nu + c_JF * F)

####  Deterministischer diskreter Anteil ("Maturation Interval") ####

def A_map(A, J):
	s = 0.5 # survival rate
	return s* (A + J)

def F_map(F):
	return F

def J_map(A_plus):
	f = 2 # fecundity of adult piscivores
	return f*A_plus


#### Euler Verfahren ####

def euler(N, A, F, J, qE_param, sigma, dt):
	global phi, print_Euler_progress
	sqrt_dt = np.sqrt(dt)
	for i in range(int(N - 1)):
		if print_Euler_progress:
			print('Step: ' + str(i))
		if i % within_year_points == 0: ## Führe zuerst diskretes Intervall ("Maturation Inverval") aus ##
			if print_Euler_progress:
				print('Compute discrete maturation step, before monitoring')
			A[i+1] = A_map(A[i], J[i])
			F[i+1] = F_map(F[i])
			J[i+1] = J_map(A[i+1])
			## Danach führe kontinuierliches Intervall aus ##
			A[i+1] = A[i+1] + dt * dA_dt(A[i+1], qE_param[i])
			F[i+1] = F[i+1] + dt * dF_dt(F[i+1], A[i+1], sigma) + sigma * np.random.normal(loc = 0.0, scale = sqrt_dt)
			J[i+1] = J[i+1] + dt * dJ_dt(J[i+1], A[i+1], F[i+1])
	
		else:
			## Führe kontinuierliches Intervall aus ("Monitoring Interval") ##
			if print_Euler_progress:
				print('Compute monitoring step only.')
			A[i+1] = A[i] + dt * dA_dt(A[i], qE_param[i])
			F[i+1] = F[i] + dt * dF_dt(F[i], A[i], sigma) + sigma * np.random.normal(loc = 0.0, scale = sqrt_dt)
			J[i+1] = J[i] + dt * dJ_dt(J[i], A[i], F[i])
	return A, F, J


print_Euler_progress = False

#### Berechne eine Realisation des Prozesses ####

start_time = 0
burn_in_period = 250 # choose 250
simulation_length = 160 # choose 160
end_time = burn_in_period + simulation_length # in years
within_year_points = 50 # high sampling 50; low sampling 1

samples = end_time * within_year_points
# print(samples)

dt = 1. / 50.

# sigma = 0.002 / dt # additive noise level low
# sigma = round(0.044 / dt, 2) # additive noise level medium
sigma = 0.09 / dt # additive noise level high


print('sigma:' + str(sigma))
time = np.linspace(start_time, end_time, samples)

qE_increment = 0.013 # per year (choose 0.013)
qE_init = 1.
print(np.ones(burn_in_period * within_year_points).shape)
print(np.arange(qE_init, qE_init + simulation_length * qE_increment, qE_increment / within_year_points).shape)
qE_param = np.append(np.ones(burn_in_period * within_year_points - 1) * qE_init, np.arange(qE_init, qE_init + simulation_length * qE_increment, qE_increment / within_year_points))
plt.plot(time, qE_param)
plt.show()

print('DELTA qE / year: ' + str(qE_param[burn_in_period * within_year_points + within_year_points] - qE_param[burn_in_period * within_year_points]))

print(qE_param.shape)
A = np.zeros(samples)
# print(A.shape)
F = np.zeros(samples)
J = np.zeros(samples)

A[0] = 200
F[0] = 25

for i in range(qE_param.size):
	if qE_param[i] >= 1.78:
		attractor_switch = i
		break

for i in range(qE_param.size):
	if qE_param[i] >= 2.23:
		point_of_no_return = i
		break


A, F, J = euler(samples, A, F, J, qE_param, sigma, dt)
A = A[burn_in_period * within_year_points:]
F = F[burn_in_period * within_year_points:]
J = J[burn_in_period * within_year_points:]
print('shape of A, F, J: ' + str(A.shape))
np.save('ecological_model_adults_sig' + str(sigma) + 'ppyear' + str(within_year_points) +'.npy', A)
np.save('ecological_model_planktivores_sig' + str(sigma) + 'ppyear' + str(within_year_points) +'.npy', F)
np.save('ecological_model_juveniles_sig' + str(sigma) + 'ppyear' + str(within_year_points) +'.npy', J)
plt.plot(time[burn_in_period * within_year_points:], A, label = 'adults')
plt.plot(time[burn_in_period * within_year_points:], F, label = 'planktivores')
plt.plot(time[burn_in_period * within_year_points:], J, label = 'juveniles')
plt.axvline(time[attractor_switch], color = 'r', ls = '--')
plt.axvline(time[point_of_no_return], color = 'g', ls = '--')
plt.legend()
plt.savefig('ecological_model_sig' + str(sigma) + 'ppyear' + str(within_year_points) + '.png')
plt.show()