"""
Voss implementation of the unscented Kalman filter
"""

# Original credit
# % Unscented Kalman Filter (UKF) applied to FitzHugh-Nagumo neuron dynamics.
# % Voltage observed, currents and inputs estimated.
# % FitzHughNagumo() is the main program and calls the other programs.
# % A detailed description is provided in H.U. Voss, J. Timmer & J. Kurths, Nonlinear dynamical system identification
# from uncertain and indirect measurements, Int. J. Bifurcation and Chaos 14, 1905-1933 (2004).
# I will be happy to email this paper on request. It contains a tutorial about the estimation of hidden states and
# unscented Kalman filtering. % For commercial use and questions, please contact me.
# Henning U. Voss, Ph.D.
# Associate Professor of Physics in Radiology
# Citigroup Biomedical Imaging Center
# Weill Medical College of Cornell University
# 516 E 72nd Street
# New York, NY 10021
# Tel. 001-212 746-5216, Fax. 001-212 746-6681
# Email: hev2006@med.cornell.edu

# ported to Python FPB April 2024


import numpy as np
from scipy.linalg import sqrtm, block_diag


class UKFModel(object):
	def __init__(self):
		pass

	def f_model(self, x, p):
		return np.array([])

	def obs_g_model(self, x):
		return np.array([])

	def n_params(self):
		return 0

	def n_variables(self):
		return 0

	def n_observables(self):
		return 0


class UKFVoss(object):
	def __init__(self, model: UKFModel):
		# Dimensions: dq for param. vector, dx augmented state, dy observation
		self.model = model
		self.dq = model.n_params()
		self.dx = self.dq + model.n_variables()
		self.dy = model.n_observables()

		self.ll = 800  # number of data samples
		self.dT = 0.2  # sampling time step (global variable)
		self.dt = 0.02  # local integration step
		self.Q = 0.015  # process noise covariance matrix
		# Q = 0.0001
		self.R0 = 0.2  # observation noise standard deviation, in units of the standard deviations of the "signal"
		self.Pxx = None
		self.Ks = None
		self.xhat = None

	def evolve_f(self, x):
		"""
		Evolve the given system using a 4th order Runge-Kutta integrator.

		:param x: The initial state of the system. Contains in the first self.dq elements the parameters.
		Dynamical variables are the further elements   (numpy.ndarray)
		:return: The evolved state of the system. (numpy.ndarray)

		"""
		# Model function F(x)

		dq = self.dq
		dt = self.dt
		fc = self.model.f_model
		nn = int(np.fix(self.dT / dt))

		p = x[:dq, :]
		xnl = x[dq:, :]

		# 4th order Runge-Kutta integrator with parameters (TODO use integrator function)
		for i in range(nn):
			# print(p)
			k1 = dt * fc(xnl, p)
			k2 = dt * fc(xnl + k1 / 2, p)
			k3 = dt * fc(xnl + k2 / 2, p)
			k4 = dt * fc(xnl + k3, p)
			xnl = xnl + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

		r = np.vstack([x[:dq, :], xnl])
		return r

	def unscented_transform(self, xhat, Pxx, y, R):
		"""
		This method performs an unscented transform for a given set of parameters.

		:param xhat: Initial state estimate
		:param Pxx: Covariance matrix of the state estimate
		:param y: Measurement vector
		:param R: Measurement noise covariance matrix

		:return: Updated state estimate, updated covariance matrix, Kalman gain
		"""
		dx = self.dx
		dy = self.dy

		fct = self.evolve_f
		obsfct = self.model.obs_g_model
		N = 2 * dx

		xsigma = sqrtm(dx * Pxx).T  # Pxx = root * root', but Pxx = chol' * chol
		Xa = xhat[:, np.newaxis] + np.hstack([xsigma, -xsigma])
		X = fct(Xa)

		x_tilde = np.mean(X, axis=1)  # same as x_tilde = np.sum(X, axis=1) / N

		Pxx = np.zeros((dx, dx))
		for i in range(N):
			Pxx += np.outer((X[:, i] - x_tilde), (X[:, i] - x_tilde)) / N

		Y = np.atleast_2d(obsfct(X))

		y_tilde = np.mean(Y, axis=1)
		Pyy = R.copy()
		for i in range(N):
			Pyy += np.outer((Y[:, i] - y_tilde), (Y[:, i] - y_tilde)) / N

		Pxy = np.zeros((dx, dy))
		for i in range(N):
			Pxy += np.outer((X[:, i] - x_tilde), (Y[:, i] - y_tilde)) / N

		K = np.dot(Pxy, np.linalg.inv(Pyy))  # same as K = np.dot(Pxy, np.linalg.inv(Pyy))
		xhat = x_tilde + np.dot(K, (y - y_tilde))
		Pxx = Pxx - np.dot(K, Pxy.T)

		return xhat, Pxx, K

	def filter(self, y):
		ll = self.ll
		dx = self.dx
		dy = self.dy

		xhat = np.zeros((dx, ll))
		xhat[:, 0] = y[:, 0]  # first guess of x_1 set to observation

		Pxx = np.zeros((dx, dx, ll))

		Pxx[:, :, 0] = block_diag(Q, R, R)

		# Variables for the estimation
		errors = np.zeros((dx, ll))
		Ks = np.zeros((dx, dy, ll))  # Kalman gains

		# Main loop for recursive estimation
		for k in range(1, ll):
			xhat[:, k], Pxx[:, :, k], Ks[:, :, k] = self.unscented_transform(xhat[:, k - 1], Pxx[:, :, k - 1], y[:, k], R)

			Pxx[0, 0, k] = Q

		self.Pxx = Pxx
		self.Ks = Ks
		self.xhat = xhat
		return xhat, Pxx, Ks

# Results
	def stats(self):
		errors = np.zeros((self.dx, self.ll))
		for k in range(self.ll):
			errors[:, k] = np.sqrt(np.diag(self.Pxx[:, :, k]))
class FNModel(UKFModel):
	def __init__(self, a=0.7, b=0.8, c=3.):
		super(FNModel, self).__init__()
		self.a = a
		self.b = b
		self.c = c

	def f_model(self, x, p):
		a, b, c = self.a, self.b, self.c
		# p = p.ravel()
		x = np.atleast_2d(x)
		# return np.array([c * (x[1,:] + x[0,:] - x[0,:]**3 / 3 + p), -(x[0,:] - a + b * x[1,:]) / c])
		rr = [np.atleast_2d(c * (x[1, :] + x[0, :] - x[0, :] ** 3 / 3 + p)),
			  np.atleast_2d(-(x[0, :] - a + b * x[1, :]) / c)]
		# print(rr)
		return np.vstack(rr)

	def obs_g_model(self, x):
		return x[1, :]

	def n_params(self):
		return 1

	def n_variables(self):
		return 2

	def n_observables(self):
		return 1
