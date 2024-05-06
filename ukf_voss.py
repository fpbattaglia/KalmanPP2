"""
Voss's implementation of the unscented Kalman filter
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
from scipy.integrate import solve_ivp
import tqdm

class UKFModel(object):
	"""A class representing a model for a  Unscented Kalman Filter.

	Attributes:
	    Q_par (float): Initial value for parameter covariance.
	    Q_var (float): Initial value for variable covariance.
	    R (ndarray): Observation covariance.
	"""
	def __init__(self):
		self.Q_par = 0.1  # initial value for parameter covariance
		self.Q_var = 0.1  # initial value for variable covariance
		self.R = np.array((1,))  # observation covariance

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
	def __init__(self, model: UKFModel, ll=800, dT=0.2, dt=0.02):
		"""
		Initialization method for the class.

		:param model: an instance of the UKFModel class
		:param ll: number of data samples (default is 800)
		:param dT: sampling time step (global variable) (default is 0.2)
		:param dt: local integration step (default is 0.02)
		"""
		# Dimensions: dq for param. vector, dx augmented state, dy observation
		self.model = model
		self.dq = model.n_params()
		self.dx = self.dq + model.n_variables()
		self.dy = model.n_observables()

		self.ll = ll  # number of data samples
		self.dT = dT  # sampling time step (global variable)
		self.dt = dt  # local integration step
		# noinspection PyUnresolvedReferences
		if type(model.Q_par) is float or model.Q_par.size > 0:
			self.Q = block_diag(model.Q_par, model.Q_var)  # process noise covariance matrix
		else:
			self.Q = model.Q_var

		self.R = model.R
		self.Pxx = None
		self.Ks = None
		self.x_hat = None
		self.use_solveivp = True

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

		pars = x[:dq, :]
		xnl = x[dq:, :]

		if self.use_solveivp:

			def ode_func(t: np.ndarray, xa):
				return fc(xa, p)

			n_var, n_points = xnl.shape
			xnl_out = np.zeros_like(xnl)
			for i in range(n_points):
				p = pars[:, [i]]
				sol = solve_ivp(ode_func, (0, self.dT), xnl[: ,i], vectorized=True)
				xnl_out[:, i] = sol.y[:, -1]
			xnl = xnl_out
		# 4th order Runge-Kutta integrator with parameters
		else:
			p = pars
			nn = int(np.fix(self.dT / dt))
			for i in range(nn):
				# print(p)
				k1 = dt * fc(xnl, p)
				k2 = dt * fc(xnl + k1 / 2, p)
				k3 = dt * fc(xnl + k2 / 2, p)
				k4 = dt * fc(xnl + k3, p)
				xnl = xnl + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

		r = np.vstack([x[:dq, :], xnl])
		return r

	def unscented_transform(self, x_hat, Pxx, y, R):
		"""
		This method performs an unscented transform for a given set of parameters.

		:param x_hat: Initial state estimate
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
		Pxx = (Pxx + Pxx.T) / 2.
		xsigma = np.real(sqrtm(dx * Pxx).T)  # Pxx = root * root', but Pxx = chol' * chol
		xsigma = (xsigma + xsigma.T) / 2.
		Xa = x_hat[:, np.newaxis] + np.hstack([xsigma, -xsigma])
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
		x_hat = x_tilde + np.dot(K, (y - y_tilde))
		Pxx = Pxx - np.dot(K, Pxy.T)
		Pxx = (Pxx + Pxx.T) / 2.
		return x_hat, Pxx, K

	def filter(self, y, initial_condition=None, disable_progress=False):
		"""
		:param y: The observed data, an array of shape (dy, ll), where dy is the number of observed variables and ll is the number of time steps.
		:param initial_condition: The initial condition for the estimation, an array of shape (dx,), where dx is the number of state variables. Defaults to None.
		:return: Tuple containing the estimated state trajectory x_hat, the covariance matrix Pxx, the Kalman gains Ks, and the estimation errors. All of them are arrays of shape (dx, ll), where dx is the number of state variables and ll is the number of time steps.

		"""
		ll = self.ll
		dx = self.dx
		dy = self.dy

		x_hat = np.zeros((dx, ll))

		if initial_condition is not None:
			x_hat[:, 0] = initial_condition
		else:
			x_hat[self.dq, 0] = y[0, 0]  # first guess of x_1 set to observation

		Pxx = np.zeros((dx, dx, ll))

		Pxx[:, :, 0] = self.Q

		# Variables for the estimation
		errors = np.zeros((dx, ll))
		Ks = np.zeros((dx, dy, ll))  # Kalman gains

		# Main loop for recursive estimation
		for k in tqdm.tqdm(range(1, ll), disable=disable_progress):
			x_hat[:, k], Pxx[:, :, k], Ks[:, :, k] = self.unscented_transform(x_hat[:, k - 1], Pxx[:, :, k - 1], y[:, k], self.R)
			# Pxx[0, 0, k] = self.model.Q_par
			Pxx[:, :, k] = self.covariance_postprocessing(Pxx[:, :, k])
			errors[:, k] = np.sqrt(np.diag(Pxx[:, :, k]))

		self.Pxx = Pxx
		self.Ks = Ks
		self.x_hat = x_hat
		return x_hat, Pxx, Ks, errors

	def covariance_postprocessing(self, P):
		"""
		Perform post-processing on the covariance matrix P, to enable for example covariance inflation.

		:param P: The original covariance matrix.
		:return: The processed covariance matrix P_out.
		"""
		P_out = P.copy()
		P_out[:self.dq, :self.dq] = self.model.Q_par
		return P_out

# Results
	def stats(self, x=None):
		"""
		Calculates the statistical errors and chi-squared value.

		Args:
		    x (optional): The true values. If provided, the chi-squared value will be calculated using the predicted values and the true values.

		Returns:
		    errors: A numpy array of shape (dx, ll) containing the statistical errors for each parameter at each time step.
		    chisq: The chi-squared value if `x` is provided, None otherwise.
		"""
		errors = np.zeros((self.dx, self.ll))
		for k in range(self.ll):
			errors[:, k] = np.sqrt(np.diag(self.Pxx[:, :, k]))
		if x is not None:
			chisq = np.mean((x-self.x_hat) **2, axis=(0,1))
		else:
			chisq = None

		return errors, chisq

class FNModel(UKFModel):
	"""
	The `FNModel` class is used to represent a FitzHugh-Nagumo model. It inherits from the `UKFModel` class.

	Methods:
	---------

	__init__(self, a=0.7, b=0.8, c=3., Q_par=0.015, Q_var=np.array((1.,)), R=1.)
		Initializes an instance of the `FNModel` class.

	f_model(self, x, p)
		Takes two parameters, `x` and `p`, which are arrays. It computes and returns a 2D array of values based on the given formulas.

	obs_g_model(self, x)
		Takes a 2D array representing the input data and returns a 1D array representing the observations (in this case the membrane potential variables).

	n_params(self)
		Returns the number of parameters.

	n_variables(self)
		Returns the number of variables.

	n_observables(self)
		Returns the number of observables.
	"""
	def __init__(self, a=0.7, b=0.8, c=3., Q_par=0.015, Q_var=np.array((1.,)), R=1.):
		"""
		Initializes an instance of the FNModel class.

		:param a: a float representing the value of parameter a (default 0.7)
		:param b: a float representing the value of parameter b (default 0.8)
		:param c: a float representing the value of parameter c (default 3.0)
		:param Q_par: a float representing the initial value for parameter covariance (default 0.015)
		:param Q_var: a numpy array representing the initial value for variable covariance (default np.array((1.,)))
		:param R: a float representing the observation covariance (default 1.0)
		"""
		super(FNModel, self).__init__()
		self.a = a
		self.b = b
		self.c = c
		self.Q = 0.015
		self.Q_par = Q_par  # initial value for parameter covariance
		self.Q_var = Q_var  # # initial value for variable covariance
		self.R = R  # observation covariance

	def f_model(self, x, p):
		"""
		:param x: the input array
		:param p: the input array
		:return: a 2D array containing computed values based on the input arrays

		This method takes in two parameters, `x` and `p`, which are arrays. It computes and returns a 2D array of values based on the given formulas.

		The parameter `x` represents an input array.
		The parameter `p` represents an input array.

		The return value is a 2D array containing computed values based on the input arrays `x` and `p`.
		"""
		a, b, c = self.a, self.b, self.c
		# p = p.ravel()
		x = np.atleast_2d(x)
		# return np.array([c * (x[1,:] + x[0,:] - x[0,:]**3 / 3 + p), -(x[0,:] - a + b * x[1,:]) / c])
		rr = [np.atleast_2d(c * (x[1, :] + x[0, :] - x[0, :] ** 3 / 3 + p[0, :])),
			  np.atleast_2d(-(x[0, :] - a + b * x[1, :]) / c)]
		# print(rr)
		return np.vstack(rr)

	def obs_g_model(self, x):
		"""
		:param x: A 2-dimensional array representing the input data. The array should have shape (n, m), where n is the number of samples and m is the number of features.
		:return: A 1-dimensional array representing the observations (in this case the membrane potential variables). The array will have shape (m,) where m is the number of features.
		"""
		return x[1, :]

	def n_params(self):
		"""
		Returns the number of parameters.

		:return: The number of parameters.
		:rtype: int
		"""
		return 1

	def n_variables(self):
		"""
		Returns the number of variables.

		:return: The number of variables defined in the method.
		:rtype: int
		"""
		return 2

	def n_observables(self):
		"""
		Returns the number of observables.

		:return: Number of observables.
		:rtype: int
		"""
		return 1


class NatureSystem(object):
	def __init__(self, ll, dT, dt, n_variables, n_params, n_observations, initial_condition=None):
		"""
		Constructor for the class.

		:param ll: The length of the time series.
		:param dT: The time step for the time series.
		:param dt: The time step for integration.
		:param n_variables: The number of variables in the system.
		:param n_params: The number of parameters in the system.
		:param n_observations: The number of observations in the time series.
		:param initial_condition: The initial condition for the variables. Defaults to None.

		"""
		self.dT = dT
		self.dt = dt
		self.ll = ll
		self.x0 = np.zeros((n_variables, ll))
		self.y = np.zeros((n_observations, ll))
		self.p = np.zeros((n_params, ll))
		if initial_condition is not None:
			self.x0[:, 0] = initial_condition

	def system(self, x, p):
		return np.array([])

	def integrate_solveivp(self):
		p = None

		def ode_func(t, x):
			return self.system(x, p)

		for n in range(self.ll - 1):
			p = self.p[:, n]
			xx = self.x0[:, n]
			sol = solve_ivp(ode_func, [0, self.dT], xx)
			self.x0[:, n + 1] = sol.y[:, -1]

	def integrateRK4(self):
		"""
		Integrates the system of ordinary differential equations using the fourth order Runge-Kutta method.

		:return: None
		"""
		nn = int(self.dT / self.dt)  # the integration time step is smaller than dT
		for n in range(self.ll - 1):
			xx = self.x0[:, n]
			for i in range(nn):
				k1 = self.dt * self.system(xx, self.p[:, n])
				k2 = self.dt * self.system(xx + k1 / 2, self.p[:, n])
				k3 = self.dt * self.system(xx + k2 / 2, self.p[:, n])
				k4 = self.dt * self.system(xx + k3, self.p[:, n])
				xx = xx + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
			self.x0[:, n + 1] = xx

	def observations(self):
		pass


class FNNature(NatureSystem):
	"""
	The `FNNature` class represents a nature for the FitzHugh-Nagumo model.
	It inherits from the `NatureSystem` class.

	Attributes:
		a (float): Parameter 'a' for the FNNature model.
		b (float): Parameter 'b' for the FNNature model.
		c (float): Parameter 'c' for the FNNature model.
		R0 (float): Initial value of R for the FNNature model.
		R (float): Current value of R for the FNNature model.
		x0 (ndarray): The state vector of the nature system.
		y (ndarray): The observed states of the nature system.
		p (ndarray): The set of parameters for the nature system.

	Methods:
		system(self, x, p): Calculates the new state of the system based on the current state 'x' and
							parameter 'p'.
		set_current(self): Calculates and sets the current 'p' in the p array for each step.
		observations(self): Generates the observations based on the current state and noise.
	"""
	def __init__(self, ll, dT, dt, a=0.7, b=0.8, c=3., R0=0.2, initial_condition: np.ndarray | None = None):
		"""
		Initialize the FNNature object.

		:param ll: Length of the time horizon in days.
		:param dT: Length of each time step in days.
		:param dt: Integration step size in days.
		:param a: Parameter a for the FNNature model (default=0.7).
		:param b: Parameter b for the FNNature model (default=0.8).
		:param c: Parameter c for the FNNature model (default=3.0).
		:param R0: Initial value of R for the FNNature model (default=0.2).
		:param initial_condition: Initial condition for the FNNature model (default=None).
		"""
		super(FNNature, self).__init__(ll, dT, dt, 2, 1, 1, initial_condition)
		self.a = a
		self.b = b
		self.c = c
		self.R0 = R0
		self.R = R0
		self.set_current()
		self.integrateRK4()
		self.observations()

	def system(self, x, p):
		"""

		"""
		return np.array([self.c * (x[1] + x[0] - x[0] ** 3 / 3 + p[0]), -(x[0] - self.a + self.b * x[1]) / self.c])

	def set_current(self):
		"""
		Sets the current value in the self.p array.

		:return: None
		"""
		# External input, estimated as parameter p later on
		z = (np.arange(self.ll) / 250) * 2 * np.pi
		z = -0.4 - 1.01 * np.abs(np.sin(z / 2))
		self.p[0, :] = z

	def observations(self):
		"""
		Calculates the observations based on the system's state and noise.

		:return: None
		"""
		self.R = self.R0 ** 2 * np.var(self.x0[0, :])
		self.y[0, :] = self.x0[0, :] + np.sqrt(self.R) * np.random.randn(self.ll)


if __name__ == '__main__':
	nature = FNNature(ll=800, dT=0.2, dt=0.02)
	# plot the nature data and the observations

	# define the model
	Q_var0 = np.diag((nature.R, nature.R))
	fn_model = FNModel(Q_var=Q_var0, R=nature.R)

	# UKF instance
	uk_filter = UKFVoss(model=fn_model)

	x_hat0, Pxx0, Ks0, errors0 = uk_filter.filter(nature.y)
