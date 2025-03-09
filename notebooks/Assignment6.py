import numpy as np
import tqdm
from scipy.linalg import sqrtm  # , block_diag
from scipy.signal import convolve2d
# from tqdm import tqdm

import ukf_voss


# """
# 		Args:
# 			ll (float): Length scale parameter.
# 			dT (float): Time step size.
# 			dt (float): Integration step size.
# 			f (int): Number of rows in the neural field.
# 			g (int): Number of columns in the neural field.
# 			k (float): Spread of coupling connections.
# 			K (float): Global connection strength.
# 			C (float): Input strength.
# 			B (float): External input strength.
# 			tau (float): Time constant of the neural field.
# 			R0 (float): Baseline firing rate of the neural field.
# 			initial_condition (numpy.ndarray | None, optional): Initial conditions for the neural field. Defaults to None.
#
# 		"""

class WilsonCowanNature(ukf_voss.ControllableNatureSystem):
	"""
	:class: WilsonCowanNature

	WilsonCowanNature is a nature class that implements the Wilson-Cowan model.

	:param ll: The length of the time vector
	:type ll: int
	:param dT: The time step size for the simulation
	:type dT: float
	:param dt: The time step size for the integration
	:type dt: float
	:param f: The number of rows in the grid
	:type f: int, optional
	:param g: The number of columns in the grid
	:type g: int, optional
	:param k: The parameter controlling the decay of the spatial kernel
	:type k: float, optional
	:param K: The global interaction strength
	:type K: float, optional
	:param C: The inverse time constant of the membrane potential
	:type C: float, optional
	:param B: potential to recovery coupling
	:type B: float, optional
	:param tau: The time constant of the adaptation dynamics
	:type tau: float, optional
	:param R0: The initial covariance matrix
	:type R0: float, optional
	:param z: The threshold for firing rate
	:type z: float, optional
	:param initial_condition: The initial condition of the system, defaults to None
	:type initial_condition: np.ndarray | None, optional
	:param run_until: The time point until which to run the simulation, defaults to None
	:type run_until: np.ndarray | None, optional

	:ivar f: The number of rows in the grid
	:vartype f: int
	:ivar g: The number of columns in the grid
	:vartype g: int
	:ivar K: The global interaction strength
	:vartype K: float
	:ivar C: The inverse time constant of the membrane potential
	:vartype C: float
	:ivar B: potential to recovery coupling
	:vartype B: float
	:ivar k: The parameter controlling the decay of the spatial kernel
	:vartype k: float
	:ivar q: The kernel size
	:vartype q: int
	:ivar ker: The spatial kernel
	:vartype ker: np.ndarray
	:ivar tau: The time constant of the adaptation dynamics
	:vartype tau: float
	:ivar use_convolve2d: Whether to use the convolution operation
	:vartype use_convolve2d: bool
	:ivar R0: The initial covariance matrix
	:vartype R0: float
	:ivar R: The covariance matrix
	:vartype R: np.ndarray
	:ivar sqrtR: The square root of the covariance matrix
	:vartype sqrtR: np.ndarray
	:ivar x0: The initial state of the system
	:vartype x0: np.ndarray
	:ivar p: The parameters of the system
	:vartype p: np.ndarray
	:ivar y: The observations generated from the system
	:vartype y: np.ndarray

	.. automethod:: __init__
	.. automethod:: system
	.. automethod:: set_control
	.. automethod:: get_control
	.. automethod:: set_threshold
	.. automethod:: get_threshold
	.. automethod:: observations
	"""
	# noinspection PyShadowingNames
	def __init__(self, ll, dT, dt, f: int = 8, g: int = 8, k: float = 0.91, K: float = 1.38, C: float = 3., B: float = 10,
				 tau: float = 4.85, R0: float = 0.2, z: float = 0.24, obs_noise: float = 0.2,
				 initial_condition: np.ndarray | None = None, run_until: np.ndarray | None = None):

		super(WilsonCowanNature, self).__init__(ll, dT, dt, n_variables=2*f*g, n_params=f*g+1, n_observations=f*g,
												initial_condition=initial_condition, run_until=run_until)
		self.f = f
		self.g = g
		self.K = K
		self.C = C
		self.B = B
		self.k = k
		self.q = 2
		self.ker = None
		self.tau = tau
		self.use_convolve2d = True
		self.R0 = R0
		self.R = R0
		self.sqrtR = None
		self.obs_noise = obs_noise
		if run_until is None:
			run_until = self.ll
		if self.use_convolve2d:
			(mesh_x, mesh_y) = np.meshgrid(np.arange(-self.q, self.q + 1), np.arange(-self.q, self.q + 1))
			self.ker = np.exp(-self.k * (mesh_x**2 + mesh_y**2))
		self.set_threshold(z)
		self.integrate_solveivp(run_until=run_until)

	def compute_sqrtR(self, from_ix, to_ix):
		self.R = self.obs_noise ** 2 * np.cov(self.x0[:(self.f * self.g), from_ix:to_ix])
		self.R += np.finfo(float).eps * np.eye(self.R.shape[0])
		self.sqrtR = sqrtm(self.R)
		self.sqrtR = np.real(self.sqrtR)
		self.sqrtR = (self.sqrtR + self.sqrtR.T) / 2.

	# noinspection PyShadowingNames
	def system(self, x, p, control=None):
		K, C, B, tau, z = self.K, self.C, self.B, self.tau, self.get_threshold()

		u = x[0:self.f * self.g].reshape((self.f, self.g))
		a = x[self.f * self.g:2 * self.f * self.g].reshape((self.f, self.g))
		# noinspection PyShadowingNames
		q = self.q

		if self.use_convolve2d:
			integ = K * convolve2d(u > z, self.ker, mode='same')
		else:
			ue = np.pad(u, ((q, q), (q, q)), mode='constant')
			integ = np.zeros((self.f, self.g))

			for i in range(-q, q + 1):
				for j in range(-q, q + 1):
					integ += K * np.exp(-self.k * (i ** 2 + j ** 2)) * (ue[i + q:i + self.f + q, j + q:j + self.g + q] > z)

		integ -= K * (u > z)
		udot = -C * u - a + integ
		if control is not None:
			udot += self.get_control().reshape((self.f, self.g))
		adot = (B * u - a) / tau
		w = np.concatenate([udot.flatten(), adot.flatten()])

		return w

	def set_threshold(self, threshold=0.):
		self.p[0, :] = threshold

	def get_threshold(self):
		return self.p[0, -1]

	def observations(self, from_ix=None, to_ix=None):
		if from_ix is None:
			from_ix = 0

		if to_ix is None:
			to_ix = self.ll

		if self.sqrtR is None:
			self.compute_sqrtR(from_ix, to_ix)
		noise = self.sqrtR @ np.random.standard_normal((self.f*self.g, to_ix-from_ix))
		self.y[:, from_ix:to_ix] = (self.x0[:(self.f*self.g), from_ix:to_ix] + noise)


class WilsonCowanModelWithParameterTracking(ukf_voss.UKFControllableModel):
	# noinspection PyShadowingNames
	def __init__(self, f=8, g=8, k=0.91, K=1.38, C=3, B=10, tau=4.85, z=0.24, Q_par: float | np.ndarray = 0.015,
				 Q_var: float | np.ndarray = 1., R: float | np.ndarray = 1., track_z=False):
		self.f = f
		self.g = g
		self.k = k
		self.K = K
		self.C = C
		self.B = B
		self.z = z
		self.tau = tau
		self.Q = 0.015
		self.q = 2
		self.use_convolve2d = True
		if self.use_convolve2d:
			(mesh_x, mesh_y) = np.meshgrid(np.arange(-self.q, self.q + 1), np.arange(-self.q, self.q + 1))
			self.ker = np.exp(-self.k * (mesh_x**2 + mesh_y**2))
		else:
			self.ker = None
		super(WilsonCowanModelWithParameterTracking, self).__init__()
		self.Q_par = Q_par  # initial value for parameter covariance
		if type(Q_var) in (int, float):
			Q_var = np.array([Q_var])
		self.Q_var = Q_var  # # initial value for variable covariance
		self.R = R  # observation covariance

		self.track_z = track_z

	def n_variables(self):
		"""
		Returns the number of variables.

		:return: The number of variables defined in the method.
		:rtype: int
		"""
		return 2*self.f*self.g

	def n_observables(self):
		"""
		Returns the number of observables.

		:return: Number of observables.
		:rtype: int
		"""
		return self.f*self.g

	def n_params(self):

		return int(self.track_z)  # TODO adapt to more parameters

	def obs_g_model(self, x):
		return x[self.n_params():(self.n_params()+self.f*self.g), :]

	# noinspection PyShadowingNames
	def f_model(self, x, p, ctrl=None):
		K, C, B, tau = self.K, self.C, self.B, self.tau
		if self.track_z:
			z = p[0]
		else:
			z = self.z

		u = x[0:self.f * self.g].reshape((self.f, self.g))
		a = x[self.f * self.g:2 * self.f * self.g].reshape((self.f, self.g))
		q = self.q

		if self.use_convolve2d:
			integ = K * convolve2d(u > z, self.ker, mode='same')
		else:
			ue = np.pad(u, ((q, q), (q, q)), mode='constant')
			integ = np.zeros((self.f, self.g))

			for i in range(-q, q + 1):
				for j in range(-q, q + 1):
					integ += K * np.exp(-self.k * (i ** 2 + j ** 2)) * (
							ue[i + q:i + self.f + q, j + q:j + self.g + q] > z)

		integ -= K * (u > z)
		udot = -C * u - a + integ
		if ctrl is not None:
			udot += ctrl.reshape((self.f, self.g))
		adot = (B * u - a) / tau

		w = np.vstack((udot.ravel()[:, np.newaxis], adot.ravel()[:, np.newaxis]))
		return w


if __name__ == '__main__':
	u0 = np.array([
		[-0.1637, -0.2244, -0.1982, -0.1410, -0.1029, -0.0811, -0.0656, -0.0396],
		[-0.2593, -0.3580, -0.3021, -0.2008, -0.1321, -0.1028, -0.0840, -0.0516],
		[-0.2444, -0.3386, -0.2746, -0.1726, -0.0950, -0.0742, -0.0637, -0.0417],
		[-0.0055, -0.0383, 0.0012, -0.0657, -0.0395, -0.0388, -0.0387, -0.0270],
		[0.2361, 0.3685, 0.2616, 0.2237, 0.0279, -0.0174, -0.0239, -0.0183],
		[0.4698, 0.7002, 0.5627, 0.2726, 0.0939, -0.0071, -0.0171, -0.0135],
		[0.3618, 0.5613, 0.4323, 0.2754, 0.0673, -0.0056, -0.0130, -0.0104],
		[0.2442, 0.2669, 0.2571, 0.1067, 0.0152, -0.0049, -0.0077, -0.0061]
	])

	a0 = np.array([
		[0.4104, 0.5339, 0.4345, 0.2786, 0.1904, 0.1454, 0.1157, 0.0694],
		[0.7058, 0.9183, 0.6974, 0.4049, 0.2427, 0.1821, 0.1465, 0.0893],
		[0.9263, 1.1846, 0.8515, 0.4104, 0.1757, 0.1283, 0.1085, 0.0707],
		[1.0750, 1.4204, 1.0146, 0.4695, 0.0921, 0.0646, 0.0636, 0.0444],
		[1.0367, 1.4612, 0.9507, 0.4993, 0.0532, 0.0309, 0.0380, 0.0293],
		[0.7878, 1.1079, 0.7914, 0.3707, 0.0468, 0.0205, 0.0267, 0.0213],
		[0.4156, 0.5573, 0.4218, 0.1886, 0.0255, 0.0150, 0.0202, 0.0162],
		[0.1386, 0.1823, 0.1406, 0.0437, 0.0083, 0.0090, 0.0119, 0.0095]
	])

	i0 = np.concatenate((u0.ravel(), a0.ravel()))
	# nature = WilsonCowanNature(ll=800, dT=0.01, dt=0.001, initial_condition=i0, run_until=400)
	#
	# f = 8
	# g = 8
	# q = 0.015
	# variance_inflate = 1.3
	# # Q_par = np.diag([q, q])
	# Q_par = np.array([])
	# # Q_var0 = np.diag((nature.R * variance_inflate, ) * 2 * f * g)
	# Q_var0 = variance_inflate * np.cov(nature.x0)
	# wilson_cowan_model = WilsonCowanModelWithParameterTracking(Q_par=Q_par, Q_var=Q_var0, R=nature.R)
	# uk_filter = ukf_voss.UKFVoss(model=wilson_cowan_model, ll=800, dT=0.01, dt=0.001)
	# x_hat0, Pxx0, Ks0, errors0 = uk_filter.filter(nature.y, initial_condition=i0)



	noise_factor = 1
	obs_noise = 0.2 * np.sqrt(noise_factor)
	# gain = 0.0003
	gain = -0.1
	variance_inflate = 2.
	baseline_run = 800
	#nature_no_ctrl = Assignment6.WilsonCowanNature(ll=4000, dT=0.01, dt=0.001, initial_condition=i0, obs_noise=obs_noise, run_until=3999)

	nature_ctrl_k = WilsonCowanNature(ll=4000, dT=0.01, dt=0.001, initial_condition=i0, run_until=baseline_run, obs_noise=obs_noise)

	Q_par = np.array([])
	#Q_var0 = np.diag((nature.R * variance_inflate, ) * 2 * f * g)
	Q_var = variance_inflate * np.cov(nature_ctrl_k.x0[:, :baseline_run])

	wilson_cowan_model = WilsonCowanModelWithParameterTracking(Q_par=Q_par, Q_var=Q_var, R=nature_ctrl_k.R)
	uk_filter = ukf_voss.UKFVoss(model=wilson_cowan_model, ll=4000, dT=0.01, dt=0.001, variance_inflation=10.)
	x_hat0, Pxx0, Ks0, errors0 = uk_filter.filter(nature_ctrl_k.y, initial_condition=i0, run_until=baseline_run,
													disable_progress=False)

	for n in tqdm.tqdm(range(baseline_run + 1, 3998)):
		nature_ctrl_k.integrate_solveivp(run_until=n)
		# control = gain*nature.y[:,n]
		control = gain * x_hat0[:64, n]
		nature_ctrl_k.set_control(control)
		wilson_cowan_model.set_control(control)
		x_hat0, Pxx0, Ks0, errors0 = uk_filter.filter(nature_ctrl_k.y, initial_condition=None, run_until=n,
													  disable_progress=True)
