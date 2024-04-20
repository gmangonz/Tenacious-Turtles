import numpy as np
from MotionModel import Unicycle
import rospy


#@title Iterative LQR
class AncillaryILQG:
  def __init__(self,
               waypoint = None,
               max_iter = 100,
               x0 = np.zeros(3),
               num_states = 3,
               num_actions = 2,
               motion_model = Unicycle(),
               K = 10,
               dt = 0.1,
               ):
    #time discretization
    self.Q = np.eye(3)
    self.R = np.eye(2)
    self.K = K
    self.dt = dt
    self.num_states = num_states
    self.num_actions = num_actions
    self.u = np.zeros((K, 2))
    self.max_iter = max_iter
    self.motion_model = motion_model
    self.eps_convergence = 1e-3
    self.lmbd_factor = 10.0
    self.lmbd_max = 1000.0
    self.nominal_states = None
    self.nominal_actions = None
    self.waypoint = waypoint

  def ilqg(self, x0, target):
    first_iter = True
    self.x0 = x0
    self.target = target
    u = self.nominal_actions
    lmbd = 1.0

    for i in range(self.max_iter):
      if first_iter:
        x_traj, f_x, f_u, l, l_x, l_xx, l_u, l_uu, l_ux, A, B, cost = self.init_ilqg(x0, u)
        old_cost = np.copy(cost)
        first_iter = False
      V = l[-1].copy()
      V_x = l_x[-1].copy()
      V_xx = l_xx[-1].copy()
      k = np.zeros((self.K, self.num_actions))
      K = np.zeros((self.K, self.num_actions, self.num_states))

      for t in range(self.K-1, -1, -1):
        Q_x = l_x[t] + np.dot(f_x[t].T, V_x)
        Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

        #might need to include second derivatives of motion model
        Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t]))
        Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
        Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

        #levenberg-marquardt heuristic
        Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
        Q_uu_evals[Q_uu_evals < 0] = 0.0
        Q_uu_evals += lmbd
        Q_uu_inv = np.dot(Q_uu_evecs, np.dot(np.diag(1.0 / Q_uu_evals), Q_uu_evecs.T))

        k[t] = -np.dot(Q_uu_inv, Q_u)
        K[t] = -np.dot(Q_uu_inv, Q_ux)

        V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
        V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))
        
      u_new = np.zeros((self.K, self.num_actions))
      x_new = x0.copy()
      for t in range(self.K - 1):
        # rospy.loginfo(f"SHAPES: {u[t].shape}, {k[t].shape}, {x_traj[t].shape}, {K[t].shape}, {x_new.shape}, {x_traj[t].shape}")
        u_new[t] = u[t] + k[t] + np.dot(K[t], x_new - x_traj[t])
        _, x_new = self.motion_model_step(x_new, u_new[t])

      x_traj_new, cost_new = self.simulate(x0, u_new)

      #levenberg-marquardt heuristic
      if cost_new < cost:
        lmbd /= self.lmbd_factor
        x_traj = np.copy(x_traj_new)
        u = np.copy(u_new)
        old_cost = np.copy(cost)
        cost = np.copy(cost_new)
        # cost = min(cost, 1e-9)
        first_iter = True

        if (i > 0) and ((np.abs(old_cost - cost)/cost) < self.eps_convergence):
          # print(f"Converged at iteration {i}")
          break
      else:
        lmbd *= self.lmbd_factor
        if lmbd > self.lmbd_max:
          # print(f"Lambda > Lambda_max, at iteration {i}")
          break

    return x_traj, u, cost

  def init_ilqg(self, x0, u):
    x_traj, cost = self.simulate(x0, u)
    old_cost = np.copy(cost)
    f_x = np.zeros((self.K, self.num_states, self.num_states))
    f_u = np.zeros((self.K, self.num_states, self.num_actions))
    l = np.zeros((self.K, 1))
    l_x = np.zeros((self.K, self.num_states))
    l_xx = np.zeros((self.K, self.num_states, self.num_states))
    l_u = np.zeros((self.K, self.num_actions))
    l_uu = np.zeros((self.K, self.num_actions, self.num_actions))
    l_ux = np.zeros((self.K, self.num_actions, self.num_states))
    for t in range(self.K-1):
      A, B = self.finite_differences(x_traj[t], u[t])
      f_x[t] = np.eye(self.num_states) + A * self.dt
      f_u[t] = B * self.dt

      l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t] = self.calculate_cost_with_derivatives(x_traj[t] - self.nominal_states[t], u[t] - self.nominal_actions[t])
      l[t] *= self.dt
      l_x[t] *= self.dt
      l_xx[t] *= self.dt
      l_u[t] *= self.dt
      l_uu[t] *= self.dt
      l_ux[t] *= self.dt

    l[-1], l_x[-1], l_xx[-1] = self.final_cost_with_derivative(x_traj[-1] - self.nominal_states[-1], u[-1] - self.nominal_actions[-1])

    return x_traj, f_x, f_u, l, l_x, l_xx, l_u, l_uu, l_ux, A, B, old_cost

  def finite_differences(self, x, u):
    eps = 1e-4

    A = np.zeros((self.num_states, self.num_states))
    for i in range(self.num_states):
      x_plus = x.copy()
      x_plus[i] += eps
      next_plus, _ = self.motion_model_step(x_plus, u)
      x_minus = x.copy()
      x_minus[i] -= eps
      next_minus, _ = self.motion_model_step(x_minus, u)
      A[:, i] = (next_plus - next_minus) / (2 * eps)

    B = np.zeros((self.num_states, self.num_actions))
    for i in range(self.num_actions):
      u_plus = u.copy()
      u_plus[i] += eps
      next_plus, _ = self.motion_model_step(x, u_plus)
      u_minus = u.copy()
      u_minus[i] -= eps
      next_minus, _ = self.motion_model_step(x, u_minus)
      B[:, i] = (next_plus - next_minus) / (2 * eps)

    return A, B

  def motion_model_step(self, x, u):
    x_next = self.motion_model.step(x, u)
    x_change = (x_next - x) / self.dt

    return x_change, x_next

  def calculate_cost_with_derivatives(self, x, u):
    l = np.linalg.norm(x) #+ np.linalg.norm(u)
    l_x = 2 * (x)
    l_xx = 2 * np.eye(3)
    l_u = np.zeros(self.num_actions)
    l_uu = np.zeros((self.num_actions, self.num_actions))
    l_ux = np.zeros((self.num_actions, self.num_states))

    return l, l_x, l_xx, l_u, l_uu, l_ux

  def calculate_cost_batch(self, x, u): 
    return np.sum(np.linalg.norm(x, axis = 1))

  def final_cost_with_derivative(self, x, u):
    l = 1000 * np.linalg.norm(x)
    l_x = 2000 * x
    l_xx = 2000 * np.eye(3)

    return l, l_x, l_xx

  def final_cost(self, x, u): 
    return 1000*np.linalg.norm(x)

  def simulate(self, x0, u):
    xs = np.zeros((self.K, self.num_states))
    xs[0] = x0
    for i in range(self.K-1):
      xs[i+1] = self.motion_model.step(xs[i], u[i])
    
    cost = self.calculate_cost_batch(self.nominal_states[:self.K-1] - xs[:self.K-1], None) * self.dt
    l_f = self.final_cost(self.nominal_states[self.K-1]-xs[self.K-1], u[-1] - self.nominal_actions[-1])
    cost += l_f

    return xs, cost