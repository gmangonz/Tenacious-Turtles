# tube_mppi_racetrack.py

import numpy as np
import rospy
import os
import sys
sys.path.append(os.path.dirname(__file__))

from ILQG import AncillaryILQG
from MotionModel import Unicycle
from MPPI import MPPIRacetrack

class TubeMPPIRacetrack:
    def __init__(self, static_map = None, num_timesteps = 25):
      #remember to set init state for nominal controller before starting
      self.static_map = static_map
      self.nominal = MPPIRacetrack(static_map = static_map, num_steps_per_rollout=num_timesteps)
      self.ancillary = AncillaryILQG(K = num_timesteps)
      self.motion_model = Unicycle()
      self.z = None
      self.threshold = 10
      self.times = []

    def update_static_map(self, new_grid):
        self.static_map = new_grid
        self.nominal.static_map = new_grid

    def simulate(self, x0, u):
      xs = [x0]
      for i in range(len(u)):
        xs.append(self.motion_model.step(xs[i], u[i]))

      return np.array(xs)

    def solve_ancillary(self, x):
      """The ancillary controller finds the control seq which minimizes the
      difference between the uncertain state and the nominal state"""
      x_traj, u, cost = self.ancillary.ilqg(x, target = None)
      return x_traj, u, cost

    def solve_nominal(self, x):
      """ Use MPPI to solve nominal problem"""
      v, score, z_traj = self.nominal.get_action(self.z.copy())
      v_actual, score_actual, z_traj_actual = self.nominal.get_action(x.copy())

      if score_actual <= score + self.threshold:
        self.z = x
        v = v_actual
        z_traj = z_traj_actual

      z_traj = self.simulate(self.z, v)
      return z_traj, v

    def get_action(self, x0):
      x0 = x0.squeeze()
      if self.z is None:
        self.z = x0

      # solve the nominal system and get the states and controls
      z_traj, v = self.solve_nominal(x0)

      # solve ancillary problem and obtain control
      self.ancillary.nominal_states = z_traj
      self.ancillary.nominal_actions = v
      x_traj, u, cost = self.solve_ancillary(x0)

      # simulate the next state of nominal system
      z_next = self.motion_model.step(self.z.copy(), v[0])
      self.z = z_next

      return u, v
    
    def update_goal(self, new_goal):
        self.nominal.waypoint = np.array(new_goal).reshape((1, -1))
        self.ancillary.waypoint = np.array(new_goal).reshape((1, -1))
