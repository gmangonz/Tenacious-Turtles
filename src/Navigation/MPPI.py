import numpy as np
from MotionModel import Unicycle
import rospy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from mapping import Mapping

class MPPIRacetrack(Mapping):
    def __init__(
        self,
        static_map,
        num_rollouts_per_iteration=100,
        num_steps_per_rollout=20,
        num_iterations=20,
        lamb=10,
        motion_model=Unicycle(),
    ):
        super(MPPIRacetrack, self).__init__()
        self.num_rollouts_per_iteration = num_rollouts_per_iteration
        self.num_steps_per_rollout = num_steps_per_rollout - 1
        self.num_iterations = num_iterations
        self.lamb = lamb

        self.motion_model = motion_model
        self.action_limits = np.vstack(
            [
                np.array([0., -np.pi]), # @SAI I changed this, not sure if this helps? 
                np.array([1.,  np.pi])
            ]
        )

        self.static_map = static_map
        self.waypoint = None

        self.nominal_actions = np.zeros(
            (self.num_steps_per_rollout,)
            + (2,),
        )
        self.nominal_actions[:, 0] = 1.0

    def score_rollouts(self, rollouts:np.ndarray, actions):
        
        goal_pos = self.waypoint # [1, 2]
        rollouts = rollouts.squeeze() # [n, m, 3]

        distances = goal_pos - rollouts[:, -1, 0:2].squeeze() # [1, 2] - [50, 2] -> [50, 2]
        speed_scores = np.linalg.norm(distances, axis=1) # [50,]

        input_shape = rollouts.shape
        rollouts_xy = rollouts[:, :, 0:2]
        rollouts_xy = rollouts_xy.reshape((-1, 2))

        rollouts_in_G = super()._world_coordinates_to_map_indices(rollouts_xy)
        in_map, rollouts_in_G = super()._in_map(rollouts_in_G)

        rollouts_in_G = rollouts_in_G.reshape(input_shape) # (n, m, 3)
        in_map = in_map.reshape(input_shape[:-1])
    
        in_collision_each_pt_along_each_rollout = self.static_map[
            rollouts_in_G[:, :, 0], rollouts_in_G[:, :, 1]
        ]

        in_collision_each_pt_along_each_rollout[in_map == False] = (
            True  # noqa: E712
        )
        in_collision_each_rollout = np.sign(
            np.sum(in_collision_each_pt_along_each_rollout, axis=-1)
        )

        collision_penalty = 1000
        collision_scores = collision_penalty * in_collision_each_rollout

        scores = speed_scores + collision_scores
        # scores = speed_scores

        return scores

    def get_action(self, initial_state: np.ndarray) -> np.ndarray:
        # Given the robot's current state, select the best next action
        best_score_so_far = np.inf
        best_rollout_so_far = None

        nominal_actions = self.nominal_actions.copy()
        for iteration in range(self.num_iterations):
            delta_actions = np.random.uniform(
                low=-2.0,
                high=2.0,
                size=(
                    self.num_rollouts_per_iteration,
                    self.num_steps_per_rollout,
                )
                + (2,),
            )
            actions = np.clip(
                nominal_actions + delta_actions,
                self.action_limits[0, :],
                self.action_limits[1, :],
            )
            delta_actions = actions - nominal_actions
            states = np.empty(
                (
                    self.num_rollouts_per_iteration,
                    self.num_steps_per_rollout + 1,
                )
                + initial_state.shape
            )
            states[:, 0, :] = initial_state
            for t in range(self.num_steps_per_rollout):
                states[:, t + 1, :] = self.motion_model.step(
                    states[:, t, :], actions[:, t, :]
                )

            scores = self.score_rollouts(states, actions)

            weights = np.exp(-scores / self.lamb)
            nominal_actions += np.sum(
                np.multiply(
                    weights[:, None, None] / np.sum(weights), delta_actions
                ),
                axis=0,
            )

            if np.min(scores) < best_score_so_far:
                best_rollout_index = np.argmin(scores)
                best_rollout_actions = actions[best_rollout_index, :, :]
                best_rollout_so_far = states[best_rollout_index, :, :]
                best_score_so_far = scores[best_rollout_index]

        # Implement 1st action (in time) of best control sequence (MPC-style)
        action = best_rollout_actions[0, :]

        self.nominal_actions = np.roll(best_rollout_actions, -1, axis=0)

        return best_rollout_actions, best_score_so_far, best_rollout_so_far