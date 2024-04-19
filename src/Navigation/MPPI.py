import numpy as np
from MotionModel import Unicycle 
import matplotlib as plt
#@title MPPI
class MPPIRacetrack:
    def __init__(
        self,
        static_map,
        num_rollouts_per_iteration=100,
        num_steps_per_rollout=20,
        num_iterations=20,
        lamb=10,
        motion_model=Unicycle(),
    ):
        """ Your implementation here """
        self.num_rollouts_per_iteration = num_rollouts_per_iteration
        self.num_steps_per_rollout = num_steps_per_rollout - 1
        self.num_iterations = num_iterations
        self.lamb = lamb

        self.motion_model = motion_model
        self.action_limits = np.vstack(
            [motion_model.action_space.low, motion_model.action_space.high]
        )

        self.static_map = static_map

        #set way points
        self.waypoints = np.array(
            [
                [-3.0, 0.0],
                [-2.0, 4.0],
                [4.0, 1.0],
                [-3.0, 0.0],
            ]
        )
        # self.waypoints = np.array(
        #     [
        #         [-10, 10]
        #     ]
        # )
        self.current_waypoint_index = 0

        self.cmap = plt.get_cmap("winter_r")

        self.nominal_actions = np.zeros(
            (self.num_steps_per_rollout,)
            + self.motion_model.action_space.shape,
        )
        self.nominal_actions[:, 0] = 1.0

    def score_rollouts(self, rollouts, actions):
        goal_pos = self.waypoints[
            self.current_waypoint_index % self.waypoints.shape[0]
        ]
        speed_scores = np.linalg.norm(goal_pos - rollouts[:, -1, 0:2], axis=1)

        if np.min(speed_scores) < 0.3:
            self.current_waypoint_index += 1
            # print(f'move to next waypoint: {self.current_waypoint_index}, {self.waypoints[self.current_waypoint_index]}')

        rollouts_in_G, in_map = (
            self.static_map.world_coordinates_to_map_indices(
                rollouts[:, :, 0:2]
            )
        )

        in_collision_each_pt_along_each_rollout = self.static_map.static_map[
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

    def get_action(self, initial_state: np.array) -> np.array:
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
                + self.motion_model.action_space.shape,
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

            # Select best rollout based on score
            if np.min(scores) < best_score_so_far:
                best_rollout_index = np.argmin(scores)
                best_rollout_actions = actions[best_rollout_index, :, :]
                best_rollout_so_far = states[best_rollout_index, :, :]
                best_score_so_far = scores[best_rollout_index]

            # self.plot_rollouts(states, scores, iteration, best_rollout_so_far)

        # print(f"{best_rollout_actions=}")

        # Implement 1st action (in time) of best control sequence (MPC-style)
        action = best_rollout_actions[0, :]

        self.nominal_actions = np.roll(best_rollout_actions, -1, axis=0)

        return best_rollout_actions, best_score_so_far, best_rollout_so_far