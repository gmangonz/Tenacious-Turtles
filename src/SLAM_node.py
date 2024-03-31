import gtsam
import numpy as np
from dataclasses import dataclass
import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan


@dataclass
class Grid:
    grid_resolution = 0.1  # meters per grid cell
    grid_size_x = 100  # number of grid cells in the x direction
    grid_size_y = 100  # number of grid cells in the y direction
    grid_origin_x = -5.0  # origin of the grid in the x direction
    grid_origin_y = -5.0  # origin of the grid in the y direction


class Mapping(Grid):
    def __init__(self, 
                 p_free: float=0.2, 
                 p_occ: float=0.8, 
                 p_prior: float=0.5):
        
        # Initialize occupancy grid with p_prior
        self.occupancy_grid = np.zeros((
            self.grid_size_x, 
            self.grid_size_y
            ))
        
        self.log_odds_free  = self._prob_to_log_odds(p_free)
        self.log_odds_occ   = self._prob_to_log_odds(p_occ)
        self.log_odds_prior = self._prob_to_log_odds(p_prior)
    
    def _log_odds_to_prob(self, log_odds):
        return 1 - 1 / (1 + np.exp(log_odds))

    def _prob_to_log_odds(self, prob):
        return np.log(prob / (1 - prob))


class Lidar(Mapping):

    def __init__(self):

        self.subscriber = rospy.Subscriber(
            '/scan', 
            LaserScan, 
            self.update_map
        )

    def _get_free_grids_from_beam(self, obs_pt_inds, hit_pt_inds):
        diff = hit_pt_inds - obs_pt_inds
        j = np.argmax( np.abs(diff) )
        D = np.abs( diff[j] )
        return obs_pt_inds + ( np.outer( np.arange(D + 1), diff ) + (D // 2) ) // D

    def _coords_to_grid_indicies(self, x, y, w):
        grid_x = int((x - self.grid_origin_x) / self.grid_resolution)
        grid_y = int((y - self.grid_origin_y) / self.grid_resolution)
        grid_w = int(w / self.grid_resolution)
        return np.array([grid_x, grid_y, grid_w])        

    def update_map(self, data):

        ranges = np.asarray(data.ranges)
        x, y, w = get_pose_of_robot()
        obs_pt_inds = self._coords_to_grid_indicies(x, y, w)

        for i in range(len(ranges)):
            # Get angle of range
            angle = i * data.angle_increment

            # Get x,y position of laser beam in the map
            beam_angle = w + angle
            hit_x = x + np.cos(beam_angle) * ranges[i]
            hit_y = y + np.sin(beam_angle) * ranges[i]

            hit_pt_inds  = self._coords_to_grid_indicies(hit_x, hit_y, beam_angle)
            free_pt_inds = self._get_free_grids_from_beam(obs_pt_inds[:2], hit_pt_inds[:2])

            self.occupancy_grid[free_pt_inds[:, 0], free_pt_inds[:, 1]] = self.occupancy_grid[free_pt_inds[:, 0], free_pt_inds[:, 1]] + self.log_odds_free - self.log_odds_prior

        self.publish()

    def get_probability_map(self):
        return super()._log_odds_to_prob(self.occupancy_grid)

    def publish(self):
        # TODO: Setup publisher for this map
         # Conver the map to a 1D array
        self.map_update = OccupancyGrid()
        self.map_update.header.frame_id = "map_laser"
        self.map_update.header.stamp = rospy.Time.now()
        # self.map_update.info = self.map_laser.info
        # self.map_update.data = self.laser_arr.flatten('F')
        # # Convert the map to type int8
        # self.map_update.data = self.map_update.data.astype(np.int8)
        # # Publish the map
        # self.pub.publish(self.map_update)


class GTSAM:

    def __init__(self):

        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1])
        )

        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1])
        )

        # # Define occupancy grid factor noise model
        self.occupancy_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]))

        # Create a factor graph
        self.graph = gtsam.NonlinearFactorGraph()

        # Create an initial estimate for the robot's pose (e.g., based on odometry)
        self.initial_estimate = gtsam.Values()
        self.current_state_index = 0
        self.poses = []

        self._initialize_graph()

    def _initialize_graph(self):

        priorMean   = gtsam.Pose2(0.0, 0.0, 0.0)
        priorFactor = gtsam.PriorFactorPose2(None, priorMean, self.prior_noise) # <<<< None
        self.graph.add(priorFactor)
        self.initial_estimate.insert(None, priorMean) # <<<< None

## ---------------------------------------------------------------- Subscribe to april tag
    def add_tag_factors(self, msg):
        
        # TODO: Process AprilTag detections and add observation factors to the factor graph
        # Should this be from the original poses or from the most recent output of april_tag_tracker.py?
        for detection in msg.detections:
            tag_id = detection.id[0]

            if tag_id not in self.landmark_set:
                self.landmark_set.add(tag_id)
                self.initial_estimates.insert(L(tag_id), gtsam.Pose2())

            relative_pose = gtsam.Pose2(detection.pose.pose.pose.position.x,
                                  detection.pose.pose.pose.position.y,
                                  detection.pose.pose.pose.orientation.z)  # Assuming yaw angle only

            self.graph.add(BetweenFactor(X(self.pose_counter), L(tag_id), relative_pose, self.apriltag_noise))
## ----------------------------------------------------------------

    def add_grid_factors(self):
        # TODO: Create occupancy grid factors for GTSAM
        grid_factors = []
        for x in range(grid_size_x):
            for y in range(grid_size_y):
                if occupancy_grid[x, y] != 0:  # Skip free cells
                    # Create factor for occupied cell
                    factor = gtsam.RangeFactor(
                        1,  # Pose variable index
                        gtsam.Point2((grid_origin_x + x * grid_resolution), (grid_origin_y + y * grid_resolution)),
                        1.0,  # Range (distance to obstacle)
                        self.occupancy_noise
                    )
                    grid_factors.append(factor)

        # Add occupancy grid factors to the factor graph
        for factor in grid_factors:
            self.graph.add(factor)

    def add_odem_factors(self):
            # TODO: Use snippets from HW_5
            pass

    def optimize(self):

        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params)
        result = optimizer.optimize()
        marginals = gtsam.Marginals(self.graph, result)
        return result, marginals
    
    def publish_poses(self):

        tag_poses = [result.atPose2(i) for i in range(1, num_tags + 1)] # TODO: Use landmark variable, NOT i
        # TODO: Get robot pose
        pass