#!/usr/bin/env python3

import gtsam
from gtsam.symbol_shorthand import X, L
import numpy as np
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
from apriltag_ros.msg import AprilTagDetectionArray
from std_msgs.msg import Bool, Int32MultiArray
import time

import os
import sys
sys.path.append(os.path.dirname(__file__))
from transformation_utils import get_quat_pose, get_matrix_pose_from_quat
from mapping import Mapping


class Lidar(Mapping):

    def __init__(self):

        super(Lidar, self).__init__()
        self._init_map()
        self.in_cam_pov = lambda angle_rad: angle_rad >= np.deg2rad(360 + 62.2 - 90) or angle_rad <= np.deg2rad(121.1 - 90)
        self.narrow_cam_pov = lambda angle_rad: angle_rad >= np.deg2rad(355) or angle_rad <= np.deg2rad(5)
        
        self.request_lidar_map  = False

        # Publish Lidar occupancy map
        self.occ_map_pub = rospy.Publisher(
            name=rospy.get_param('~occupancy_map_topic'),
            data_class=OccupancyGrid,
            queue_size=10
        )

        self.occ_frontiers_pub = rospy.Publisher(
            name='/occ_frontiers',
            data_class=Int32MultiArray, 
            queue_size=10
        )

        self.unocc_frontiers_pub = rospy.Publisher(
            name='/unocc_frontiers',
            data_class=Int32MultiArray, 
            queue_size=10
        )

        # Subscribe to the lidar messages from turtlebot's 2D lidar
        rospy.Subscriber(
            name=rospy.get_param('~lidar_topic'), 
            data_class=LaserScan, 
            callback=self.process_lidar
        )

        # Subscribe to the lidar request from update_goal node
        rospy.Subscriber(
            name='/lidar_request', 
            data_class=Bool, 
            callback=self.lidar_request
        )
 
    def _get_free_grids_from_beam(self, obs_pt_inds, hit_pt_inds) -> np.ndarray:
        diff = hit_pt_inds - obs_pt_inds
        j = np.argmax( np.abs(diff) )
        D = np.abs( diff[j] )
        return obs_pt_inds + ( np.outer( np.arange(D + 1), diff ) + (D // 2) ) // D
    
    def _update_pose(self, odom_msg):
        """ Update current pose using odometry """
        self.current_pose = get_matrix_pose_from_quat(odom_msg.pose.pose, return_matrix=False)

    def lidar_request(self, msg):
        self.request_lidar_map = msg.data # True
        if self.request_lidar_map:
            rospy.loginfo("REQUESTING NEW LIDAR MAP...")

    def process_lidar(self, lidar_msg):

        ranges  = np.asarray(lidar_msg.ranges)
        x, y, w = list(self.current_pose)
        obs_pt_inds = super()._world_coordinates_to_map_indices([x, y])
        
        narrow_pov_ranges = 0
        narrow_pov_counts = 0

        unoccupied_frontiers = []
        occupied_frontiers = []
        
        for i in range(len(ranges)):
            # Get angle of range
            angle_rad = i * lidar_msg.angle_increment
            beam_angle_rad = w + angle_rad

            send_unoccupied_frontiers = False
            if ranges[i] <= lidar_msg.range_min or ranges[i] == np.inf:
                # 10% chance to send hit_pt_inds as unoccupied frontiers
                if ranges[i] == np.inf and self.request_lidar_map:
                    if np.random.uniform() < 0.1:
                        ranges[i] = 3.5
                        send_unoccupied_frontiers = True
                else:
                    continue
            
            if self.narrow_cam_pov(angle_rad):
                narrow_pov_ranges += ranges[i]
                narrow_pov_counts += 1

            if self.request_lidar_map:
                # Get x,y position of laser beam in the map
                hit_x = x + np.cos(beam_angle_rad) * ranges[i]
                hit_y = y + np.sin(beam_angle_rad) * ranges[i]

                hit_pt_inds  = super()._world_coordinates_to_map_indices([hit_x, hit_y])
                free_pt_inds = self._get_free_grids_from_beam(obs_pt_inds[:2], hit_pt_inds[:2])

                self.occupancy_grid_logodds[hit_pt_inds[0], hit_pt_inds[1]] = self.occupancy_grid_logodds[hit_pt_inds[0], hit_pt_inds[1]] + self.log_odds_occ - self.log_odds_prior
                self.occupancy_grid_logodds[free_pt_inds[:, 0], free_pt_inds[:, 1]] = self.occupancy_grid_logodds[free_pt_inds[:, 0], free_pt_inds[:, 1]] + self.log_odds_free - self.log_odds_prior

                if send_unoccupied_frontiers:
                    unoccupied_frontiers.append(hit_pt_inds)

                if not self.in_cam_pov(angle_rad) and not send_unoccupied_frontiers:
                    # 10% chance to send hit_pt_inds as occupied frontiers
                    if np.random.uniform() < 0.1:
                        occupied_frontiers.append(hit_pt_inds)

        if narrow_pov_counts > 0:
            avg_campov_range = narrow_pov_ranges / (narrow_pov_counts + 1e-9)
            # TODO: PUBLISH THE NARROW POV DISTANCE AVG AS A NUMBER
            pass
        else:
            # TODO: PUBLISH -1 to symbolize np.inf
            pass

        if self.request_lidar_map:
            self.publish_occupancy_map(
                input_grid=super()._log_odds_to_prob(
                    log_odds=np.clip(self.occupancy_grid_logodds, a_min=-15, a_max=15)
                    ) * 100, 
                cam_pub=self.occ_map_pub
            )
            
            self.publish_frontiers(occupied_frontiers, self.occ_frontiers_pub)
            self.publish_frontiers(unoccupied_frontiers, self.unocc_frontiers_pub)

        self.request_lidar_map = False

    def _init_map(self):

        self.map_init = self._init_occupancy_map(
            input_grid=super()._log_odds_to_prob(
                log_odds=self.occupancy_grid_logodds
            ) * 100
        )

    def _init_occupancy_map(self, input_grid: np.ndarray):

        map_init = OccupancyGrid()
        map_init.info.width  = np.array(self.grid_size_x, np.uint32)
        map_init.info.height = np.array(self.grid_size_y, np.uint32)
        map_init.info.resolution = np.array(self.grid_resolution, np.float32)
        map_init.info.origin.position.x = float(self.grid_origin_x)
        map_init.info.origin.position.y = float(self.grid_origin_y)
        map_init.info.origin.position.z = 0.
        map_init.info.origin.orientation.x = 0.
        map_init.info.origin.orientation.y = 0.
        map_init.info.origin.orientation.z = 0.
        map_init.info.origin.orientation.w = 1.
        map_init.data = input_grid.flatten().astype(np.int8)
        return map_init

    def publish_occupancy_map(self, input_grid: np.ndarray, cam_pub: rospy.Publisher):

        # Conver the map to a 1D array
        map_update = OccupancyGrid()
        map_update.info = self.map_init.info
        map_update.header.frame_id = 'occupancy_grid' if cam_pub.name==rospy.get_param('~occupancy_map_topic') else 'occupancy_grid_camera'
        map_update.header.stamp = rospy.Time.now()
        map_update.data = input_grid.flatten().astype(np.int8)
        # Publish the map
        cam_pub.publish(map_update)

    def publish_frontiers(frontiers_list, frontiers_pub):

        frontiers = np.array(frontiers_list).flatten().astype(np.int32) # (n, 2) -> (n*2,)
        frontiers_msg = Int32MultiArray()
        frontiers_msg.data = frontiers
        frontiers_pub.publish(frontiers_msg)


class GTSAM(Lidar):
    """ Encapsulates functions for GTSAM. """

    def __init__(self):

        rospy.init_node('slam_node', anonymous=True)
        self.dx, self.dy, self.dw = [0., 0., 0.]

        # Subscribe to odometry
        self.odom_sub = rospy.Subscriber(
            name=rospy.get_param('~odom_topic'), 
            data_class=Odometry, 
            callback=self.update_odom
        )

        super(GTSAM, self).__init__()

        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1])
        )

        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1])
        )

        self.apriltag_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1])
        )

        # Publish predicted pose
        self.pose_pub = rospy.Publisher(
            name=rospy.get_param('~pose_topic'),
            data_class=PoseStamped if rospy.get_param('~pose_stamped') else Pose,
            queue_size=10
        )

        # Define occupancy grid factor noise model
        self.occupancy_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]))

        # Create a factor graph
        self.graph = gtsam.NonlinearFactorGraph()

        # Create an initial estimate for the robot's pose (e.g., based on odometry)
        self.initial_estimate = gtsam.Values()
        self.current_pose_idx = 0

        # Subscribe to the detected tags
        self.landmark_set = set()
        self.tag_sub = rospy.Subscriber(
            name=rospy.get_param('~tags_topic'), 
            data_class=AprilTagDetectionArray, 
            callback=self.add_tag_factors
        )

    def _initialize_graph(self):

        assert self.current_pose_idx == 0
        priorMean   = gtsam.Pose2(*self.current_pose) if hasattr(self, 'current_pose') else gtsam.Pose2(0., 0., 0.)
        priorFactor = gtsam.PriorFactorPose2(X(self.current_pose_idx), priorMean, self.prior_noise)
        self.graph.add(priorFactor)
        self.initial_estimate.insert(X(self.current_pose_idx), priorMean)

    def update_odom(self, odom_msg):
        """ Using odom message, will update current pose and dx, dy, dw """
        super()._update_pose(odom_msg)
        twist_msg = odom_msg.twist.twist
        self.dx += twist_msg.linear.x
        self.dy += twist_msg.linear.y
        self.dw += twist_msg.angular.z

    def add_tag_factors(self, apriltag_msg):
        
        added_factors = False
        for detection in apriltag_msg.detections:
            added_factors = True
            tag_id = detection.id[0]

            if tag_id not in self.landmark_set:
                self.landmark_set.add(tag_id)
                self.initial_estimate.insert(L(tag_id), gtsam.Pose2())

            relative_pose = gtsam.Pose2(detection.pose.pose.pose.position.x,
                                        detection.pose.pose.pose.position.y,
                                        detection.pose.pose.pose.orientation.z)

            self.graph.add(gtsam.BetweenFactorPose2(X(self.current_pose_idx), L(tag_id), relative_pose, self.apriltag_noise))
        rospy.loginfo(f"ADDED TAG FACTORS: {added_factors}")

    def add_odem_factors(self):

        assert self.current_pose_idx >= 1
        odometry = gtsam.Pose2(self.dx, self.dy, self.dw)

        self.initial_estimate.insert(
            X(self.current_pose_idx), 
            self.initial_estimate.atPose2( X(self.current_pose_idx-1) ).compose(odometry)
        )

        self.graph.add(
            gtsam.BetweenFactorPose2(
                X(self.current_pose_idx-1), 
                X(self.current_pose_idx), 
                odometry, 
                self.odometry_noise
            )
        )
        self.dx, self.dy, self.dw = [0., 0., 0.]

    def optimize(self):

        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params)
        result = optimizer.optimize()
        # marginals = gtsam.Marginals(self.graph, result)
        return result, None
    
    def publish_poses(self, result):

        current_pose = result.atPose2( X(self.current_pose_idx) )
        x, y, w = current_pose.x(), current_pose.y(), current_pose.theta()
        pose_msg  = get_quat_pose(*self.current_pose, stamped=rospy.get_param('~pose_stamped')) # @SAI I am cheating here... it appears SLAM is suboptimal
        self.pose_pub.publish(pose_msg)
        self.current_pose_idx += 1

    def run_SLAM(self):

        self.add_odem_factors()
        result, _ = self.optimize()
        return result
        
    def run(self):

        rate = rospy.Rate(5)  # 1 Hz
        self._initialize_graph()
        result, _ = self.optimize()
        self.publish_poses(result)
        while not rospy.is_shutdown():
            result = self.run_SLAM()
            self.publish_poses(result)
            rate.sleep()

if __name__ == '__main__':
    
    try:
        slam = GTSAM()
        slam.run()
    except rospy.ROSInterruptException: 
    	rospy.loginfo("Shutting slam_node down ...")