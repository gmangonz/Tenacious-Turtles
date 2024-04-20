#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Pose, Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Int32MultiArray, Float64
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import os
import sys
sys.path.append(os.path.dirname(__file__))
from transformation_utils import get_matrix_pose_from_quat, get_quat_pose, calculate_yaw
from mapping import Mapping


class GoalUpdater(Mapping):
    def __init__(self):
        super(GoalUpdater, self).__init__()
        rospy.init_node('update_goal', anonymous=True)

        self.goal_pose = None
        self.distance_threshold = 0.85
        self.count = 10
        cv2.namedWindow("Frontiers")
        
        # Publish to turtlebot control
        self.cmd_vel = rospy.Publisher(
            name='/cmd_vel', 
            data_class=Twist, 
            queue_size=10
        )

        # Publish new goal pose to TubeMPPI
        self.goal_publisher = rospy.Publisher(
            name=rospy.get_param("~goal_update"), 
            data_class=PoseStamped if rospy.get_param('~pose_stamped') else Pose, 
            queue_size=10
        )

        # Request new lidar map
        self.lidar_request = rospy.Publisher(
            name='/lidar_request', 
            data_class=Bool, 
            queue_size=10
        )
        self.set_defaults()
        self.reset_goal_bool = True # Look for new goal at startup and request new lidar

        # Subscribe to turtle pose
        rospy.Subscriber(
            name=rospy.get_param('~pose_topic'),
            data_class=PoseStamped if rospy.get_param('~pose_stamped') else Pose,
            callback=self.turtle_pose_callback,
            queue_size=10
        )

        # 
        rospy.Subscriber(
            name='/cam_pov_ranges',
            data_class=Float64,
            callback=self.lidar_cam_pov_callback,
            queue_size=10
        )

        # Subscribe to LiDAR occupancy map
        rospy.Subscriber(
            name=rospy.get_param('~occupancy_map_topic'),
            data_class=OccupancyGrid,
            callback=self.occupancy_grid_callback_lidar,
            queue_size=10
        )

        # Receive the unoccupied frontieres from lidar
        rospy.Subscriber(
            name='/unocc_frontiers',
            data_class=Int32MultiArray, 
            callback=self.unocc_frontiers_callback, 
            queue_size=10
        )

        # Recieve the occupied frontiers from lidar
        rospy.Subscriber(
            name='/occ_frontiers',
            data_class=Int32MultiArray, 
            callback=self.occ_frontiers_callback, 
            queue_size=10
        )
    
    def turtle_pose_callback(self, pose_msg):
        if rospy.get_param('~pose_stamped'):
            pose_msg = pose_msg.pose
        self.turtle_pose = get_matrix_pose_from_quat(pose_msg, return_matrix=False) # [x, y, yaw]

    def occupancy_grid_callback_lidar(self, grid_msg):
        map_array = np.array(grid_msg.data).reshape((grid_msg.info.height, grid_msg.info.width))
        rospy.loginfo(f"MAP ARRAY SHAPE: {map_array.shape}")
        self.lidar_map = cv2.normalize(map_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rospy.loginfo(f"LIDAR MAP SHAPE: {self.lidar_map.shape}")
        self.received_data_0 = True

    def unocc_frontiers_callback(self, msg):
        self.unoccupied_frontiers = np.array(msg.data).reshape((-1, 2))
        self.received_data_1 = True
    
    def occ_frontiers_callback(self, msg):
        self.occupied_frontiers = np.array(msg.data).reshape((-1, 2))
        self.received_data_2 = True

    def lidar_cam_pov_callback(self, msg):
        avg_campov_range = msg.data

        if avg_campov_range == -1.:
            return

        # If something is close to the cam and our goal distance is getting close
        if avg_campov_range < self.distance_threshold and self.goal_distance() < 0.35:
            rospy.loginfo("CLOSE TO WALL AND REACHED THE GOAL...GOAL RESET")
            self.reset_goal_bool = True

        # Outside the for loop
        if avg_campov_range < self.distance_threshold :
            self.count -= 1
            if self.count <= 1:
                rospy.loginfo("CLOSE TO WALL...GETTING STUCK...GOAL RESET")
                self.count=10
                self.reset_goal_bool = True
 
    def goal_distance(self):
        if getattr(self, 'goal_pose') is not None and hasattr(self, 'turtle_pose'):
            xc, yc, _ = self.turtle_pose
            xg, yg, _ = self.goal_pose
            distance = np.sqrt((xc - xg) ** 2 + (yc - yg) ** 2)
            return distance
        else:
            return np.inf
    
    def display_pose_and_goal(self, current_position, new_target_position, occupancy_map):
        
        rospy.loginfo(f"LIDAR MAP SHAPE: {occupancy_map.shape}")
        image1_with_positions = cv2.cvtColor(occupancy_map, cv2.COLOR_GRAY2RGB).astype(float)
        image1_with_positions = cv2.circle(image1_with_positions, (current_position[1], current_position[0]), 3, (0, 0, 255), -1)  # Red circle for current position

        for sampled_grid_ids in self.occupied_frontiers:
            image1_with_positions = cv2.circle(image1_with_positions, (sampled_grid_ids[1], sampled_grid_ids[0]), 3, (255, 0, 0), -1)  # Blue circle for all occupied frontiers
        for sampled_grid_ids in self.unoccupied_frontiers:
            image1_with_positions = cv2.circle(image1_with_positions, (sampled_grid_ids[1], sampled_grid_ids[0]), 3, (0, 165, 255), -1)  # Ornge circle for all unoccupied frontiers
        if new_target_position is not None:
            image1_with_positions = cv2.circle(image1_with_positions, (new_target_position[1], new_target_position[0]), 3, (0, 255, 0), -1)  # Green circle for final position
        
        cv2.imshow("Frontiers", image1_with_positions)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
        if key == ord('p'):
            cv2.waitKey(-1)

    def find_next_goal(self, turtle_grid_pose):

        if self.unoccupied_frontiers.size != 0:
            distances = np.linalg.norm(self.unoccupied_frontiers - turtle_grid_pose, axis=1)
            index  = np.argsort(distances)[int(len(distances) // 2)]
            goal_1 = self.unoccupied_frontiers[index]

        if self.occupied_frontiers.size != 0:
            distances = np.linalg.norm(self.occupied_frontiers - turtle_grid_pose, axis=1)
            index  = np.argsort(distances)[int(len(distances) // 2)]
            goal_2 = self.occupied_frontiers[index]
        
        if np.random.uniform() < 0.5:
            return goal_2
        return goal_1
    
    def request_lidar(self, msg:bool = True):

        lidar_request = Bool()
        lidar_request.data = msg
        self.lidar_request.publish(lidar_request)

    def set_defaults(self):
        self.request_lidar(msg=False)
        self.lidar_map = None
        self.unoccupied_frontiers = []
        self.occupied_frontiers   = []
        self.reset_goal_bool = False
        self.received_data_0 = False
        self.received_data_1 = False
        self.received_data_2 = False

    def _reset_goal(self, stop:Bool=True):
        
        if stop: self.cmd_vel.publish(Twist())
        self.request_lidar()

        turtle_pose = self.turtle_pose
        cor_x, cor_y, _  = super()._world_coordinates_to_map_indices(turtle_pose[:2])
        turtle_grid_pose = np.array([cor_x, cor_y])
        rospy.loginfo(f"Recieved a request to update the goal... current turtle pose: {turtle_pose}, grid coords: {cor_x, cor_y}")

        # Wait until LIDAR and fontiers are up to date...TODO: Other way to do this?
        rospy.loginfo(f"Waiting for LIDAR data...")
        while self.lidar_map is None and not self.received_data_1 and not self.received_data_2:
            pass
     
        rospy.loginfo(f"Received LIDAR data...")
        new_target_grid = self.find_next_goal(turtle_grid_pose)
        if new_target_grid is not None:

            # Get new Pose
            self.display_pose_and_goal(turtle_grid_pose, new_target_grid, self.lidar_map)
            goal_heading = calculate_yaw(new_target_grid[0], new_target_grid[1], *turtle_pose[:2])
            self.goal_pose = super()._grid_indices_to_coords(*new_target_grid, goal_heading, sign=-1)

            # Publish pose message
            rospy.loginfo(f"NEW TARGET POSE: {self.goal_pose}, NEW TARGET COORDS: {new_target_grid}")
            goal_pose_msg = get_quat_pose(self.goal_pose[0], self.goal_pose[1], goal_heading, stamped=rospy.get_param('~pose_stamped'))
            self.goal_publisher.publish(goal_pose_msg)
            
            # Do not reset goal
            self.set_defaults()

    def main(self):
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():

            # We have reached the goal
            if self.goal_distance() < 0.35:
                rospy.loginfo("CLOSE TO GOAL...GOAL RESET")
                self._reset_goal()

            # We have recieved an external goal reset command
            elif self.reset_goal_bool and hasattr(self, 'turtle_pose'):
                self._reset_goal()

            rate.sleep()

if __name__ == '__main__':
    try:
        goal_updater=GoalUpdater()
        goal_updater.main()
    except rospy.ROSInterruptException:
        pass
