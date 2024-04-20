#!/usr/bin/env python

# from actionlib import SimpleActionClient
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Int32MultiArray, Float64
from squirtle.msg import lidar_frontiers#, LidarRequestAction, LidarRequestGoal

import numpy as np
import cv2
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

        self.reset_goal_bool = True # Look for new goal at startup and request new lidar
        
        # Request new lidar map
        self.lidar_request = rospy.Publisher(
            name='/lidar_request', 
            data_class=Bool, 
            queue_size=10
        )

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

        # Subscribe to turtle pose
        rospy.Subscriber(
            name=rospy.get_param('~pose_topic'),
            data_class=PoseStamped if rospy.get_param('~pose_stamped') else Pose,
            callback=self.turtle_pose_callback,
            queue_size=10
        )

        # Subscribe to get distance of objects within cam pov
        rospy.Subscriber(
            name='/cam_pov_ranges',
            data_class=Float64,
            callback=self.lidar_cam_pov_callback,
            queue_size=10
        )

    def turtle_pose_callback(self, pose_msg):
        if rospy.get_param('~pose_stamped'):
            pose_msg = pose_msg.pose
        self.turtle_pose = get_matrix_pose_from_quat(pose_msg, return_matrix=False) # [x, y, yaw]

    def goal_distance(self):
        if getattr(self, 'goal_pose') is not None and hasattr(self, 'turtle_pose'):
            xc, yc, _ = self.turtle_pose
            xg, yg, _ = self.goal_pose
            distance = np.sqrt((xc - xg) ** 2 + (yc - yg) ** 2)
            return distance
        else:
            return np.inf

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

    def find_next_goal(self, turtle_grid_pose: np.ndarray, unoccupied_frontiers: np.ndarray, occupied_frontiers: np.ndarray):

        if unoccupied_frontiers.size != 0:
            distances = np.linalg.norm(unoccupied_frontiers - turtle_grid_pose, axis=1)
            index  = np.argsort(distances)[int(len(distances) // 2)]
            goal_1 = unoccupied_frontiers[index]
            if np.random.uniform() < 0.0:
                return goal_1

        if occupied_frontiers.size != 0:
            distances = np.linalg.norm(occupied_frontiers - turtle_grid_pose, axis=1)
            index  = np.argsort(distances)[int(len(distances) // 2)]
            goal_2 = occupied_frontiers[index]
            return goal_2
        
        """
        unoccupied_frontiers has data. But was not chosen
        as the goal. However, occupied_frontiers did not
        have data. Return goal_1.
        """
        return goal_1

    def _request_lidar(self):

        rospy.loginfo("[GOAL CLIENT] REQUESTING NEW LIDAR MAP...")
        msg = Bool()
        msg.data = True
        self.lidar_request.publish(msg)
        return rospy.wait_for_message(
            topic='/lidar_frontiers', 
            topic_type=lidar_frontiers
        )

    def _parse_data(self, data_msg):
        
        occupied_frontiers = np.array(
            data_msg.occupied_frontiers.data
            ).reshape((-1, 2))
        
        unoccupied_frontiers = np.array(
            data_msg.unoccupied_frontiers.data
            ).reshape((-1, 2))
        
        return unoccupied_frontiers, occupied_frontiers

    def send_request_and_decode(self):
        self.cmd_vel.publish(Twist())
        return self._parse_data(self._request_lidar())

    def reset_goal(self):
        
        # Send lidar request and wait for data
        unoccupied_frontiers, occupied_frontiers = self.send_request_and_decode()
        rospy.loginfo("[GOAL CLIENT] SUCCESSFULLY RECIEVED LIDAR MAP AND FRONTIERS...")
        turtle_pose = self.turtle_pose

        # Get turtle pose in occupancy map, get new goal
        cor_x, cor_y, _  = super()._world_coordinates_to_map_indices(turtle_pose[:2])
        turtle_grid_pose = np.array([cor_x, cor_y])
        new_target_grid  = self.find_next_goal(turtle_grid_pose, unoccupied_frontiers, occupied_frontiers)
        rospy.loginfo(f"CURRENT TURTLE POSE: {turtle_pose}, GRID COORDS: {cor_x, cor_y}")

        if new_target_grid is not None:
            # Get new Pose
            # self.display_pose_and_goal(turtle_grid_pose, new_target_grid, occupancy_map, unoccupied_frontiers, occupied_frontiers)
            goal_heading = calculate_yaw(new_target_grid[0], new_target_grid[1], *turtle_pose[:2])
            self.goal_pose = super()._grid_indices_to_coords(*new_target_grid, goal_heading, sign=-1)

            # Publish pose message
            rospy.loginfo(f"NEW TARGET POSE: {self.goal_pose}, NEW TARGET COORDS: {new_target_grid}")
            goal_pose_msg = get_quat_pose(self.goal_pose[0], self.goal_pose[1], goal_heading, stamped=rospy.get_param('~pose_stamped'))
            self.goal_publisher.publish(goal_pose_msg)

        self.reset_goal_bool = False

    def run(self):
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():

            # We have reached the goal
            if self.goal_distance() < 0.35:
                rospy.loginfo("CLOSE TO GOAL...GOAL RESET")
                self.reset_goal()

            # We have recieved an external goal reset command
            elif self.reset_goal_bool and hasattr(self, 'turtle_pose'):
                self.reset_goal()

            rate.sleep()

if __name__ == '__main__':
    try:
        goal_updater=GoalUpdater()
        goal_updater.run()
    except rospy.ROSInterruptException:
        pass
