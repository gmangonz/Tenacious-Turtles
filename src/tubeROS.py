import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid, Odometry
import time

import os
import sys
sys.path.append(os.path.dirname(__file__))
from Navigation.tubeMPPI import TubeMPPIRacetrack
from transformation_utils import get_matrix_pose_from_quat

class TubeMPPIROSNode:
    def __init__(self):
        rospy.init_node('tube_mppi_path_planner')

        self.path_planner = TubeMPPIRacetrack()
        self.current_pose = np.zeros(3) # x, y, theta
        rospy.on_shutdown(self.shutdown)
        self.goal_position = None

        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', 
            Twist, 
            queue_size=10
        )
        
        self.goal_subscriber = rospy.Subscriber(
            '/goal_update', 
            PoseStamped, 
            self.goal_update_callback
        )
        
        self.pose_subscriber = rospy.Subscriber(
            '/turtle_pose', 
            PoseStamped if rospy.get_param('~pose_stamped') else Pose,
            self.pose_callback
        )
        
        self.map_subscriber = rospy.Subscriber(
            '/occupancy_map', 
            OccupancyGrid, 
            self.map_callback
        )

    def goal_update_callback(self, msg):
        if rospy.get_param('~pose_stamped'):
            msg = msg.pose
        self.goal_position = np.array([msg.position.x, msg.position.y])
        self.path_planner.update_goal(self.goal_position)

    def map_callback(self, msg):
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.path_planner.update_static_map(grid)

    def pose_callback(self, msg):
        """ SLAM predicted position """
        if rospy.get_param('~pose_stamped'):
            msg = msg.pose
        self.current_pose = get_matrix_pose_from_quat(msg, return_matrix=False)
        self.current_pose = np.array([self.current_pose])

    def plan_and_execute(self):
        rospy.loginfo(f"PLANNING AND EXECUTING... TO GO TO {self.goal_position}")
        self.velocity_publisher.publish(Twist())
        control_actions, _ = self.path_planner.get_action(self.current_pose)
        for control in control_actions[:1]:
            self.publish_vel(control)
            rospy.sleep(0.1) # @SAI I think this should be dt... need to confirm dt that sunny used (I believe he said 0.1)
            rospy.loginfo(f"publishing Controls: {control}")
        
    def publish_vel(self, control):
        twist_msg = Twist()
        twist_msg.linear.x = control[0]
        twist_msg.linear.y = 0.
        twist_msg.linear.z = 0.
        twist_msg.angular.x = 0.
        twist_msg.angular.y = 0.
        twist_msg.angular.z = control[1]
        self.velocity_publisher.publish(twist_msg)
    
    def shutdown(self):
        rospy.loginfo("Stopping TurtleBot")
        self.velocity_publisher.publish(Twist())
        rospy.sleep(1)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.goal_position is not None:
                self.plan_and_execute()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = TubeMPPIROSNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo('Tube MPPI path planner node shut down.')
