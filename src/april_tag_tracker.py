#!/usr/bin/env python

import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Union

import rospy
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import os
import sys
sys.path.append(os.path.dirname(__file__))
from transformation_utils import T_RC, T_CR, get_matrix_pose_from_quat
from mapping import Mapping


class AprilTagTracker(Mapping):

    def __init__(self):
        super(AprilTagTracker, self).__init__()
        rospy.init_node('april_tag_tracker', anonymous=True)

        self.T_OR = np.eye(4)
        self.T_RO = np.eye(4)

        self.unique_tags = {}
        self.xy_to_ids   = {}
        self.current_id  = 0

        # Publish tracked AprilTag poses
        self.tag_track_pub = rospy.Publisher(
            name=rospy.get_param('~track_topic'), 
            data_class=AprilTagDetectionArray, 
            queue_size=10
        )

        # Subscribe to SLAM topic and update self.T_OR and self.T_RO
        self.turtle_sub = rospy.Subscriber(
            name=rospy.get_param('~pose_topic'),
            data_class=PoseStamped if rospy.get_param('~pose_stamped') else Pose,
            callback=self.turtle_pose_update,
            queue_size=10
        )

        # Subscribe to tag_detection topic
        self.tag_sub = rospy.Subscriber(
            name=rospy.get_param('~tags_topic'),
            data_class=AprilTagDetectionArray,
            callback=self.tag_update_callback,
            queue_size=10
        )

    def turtle_pose_update(self, turtle_pose_msg: Union[Pose, PoseStamped]):
        
        if rospy.get_param('~pose_stamped'):
            turtle_pose_msg = turtle_pose_msg.pose
        # SLAM output will be T_OR. Or Robot w.r.t. to Origin
        t = [turtle_pose_msg.position.x, 
             turtle_pose_msg.position.y, 
             turtle_pose_msg.position.z]

        q = [turtle_pose_msg.orientation.x, 
             turtle_pose_msg.orientation.y, 
             turtle_pose_msg.orientation.z, 
             turtle_pose_msg.orientation.w]
        r = R.from_quat(q).as_matrix()

        self.T_OR[:3, :3] = r
        self.T_OR[:3, -1] = np.array(t)

        # Transform T_OR to T_RO
        self.T_RO[:3, :3] = r.T
        self.T_RO[:3, -1] = -(self.T_OR[:3, -1].T) @ np.array(self.T_OR[:3, :3])

    def tag_update_callback(self, tag_detections: AprilTagDetection):
        
        for tag_detection in tag_detections.detections:
            tag_pose = tag_detection.pose.pose.pose
            T_CA = get_matrix_pose_from_quat(tag_pose)
            T_OA = self.T_OR @ T_RC @ T_CA
            x, y = T_OA[:2, -1]

            occ_x, occ_y, _  = super()._world_coordinates_to_map_indices([x, y])
            occ_map_location = (occ_x, occ_y)

            self.unique_tags[occ_map_location] = T_OA
            if self.xy_to_ids.get(occ_map_location, None) is None:
                self.xy_to_ids[occ_map_location] = self.current_id
                self.current_id += 1
            # self.update_goal_pub.publish() # TODO
        
        # Publish all tracked tags and their T_CA poses
        tag_msg = AprilTagDetectionArray()
        tag_msg.header = tag_detections.header
        for unique_xy, T_OA_tag in self.unique_tags.items():
            tag = AprilTagDetection()

            T_CA = T_CR @ self.T_RO @ T_OA_tag                
            q = R.from_matrix(T_CA[:3, :3]).as_quat()
            t = T_CA[:3, -1]

            tag.id.append(self.xy_to_ids[unique_xy])
            tag.pose.pose.pose.position.x = t[0]
            tag.pose.pose.pose.position.y = t[1]
            tag.pose.pose.pose.position.z = t[2]
            tag.pose.pose.pose.orientation.x = q[0]
            tag.pose.pose.pose.orientation.y = q[1]
            tag.pose.pose.pose.orientation.z = q[2]
            tag.pose.pose.pose.orientation.w = q[3]
            tag_msg.detections.append(tag)
        self.tag_track_pub.publish(tag_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    
    try:
        tag_tracker = AprilTagTracker()
        tag_tracker.run()
    except rospy.ROSInterruptException: 
    	rospy.loginfo("Shutting april_tag_detector down ...")