import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, PoseStamped
import rospy


T_RC = np.array([
    [1, 0, 0, 0.032],
    [0, 1, 0, 0],
    [0, 0, 1, 0.11],
    [0, 0, 0, 1]
])
T_CR = np.array([
    [1, 0, 0, -0.032],
    [0, 1, 0, 0],
    [0, 0, 1, -0.11],
    [0, 0, 0, 1]
])

base_to_scan = np.array([
    [1, 0, 0, -0.032],
    [0, 1, 0, 0],
    [0, 0, 1, 0.172],
    [0, 0, 0, 1]])
scan_to_base = np.array([
    [1, 0, 0, 0.032],
    [0, 1, 0, 0],
    [0, 0, 1, -0.172],
    [0, 0, 0, 1]])
scan_to_camera = np.array([
    [1, 0, 0, 0.0624],
    [0, 1, 0, 0],
    [0, 0, 1, -0.062],
    [0, 0, 0, 1]])
camera_to_scan = np.array([
    [1, 0, 0, -0.064],
    [0, 1, 0, 0],
    [0, 0, 1, 0.062],
    [0, 0, 0, 1]])


def get_matrix_pose_from_quat(pose, return_matrix=True):
    t = [pose.position.x, 
         pose.position.y, 
         pose.position.z]
    
    r = R.from_quat([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])

    if not return_matrix:
        _, _, yaw = r.as_euler('xyz', degrees=False)
        return [t[0], t[1], yaw]

    r = r.as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, -1] = np.array(t)
    return T

def get_quat_pose(x, y, yaw, stamped=False):
    
    pose_msg = Pose() if not stamped else PoseStamped()

    q = R.from_euler(
        seq='xyz', 
        angles=[0., 0., yaw], 
        degrees=False
    ).as_quat()
    
    pose_msg.header.frame_id = 'pose_stamped' if stamped else 'pose'
    pose_msg.header.stamp = rospy.Time.now()
    
    if stamped:
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = 0
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
    else:
        pose_msg.position.x = x
        pose_msg.position.y = y
        pose_msg.position.z = 0
        pose_msg.orientation.x = q[0]
        pose_msg.orientation.y = q[1]
        pose_msg.orientation.z = q[2]
        pose_msg.orientation.w = q[3]
    return pose_msg



def calculate_yaw(x_target, y_target, x_source, y_source):
    # Calculate the difference in x and y coordinates
    dx = x_target - x_source
    dy = y_target - y_source
    
    # Calculate the yaw using arctan2
    yaw = np.arctan2(dy, dx)
    
    return yaw