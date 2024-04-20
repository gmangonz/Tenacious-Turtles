import cv2
import numpy as np


def draw_bbox(image, corners, center, tag_fam):

    (ptA, ptB, ptC, ptD) = corners

    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))

    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)

    (cX, cY) = (int(center[0]), int(center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
    cv2.putText(
        image, 
        tag_fam, 
        (ptA[0], ptA[1] - 15), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 255, 0), 
        2
    )
    return image


def filter_frontier(image):
    _, img_thesholded = cv2.threshold(image, thresh=180, maxval=255, type=cv2.THRESH_BINARY_INV)
    return img_thesholded


def find_next_pixel(current_position, diff_image):
    """ 
    Given the current position in the occupancy map and the 
    occupancy map corresponding to the ~camera_pov...
    """

    # In diff_image, the white pixels are the walls outside the camera's pov
    white_pixels_indices = np.argwhere(diff_image > 240)
    if len(white_pixels_indices) == 0:
        return None, None
    
    # Randomly sample n% of white pixels
    num_samples = max(1, int(len(white_pixels_indices) // 20))  # Ensure at least one sample
    sampled_indices = np.random.choice(len(white_pixels_indices), num_samples, replace=False)
    sampled_pixels = white_pixels_indices[sampled_indices].astype(np.float32)

    distances = np.linalg.norm(sampled_pixels - current_position, axis=1)
    index = np.argsort(distances)[int(len(distances) // 2)]
    
    goal_pixel = sampled_pixels[index]        
    return goal_pixel.astype(int), sampled_pixels.astype(int)  # Return as integer coordinates


def find_goal_position(current_position, lidar_map, camera_map):
    """ 
    Given the turtle's position in the occupancy map, occupancy map from the
    lidar and the occupancy from the camera's POV...
    """
    
    filtered_image1 = filter_frontier(lidar_map)
    filtered_image2 = filter_frontier(camera_map)
    diff_image = cv2.bitwise_xor(filtered_image1, filtered_image2)
    
    goal_pixel, sampled_pixels = find_next_pixel(current_position, diff_image)
    return goal_pixel, sampled_pixels


def get_frontiers(occupancy_map: np.ndarray, cam_occupancy_map: np.ndarray, current_position):
    occupancy_map = cv2.normalize(occupancy_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cam_occupancy_map = cv2.normalize(cam_occupancy_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    goal_pixel, sampled_pixels = find_goal_position(current_position[:2], occupancy_map, cam_occupancy_map)
    display_pose_and_goal(current_position, goal_pixel, occupancy_map, [], sampled_pixels)
    return goal_pixel, sampled_pixels


def display_pose_and_goal(current_position, new_target_position, occupancy_map, unoccupied_frontiers, occupied_frontiers):
    
    image1_with_positions = cv2.cvtColor(occupancy_map, cv2.COLOR_GRAY2RGB).astype(float)
    image1_with_positions = cv2.circle(image1_with_positions, (int(current_position[1]), int(current_position[0])), 3, (0, 0, 255), -1)  # Red circle for current position

    for sampled_grid_ids in occupied_frontiers:
        image1_with_positions = cv2.circle(image1_with_positions, (sampled_grid_ids[1], sampled_grid_ids[0]), 3, (255, 0, 0), -1)  # Blue circle for all occupied frontiers
    
    for sampled_grid_ids in unoccupied_frontiers:
        image1_with_positions = cv2.circle(image1_with_positions, (sampled_grid_ids[1], sampled_grid_ids[0]), 3, (0, 165, 255), -1)  # Ornge circle for all unoccupied frontiers
    
    if new_target_position is not None:
        image1_with_positions = cv2.circle(image1_with_positions, (new_target_position[1], new_target_position[0]), 3, (0, 255, 0), -1)  # Green circle for final position
    
    cv2.namedWindow("Frontiers")
    cv2.imshow("Frontiers", image1_with_positions)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
    if key == ord('p'):
        cv2.waitKey(-1)