# utils.py

from typing import Any
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np

from calib_data import CalibData
from areas_theta_compute import NotchAngleComputer, get_default_weights_path
from ots_angle_estimator import OTSAngleEstimator

def rotation_correction_ccw(theta_degrees, ox, oy):
    """
    Generates a 3x3 homogeneous transformation matrix for a 2D clockwise
    rotation around a specified center point.

    Args:
        theta_degrees (float): The clockwise rotation angle in degrees.
        ox (float): The x-coordinate of the center of rotation (origin).
        oy (float): The y-coordinate of the center of rotation (origin).

    Returns:
        np.ndarray: A 3x3 NumPy array representing the transformation matrix.
    """
    # Convert the angle from degrees to radians for trigonometric functions
    theta_rad = np.deg2rad(theta_degrees)

    # Calculate the cosine and sine of the angle once for efficiency
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Construct the matrix
    # R = [ cos(t)  -sin(t)  (1-cos(t))ox + sin(t)oy ]
    #     [ sin(t)  cos(t)   -sin(t)ox + (1-cos(t))oy ]
    #     [   0       0                  1           ]
    rotation_matrix = np.array([
        [cos_theta,   -sin_theta,  (1 - cos_theta) * ox + sin_theta * oy],
        [sin_theta,  cos_theta,  -sin_theta * ox + (1 - cos_theta) * oy],
        [0,           0,          1]
    ])

    return rotation_matrix

def rotation_correction_cw(theta_degrees, ox, oy):
    """
    Generates a 3x3 homogeneous transformation matrix for a 2D clockwise
    rotation around a specified center point.

    Args:
        theta_degrees (float): The clockwise rotation angle in degrees.
        ox (float): The x-coordinate of the center of rotation (origin).
        oy (float): The y-coordinate of the center of rotation (origin).

    Returns:
        np.ndarray: A 3x3 NumPy array representing the transformation matrix.
    """
    # Convert the angle from degrees to radians for trigonometric functions
    theta_rad = np.deg2rad(theta_degrees)

    # Calculate the cosine and sine of the angle once for efficiency
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Construct the matrix
    # R = [ cos(t)  sin(t)  (1-cos(t))ox - sin(t)oy ]
    #     [-sin(t)  cos(t)   sin(t)ox + (1-cos(t))oy ]
    #     [   0       0                  1           ]
    rotation_matrix = np.array([
        [cos_theta,   sin_theta,  (1 - cos_theta) * ox - sin_theta * oy],
        [-sin_theta,  cos_theta,  sin_theta * ox + (1 - cos_theta) * oy],
        [0,           0,          1]
    ])

    return rotation_matrix

def display_calib(video_path: Path, rb_data: dict[str, Any], calib_data: CalibData, use_notch: bool=False, pen_mode: bool=False) -> None:
    """
    Opens and displays a video file in a window.

    Playback continues until the end of the video is reached or the user
    presses the 'q' key.

    Args:
        video_path: The absolute path to the video file.
    """
    cap = cv2.VideoCapture(str(video_path))

    if use_notch:
        notch_detector = NotchDetector()
        initial_theta: float = -7.619634951237694
        previous_theta = initial_theta
    else:
        ots_angle_estimator = OTSAngleEstimator(ref_rb_pose=calib_data.ots_ref_pose, cam_rb_data=rb_data["Cam_RB"], lens_rb_data=rb_data["Lens_RB"])
        if not ots_angle_estimator.is_axis_estimated():
            print("Warning: OTS rotation axis could not be estimated reliably. Angle estimates may be inaccurate.")

    camera_center_x: float = calib_data.camera_model.get_center()[0]
    camera_center_y: float = calib_data.camera_model.get_center()[1]
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    print("Starting video playback. Press 'q' to quit.")

    # define the starting frame ID
    start_frame_id: int = 0

    # seek the file to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
    frame_id: int = start_frame_id

    while True:

        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        
        if use_notch:
            current_theta_detection: dict[str, Any] = notch_detector.process_frame(frame)
            current_theta = current_theta_detection.get("angle", initial_theta)
            if current_theta is None:
                current_theta = previous_theta
            theta = current_theta - initial_theta
            previous_theta = current_theta
        else:
            theta = ots_angle_estimator.estimate_angle(np.linalg.inv(rb_data["Cam_RB"][frame_id].get_transform()) @ rb_data["Lens_RB"][frame_id].get_transform())


        # TODO Handle how to choose which correction matrix to apply
        
        #R_cor = rotation_correction_cw(theta, camera_center_x, camera_center_y)
        R_cor = rotation_correction_ccw(theta, camera_center_x, camera_center_y)

        draw_marker(frame,
                    frame_id,
                    rb_data,
                    calib_data,
                    use_notch,
                    ots_angle_estimator,
                    R_cor,
                    pen_mode=pen_mode)


        # Display the theta value in the top left corner of the window in degrees and radians
        cv2.putText(frame, f"Theta: {theta:.2f} deg / {np.deg2rad(theta):.2f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the frame in a window named "Video Playback"
        cv2.imshow("Video Playback", frame)

        # Wait 1ms for a key press. The 0xFF mask is a good practice on 64-bit systems.
        if cv2.waitKey(16) & 0xFF == ord('q'):
            print("Playback stopped by user.")
            break
        frame_id += 1

    
    # Clean up and close the window
    cap.release()
    cv2.destroyAllWindows()

def draw_marker(frame, frame_id, rb_data, calib_data, use_notch, ots_angle_estimator, R_cor, pen_mode=False):
    T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
    RT = np.linalg.inv(T_World_Lens @ calib_data.RT)

    if pen_mode:
        obj_pts = np.array(rb_data["Pen_RB"][frame_id].data.position).reshape(1, 3)
    else:
        obj_pts = np.vstack([np.array(value) for value in rb_data["Calib_RB"][frame_id].data.marker_positions.values()])

    proj_marker_2d = cv2.projectPoints(obj_pts, 
                                        cv2.Rodrigues(RT[:3, :3])[0],
                                        RT[:3, 3],
                                        calib_data.K,
                                        calib_data.dist_coeffs)[0]

    # Get homogeneous 2D marker positions and apply corrected rotation

    homog_marker_2d = np.hstack([proj_marker_2d.reshape(-1, 2), np.ones((proj_marker_2d.shape[0], 1))]).T

    # Apply correction
    homog_marker_2d_cor = R_cor @ homog_marker_2d

    # Draw corrected positions

    # handle case when markers are out of the image
    if homog_marker_2d_cor is not None:
        # draw only if when all rigid bodies are visible
        is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
        is_cam_visible = rb_data["Cam_RB"][frame_id].data.is_visible
        is_rb_visible = rb_data["Pen_RB"][frame_id].data.is_visible if pen_mode else rb_data["Calib_RB"][frame_id].data.is_visible
        if is_cam_visible and is_lens_visible and is_rb_visible:
            # Draw the corrected marker positions
            if homog_marker_2d_cor is not None:
                try:
                    for i in range(homog_marker_2d_cor.shape[1]):
                        cv2.circle(frame, tuple(np.round(homog_marker_2d_cor[:2, i].flatten()).astype(int)), 5, (0, 0, 255), -1)
                except Exception as e:
                    print(f"Error drawing marker: {e}")