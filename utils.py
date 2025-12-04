# utils.py

from typing import Any
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np

from calib_data import CalibData
from areas_theta_compute import NotchAngleComputer, get_default_weights_path
from pyramid_manager import PyramidManager


def rotation_correction_ccw(theta_degrees, ox, oy):
    """
    Generates a 3x3 homogeneous transformation matrix for a 2D counter-clockwise
    rotation around a specified center point.

    Args:
        theta_degrees (float): The counter-clockwise rotation angle in degrees.
        ox (float): The x-coordinate of the center of rotation (origin).
        oy (float): The y-coordinate of the center of rotation (origin).

    Returns:
        np.ndarray: A 3x3 NumPy array representing the transformation matrix.
    """
    theta_rad = np.deg2rad(theta_degrees)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, (1 - cos_theta) * ox + sin_theta * oy],
        [sin_theta, cos_theta, -sin_theta * ox + (1 - cos_theta) * oy],
        [0, 0, 1]
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
    theta_rad = np.deg2rad(theta_degrees)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    rotation_matrix = np.array([
        [cos_theta, sin_theta, (1 - cos_theta) * ox - sin_theta * oy],
        [-sin_theta, cos_theta, sin_theta * ox + (1 - cos_theta) * oy],
        [0, 0, 1]
    ])

    return rotation_matrix


def display_pyramid(
        video_path: Path,
        rb_data: dict[str, Any],
        calib_data: CalibData,
        pyramid_json_path: Path,
        use_notch: bool = True
) -> None:
    """
    Display pyramid summits overlaid on video frames.

    Transforms pyramid summit points from pyramid local frame through OptiTrack
    frame to camera image frame and displays them on the video.

    Args:
        video_path: Path to the video file
        rb_data: Dictionary containing rigid body tracking data
        calib_data: Camera calibration data
        pyramid_json_path: Path to pyramid geometry JSON file
        use_notch: Whether to use notch detection for angle estimation
    """
    # Load pyramid geometry and get summits in OptiTrack frame
    pyramid_manager = PyramidManager(pyramid_json_path)
    summits_optitrack_frame = pyramid_manager.get_summits_in_optitrack_frame()

    print(f"Loaded pyramid with {len(summits_optitrack_frame)} summit points")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get camera center for rotation correction
    camera_center_x: float = calib_data.camera_model.get_center()[0]
    camera_center_y: float = calib_data.camera_model.get_center()[1]

    # Initialize notch detector if needed
    if use_notch:
        notch_computer = NotchAngleComputer(
            notch_model="pose",
            circle_method="hough",
            verbose=True
        )
        notch_computer.load_models(
            notch_model_path=str(get_default_weights_path()),
            device="auto"
        )
        initial_theta = 170

    print("Starting video playback. Press 'q' to quit.")

    # Define the starting frame ID
    start_frame_id: int = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
    frame_id: int = start_frame_id

    # Track whether we have a valid notch detection
    notch_visible = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Estimate rotation angle using notch detector
        if use_notch:
            # Process the frame using NotchAngleComputer
            results = notch_computer.run(images=[frame], angle_unit="degrees")

            # Check visibility and extract angle from results
            if results and len(results) > 0:
                result = results[0]

                # Update initial_theta if not set and notch is visible
                if initial_theta is None and result.visibility == 1 and result.success and result.angle is not None:
                    initial_theta = result.angle
                    print(f"Initial theta set from first detection: {initial_theta:.2f} degrees")

                # Calculate theta relative to initial position
                if result.visibility == 1 and initial_theta is not None and result.success and result.angle is not None:
                    theta = initial_theta - result.angle
                    notch_visible = True
                else:
                    theta = 0.0
                    notch_visible = False

                    if initial_theta is None:
                        cv2.putText(frame, "Waiting for initial notch detection",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    else:
                        cv2.putText(frame, "Cannot detect notch - no points displayed",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                theta = 0.0
                notch_visible = False
                cv2.putText(frame, "Cannot detect notch - no points displayed",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Without notch, assume no rotation
            theta = 0.0
            notch_visible = True

        # Compute rotation correction matrix
        R_cor = rotation_correction_cw(theta, camera_center_x, camera_center_y)

        # Only draw pyramid points if notch is visible
        if notch_visible:
            draw_pyramid_summits(
                frame,
                frame_id,
                rb_data,
                calib_data,
                summits_optitrack_frame,
                R_cor
            )

        # Display the theta value in the top left corner
        cv2.putText(frame, f"Theta: {theta:.2f} deg / {np.deg2rad(theta):.2f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Pyramid Summit Overlay", frame)

        # Wait for key press (16ms ≈ 60fps)
        if cv2.waitKey(16) & 0xFF == ord('q'):
            print("Playback stopped by user.")
            break

        frame_id += 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


def draw_pyramid_summits(
        frame: np.ndarray,
        frame_id: int,
        rb_data: dict[str, Any],
        calib_data: CalibData,
        summits_optitrack: np.ndarray,
        R_cor: np.ndarray
) -> None:
    """
    Draw pyramid summit points on the video frame.

    Pipeline:
    1. Get Pyramid_RB pose in world frame (from OptiTrack tracking)
    2. Transform summits from OptiTrack rigid body frame to world frame
    3. Transform from world to camera frame
    4. Project to image coordinates
    5. Apply rotation correction
    6. Draw on frame

    Args:
        frame: Video frame to draw on
        frame_id: Current frame index
        rb_data: Dictionary containing rigid body tracking data
        calib_data: Camera calibration data
        summits_optitrack: Summit positions in OptiTrack rigid body frame (Nx3)
        R_cor: 3x3 rotation correction matrix for image plane
    """
    # Check if all required rigid bodies are visible
    is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
    is_cam_visible = rb_data["Cam_RB"][frame_id].data.is_visible
    is_pyramid_visible = rb_data["Pyramid_RB"][frame_id].data.is_visible

    if not (is_cam_visible and is_lens_visible and is_pyramid_visible):
        # Draw a message if pyramid is not visible
        if not is_pyramid_visible:
            cv2.putText(frame, "Pyramid not visible in OptiTrack",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    try:
        # Step 1: Get transformation matrices
        # T_World_Pyramid: Pyramid rigid body pose in world frame
        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()

        # T_World_Lens: Camera lens pose in world frame
        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()

        # Step 2: Transform summits from OptiTrack rigid body frame to world frame
        # summits_world = T_World_Pyramid @ summits_optitrack
        n_summits = summits_optitrack.shape[0]
        summits_hom = np.hstack([summits_optitrack, np.ones((n_summits, 1))])
        summits_world = (T_World_Pyramid @ summits_hom.T).T[:, 0:3]

        # Step 3: Transform from world to camera frame
        # RT is the extrinsic camera calibration (camera to lens rigid body)
        # T_Cam_World = inv(T_World_Lens @ RT)
        RT = np.linalg.inv(T_World_Lens @ calib_data.RT)

        # Step 4: Project summits to image coordinates
        proj_summits_2d = cv2.projectPoints(
            summits_world,
            cv2.Rodrigues(RT[:3, :3])[0],
            RT[:3, 3],
            calib_data.K,
            calib_data.dist_coeffs
        )[0]

        # Step 5: Apply rotation correction
        # Convert to homogeneous 2D coordinates
        homog_summits_2d = np.hstack([
            proj_summits_2d.reshape(-1, 2),
            np.ones((proj_summits_2d.shape[0], 1))
        ]).T

        # Apply rotation correction
        homog_summits_2d_cor = R_cor @ homog_summits_2d

        # Step 6: Draw corrected summit positions
        for i in range(homog_summits_2d_cor.shape[1]):
            x, y = homog_summits_2d_cor[:2, i].flatten()
            x_int, y_int = int(round(x)), int(round(y))

            # Check if point is within frame bounds
            if 0 <= x_int < frame.shape[1] and 0 <= y_int < frame.shape[0]:
                # Draw the summit point
                cv2.circle(frame, (x_int, y_int), 5, (0, 255, 0), -1)  # Green filled circle

                # Draw the summit index number
                cv2.putText(frame, str(i), (x_int + 8, y_int - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Draw info text
        cv2.putText(frame, f"Pyramid summits: {n_summits} points visible",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error drawing pyramid summits: {e}")
        cv2.putText(frame, f"Error: {str(e)[:50]}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def display_calib(
        video_path: Path,
        rb_data: dict[str, Any],
        calib_data: CalibData,
        use_notch: bool = False,
        pen_mode: bool = False,
        pyramid_mode: bool = False
) -> None:
    """
    Opens and displays a video file with marker overlays.

    Args:
        video_path: The absolute path to the video file.
        rb_data: Dictionary containing rigid body tracking data.
        calib_data: Camera calibration data.
        use_notch: Whether to use notch detection for angle estimation.
        pen_mode: Whether to display pen marker.
        pyramid_mode: Whether to display pyramid mode (not used in this function).
    """
    cap = cv2.VideoCapture(str(video_path))

    if use_notch:
        # Initialize the NotchAngleComputer
        notch_computer = NotchAngleComputer(
            notch_model="pose",
            circle_method="hough",
            verbose=True
        )
        notch_computer.load_models(
            notch_model_path=str(get_default_weights_path()),
            device="auto"
        )
        initial_theta = None

    camera_center_x: float = calib_data.camera_model.get_center()[0]
    camera_center_y: float = calib_data.camera_model.get_center()[1]

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    print("Starting video playback. Press 'q' to quit.")

    start_frame_id: int = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
    frame_id: int = start_frame_id

    notch_visible = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        if use_notch:
            # Process the frame using NotchAngleComputer
            results = notch_computer.run(images=[frame], angle_unit="degrees")

            if results and len(results) > 0:
                result = results[0]

                if initial_theta is None and result.visibility == 1 and result.success and result.angle is not None:
                    initial_theta = result.angle
                    print(f"Initial theta set from first detection: {initial_theta:.2f} degrees")

                if result.visibility == 1 and initial_theta is not None:
                    theta = initial_theta - result.angle
                    notch_visible = True
                else:
                    theta = 0.0
                    notch_visible = False
                    if initial_theta is None:
                        cv2.putText(frame, "Waiting for initial notch detection",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    else:
                        cv2.putText(frame, "Cannot detect notch",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            theta = 0.0
            notch_visible = True

        R_cor = rotation_correction_cw(theta, camera_center_x, camera_center_y)

        should_draw = True
        if use_notch:
            if not notch_visible:
                should_draw = False

        if should_draw:
            draw_marker(frame,
                        frame_id,
                        rb_data,
                        calib_data,
                        R_cor,
                        pen_mode=pen_mode,
                        theta_deg=theta)

        cv2.putText(frame, f"Theta: {theta:.2f} deg / {np.deg2rad(theta):.2f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Video Playback", frame)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            print("Playback stopped by user.")
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


def draw_marker(
        frame: np.ndarray,
        frame_id: int,
        rb_data: dict[str, Any],
        calib_data: CalibData,
        R_cor: np.ndarray,
        pen_mode: bool = False,
        theta_deg: float = 0.0
) -> None:
    """
    Draw calibration markers on the frame.

    Args:
        frame: Video frame to draw on.
        frame_id: Current frame index.
        rb_data: Dictionary containing rigid body tracking data.
        calib_data: Camera calibration data.
        R_cor: Rotation correction matrix.
        pen_mode: Whether to draw pen marker instead of calibration markers.
        theta_deg: Rotation angle in degrees.
    """
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

    homog_marker_2d = np.hstack([proj_marker_2d.reshape(-1, 2), np.ones((proj_marker_2d.shape[0], 1))]).T
    homog_marker_2d_cor = R_cor @ homog_marker_2d

    if homog_marker_2d_cor is not None:
        is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
        is_cam_visible = rb_data["Cam_RB"][frame_id].data.is_visible
        is_rb_visible = rb_data["Pen_RB"][frame_id].data.is_visible if pen_mode else rb_data["Calib_RB"][
            frame_id].data.is_visible

        if is_cam_visible and is_lens_visible and is_rb_visible:
            try:
                for i in range(homog_marker_2d_cor.shape[1]):
                    cv2.circle(frame, tuple(np.round(homog_marker_2d_cor[:2, i].flatten()).astype(int)), 5, (0, 0, 255),
                               -1)
            except Exception as e:
                print(f"Error drawing marker: {e}")