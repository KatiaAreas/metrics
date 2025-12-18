# utils.py


from typing import Any, Optional
import cv2
from pathlib import Path
import numpy as np
import numpy.typing as npt
import re
import json

from calib_data import CalibData
from areas_theta_compute import NotchAngleComputer, get_default_weights_path
from calibrate_pyramid_to_optitrack import complete_workflow_with_visualization
from pyramid_transformer import PyramidTransformer, extract_marker_positions_from_rb_data, plot_svd_fit_quality
from verification_script import verify_pyramid_transformation, visualize_pyramid_frame_and_points, verif_svd
from metrics import Metrics


def load_keypoints_from_json(json_path: Path) -> dict:
    """
    Load 2D keypoints from JSON file.

    Args:
        json_path: Path to the JSON file containing keypoint data

    Returns:
        Dictionary mapping frame_id to list of keypoints
        Format: {frame_id: [{id, x, y, visibility}, ...]}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    keypoints_dict = {}
    for frame in data['frames']:
        frame_id = frame['frame_id']
        keypoints_dict[frame_id] = frame['keypoints']

    return keypoints_dict


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


def parse_vectors_log(vectors_log_path: Path) -> Optional[float]:
    """
    Parse the vectors.log file to extract the initial theta angle.

    Expected format:
    "Timestamp: 2025-11-21 16:21:57, RefVector: [1.0, 0.0], Center: (955, 535),
     Clicked: (384, 632), Vector: [-571, 97], Normalized: [-0.9859, 0.1675], Angle: 170.36"

    Args:
        vectors_log_path: Path to the vectors.log file

    Returns:
        Initial theta angle in degrees, or None if parsing fails
    """
    if not vectors_log_path.exists():
        print(f"Warning: vectors.log file not found at {vectors_log_path}")
        return None

    try:
        with open(vectors_log_path, 'r') as f:
            content = f.read()

        # Extract angle using regex
        # Pattern matches "Angle: " followed by a number (integer or float)
        angle_match = re.search(r'Angle:\s*(-?\d+\.?\d*)', content)

        if angle_match:
            initial_theta = float(angle_match.group(1))
            print(f"✓ Loaded initial theta from vectors.log: {initial_theta:.2f}°")
            return initial_theta
        else:
            print("Warning: Could not find 'Angle:' field in vectors.log")
            return None

    except Exception as e:
        print(f"Error parsing vectors.log: {e}")
        return None


def draw_keypoints_on_frame(
        frame: np.ndarray,
        keypoints: list,
        color: tuple = (255, 0, 0),  # Blue color for keypoints (BGR)
        radius: int = 3,
        show_ids: bool = True
) -> int:
    """
    Draw 2D keypoints on the frame.

    Args:
        frame: Video frame to draw on
        keypoints: List of keypoint dictionaries with keys: id, x, y, visibility
        color: BGR color tuple for keypoints (default: blue)
        radius: Radius of keypoint circles
        show_ids: Whether to show keypoint IDs as text

    Returns:
        Number of visible keypoints drawn
    """
    visible_count = 0

    for kp in keypoints:
        # Check visibility (typically 0=not visible, 1=occluded, 2=visible)
        if kp.get('visibility', 2) > 0:
            x = int(round(kp['x']))
            y = int(round(kp['y']))

            # Check if point is within frame bounds
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Draw keypoint circle
                cv2.circle(frame, (x, y), radius, color, -1)

                # Draw keypoint ID
                if show_ids:
                    cv2.putText(frame, str(kp['id']), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                visible_count += 1

    return visible_count


def display_pyramid(
        video_path: Path,
        rb_data: dict[str, Any],
        calib_data,  # CalibData type
        pyramid_json_path: Path,
        keypoints_json_path: Optional[Path] = None,
        use_notch: bool = False,
        R_const_to_opt: npt.NDArray[np.float64] | None = None,
        vectors_log_path: Optional[Path] = None,
        verify_transformation: bool = True,
        compute_metrics: bool = False,
        metrics_output_path: Optional[Path] = None,
        reference_frame: str = "optitrack",
        error_unit: str = "m",
        show_per_point_errors: bool = True,
        show_frame_stats: bool = True,
        show_cumulative_stats: bool = True,
        enable_realtime_plot: bool = True,
        show_plot_in_video: bool = True,
) -> None:
    """
    Display pyramid points overlaid on video frames with optional metrics calculation.

    NEW: Can compute and display error metrics between 2D keypoints and reprojected pyramid points.
    NEW: Can show real-time cumulative error plots.

    Pipeline:
    1. Load pyramid geometry (points 0-17 from JSON)
    2. Transform from pyramid frame to OptiTrack rigid body frame
    3. For each video frame:
       a. Transform from OptiTrack rigid body frame to world frame (using Pyramid_RB pose)
       b. Transform from world frame to camera frame
       c. Project to image coordinates
       d. Apply notch rotation correction (optional)
       e. Draw pyramid points on frame
       f. Draw 2D keypoints on frame (if provided)
       g. Compute and display metrics (if enabled)
       h. Update real-time plots (if enabled)

    Args:
        video_path: Path to the video file
        rb_data: Dictionary containing rigid body tracking data
        calib_data: Camera calibration data
        pyramid_json_path: Path to pyramid geometry JSON file
        keypoints_json_path: Path to 2D keypoints JSON file (optional)
        use_notch: Whether to use notch detection for angle estimation (default: False)
        R_const_to_opt: Rotation from constellation to OptiTrack frame
        vectors_log_path: Path to vectors.log file for initial theta
        verify_transformation: Whether to run verification plots
        compute_metrics: Whether to compute error metrics (requires keypoints_json_path)
        metrics_output_path: Path to save metrics JSON file (if None, only displays)
        reference_frame: Reference frame for 3D error calculation ("optitrack" or "pyramid")
        error_unit: Unit for 3D errors ("mm" or "m")
        show_per_point_errors: Whether to show errors for individual points
        show_frame_stats: Whether to show per-frame error statistics
        show_cumulative_stats: Whether to show cumulative error statistics
        enable_realtime_plot: Whether to show real-time cumulative error plots
        show_plot_in_video: Whether to embed the plot in the video frame (requires enable_realtime_plot)
    """
    # =========================================================================
    # STEP 1: Load pyramid geometry and compute transformation
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"LOADING PYRAMID GEOMETRY")
    print("=" * 70)

    # 1. Initialize transformer
    transformer = PyramidTransformer(pyramid_json_path)

    # 2. Extract OptiTrack marker positions
    marker_positions_m, rb_position_m, rb_quaternion = extract_marker_positions_from_rb_data(
        rb_data,
        frame_id=0
    )

    # 3. Define your known matching
    matching = {
        'Marker 002': 20,
        'Marker 001': 21,
        'Marker 003': 18,
        'Marker 004': 19
    }

    # 4. Compute rotation using SVD
    R_constellation_to_optitrack = transformer.compute_optitrack_rotation_from_markers(
        marker_positions_m,
        matching
    )
    R_pyramid_to_optitrack = transformer.R_pyramid_to_optitrack

    T_pyramid_to_optitrack = transformer.T_pyramid_to_optitrack  # 4x4 with translation
    print('T_pyramid_to_optitrack = ', T_pyramid_to_optitrack)

    # Get all points and convert: world → pyramid → OptiTrack
    points_world = transformer.points_m  # All 22 points

    # World → pyramid frame
    R_pyramid = transformer.R_pyramid
    pyramid_origin = transformer.pyramid_origin_m
    points_pyramid = (R_pyramid.T @ (points_world - pyramid_origin).T).T

    # Pyramid → OptiTrack frame : Transform points from pyramid frame to OptiTrack frame
    points_optitrack = transformer.transform_pyramid_to_optitrack(points_pyramid)

    ####################### Test transforms##################################
    if verify_transformation:
        transformer.plot_constellation_frame()
        verif_svd(rb_data, pyramid_json_path)
        visualize_pyramid_frame_and_points(transformer, interactive=True, save_path=None)
        verify_pyramid_transformation(
            rb_data=rb_data,
            calib_data=calib_data,
            transformer=transformer,
            frame_id=0
        )

    ########################################

    # =========================================================================
    # STEP 1.5: Load 2D keypoints if provided
    # =========================================================================
    keypoints_dict = None
    if keypoints_json_path is not None:
        print("\n" + "=" * 70)
        print(f"LOADING 2D KEYPOINTS")
        print("=" * 70)
        try:
            keypoints_dict = load_keypoints_from_json(keypoints_json_path)
            print(f"✓ Loaded keypoints for {len(keypoints_dict)} frames")
            print(f"✓ Number of keypoints per frame: {len(keypoints_dict[list(keypoints_dict.keys())[0]])}")
        except Exception as e:
            print(f"Error loading keypoints: {e}")
            keypoints_dict = None

    # # =========================================================================
    # # Verify scale
    # #==========================================================================
    # from scale_diagnostics import diagnose_scale_issues
    #
    # diagnose_scale_issues(
    #     calib_data=calib_data,
    #     transformer=transformer,
    #     rb_data=rb_data,
    #     keypoints_dict=keypoints_dict,
    #     frame_id=0
    # )

    # =========================================================================
    # STEP 1.6: Initialize Metrics if needed
    # =========================================================================
    metrics = None
    if compute_metrics and keypoints_dict is not None:
        print("\n" + "=" * 70)
        print(f"INITIALIZING METRICS COMPUTATION")
        print("=" * 70)
        metrics = Metrics(
            calib_data=calib_data,
            transformer=transformer,
            error_unit=error_unit,
            enable_realtime_plot=enable_realtime_plot  # NEW
        )
        print(f"✓ Metrics initialized")
        print(f"  - Reference frame: {reference_frame}")
        print(f"  - 3D error unit: {error_unit}")
        if enable_realtime_plot:
            print(f"  - Real-time plotting: ENABLED")
            if show_plot_in_video:
                print(f"  - Plot embedded in video: YES")
            else:
                print(f"  - Plot in separate window: YES")
        if metrics_output_path:
            print(f"  - Will save to: {metrics_output_path}")
    elif compute_metrics and keypoints_dict is None:
        print("\n⚠ WARNING: compute_metrics=True but no keypoints provided.")
        print("           Metrics computation disabled.")

    # =========================================================================
    # STEP 2: Initialize notch detector if needed
    # =========================================================================
    notch_computer = None
    initial_theta = None

    # Load initial theta from vectors.log if provided
    if vectors_log_path is not None:
        initial_theta = parse_vectors_log(vectors_log_path)
        if initial_theta is None:
            print("Warning: Failed to load initial theta from vectors.log")
            if use_notch:
                print("         Will wait for first notch detection to set initial theta")

    if use_notch:
        print("\nInitializing notch detector...")
        notch_computer = NotchAngleComputer(
            notch_model="pose",
            circle_method="hough",
            verbose=True
        )
        notch_computer.load_models(
            notch_model_path=str(get_default_weights_path()),
            device="auto"
        )
        print("✓ Notch detector initialized")

        if initial_theta is not None:
            print(f"✓ Using initial theta from vectors.log: {initial_theta:.2f}°")
        else:
            print("⚠ No initial theta loaded - will use first detection")

    # Get camera center for rotation correction
    camera_center_x: float = calib_data.camera_model.get_center()[0]
    camera_center_y: float = calib_data.camera_model.get_center()[1]

    # =========================================================================
    # STEP 3: Open video and start playback
    # =========================================================================
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    print("\nStarting video playback. Press 'q' to quit.")
    if use_notch:
        print("Notch detection: ENABLED")
        if initial_theta is not None:
            print(f"Initial theta: {initial_theta:.2f}° (from vectors.log)")
        else:
            print("Initial theta: Will be set on first detection")
    else:
        print("Notch detection: DISABLED (theta = 0)")

    if keypoints_dict is not None:
        print("2D Keypoints: ENABLED (will be drawn in BLUE)")

    if metrics is not None:
        print("Metrics computation: ENABLED")

    # Define the starting frame ID
    start_frame_id: int = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
    frame_id: int = start_frame_id

    notch_visible = False

    # =========================================================================
    # STEP 4: Video playback loop
    # =========================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # =====================================================================
        # Compute theta (rotation angle) from notch detector
        # =====================================================================
        if use_notch:
            # Process the frame using NotchAngleComputer
            results = notch_computer.run(images=[frame], angle_unit="degrees")

            if results and len(results) > 0:
                result = results[0]

                # Set initial theta on first successful detection
                if initial_theta is None and result.visibility == 1 and result.success and result.angle is not None:
                    initial_theta = result.angle
                    print(f"Initial theta set from first detection: {initial_theta:.2f} degrees")

                # Compute relative angle
                if result.visibility == 1 and initial_theta is not None:
                    theta = initial_theta - result.angle
                    notch_visible = True
                else:
                    theta = 0.0
                    notch_visible = False

                    # Draw warning messages
                    if initial_theta is None:
                        cv2.putText(frame, "Waiting for initial notch detection",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    else:
                        cv2.putText(frame, "Cannot detect notch",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            theta = 0.0
            notch_visible = True

        # Compute rotation correction matrix
        R_cor = rotation_correction_cw(theta, camera_center_x, camera_center_y)

        # =====================================================================
        # Draw pyramid points and get reprojected coordinates
        # =====================================================================
        reprojected_points_2d = None
        should_draw = True
        if use_notch and not notch_visible:
            should_draw = False

        if should_draw:
            # Modified function that returns coordinates
            reprojected_points_2d = draw_pyramid_points_and_get_coords(
                frame,
                frame_id,
                rb_data,
                calib_data,
                points_optitrack,
                R_cor=R_cor
            )

        # =====================================================================
        # Draw 2D keypoints
        # =====================================================================
        if keypoints_dict is not None and frame_id in keypoints_dict:
            keypoints = keypoints_dict[frame_id]
            visible_kp_count = draw_keypoints_on_frame(
                frame,
                keypoints,
                color=(255, 0, 0),  # Blue for keypoints
                radius=3,
                show_ids=True
            )

            # Display keypoint info
            cv2.putText(frame, f"2D Keypoints: {visible_kp_count}/{len(keypoints)} visible",
                        (10, frame.shape[0] - (160 if metrics is not None else 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # =================================================================
            # Compute and display metrics
            # =================================================================
            if metrics is not None and reprojected_points_2d is not None:
                try:
                    # Compute errors for this frame
                    frame_errors = metrics.update_frame_errors(
                        keypoints=keypoints,
                        reprojected_points_2d=reprojected_points_2d,
                        points_pyramid_frame=points_pyramid,
                        frame_id=frame_id,
                        rb_data=rb_data,
                        reference_frame=reference_frame
                    )

                    # Draw errors on frame
                    metrics.draw_errors_on_frame(
                        frame=frame,
                        frame_errors=frame_errors,
                        keypoints=keypoints,
                        reprojected_points_2d=reprojected_points_2d,
                        show_per_point=show_per_point_errors,
                        show_frame_stats=show_frame_stats,
                        show_cumulative_stats=show_cumulative_stats
                    )

                    # Update real-time plot (NEW)
                    if enable_realtime_plot:
                        metrics.update_realtime_plot()

                        # Optionally embed plot in video frame
                        if show_plot_in_video:
                            plot_img = metrics.get_plot_as_image(
                                width=frame.shape[1] // 2,
                                height=frame.shape[0] // 2
                            )
                            if plot_img is not None:
                                # Overlay plot on top-right corner of frame
                                h, w = plot_img.shape[:2]
                                frame[0:h, frame.shape[1] - w:frame.shape[1]] = plot_img

                except Exception as e:
                    print(f"Error computing metrics for frame {frame_id}: {e}")
                    cv2.putText(frame, f"Metrics error: {str(e)[:30]}",
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display theta information
        cv2.putText(frame, f"Theta: {theta:.2f} deg / {np.deg2rad(theta):.2f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display initial theta source
        if use_notch and initial_theta is not None:
            theta_source = "vectors.log" if vectors_log_path is not None else "first detection"
            cv2.putText(frame, f"Init theta: {initial_theta:.2f}deg ({theta_source})",
                        (10, frame.shape[0] - (130 if metrics is not None else 80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Display the frame
        cv2.imshow("Pyramid Points Overlay with Metrics", frame)

        # Wait for key press (16ms ≈ 60fps)
        if cv2.waitKey(16) & 0xFF == ord('q'):
            print("Playback stopped by user.")
            break

        frame_id += 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # =========================================================================
    # Save and print metrics summary
    # =========================================================================
    if metrics is not None:
        print("\n" + "=" * 70)
        print("FINALIZING METRICS")
        print("=" * 70)

        # Print summary to console
        metrics.print_summary()

        # Save to file if path provided
        if metrics_output_path is not None:
            metrics.save_statistics(metrics_output_path)

        # Close plot window
        if enable_realtime_plot:
            print("\nClose the plot window to continue...")
            metrics.close_plot()


def draw_pyramid_points_and_get_coords(
        frame: np.ndarray,
        frame_id: int,
        rb_data: dict[str, Any],
        calib_data,  # CalibData type
        points_optitrack_m: np.ndarray,
        R_cor: np.ndarray
) -> Optional[np.ndarray]:
    """
    Draw pyramid points on the video frame AND return the 2D coordinates.

    This is a modified version of draw_pyramid_points that also returns
    the corrected 2D coordinates for metrics calculation.

    Args:
        frame: Video frame to draw on
        frame_id: Current frame index
        rb_data: Dictionary containing rigid body tracking data
        calib_data: Camera calibration data
        points_optitrack_m: Point positions in OptiTrack rigid body frame (Nx3, meters)
        R_cor: 3x3 rotation correction matrix for 2D homogeneous coordinates

    Returns:
        Nx2 array of corrected 2D coordinates, or None if rigid bodies not visible
    """
    # Check if all required rigid bodies are visible
    is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
    is_cam_visible = rb_data["Cam_RB"][frame_id].data.is_visible
    is_pyramid_visible = rb_data["Pyramid_RB"][frame_id].data.is_visible

    if not (is_cam_visible and is_lens_visible and is_pyramid_visible):
        # Draw visibility warnings
        if not is_pyramid_visible:
            cv2.putText(frame, "Pyramid not visible in OptiTrack",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not is_lens_visible:
            cv2.putText(frame, "Lens not visible in OptiTrack",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not is_cam_visible:
            cv2.putText(frame, "Camera not visible in OptiTrack",
                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None

    try:
        # Get transformation matrices
        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
        RT = np.linalg.inv(T_World_Lens @ calib_data.RT)

        # Transform points to world frame
        n_points = points_optitrack_m.shape[0]
        points_hom = np.hstack([points_optitrack_m, np.ones((n_points, 1))])  # Nx4
        points_world_hom = (T_World_Pyramid @ points_hom.T).T  # Nx4
        obj_pts = points_world_hom[:, 0:3]  # Nx3

        # Project to image coordinates
        proj_marker_2d = cv2.projectPoints(
            obj_pts,
            cv2.Rodrigues(RT[:3, :3])[0],
            RT[:3, 3],
            calib_data.K,
            calib_data.dist_coeffs
        )[0]

        # Apply rotation correction
        homog_marker_2d = np.hstack([proj_marker_2d.reshape(-1, 2), np.ones((proj_marker_2d.shape[0], 1))]).T
        homog_marker_2d_cor = R_cor @ homog_marker_2d

        # Extract 2D coordinates
        corrected_2d = homog_marker_2d_cor[:2, :].T  # Nx2

        # Draw points on frame
        if homog_marker_2d_cor is not None:
            try:
                points_in_frame = 0
                for i in range(homog_marker_2d_cor.shape[1]):
                    x, y = homog_marker_2d_cor[:2, i].flatten()
                    x_int, y_int = int(round(x)), int(round(y))

                    # Check if point is within frame bounds
                    if 0 <= x_int < frame.shape[1] and 0 <= y_int < frame.shape[0]:
                        points_in_frame += 1

                        # Draw the point (green circle)
                        cv2.circle(frame, (x_int, y_int), 5, (0, 255, 0), -1)

                        # Draw the point index number
                        cv2.putText(frame, str(i), (x_int + 8, y_int - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                # Draw info text
                cv2.putText(frame, f"Pyramid points: {points_in_frame}/{n_points} visible",
                            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error drawing pyramid points at frame {frame_id}: {e}")
                cv2.putText(frame, f"Draw error: {str(e)[:30]}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return corrected_2d

    except Exception as e:
        print(f"Error in draw_pyramid_points_and_get_coords at frame {frame_id}: {e}")
        import traceback
        traceback.print_exc()
        cv2.putText(frame, f"Error: {str(e)[:50]}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return None


def draw_pyramid_points(
        frame: np.ndarray,
        frame_id: int,
        rb_data: dict[str, Any],
        calib_data,  # CalibData type
        points_optitrack_m: np.ndarray,
        R_cor: np.ndarray
) -> None:
    """
    Draw pyramid points on the video frame.
    Uses the SAME transformation pipeline as the validated draw_marker function.

    Pipeline (following validated draw_marker):
    1. Get Pyramid_RB pose in world frame (from OptiTrack tracking)
    2. Transform points from OptiTrack rigid body frame to world frame
    3. Transform from world frame to camera frame using validated RT matrix
    4. Project to image coordinates using cv2.projectPoints
    5. Apply rotation correction (R_cor) to 2D homogeneous coordinates
    6. Draw on frame

    Args:
        frame: Video frame to draw on
        frame_id: Current frame index
        rb_data: Dictionary containing rigid body tracking data
        calib_data: Camera calibration data
        points_optitrack_m: Point positions in OptiTrack rigid body frame (Nx3, meters)
        R_cor: 3x3 rotation correction matrix for 2D homogeneous coordinates
    """
    # Check if all required rigid bodies are visible
    is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
    is_cam_visible = rb_data["Cam_RB"][frame_id].data.is_visible
    is_pyramid_visible = rb_data["Pyramid_RB"][frame_id].data.is_visible

    if not (is_cam_visible and is_lens_visible and is_pyramid_visible):
        # Draw visibility warnings
        if not is_pyramid_visible:
            cv2.putText(frame, "Pyramid not visible in OptiTrack",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not is_lens_visible:
            cv2.putText(frame, "Lens not visible in OptiTrack",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not is_cam_visible:
            cv2.putText(frame, "Camera not visible in OptiTrack",
                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    try:
        # =====================================================================
        # STEP 1: Get transformation matrices (SAME AS VALIDATED draw_marker)
        # =====================================================================
        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()

        # RT matrix: same calculation as validated draw_marker
        RT = np.linalg.inv(T_World_Lens @ calib_data.RT)

        # =====================================================================
        # STEP 2: Transform points to world frame
        # =====================================================================
        # points_optitrack_m are in Pyramid_RB local frame (meters)
        # Transform: points_world = T_World_Pyramid @ [points | 1]
        n_points = points_optitrack_m.shape[0]
        points_hom = np.hstack([points_optitrack_m, np.ones((n_points, 1))])  # Nx4
        points_world_hom = (T_World_Pyramid @ points_hom.T).T  # Nx4
        obj_pts = points_world_hom[:, 0:3]  # Nx3 - same format as draw_marker

        # =====================================================================
        # STEP 3: Project to image coordinates (SAME AS VALIDATED draw_marker)
        # =====================================================================
        proj_marker_2d = cv2.projectPoints(
            obj_pts,
            cv2.Rodrigues(RT[:3, :3])[0],
            RT[:3, 3],
            calib_data.K,
            calib_data.dist_coeffs
        )[0]

        # =====================================================================
        # STEP 4: Apply rotation correction (SAME AS VALIDATED draw_marker)
        # =====================================================================
        # Get homogeneous 2D marker positions
        homog_marker_2d = np.hstack([proj_marker_2d.reshape(-1, 2), np.ones((proj_marker_2d.shape[0], 1))]).T

        # Apply correction
        homog_marker_2d_cor = R_cor @ homog_marker_2d

        # =====================================================================
        # STEP 5: Draw points on frame (SAME AS VALIDATED draw_marker)
        # =====================================================================
        if homog_marker_2d_cor is not None:
            try:
                points_in_frame = 0
                for i in range(homog_marker_2d_cor.shape[1]):
                    x, y = homog_marker_2d_cor[:2, i].flatten()
                    x_int, y_int = int(round(x)), int(round(y))

                    # Check if point is within frame bounds
                    if 0 <= x_int < frame.shape[1] and 0 <= y_int < frame.shape[0]:
                        points_in_frame += 1

                        # Draw the point (green circle)
                        cv2.circle(frame, (x_int, y_int), 5, (0, 255, 0), -1)

                        # Draw the point index number
                        cv2.putText(frame, str(i), (x_int + 8, y_int - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                # Draw info text
                cv2.putText(frame, f"Pyramid points: {points_in_frame}/{n_points} visible",
                            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error drawing pyramid points at frame {frame_id}: {e}")
                cv2.putText(frame, f"Draw error: {str(e)[:30]}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    except Exception as e:
        print(f"Error in draw_pyramid_points at frame {frame_id}: {e}")
        import traceback
        traceback.print_exc()
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
    This is the VALIDATED function that works correctly.

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