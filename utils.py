"""
Enhanced utils.py with integrated real-time uncertainty analysis

This version includes:
- Real-time uncertainty plotting during video playback
- Per-frame uncertainty tracking
- Live visualization overlays
- Integrated uncertainty analysis with existing metrics
- RANSAC filtering for pyramid points (obj_pts)
- Temporal mean filtering for pyramid points over time
- Exclusion of pyramid points 18-21
"""

from typing import Any, Optional, Dict, Tuple, List
import cv2
from pathlib import Path
import numpy as np
import numpy.typing as npt
import re
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque

from calib_data import CalibData
from areas_theta_compute import NotchAngleComputer, get_default_weights_path
from pyramid_transformer import PyramidTransformer, extract_marker_positions_from_rb_data
from uncertainty_analysis import UncertaintyAnalyzer, TransformationUncertainty


class TemporalPointFilter:
    """
    Temporal mean filter for 3D points over time.
    Maintains a sliding window of point positions and computes running average.
    """

    def __init__(self, window_size: int = 5, n_points: int = 18):
        """
        Initialize temporal filter.

        Args:
            window_size: Number of frames to average over
            n_points: Number of points to track (default 18 for points 0-17)
        """
        self.window_size = window_size
        self.n_points = n_points
        # Deque for each point: stores (frame_id, position) tuples
        self.point_history = [deque(maxlen=window_size) for _ in range(n_points)]

    def update(self, points: np.ndarray, frame_id: int, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update filter with new points and return filtered positions.

        Args:
            points: Nx3 array of 3D point positions
            frame_id: Current frame ID
            valid_mask: Optional boolean mask indicating which points are valid

        Returns:
            Nx3 array of filtered point positions
        """
        if valid_mask is None:
            valid_mask = np.ones(len(points), dtype=bool)

        filtered_points = np.zeros_like(points)

        for i in range(len(points)):
            if valid_mask[i]:
                # Add new observation
                self.point_history[i].append((frame_id, points[i].copy()))

                # Compute mean of recent observations
                if len(self.point_history[i]) > 0:
                    recent_positions = np.array([pos for _, pos in self.point_history[i]])
                    filtered_points[i] = np.mean(recent_positions, axis=0)
                else:
                    filtered_points[i] = points[i]
            else:
                # Use most recent filtered value if available
                if len(self.point_history[i]) > 0:
                    filtered_points[i] = self.point_history[i][-1][1]
                else:
                    filtered_points[i] = points[i]

        return filtered_points

    def reset(self):
        """Clear all history."""
        self.point_history = [deque(maxlen=self.window_size) for _ in range(self.n_points)]


class RealTimeUncertaintyPlotter:
    """
    Real-time plotter for uncertainty metrics during video playback.
    """

    def __init__(
            self,
            max_history: int = 100,
            plot_width: int = 600,
            plot_height: int = 400
    ):
        """
        Initialize real-time uncertainty plotter.

        Args:
            max_history: Maximum number of frames to keep in history
            plot_width: Width of plot image in pixels
            plot_height: Height of plot image in pixels
        """
        self.max_history = max_history
        self.plot_width = plot_width
        self.plot_height = plot_height

        # Storage for temporal data
        self.frame_ids = deque(maxlen=max_history)
        self.stage1_errors_mm = deque(maxlen=max_history)
        self.stage2_errors_mm = deque(maxlen=max_history)
        self.stage3_errors_px = deque(maxlen=max_history)
        self.combined_3d_errors_mm = deque(maxlen=max_history)

        # Create figure for plotting
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.suptitle('Real-Time Uncertainty Analysis', fontsize=14, fontweight='bold')

        # Configure axes
        self._setup_axes()

        plt.tight_layout()

    def _setup_axes(self):
        """Configure the plot axes."""
        # Plot 1: Stage 1 (SVD) error
        self.axes[0, 0].set_title('Stage 1: SVD Fitting', fontsize=10, fontweight='bold')
        self.axes[0, 0].set_xlabel('Frame', fontsize=9)
        self.axes[0, 0].set_ylabel('3D Error (mm)', fontsize=9)
        self.axes[0, 0].grid(alpha=0.3)

        # Plot 2: Stage 2 (Tracking) error
        self.axes[0, 1].set_title('Stage 2: OptiTrack Tracking', fontsize=10, fontweight='bold')
        self.axes[0, 1].set_xlabel('Frame', fontsize=9)
        self.axes[0, 1].set_ylabel('Tracking Noise (mm)', fontsize=9)
        self.axes[0, 1].grid(alpha=0.3)

        # Plot 3: Stage 3 (Reprojection) error
        self.axes[1, 0].set_title('Stage 3: Reprojection', fontsize=10, fontweight='bold')
        self.axes[1, 0].set_xlabel('Frame', fontsize=9)
        self.axes[1, 0].set_ylabel('Error (px)', fontsize=9)
        self.axes[1, 0].grid(alpha=0.3)

        # Plot 4: Combined 3D error
        self.axes[1, 1].set_title('Combined 3D Error (RSS)', fontsize=10, fontweight='bold')
        self.axes[1, 1].set_xlabel('Frame', fontsize=9)
        self.axes[1, 1].set_ylabel('3D Error (mm)', fontsize=9)
        self.axes[1, 1].grid(alpha=0.3)

    def update(
            self,
            frame_id: int,
            stage1_error_mm: Optional[float] = None,
            stage2_error_mm: Optional[float] = None,
            stage3_error_px: Optional[float] = None
    ) -> None:
        """
        Update plots with new frame data.

        Args:
            frame_id: Current frame ID
            stage1_error_mm: Stage 1 (SVD) error in millimeters
            stage2_error_mm: Stage 2 (tracking) error in millimeters
            stage3_error_px: Stage 3 (reprojection) error in pixels
        """
        # Store data
        self.frame_ids.append(frame_id)

        if stage1_error_mm is not None:
            self.stage1_errors_mm.append(stage1_error_mm)

        if stage2_error_mm is not None:
            self.stage2_errors_mm.append(stage2_error_mm)

        if stage3_error_px is not None:
            self.stage3_errors_px.append(stage3_error_px)

        # Calculate combined 3D error (RSS)
        if stage1_error_mm is not None and stage2_error_mm is not None:
            combined = np.sqrt(stage1_error_mm ** 2 + stage2_error_mm ** 2)
            self.combined_3d_errors_mm.append(combined)

        # Update plots
        self._redraw_plots()

    def _redraw_plots(self):
        """Redraw all plots with current data."""
        frame_list = list(self.frame_ids)

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Reconfigure axes
        self._setup_axes()

        # Plot 1: Stage 1 errors
        if len(self.stage1_errors_mm) > 0:
            errors = list(self.stage1_errors_mm)
            self.axes[0, 0].plot(frame_list[:len(errors)], errors, 'b-', linewidth=2)

            # Add horizontal line for quality threshold
            self.axes[0, 0].axhline(2.0, color='orange', linestyle='--',
                                    linewidth=1, label='Good threshold')
            self.axes[0, 0].axhline(5.0, color='red', linestyle='--',
                                    linewidth=1, label='Acceptable threshold')

            # Add mean line
            mean_val = np.mean(errors)
            self.axes[0, 0].axhline(mean_val, color='green', linestyle=':',
                                    linewidth=2, label=f'Mean: {mean_val:.2f}mm')
            self.axes[0, 0].legend(fontsize=7, loc='upper right')

        # Plot 2: Stage 2 errors
        if len(self.stage2_errors_mm) > 0:
            errors = list(self.stage2_errors_mm)
            self.axes[0, 1].plot(frame_list[:len(errors)], errors, 'purple', linewidth=2)

            # Add threshold lines
            self.axes[0, 1].axhline(1.0, color='orange', linestyle='--',
                                    linewidth=1, label='Good threshold')
            self.axes[0, 1].axhline(2.0, color='red', linestyle='--',
                                    linewidth=1, label='Acceptable threshold')

            mean_val = np.mean(errors)
            self.axes[0, 1].axhline(mean_val, color='green', linestyle=':',
                                    linewidth=2, label=f'Mean: {mean_val:.2f}mm')
            self.axes[0, 1].legend(fontsize=7, loc='upper right')

        # Plot 3: Stage 3 errors
        if len(self.stage3_errors_px) > 0:
            errors = list(self.stage3_errors_px)
            self.axes[1, 0].plot(frame_list[:len(errors)], errors, 'r-', linewidth=2)

            # Add threshold lines
            self.axes[1, 0].axhline(2.0, color='orange', linestyle='--',
                                    linewidth=1, label='Good threshold')
            self.axes[1, 0].axhline(5.0, color='red', linestyle='--',
                                    linewidth=1, label='Acceptable threshold')

            mean_val = np.mean(errors)
            self.axes[1, 0].axhline(mean_val, color='green', linestyle=':',
                                    linewidth=2, label=f'Mean: {mean_val:.2f}px')
            self.axes[1, 0].legend(fontsize=7, loc='upper right')

        # Plot 4: Combined 3D errors
        if len(self.combined_3d_errors_mm) > 0:
            errors = list(self.combined_3d_errors_mm)
            self.axes[1, 1].plot(frame_list[:len(errors)], errors, 'orange', linewidth=2)

            mean_val = np.mean(errors)
            self.axes[1, 1].axhline(mean_val, color='green', linestyle=':',
                                    linewidth=2, label=f'Mean: {mean_val:.2f}mm')
            self.axes[1, 1].legend(fontsize=7, loc='upper right')

        plt.tight_layout()

    def get_plot_image(self) -> npt.NDArray[np.uint8]:
        """
        Render current plot to numpy array for overlay on video.

        Returns:
            BGR image array of the plot
        """
        # Draw the figure to canvas
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()

        # Convert to numpy array
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)

        # Convert RGBA to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Resize to desired dimensions
        image = cv2.resize(image, (self.plot_width, self.plot_height))

        return image

    def close(self):
        """Close the matplotlib figure."""
        plt.close(self.fig)


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


def is_point_inside_circle(x: float, y: float, circle_center: Tuple[float, float], circle_radius: float) -> bool:
    """
    Check if a point is inside a circle.

    Args:
        x: X coordinate of the point
        y: Y coordinate of the point
        circle_center: (cx, cy) tuple of circle center coordinates
        circle_radius: Radius of the circle

    Returns:
        True if point is inside circle, False otherwise
    """
    cx, cy = circle_center
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return distance <= circle_radius


def rotation_correction_ellipse_ccw(theta_degrees, ox, oy, rx, ry):
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
        [rx * cos_theta, -ry * sin_theta, (1 - cos_theta) * ox + sin_theta * oy],
        [rx * sin_theta, ry * cos_theta, -sin_theta * ox + (1 - cos_theta) * oy],
        [0, 0, 1]
    ])

    return rotation_matrix


def rotation_correction_ellipse_cw(theta_degrees, ox, oy, rx, ry):
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
        [rx * cos_theta, ry * sin_theta, (1 - cos_theta) * ox - sin_theta * oy],
        [-rx * sin_theta, ry * cos_theta, sin_theta * ox + (1 - cos_theta) * oy],
        [0, 0, 1]
    ])

    return rotation_matrix


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
        enable_uncertainty_analysis: bool = True,
        show_uncertainty_plot: bool = True,
        uncertainty_plot_position: str = "bottom_right",
        uncertainty_update_interval: int = 5,
        enable_ransac: bool = True,
        ransac_threshold: float = 0.01,
        enable_temporal_filter: bool = True,
        temporal_window_size: int = 5,
) -> None:
    """
    Display pyramid points overlaid on video frames with optional metrics and uncertainty analysis.

    NEW FEATURES:
    - RANSAC filtering applied to pyramid 3D points (obj_pts) before projection
    - Temporal mean filtering for pyramid points over time
    - Exclusion of pyramid points 18-21

    Pipeline:
    1. Load pyramid geometry (points 0-17 from JSON, EXCLUDING 18-21)
    2. Transform from pyramid frame to OptiTrack rigid body frame
    3. For each video frame:
       a. Detect circle (if use_notch=True)
       b. Transform from OptiTrack rigid body frame to world frame (using Pyramid_RB pose)
       c. Apply RANSAC filtering to obj_pts (optional)
       d. Apply temporal mean filtering to obj_pts (optional)
       e. Transform from world frame to camera frame
       f. Project to image coordinates
       g. Apply notch rotation correction (optional)
       h. Draw pyramid points on frame (EXCLUDING 18-21)
       i. Compute and display metrics (if enabled)
       j. Compute and display uncertainty (if enabled)
       k. Update real-time plots (if enabled)

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
        enable_uncertainty_analysis: Whether to compute real-time uncertainty analysis
        show_uncertainty_plot: Whether to show real-time uncertainty plot
        uncertainty_plot_position: Position of uncertainty plot ("top_right", "bottom_right", "top_left", "bottom_left")
        uncertainty_update_interval: Update uncertainty plot every N frames (for performance)
        enable_ransac: Whether to apply RANSAC filtering to pyramid points
        ransac_threshold: RANSAC inlier threshold in meters (default: 0.01m = 10mm)
        enable_temporal_filter: Whether to apply temporal mean filtering
        temporal_window_size: Number of frames to average over for temporal filtering
    """
    # =========================================================================
    # STEP 1: Load pyramid geometry and compute transformation
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"LOADING PYRAMID GEOMETRY (EXCLUDING POINTS 18-21)")
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

    # =========================================================================
    # EXCLUDE POINTS 18-21 from visualization
    # =========================================================================
    # Create indices for points 0-17 (excluding 18-21)
    valid_indices = list(range(0, 18))  # Points 0 to 17
    points_optitrack_filtered = points_optitrack[valid_indices]
    points_pyramid_filtered = points_pyramid[valid_indices]

    print(f"✓ Using {len(valid_indices)} pyramid points (0-17, excluding 18-21)")

    ####################### Test transforms##################################
    if verify_transformation:
        from verification_script import verif_svd, visualize_pyramid_frame_and_points, verify_pyramid_transformation
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

    # =========================================================================
    # STEP 1.6: Initialize Metrics if needed
    # =========================================================================
    metrics = None
    if compute_metrics and keypoints_dict is not None:
        from metrics import Metrics
        print("\n" + "=" * 70)
        print(f"INITIALIZING METRICS COMPUTATION")
        print("=" * 70)
        metrics = Metrics(
            calib_data=calib_data,
            transformer=transformer,
            error_unit=error_unit,
            enable_realtime_plot=enable_realtime_plot
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
    # STEP 1.7: Initialize Uncertainty Analyzer if needed
    # =========================================================================
    uncertainty_analyzer = None
    uncertainty_plotter = None

    if enable_uncertainty_analysis:
        print("\n" + "=" * 70)
        print(f"INITIALIZING UNCERTAINTY ANALYSIS")
        print("=" * 70)

        uncertainty_analyzer = UncertaintyAnalyzer(
            transformer=transformer,
            calib_data=calib_data,
            verbose=False  # Disable verbose to avoid cluttering console during video
        )

        # Perform initial analysis (frame 0) to get baseline metrics
        try:
            initial_report = uncertainty_analyzer.analyze_full_pipeline(
                rb_data=rb_data,
                marker_positions_m=marker_positions_m,
                matching=matching,
                frame_id=0,
                keypoints_2d=None,
                R_cor=None
            )
            print(f"✓ Uncertainty analyzer initialized")
            print(f"  - Stage 1 (SVD) error: {initial_report.model_to_optitrack.mean_error_3d_m * 1000:.3f} mm")
            print(
                f"  - Stage 2 (Tracking) noise: {initial_report.optitrack_to_camera.translation_error_m * 1000:.3f} mm")

            if show_uncertainty_plot:
                print(f"  - Real-time uncertainty plot: ENABLED")
                print(f"  - Plot position: {uncertainty_plot_position}")
                print(f"  - Update interval: every {uncertainty_update_interval} frames")

                uncertainty_plotter = RealTimeUncertaintyPlotter(
                    max_history=100,
                    plot_width=600,
                    plot_height=400
                )

        except Exception as e:
            print(f"⚠ Error initializing uncertainty analysis: {e}")
            enable_uncertainty_analysis = False

    # =========================================================================
    # STEP 1.8: Initialize temporal filter if needed
    # =========================================================================
    temporal_filter = None
    if enable_temporal_filter:
        print("\n" + "=" * 70)
        print(f"INITIALIZING TEMPORAL FILTER")
        print("=" * 70)
        temporal_filter = TemporalPointFilter(
            window_size=temporal_window_size,
            n_points=len(valid_indices)
        )
        print(f"✓ Temporal filter initialized with window size {temporal_window_size}")

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

    print("\nStarting video playback. Press 'q' to quit, SPACEBAR to pause/resume.")
    if use_notch:
        print("Notch detection: ENABLED")
        if initial_theta is not None:
            print(f"Initial theta: {initial_theta:.2f}° (from vectors.log)")
        else:
            print("Initial theta: Will be set on first detection")
    else:
        print("Notch detection: DISABLED (theta = 0)")

    if enable_ransac:
        print(f"RANSAC filtering: ENABLED (threshold={ransac_threshold*1000:.1f}mm)")

    if enable_temporal_filter:
        print(f"Temporal filtering: ENABLED (window={temporal_window_size} frames)")

    if keypoints_dict is not None:
        print("2D Keypoints: ENABLED (will be drawn in BLUE)")

    if metrics is not None:
        print("Metrics computation: ENABLED")

    if enable_uncertainty_analysis:
        print("Uncertainty analysis: ENABLED")

    # Define the starting frame ID
    start_frame_id: int = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
    frame_id: int = start_frame_id

    notch_visible = False
    circle_center = None
    circle_radius = None

    # Pause control
    paused = False

    # =========================================================================
    # STEP 4: Video playback loop
    # =========================================================================
    while True:
        # Only read new frame if not paused
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
        # If paused, keep displaying the same frame

        # Make a copy of the frame for processing (so we can redraw on the same frame when paused)
        display_frame = frame.copy()

        # =====================================================================
        # Compute theta (rotation angle) from notch detector
        # =====================================================================
        if use_notch:
            # Process the frame using NotchAngleComputer
            results = notch_computer.run(images=[display_frame], angle_unit="degrees")

            if results and len(results) > 0:
                result = results[0]

                # Extract circle information for filtering pyramid points
                if hasattr(result, 'circle_center') and hasattr(result, 'circle_radius'):
                    circle_center = result.circle_center
                    circle_radius = result.circle_radius
                else:
                    circle_center = None
                    circle_radius = None

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
                        cv2.putText(display_frame, "Waiting for initial notch detection",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    else:
                        cv2.putText(display_frame, "Cannot detect notch",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            theta = 0.0
            notch_visible = True
            circle_center = None
            circle_radius = None

        # Compute rotation correction matrix
        R_cor = rotation_correction_cw(theta, camera_center_x, camera_center_y)

        # =====================================================================
        # Draw pyramid points and get reprojected coordinates (EXCLUDING 18-21)
        # with RANSAC, temporal filtering, and circle filtering
        # =====================================================================
        reprojected_points_2d = None
        should_draw = True
        if use_notch and not notch_visible:
            should_draw = False

        if should_draw:
            # Modified function that returns coordinates (with filtered points)
            # Pass circle info for filtering
            reprojected_points_2d, ransac_mask = draw_pyramid_points_and_get_coords(
                display_frame,
                frame_id,
                rb_data,
                calib_data,
                points_optitrack_filtered,  # Use filtered points (0-17 only)
                R_cor=R_cor,
                enable_ransac=enable_ransac,
                ransac_threshold=ransac_threshold,
                temporal_filter=temporal_filter if enable_temporal_filter else None,
                circle_center=circle_center if use_notch else None,
                circle_radius=circle_radius if use_notch else None
            )

        # =====================================================================
        # Extract and draw 2D keypoints
        # =====================================================================
        keypoints_2d = None
        if keypoints_dict is not None and frame_id in keypoints_dict:
            keypoints = keypoints_dict[frame_id]

            # Extract keypoints as numpy array
            keypoints_2d = np.array([[kp['x'], kp['y']] for kp in keypoints if kp.get('visibility', 2) > 0])

            # Draw keypoints on frame
            for kp in keypoints:
                if kp.get('visibility', 2) > 0:
                    x, y = int(round(kp['x'])), int(round(kp['y']))
                    if 0 <= x < display_frame.shape[1] and 0 <= y < display_frame.shape[0]:
                        cv2.circle(display_frame, (x, y), 3, (255, 0, 0), -1)  # Blue
                        cv2.putText(display_frame, str(kp['id']), (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # =================================================================
            # Compute and display metrics
            # =================================================================
            if metrics is not None and reprojected_points_2d is not None:
                try:
                    # Compute errors for this frame
                    frame_errors = metrics.update_frame_errors(
                        keypoints=keypoints,
                        reprojected_points_2d=reprojected_points_2d,
                        points_pyramid_frame=points_pyramid_filtered,  # Use filtered points
                        frame_id=frame_id,
                        rb_data=rb_data,
                        reference_frame=reference_frame
                    )

                    # Draw errors on frame
                    metrics.draw_errors_on_frame(
                        frame=display_frame,
                        frame_errors=frame_errors,
                        keypoints=keypoints,
                        reprojected_points_2d=reprojected_points_2d,
                        show_per_point=show_per_point_errors,
                        show_frame_stats=show_frame_stats,
                        show_cumulative_stats=show_cumulative_stats
                    )

                    # Update real-time plot
                    if enable_realtime_plot:
                        metrics.update_realtime_plot()

                        # Optionally embed plot in video frame
                        if show_plot_in_video:
                            plot_img = metrics.get_plot_as_image(
                                width=display_frame.shape[1] // 2,
                                height=display_frame.shape[0] // 2
                            )
                            if plot_img is not None:
                                # Overlay plot on top-right corner of frame
                                h, w = plot_img.shape[:2]
                                display_frame[0:h, display_frame.shape[1] - w:display_frame.shape[1]] = plot_img

                except Exception as e:
                    print(f"Error computing metrics for frame {frame_id}: {e}")
                    cv2.putText(display_frame, f"Metrics error: {str(e)[:30]}",
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # =====================================================================
        # Compute and display uncertainty analysis
        # =====================================================================
        if enable_uncertainty_analysis and uncertainty_analyzer is not None:
            # Update uncertainty analysis every N frames (for performance)
            if frame_id % uncertainty_update_interval == 0:
                try:
                    # Run uncertainty analysis for current frame
                    report = uncertainty_analyzer.analyze_full_pipeline(
                        rb_data=rb_data,
                        marker_positions_m=marker_positions_m,
                        matching=matching,
                        frame_id=frame_id,
                        keypoints_2d=keypoints_2d,
                        R_cor=R_cor
                    )

                    # Extract uncertainty metrics
                    stage1_error_mm = None
                    stage2_error_mm = None
                    stage3_error_px = None

                    if report.model_to_optitrack is not None:
                        stage1_error_mm = report.model_to_optitrack.mean_error_3d_m * 1000

                    if report.optitrack_to_camera is not None:
                        stage2_error_mm = report.optitrack_to_camera.translation_error_m * 1000

                    if report.camera_projection is not None and report.camera_projection.mean_reprojection_error_px is not None:
                        stage3_error_px = report.camera_projection.mean_reprojection_error_px

                    # Update real-time uncertainty plot
                    if show_uncertainty_plot and uncertainty_plotter is not None:
                        uncertainty_plotter.update(
                            frame_id=frame_id,
                            stage1_error_mm=stage1_error_mm,
                            stage2_error_mm=stage2_error_mm,
                            stage3_error_px=stage3_error_px
                        )

                        # Get plot image and overlay on frame
                        uncertainty_plot_img = uncertainty_plotter.get_plot_image()

                        # Determine plot position
                        plot_h, plot_w = uncertainty_plot_img.shape[:2]
                        frame_h, frame_w = display_frame.shape[:2]

                        if uncertainty_plot_position == "top_right":
                            y1, y2 = 0, plot_h
                            x1, x2 = frame_w - plot_w, frame_w
                        elif uncertainty_plot_position == "top_left":
                            y1, y2 = 0, plot_h
                            x1, x2 = 0, plot_w
                        elif uncertainty_plot_position == "bottom_left":
                            y1, y2 = frame_h - plot_h, frame_h
                            x1, x2 = 0, plot_w
                        else:  # bottom_right (default)
                            y1, y2 = frame_h - plot_h, frame_h
                            x1, x2 = frame_w - plot_w, frame_w

                        # Ensure plot fits in frame
                        if y2 <= frame_h and x2 <= frame_w and y1 >= 0 and x1 >= 0:
                            display_frame[y1:y2, x1:x2] = uncertainty_plot_img

                    # Display uncertainty text overlay
                    if stage1_error_mm is not None or stage2_error_mm is not None or stage3_error_px is not None:
                        y_offset = 100
                        if stage1_error_mm is not None:
                            quality_1 = "✓" if stage1_error_mm < 2 else "⚠" if stage1_error_mm < 5 else "✗"
                            cv2.putText(display_frame, f"{quality_1} SVD: {stage1_error_mm:.2f}mm",
                                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y_offset += 20

                        if stage2_error_mm is not None:
                            quality_2 = "✓" if stage2_error_mm < 1 else "⚠" if stage2_error_mm < 2 else "✗"
                            cv2.putText(display_frame, f"{quality_2} Track: {stage2_error_mm:.2f}mm",
                                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y_offset += 20

                        if stage3_error_px is not None:
                            quality_3 = "✓" if stage3_error_px < 2 else "⚠" if stage3_error_px < 5 else "✗"
                            cv2.putText(display_frame, f"{quality_3} Reproj: {stage3_error_px:.2f}px",
                                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                except Exception as e:
                    print(f"Error computing uncertainty for frame {frame_id}: {e}")
                    cv2.putText(display_frame, f"Uncertainty error: {str(e)[:30]}",
                                (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display theta information
        cv2.putText(display_frame, f"Theta: {theta:.2f} deg / {np.deg2rad(theta):.2f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display initial theta source
        if use_notch and initial_theta is not None:
            theta_source = "vectors.log" if vectors_log_path is not None else "first detection"
            cv2.putText(display_frame, f"Init theta: {initial_theta:.2f}deg ({theta_source})",
                        (10, display_frame.shape[0] - (130 if metrics is not None else 80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Display pause status
        if paused:
            cv2.putText(display_frame, "PAUSED - Press SPACEBAR to resume",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display the frame
        window_title = "Pyramid Points Overlay (Points 0-17)"
        if metrics is not None:
            window_title += " + Metrics"
        if enable_uncertainty_analysis:
            window_title += " + Uncertainty"

        cv2.imshow(window_title, display_frame)

        # Wait for key press (16ms ≈ 60fps)
        key = cv2.waitKey(16) & 0xFF

        # Handle key presses
        if key == ord('q'):
            print("Playback stopped by user.")
            break
        elif key == ord(' '):  # Spacebar
            paused = not paused
            if paused:
                print(f"Video paused at frame {frame_id}")
            else:
                print(f"Video resumed from frame {frame_id}")

        # Only increment frame_id if not paused
        if not paused:
            frame_id += 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    if uncertainty_plotter is not None:
        uncertainty_plotter.close()

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
        R_cor: np.ndarray,
        enable_ransac: bool = True,
        ransac_threshold: float = 0.01,
        temporal_filter: Optional[TemporalPointFilter] = None,
        circle_center: Optional[Tuple[float, float]] = None,
        circle_radius: Optional[float] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Draw pyramid points on the video frame AND return the 2D coordinates.
    Applies RANSAC filtering and temporal averaging to obj_pts.
    Only draws points inside the detected circle (if circle_center and circle_radius provided).

    NOTE: This function now expects points_optitrack_m to be pre-filtered (points 0-17 only).

    Args:
        frame: Video frame to draw on
        frame_id: Current frame index
        rb_data: Dictionary containing rigid body tracking data
        calib_data: Camera calibration data
        points_optitrack_m: Point positions in OptiTrack rigid body frame (Nx3, meters)
                           Should be pre-filtered to exclude points 18-21
        R_cor: 3x3 rotation correction matrix for 2D homogeneous coordinates
        enable_ransac: Whether to apply RANSAC filtering
        ransac_threshold: RANSAC inlier threshold in meters
        temporal_filter: Optional temporal filter instance
        circle_center: (cx, cy) tuple of circle center for filtering, or None
        circle_radius: Radius of circle for filtering, or None

    Returns:
        Tuple of (Nx2 array of corrected 2D coordinates, boolean mask of inliers),
        or (None, None) if rigid bodies not visible
    """
    # Check if all required rigid bodies are visible
    is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
    is_pyramid_visible = rb_data["Pyramid_RB"][frame_id].data.is_visible

    if not (is_lens_visible and is_pyramid_visible):
        # Draw visibility warnings
        if not is_pyramid_visible:
            cv2.putText(frame, "Pyramid not visible in OptiTrack",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not is_lens_visible:
            cv2.putText(frame, "Lens not visible in OptiTrack",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None

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

        # ########## CHECK VISIBILITY KEYPOINTS ##############
        #
        # # 1. Transform to camera frame
        # points_camera_hom = (RT @ np.hstack([obj_pts, np.ones((n_points, 1))]).T).T
        # points_camera = points_camera_hom[:, :3]
        #
        # # 2. Check if points are in front of camera (positive Z)
        # in_front_mask = points_camera[:, 2] > 0.01  # At least 1cm in front
        #
        # # 3. Project to get pixel coordinates (without distortion first)
        # points_normalized = points_camera[:, :2] / points_camera[:, 2:3]
        # points_pixel = (calib_data.K[:2, :2] @ points_normalized.T).T + calib_data.K[:2, 2]
        #
        # # 4. Check if points are within frame bounds (with margin)
        # margin = 50  # pixels
        # h, w = frame.shape[:2]
        # in_frame_mask = (
        #         (points_pixel[:, 0] >= -margin) &
        #         (points_pixel[:, 0] < w + margin) &
        #         (points_pixel[:, 1] >= -margin) &
        #         (points_pixel[:, 1] < h + margin)
        # )
        #
        # # 5. Combined visibility mask
        # visibility_mask = in_front_mask & in_frame_mask
        #
        # # 6. If too few points are visible, return None
        # if np.sum(visibility_mask) < 4:
        #     cv2.putText(frame, "Insufficient visible points",
        #                 (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        #     return None, None
        #
        # # 7. Filter points to only visible ones
        # obj_pts = obj_pts[visibility_mask]
        # n_points = len(obj_pts)

        # =====================================================================
        # RANSAC FILTERING (applied to obj_pts in 3D space)
        # =====================================================================
        ransac_mask = np.ones(n_points, dtype=bool)

        if enable_ransac and n_points >= 4:  # Need at least 4 points for RANSAC
            try:
                # Compute centroid
                centroid = np.mean(obj_pts, axis=0)

                # Compute distances from centroid
                distances = np.linalg.norm(obj_pts - centroid, axis=1)

                # Use RANSAC-like approach: find consensus set
                median_dist = np.median(distances)
                mad = np.median(np.abs(distances - median_dist))  # Median Absolute Deviation

                # Inliers are points within threshold of median
                # Using MAD-based threshold for robustness
                threshold_dist = ransac_threshold + 3 * mad  # Adaptive threshold
                ransac_mask = distances < threshold_dist

                n_inliers = np.sum(ransac_mask)
                n_outliers = n_points - n_inliers

                if n_outliers > 0:
                    # Display RANSAC info
                    cv2.putText(frame, f"RANSAC: {n_inliers}/{n_points} inliers ({n_outliers} outliers)",
                                (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

            except Exception as e:
                print(f"RANSAC error at frame {frame_id}: {e}")
                ransac_mask = np.ones(n_points, dtype=bool)

        # =====================================================================
        # TEMPORAL FILTERING (applied to obj_pts in 3D space)
        # =====================================================================
        if temporal_filter is not None:
            obj_pts = temporal_filter.update(obj_pts, frame_id, ransac_mask)

        # =====================================================================
        # PROJECT TO IMAGE COORDINATES
        # =====================================================================
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

        # =====================================================================
        # DRAW POINTS ON FRAME (WITH CIRCLE FILTERING)
        # =====================================================================
        if homog_marker_2d_cor is not None:
            try:
                points_in_frame = 0
                points_inside_circle = 0
                inliers_drawn = 0

                for i in range(homog_marker_2d_cor.shape[1]):
                    x, y = homog_marker_2d_cor[:2, i].flatten()
                    x_int, y_int = int(round(x)), int(round(y))

                    # Check if point is within frame bounds
                    if 0 <= x_int < frame.shape[1] and 0 <= y_int < frame.shape[0]:
                        points_in_frame += 1

                        # Check circle filtering
                        should_draw = True
                        if circle_center is not None and circle_radius is not None:
                            # Calculate distance from circle center
                            dist = np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2)
                            if dist <= circle_radius:
                                points_inside_circle += 1
                            else:
                                should_draw = False  # Skip drawing if outside circle
                        else:
                            points_inside_circle += 1

                        if should_draw:
                            # All points same color (green)
                            color = (0, 255, 0)  # Green for all points
                            radius = 5
                            if ransac_mask[i]:
                                inliers_drawn += 1

                            # Draw the point
                            cv2.circle(frame, (x_int, y_int), radius, color, -1)

                            # Draw the point index number
                            cv2.putText(frame, str(i), (x_int + 8, y_int - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                # Draw info text
                if circle_center is not None and circle_radius is not None:
                    info_text = f"Pyramid points: {points_inside_circle} inside circle (total: {n_points})"
                    cv2.putText(frame, info_text,
                                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw the circle boundary in CYAN with thick line for visibility
                    cx, cy = int(circle_center[0]), int(circle_center[1])
                    cv2.circle(frame, (cx, cy), int(circle_radius), (255, 255, 0), 3)  # Yellow, thickness 3

                    # # Draw center point in RED
                    # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    # # Draw radius value
                    # cv2.putText(frame, f"R={circle_radius:.1f}px",
                    #            (cx + 10, cy - 10),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                else:
                    info_text = f"Pyramid points: {inliers_drawn}/{points_in_frame} inliers (total: {n_points})"
                    cv2.putText(frame, info_text,
                                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error drawing pyramid points at frame {frame_id}: {e}")
                cv2.putText(frame, f"Draw error: {str(e)[:30]}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return corrected_2d, ransac_mask

    except Exception as e:
        print(f"Error in draw_pyramid_points_and_get_coords at frame {frame_id}: {e}")
        import traceback
        traceback.print_exc()
        cv2.putText(frame, f"Error: {str(e)[:50]}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return None, None


def draw_pyramid_points(
        frame: np.ndarray,
        frame_id: int,
        rb_data: dict[str, Any],
        calib_data,  # CalibData type
        points_optitrack_m: np.ndarray,
        R_cor: np.ndarray,
        enable_ransac: bool = True,
        ransac_threshold: float = 0.01,
        temporal_filter: Optional[TemporalPointFilter] = None,
        circle_center: Optional[Tuple[float, float]] = None,
        circle_radius: Optional[float] = None
) -> None:
    """
    Draw pyramid points on the video frame with RANSAC and temporal filtering.

    This is a wrapper around draw_pyramid_points_and_get_coords for backward compatibility.

    Args:
        frame: Video frame to draw on
        frame_id: Current frame index
        rb_data: Dictionary containing rigid body tracking data
        calib_data: Camera calibration data
        points_optitrack_m: Point positions in OptiTrack rigid body frame (Nx3, meters)
        R_cor: 3x3 rotation correction matrix for 2D homogeneous coordinates
        enable_ransac: Whether to apply RANSAC filtering
        ransac_threshold: RANSAC inlier threshold in meters
        temporal_filter: Optional temporal filter instance
        circle_center: (cx, cy) tuple of circle center for filtering, or None
        circle_radius: Radius of circle for filtering, or None
    """
    draw_pyramid_points_and_get_coords(
        frame=frame,
        frame_id=frame_id,
        rb_data=rb_data,
        calib_data=calib_data,
        points_optitrack_m=points_optitrack_m,
        R_cor=R_cor,
        enable_ransac=enable_ransac,
        ransac_threshold=ransac_threshold,
        temporal_filter=temporal_filter,
        circle_center=circle_center,
        circle_radius=circle_radius
    )


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
        is_rb_visible = rb_data["Pen_RB"][frame_id].data.is_visible if pen_mode else rb_data["Calib_RB"][
            frame_id].data.is_visible

        if is_lens_visible and is_rb_visible:
            try:
                for i in range(homog_marker_2d_cor.shape[1]):
                    cv2.circle(frame, tuple(np.round(homog_marker_2d_cor[:2, i].flatten()).astype(int)), 5, (0, 0, 255),
                               -1)
            except Exception as e:
                print(f"Error drawing marker: {e}")