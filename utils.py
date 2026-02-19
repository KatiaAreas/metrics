"""
utils.py - minimal changes from working version (document 1):
  1. Removed matplotlib/FigureCanvasAgg imports (not needed here anymore)
  2. Changed uncertainty_analysis import to: UncertaintyAnalyzer, UncertaintyPlotter,
     FrameRecord, frame_record_from_analyzer
  3. Removed RealTimeUncertaintyPlotter class (now in uncertainty_analysis.py)
  4. display_pyramid: plotter init + update → separate cv2 window, no overlay
Everything else is byte-for-byte identical to the working version.
"""

from typing import Any, Optional, Dict, Tuple, List
import cv2
from pathlib import Path
import numpy as np
import numpy.typing as npt
import re
import json
from collections import deque

from calib_data import CalibData
from areas_theta_compute import NotchAngleComputer, get_default_weights_path
from pyramid_transformer import PyramidTransformer, extract_marker_positions_from_rb_data

# CHANGED: import new classes — UncertaintyPlotter lives in uncertainty_analysis now
from uncertainty_analysis import (
    UncertaintyAnalyzer,
    UncertaintyPlotter,
    FrameRecord,
    frame_record_from_analyzer,
)


# ---------------------------------------------------------------------------
# TemporalPointFilter  (unchanged)
# ---------------------------------------------------------------------------

class TemporalPointFilter:
    """
    Temporal mean filter for 3D points over time.
    Maintains a sliding window of point positions and computes running average.
    """

    def __init__(self, window_size: int = 5, n_points: int = 18):
        self.window_size = window_size
        self.n_points = n_points
        self.point_history = [deque(maxlen=window_size) for _ in range(n_points)]

    def update(self, points: np.ndarray, frame_id: int,
               valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if valid_mask is None:
            valid_mask = np.ones(len(points), dtype=bool)

        filtered_points = np.zeros_like(points)

        for i in range(len(points)):
            if valid_mask[i]:
                self.point_history[i].append((frame_id, points[i].copy()))
                if len(self.point_history[i]) > 0:
                    recent_positions = np.array([pos for _, pos in self.point_history[i]])
                    filtered_points[i] = np.mean(recent_positions, axis=0)
                else:
                    filtered_points[i] = points[i]
            else:
                if len(self.point_history[i]) > 0:
                    filtered_points[i] = self.point_history[i][-1][1]
                else:
                    filtered_points[i] = points[i]

        return filtered_points

    def reset(self):
        self.point_history = [deque(maxlen=self.window_size) for _ in range(self.n_points)]


# NOTE: RealTimeUncertaintyPlotter has been removed.
# Use UncertaintyPlotter from uncertainty_analysis.py — it opens its own cv2 window.


# ---------------------------------------------------------------------------
# Helpers  (unchanged)
# ---------------------------------------------------------------------------

def load_keypoints_from_json(json_path: Path) -> dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints_dict = {}
    for frame in data['frames']:
        frame_id = frame['frame_id']
        keypoints_dict[frame_id] = frame['keypoints']
    return keypoints_dict


def is_point_inside_circle(x: float, y: float,
                            circle_center: Tuple[float, float],
                            circle_radius: float) -> bool:
    cx, cy = circle_center
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return distance <= circle_radius


def rotation_correction_ellipse_ccw(theta_degrees, ox, oy, rx, ry):
    theta_rad = np.deg2rad(theta_degrees)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    return np.array([
        [rx * cos_theta, -ry * sin_theta, (1 - cos_theta) * ox + sin_theta * oy],
        [rx * sin_theta,  ry * cos_theta, -sin_theta * ox + (1 - cos_theta) * oy],
        [0, 0, 1]
    ])


def rotation_correction_ellipse_cw(theta_degrees, ox, oy, rx, ry):
    theta_rad = np.deg2rad(theta_degrees)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    return np.array([
        [ rx * cos_theta, ry * sin_theta, (1 - cos_theta) * ox - sin_theta * oy],
        [-rx * sin_theta, ry * cos_theta,  sin_theta * ox + (1 - cos_theta) * oy],
        [0, 0, 1]
    ])


def rotation_correction_ccw(theta_degrees, ox, oy):
    theta_rad = np.deg2rad(theta_degrees)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    return np.array([
        [cos_theta, -sin_theta, (1 - cos_theta) * ox + sin_theta * oy],
        [sin_theta,  cos_theta, -sin_theta * ox + (1 - cos_theta) * oy],
        [0, 0, 1]
    ])


def rotation_correction_cw(theta_degrees, ox, oy):
    theta_rad = np.deg2rad(theta_degrees)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    return np.array([
        [ cos_theta, sin_theta, (1 - cos_theta) * ox - sin_theta * oy],
        [-sin_theta, cos_theta,  sin_theta * ox + (1 - cos_theta) * oy],
        [0, 0, 1]
    ])


def parse_vectors_log(vectors_log_path: Path) -> Optional[float]:
    if not vectors_log_path.exists():
        print(f"Warning: vectors.log file not found at {vectors_log_path}")
        return None
    try:
        with open(vectors_log_path, 'r') as f:
            content = f.read()
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


# ---------------------------------------------------------------------------
# display_pyramid  (4 changed lines inside, everything else identical)
# ---------------------------------------------------------------------------

def display_pyramid(
        video_path: Path,
        rb_data: dict[str, Any],
        calib_data,
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
        uncertainty_update_interval: int = 5,
        enable_ransac: bool = True,
        ransac_threshold: float = 0.01,
        enable_temporal_filter: bool = True,
        temporal_window_size: int = 5,
        start_frame_id: int = 0,
) -> None:
    """
    Display pyramid points overlaid on video frames.
    Uncertainty plot is shown in a SEPARATE cv2 window, not overlaid on the video.
    Press SPACEBAR on either the video window or the uncertainty window to pause/resume.
    """
    # =========================================================================
    # STEP 1: Load pyramid geometry
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING PYRAMID GEOMETRY (EXCLUDING POINTS 18-21)")
    print("=" * 70)

    transformer = PyramidTransformer(pyramid_json_path)

    marker_positions_m, rb_position_m, rb_quaternion = extract_marker_positions_from_rb_data(
        rb_data, frame_id=0)

    matching = {
        'Marker 002': 20,
        'Marker 001': 21,
        'Marker 003': 18,
        'Marker 004': 19
    }

    R_constellation_to_optitrack = transformer.compute_optitrack_rotation_from_markers(
        marker_positions_m, matching)
    R_pyramid_to_optitrack = transformer.R_pyramid_to_optitrack
    T_pyramid_to_optitrack = transformer.T_pyramid_to_optitrack
    print('T_pyramid_to_optitrack = ', T_pyramid_to_optitrack)

    points_world = transformer.points_m
    R_pyramid = transformer.R_pyramid
    pyramid_origin = transformer.pyramid_origin_m
    points_pyramid = (R_pyramid.T @ (points_world - pyramid_origin).T).T
    points_optitrack = transformer.transform_pyramid_to_optitrack(points_pyramid)

    valid_indices = list(range(0, 18))
    points_optitrack_filtered = points_optitrack[valid_indices]
    points_pyramid_filtered = points_pyramid[valid_indices]
    print(f"✓ Using {len(valid_indices)} pyramid points (0-17, excluding 18-21)")

    if verify_transformation:
        from verification_script import verif_svd, visualize_pyramid_frame_and_points, verify_pyramid_transformation
        transformer.plot_constellation_frame()
        verif_svd(rb_data, pyramid_json_path)
        visualize_pyramid_frame_and_points(transformer, interactive=True, save_path=None)
        verify_pyramid_transformation(rb_data=rb_data, calib_data=calib_data,
                                      transformer=transformer, frame_id=0)

    # =========================================================================
    # STEP 1.5: Load 2D keypoints
    # =========================================================================
    keypoints_dict = None
    if keypoints_json_path is not None:
        print("\n" + "=" * 70)
        print("LOADING 2D KEYPOINTS")
        print("=" * 70)
        try:
            keypoints_dict = load_keypoints_from_json(keypoints_json_path)
            print(f"✓ Loaded keypoints for {len(keypoints_dict)} frames")
            print(f"✓ Number of keypoints per frame: {len(keypoints_dict[list(keypoints_dict.keys())[0]])}")
        except Exception as e:
            print(f"Error loading keypoints: {e}")
            keypoints_dict = None

    # =========================================================================
    # STEP 1.6: Initialize Metrics
    # =========================================================================
    metrics = None
    if compute_metrics and keypoints_dict is not None:
        from metrics import Metrics
        print("\n" + "=" * 70)
        print("INITIALIZING METRICS COMPUTATION")
        print("=" * 70)
        metrics = Metrics(calib_data=calib_data, transformer=transformer,
                          error_unit=error_unit, enable_realtime_plot=enable_realtime_plot)
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
    # STEP 1.7: Initialize Uncertainty Analyzer
    # =========================================================================
    uncertainty_analyzer = None
    uncertainty_plotter = None

    if enable_uncertainty_analysis:
        print("\n" + "=" * 70)
        print("INITIALIZING UNCERTAINTY ANALYSIS")
        print("=" * 70)

        uncertainty_analyzer = UncertaintyAnalyzer(
            transformer=transformer, calib_data=calib_data, verbose=False)

        try:
            initial_report = uncertainty_analyzer.analyze_full_pipeline(
                rb_data=rb_data, marker_positions_m=marker_positions_m,
                matching=matching, frame_id=0, keypoints_2d=None, R_cor=None)

            print(f"✓ Uncertainty analyzer initialized")
            print(f"  - Stage 1 (SVD) error: "
                  f"{initial_report.model_to_optitrack.mean_error_3d_m * 1000:.3f} mm")
            print(f"  - Stage 2 (Tracking) noise: "
                  f"{initial_report.optitrack_to_camera.translation_error_m * 1000:.3f} mm")

            # CHANGED: open a separate cv2 window — no overlay on video frame
            if show_uncertainty_plot:
                print(f"  - Uncertainty plot: SEPARATE WINDOW ('Uncertainty Analysis')")
                print(f"  - Update interval: every {uncertainty_update_interval} frames")
                uncertainty_plotter = UncertaintyPlotter(
                    history_len=200,
                    window_name="Uncertainty Analysis",
                )

        except Exception as e:
            print(f"⚠ Error initializing uncertainty analysis: {e}")
            enable_uncertainty_analysis = False

    # =========================================================================
    # STEP 1.8: Initialize temporal filter
    # =========================================================================
    temporal_filter = None
    if enable_temporal_filter:
        print("\n" + "=" * 70)
        print("INITIALIZING TEMPORAL FILTER")
        print("=" * 70)
        temporal_filter = TemporalPointFilter(
            window_size=temporal_window_size, n_points=len(valid_indices))
        print(f"✓ Temporal filter initialized with window size {temporal_window_size}")

    # =========================================================================
    # STEP 2: Initialize notch detector
    # =========================================================================
    notch_computer = None
    initial_theta = None

    if vectors_log_path is not None:
        initial_theta = parse_vectors_log(vectors_log_path)
        if initial_theta is None:
            print("Warning: Failed to load initial theta from vectors.log")
            if use_notch:
                print("         Will wait for first notch detection to set initial theta")

    if use_notch:
        print("\nInitializing notch detector...")
        notch_computer = NotchAngleComputer(
            notch_model="pose", circle_method="hough", verbose=True)
        notch_computer.load_models(
            notch_model_path=str(get_default_weights_path()), device="auto")
        print("✓ Notch detector initialized")
        if initial_theta is not None:
            print(f"✓ Using initial theta from vectors.log: {initial_theta:.2f}°")
        else:
            print("⚠ No initial theta loaded - will use first detection")

    camera_center_x: float = calib_data.camera_model.get_center()[0]
    camera_center_y: float = calib_data.camera_model.get_center()[1]

    # =========================================================================
    # STEP 3: Open video
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
        print("Uncertainty analysis: ENABLED (separate window)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
    frame_id: int = start_frame_id
    if start_frame_id > 0:
        print(f"Starting from frame {start_frame_id}")

    notch_visible = False
    circle_center = None
    circle_radius = None
    paused = False

    # =========================================================================
    # STEP 4: Playback loop
    # =========================================================================
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

        display_frame = frame.copy()

        # =====================================================================
        # Compute theta
        # =====================================================================
        if use_notch:
            results = notch_computer.run(images=[display_frame], angle_unit="degrees")
            if results and len(results) > 0:
                result = results[0]

                if hasattr(result, 'circle_center') and hasattr(result, 'circle_radius'):
                    circle_center = result.circle_center
                    circle_radius = result.circle_radius
                else:
                    circle_center = None
                    circle_radius = None

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

        R_cor = rotation_correction_cw(theta, camera_center_x, camera_center_y)

        # =====================================================================
        # Draw pyramid points
        # =====================================================================
        reprojected_points_2d = None
        should_draw = True
        if use_notch and not notch_visible:
            should_draw = False

        if should_draw:
            reprojected_points_2d, ransac_mask = draw_pyramid_points_and_get_coords(
                display_frame, frame_id, rb_data, calib_data,
                points_optitrack_filtered, R_cor=R_cor,
                enable_ransac=enable_ransac, ransac_threshold=ransac_threshold,
                temporal_filter=temporal_filter if enable_temporal_filter else None,
                circle_center=circle_center if use_notch else None,
                circle_radius=circle_radius if use_notch else None
            )

        # =====================================================================
        # 2D keypoints
        # =====================================================================
        keypoints_2d = None
        if keypoints_dict is not None and frame_id in keypoints_dict:
            keypoints = keypoints_dict[frame_id]
            keypoints_2d = np.array([[kp['x'], kp['y']] for kp in keypoints
                                     if kp.get('visibility', 2) > 0])

            for kp in keypoints:
                if kp.get('visibility', 2) > 0:
                    x, y = int(round(kp['x'])), int(round(kp['y']))
                    if 0 <= x < display_frame.shape[1] and 0 <= y < display_frame.shape[0]:
                        cv2.circle(display_frame, (x, y), 3, (255, 0, 0), -1)
                        cv2.putText(display_frame, str(kp['id']), (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # Metrics
            if metrics is not None and reprojected_points_2d is not None:
                try:
                    frame_errors = metrics.update_frame_errors(
                        keypoints=keypoints,
                        reprojected_points_2d=reprojected_points_2d,
                        points_pyramid_frame=points_pyramid_filtered,
                        frame_id=frame_id, rb_data=rb_data,
                        reference_frame=reference_frame)

                    metrics.draw_errors_on_frame(
                        frame=display_frame, frame_errors=frame_errors,
                        keypoints=keypoints,
                        reprojected_points_2d=reprojected_points_2d,
                        show_per_point=show_per_point_errors,
                        show_frame_stats=show_frame_stats,
                        show_cumulative_stats=show_cumulative_stats)

                    if enable_realtime_plot:
                        metrics.update_realtime_plot()
                        if show_plot_in_video:
                            plot_img = metrics.get_plot_as_image(
                                width=display_frame.shape[1] // 2,
                                height=display_frame.shape[0] // 2)
                            if plot_img is not None:
                                h, w = plot_img.shape[:2]
                                display_frame[0:h, display_frame.shape[1] - w:] = plot_img

                except Exception as e:
                    print(f"Error computing metrics for frame {frame_id}: {e}")
                    cv2.putText(display_frame, f"Metrics error: {str(e)[:30]}",
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # =====================================================================
        # Uncertainty analysis  — CHANGED: separate window, no overlay
        # =====================================================================
        if enable_uncertainty_analysis and uncertainty_analyzer is not None:
            if frame_id % uncertainty_update_interval == 0:
                try:
                    report = uncertainty_analyzer.analyze_full_pipeline(
                        rb_data=rb_data, marker_positions_m=marker_positions_m,
                        matching=matching, frame_id=frame_id,
                        keypoints_2d=keypoints_2d, R_cor=R_cor)

                    # CHANGED: push to separate window
                    if show_uncertainty_plot and uncertainty_plotter is not None:
                        rec = frame_record_from_analyzer(frame_id, uncertainty_analyzer)
                        uncertainty_plotter.update(rec)  # refreshes its own cv2 window

                    # Keep only a tiny text overlay on the video (3 numbers)
                    u1 = report.model_to_optitrack
                    u2 = report.optitrack_to_camera
                    u3 = report.camera_projection
                    y_off = 100

                    if u1 and u1.mean_error_3d_m is not None:
                        val = u1.mean_error_3d_m * 1000
                        q = "[OK]" if val < 2 else "[!]" if val < 5 else "[X]"
                        cv2.putText(display_frame, f"{q} SVD: {val:.2f}mm",
                                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_off += 20

                    if u2 and u2.translation_error_m is not None:
                        val = u2.translation_error_m * 1000
                        q = "[OK]" if val < 0.5 else "[!]" if val < 2 else "[X]"
                        cv2.putText(display_frame, f"{q} Track: {val:.2f}mm",
                                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_off += 20

                    if u3 and u3.mean_reprojection_error_px is not None:
                        val = u3.mean_reprojection_error_px
                        q = "[OK]" if val < 2 else "[!]" if val < 5 else "[X]"
                        cv2.putText(display_frame, f"{q} Reproj: {val:.2f}px",
                                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                except Exception as e:
                    print(f"Error computing uncertainty for frame {frame_id}: {e}")
                    cv2.putText(display_frame, f"Uncertainty error: {str(e)[:30]}",
                                (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display theta
        cv2.putText(display_frame, f"Theta: {theta:.2f} deg / {np.deg2rad(theta):.2f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if use_notch and initial_theta is not None:
            theta_source = "vectors.log" if vectors_log_path is not None else "first detection"
            cv2.putText(display_frame,
                        f"Init theta: {initial_theta:.2f}deg ({theta_source})",
                        (10, display_frame.shape[0] - (130 if metrics is not None else 80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if paused:
            cv2.putText(display_frame, "PAUSED - Press SPACEBAR to resume",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        window_title = "Pyramid Points Overlay (Points 0-17)"
        if metrics is not None:
            window_title += " + Metrics"
        if enable_uncertainty_analysis:
            window_title += " + Uncertainty"

        cv2.imshow(window_title, display_frame)

        # cv2.waitKey handles ALL open windows (video + uncertainty plot)
        key = cv2.waitKey(16) & 0xFF

        if key == ord('q'):
            print("Playback stopped by user.")
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                print(f"Video paused at frame {frame_id}")
            else:
                print(f"Video resumed from frame {frame_id}")

        if not paused:
            frame_id += 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    if uncertainty_plotter is not None:
        uncertainty_plotter.close()

    # Metrics summary
    if metrics is not None:
        print("\n" + "=" * 70)
        print("FINALIZING METRICS")
        print("=" * 70)
        metrics.print_summary()
        if metrics_output_path is not None:
            metrics.save_statistics(metrics_output_path)
        if enable_realtime_plot:
            print("\nClose the plot window to continue...")
            metrics.close_plot()


# ---------------------------------------------------------------------------
# draw_pyramid_points_and_get_coords  (unchanged from document 1)
# ---------------------------------------------------------------------------

def draw_pyramid_points_and_get_coords(
        frame: np.ndarray,
        frame_id: int,
        rb_data: dict[str, Any],
        calib_data,
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
    """
    is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
    is_pyramid_visible = rb_data["Pyramid_RB"][frame_id].data.is_visible

    if not (is_lens_visible and is_pyramid_visible):
        if not is_pyramid_visible:
            cv2.putText(frame, "Pyramid not visible in OptiTrack",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not is_lens_visible:
            cv2.putText(frame, "Lens not visible in OptiTrack",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None

    try:
        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
        RT = np.linalg.inv(T_World_Lens @ calib_data.RT)

        n_points = points_optitrack_m.shape[0]
        points_hom = np.hstack([points_optitrack_m, np.ones((n_points, 1))])
        points_world_hom = (T_World_Pyramid @ points_hom.T).T
        obj_pts = points_world_hom[:, 0:3]

        # RANSAC
        ransac_mask = np.ones(n_points, dtype=bool)
        if enable_ransac and n_points >= 4:
            try:
                centroid = np.mean(obj_pts, axis=0)
                distances = np.linalg.norm(obj_pts - centroid, axis=1)
                median_dist = np.median(distances)
                mad = np.median(np.abs(distances - median_dist))
                threshold_dist = ransac_threshold + 3 * mad
                ransac_mask = distances < threshold_dist
                n_inliers = np.sum(ransac_mask)
                n_outliers = n_points - n_inliers
                if n_outliers > 0:
                    cv2.putText(frame,
                                f"RANSAC: {n_inliers}/{n_points} inliers ({n_outliers} outliers)",
                                (10, frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            except Exception as e:
                print(f"RANSAC error at frame {frame_id}: {e}")
                ransac_mask = np.ones(n_points, dtype=bool)

        # Temporal filter
        if temporal_filter is not None:
            obj_pts = temporal_filter.update(obj_pts, frame_id, ransac_mask)

        # Project
        proj_marker_2d = cv2.projectPoints(
            obj_pts, cv2.Rodrigues(RT[:3, :3])[0], RT[:3, 3],
            calib_data.K, calib_data.dist_coeffs)[0]

        homog_marker_2d = np.hstack([
            proj_marker_2d.reshape(-1, 2),
            np.ones((proj_marker_2d.shape[0], 1))]).T
        homog_marker_2d_cor = R_cor @ homog_marker_2d
        corrected_2d = homog_marker_2d_cor[:2, :].T

        # Draw
        if homog_marker_2d_cor is not None:
            try:
                points_in_frame = 0
                points_inside_circle = 0
                inliers_drawn = 0

                for i in range(homog_marker_2d_cor.shape[1]):
                    x, y = homog_marker_2d_cor[:2, i].flatten()
                    x_int, y_int = int(round(x)), int(round(y))

                    if 0 <= x_int < frame.shape[1] and 0 <= y_int < frame.shape[0]:
                        points_in_frame += 1
                        should_draw = True

                        if circle_center is not None and circle_radius is not None:
                            dist = np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2)
                            if dist <= circle_radius:
                                points_inside_circle += 1
                            else:
                                should_draw = False
                        else:
                            points_inside_circle += 1

                        if should_draw:
                            color = (0, 255, 0)
                            radius = 5
                            if ransac_mask[i]:
                                inliers_drawn += 1
                            cv2.circle(frame, (x_int, y_int), radius, color, -1)
                            cv2.putText(frame, str(i), (x_int + 8, y_int - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                if circle_center is not None and circle_radius is not None:
                    info_text = (f"Pyramid points: {points_inside_circle} inside circle "
                                 f"(total: {n_points})")
                    cv2.putText(frame, info_text, (10, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cx, cy = int(circle_center[0]), int(circle_center[1])
                    cv2.circle(frame, (cx, cy), int(circle_radius), (255, 255, 0), 3)
                else:
                    info_text = (f"Pyramid points: {inliers_drawn}/{points_in_frame} "
                                 f"inliers (total: {n_points})")
                    cv2.putText(frame, info_text, (10, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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


# ---------------------------------------------------------------------------
# draw_pyramid_points  (unchanged)
# ---------------------------------------------------------------------------

def draw_pyramid_points(
        frame: np.ndarray,
        frame_id: int,
        rb_data: dict[str, Any],
        calib_data,
        points_optitrack_m: np.ndarray,
        R_cor: np.ndarray,
        enable_ransac: bool = True,
        ransac_threshold: float = 0.01,
        temporal_filter: Optional[TemporalPointFilter] = None,
        circle_center: Optional[Tuple[float, float]] = None,
        circle_radius: Optional[float] = None
) -> None:
    """Wrapper around draw_pyramid_points_and_get_coords for backward compatibility."""
    draw_pyramid_points_and_get_coords(
        frame=frame, frame_id=frame_id, rb_data=rb_data, calib_data=calib_data,
        points_optitrack_m=points_optitrack_m, R_cor=R_cor,
        enable_ransac=enable_ransac, ransac_threshold=ransac_threshold,
        temporal_filter=temporal_filter, circle_center=circle_center,
        circle_radius=circle_radius)


# ---------------------------------------------------------------------------
# display_calib  (unchanged)
# ---------------------------------------------------------------------------

def display_calib(
        video_path: Path,
        rb_data: dict[str, Any],
        calib_data: CalibData,
        use_notch: bool = False,
        pen_mode: bool = False,
        pyramid_mode: bool = False
) -> None:
    cap = cv2.VideoCapture(str(video_path))

    if use_notch:
        notch_computer = NotchAngleComputer(
            notch_model="pose", circle_method="hough", verbose=True)
        notch_computer.load_models(
            notch_model_path=str(get_default_weights_path()), device="auto")
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
            results = notch_computer.run(images=[frame], angle_unit="degrees")
            if results and len(results) > 0:
                result = results[0]
                if (initial_theta is None and result.visibility == 1
                        and result.success and result.angle is not None):
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
        if use_notch and not notch_visible:
            should_draw = False

        if should_draw:
            draw_marker(frame, frame_id, rb_data, calib_data, R_cor,
                        pen_mode=pen_mode, theta_deg=theta)

        cv2.putText(frame, f"Theta: {theta:.2f} deg / {np.deg2rad(theta):.2f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Video Playback", frame)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            print("Playback stopped by user.")
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# draw_marker  (unchanged)
# ---------------------------------------------------------------------------

def draw_marker(
        frame: np.ndarray,
        frame_id: int,
        rb_data: dict[str, Any],
        calib_data: CalibData,
        R_cor: np.ndarray,
        pen_mode: bool = False,
        theta_deg: float = 0.0
) -> None:
    T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
    RT = np.linalg.inv(T_World_Lens @ calib_data.RT)

    if pen_mode:
        obj_pts = np.array(rb_data["Pen_RB"][frame_id].data.position).reshape(1, 3)
    else:
        obj_pts = np.vstack([np.array(value)
                             for value in rb_data["Calib_RB"][frame_id].data.marker_positions.values()])

    proj_marker_2d = cv2.projectPoints(
        obj_pts, cv2.Rodrigues(RT[:3, :3])[0], RT[:3, 3],
        calib_data.K, calib_data.dist_coeffs)[0]

    homog_marker_2d = np.hstack([
        proj_marker_2d.reshape(-1, 2),
        np.ones((proj_marker_2d.shape[0], 1))]).T
    homog_marker_2d_cor = R_cor @ homog_marker_2d

    if homog_marker_2d_cor is not None:
        is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
        is_rb_visible = (rb_data["Pen_RB"][frame_id].data.is_visible if pen_mode
                         else rb_data["Calib_RB"][frame_id].data.is_visible)

        if is_lens_visible and is_rb_visible:
            try:
                for i in range(homog_marker_2d_cor.shape[1]):
                    cv2.circle(frame,
                               tuple(np.round(homog_marker_2d_cor[:2, i].flatten()).astype(int)),
                               5, (0, 0, 255), -1)
            except Exception as e:
                print(f"Error drawing marker: {e}")