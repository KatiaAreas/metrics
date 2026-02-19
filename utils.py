"""
Enhanced utils.py with integrated real-time uncertainty analysis

CHANGES vs previous version:
- RealTimeUncertaintyPlotter removed; replaced by UncertaintyPlotter from
  uncertainty_analysis.py (4 physically-meaningful diagnostic panels).
- display_pyramid: uses frame_record_from_analyzer() to build FrameRecord
  and UncertaintyPlotter.update() / .get_plot_image() instead of the old
  stage1/2/3 scalar approach.
- Sensitivity analysis runs on a separate, sparser interval
  (uncertainty_sensitivity_interval) to avoid per-frame overhead.
- Cached plot image and cached text overlay drawn every frame (not only on
  update interval).
- ASCII symbols [OK]/[!]/[X] used for cv2.putText (no Unicode issues).
"""

from typing import Any, Optional, Dict, Tuple, List
import cv2
from pathlib import Path
import numpy as np
import numpy.typing as npt
import re
import json

# CRITICAL: Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque

from calib_data import CalibData
from areas_theta_compute import NotchAngleComputer, get_default_weights_path
from pyramid_transformer import PyramidTransformer, extract_marker_positions_from_rb_data
from uncertainty_analysis import (
    UncertaintyAnalyzer,
    TransformationUncertainty,
    UncertaintyPlotter,
    FrameRecord,
    frame_record_from_analyzer,
)


class TemporalPointFilter:
    """Temporal mean filter for 3D points over time."""

    def __init__(self, window_size: int = 5, n_points: int = 18):
        self.window_size = window_size
        self.n_points = n_points
        self.point_history = [deque(maxlen=window_size) for _ in range(n_points)]

    def update(
        self,
        points: np.ndarray,
        frame_id: int,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
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
                self.point_history[i].clear()
                filtered_points[i] = points[i]
        return filtered_points

    def reset(self):
        self.point_history = [deque(maxlen=self.window_size) for _ in range(self.n_points)]


def load_keypoints_from_json(json_path: Path) -> dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints_dict = {}
    for frame in data['frames']:
        frame_id = frame['frame_id']
        keypoints_dict[frame_id] = frame['keypoints']
    return keypoints_dict


def is_point_inside_circle(
    x: float, y: float,
    circle_center: Tuple[float, float],
    circle_radius: float,
) -> bool:
    cx, cy = circle_center
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= circle_radius


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
        [rx * cos_theta,  ry * sin_theta, (1 - cos_theta) * ox - sin_theta * oy],
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
        [cos_theta,  sin_theta, (1 - cos_theta) * ox - sin_theta * oy],
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
            print(f"[OK] Loaded initial theta from vectors.log: {initial_theta:.2f} deg")
            return initial_theta
        print("Warning: Could not find 'Angle:' field in vectors.log")
        return None
    except Exception as e:
        print(f"Error parsing vectors.log: {e}")
        return None


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
        uncertainty_plot_position: str = "top_left",
        # How often to run the lightweight pipeline (stage 1-3 + budget from
        # cached sensitivities).  Runs every N frames.
        uncertainty_update_interval: int = 5,
        # How often to run the expensive perturbation-based sensitivity
        # analysis.  Set to 0 to disable (budget will use jitter estimates).
        uncertainty_sensitivity_interval: int = 60,
        enable_ransac: bool = True,
        ransac_threshold: float = 0.01,
        enable_temporal_filter: bool = True,
        temporal_window_size: int = 5,
        save_video: bool = False,
        output_video_path: Optional[Path] = None,
        output_fps: Optional[float] = None,
        output_codec: str = 'mp4v',
) -> None:
    """
    Display pyramid points overlaid on video frames with optional metrics and
    uncertainty analysis.

    Uncertainty analysis now shows 4 physically-meaningful diagnostic panels:
      1. Intrinsics impact (px per unit Δparam)
      2. Extrinsics translation sensitivity (px/mm per axis)
      3. Extrinsics rotation + time-sync sensitivity (px/° and px/frame)
      4. Error budget: predicted contributions vs actual reprojection
    """
    # =========================================================================
    # STEP 1: Load pyramid geometry and compute transformation
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING PYRAMID GEOMETRY (EXCLUDING POINTS 18-21)")
    print("=" * 70)

    transformer = PyramidTransformer(pyramid_json_path)

    marker_positions_m, rb_position_m, rb_quaternion = \
        extract_marker_positions_from_rb_data(rb_data, frame_id=0)

    matching = {
        'Marker 002': 20, 'Marker 001': 21,
        'Marker 003': 18, 'Marker 004': 19
    }

    R_constellation_to_optitrack = transformer.compute_optitrack_rotation_from_markers(
        marker_positions_m, matching
    )
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
    print(f"[OK] Using {len(valid_indices)} pyramid points (0-17, excluding 18-21)")

    if verify_transformation:
        from verification_script import (
            verif_svd,
            visualize_pyramid_frame_and_points,
            verify_pyramid_transformation,
        )
        transformer.plot_constellation_frame()
        verif_svd(rb_data, pyramid_json_path)
        visualize_pyramid_frame_and_points(transformer, interactive=True, save_path=None)
        verify_pyramid_transformation(
            rb_data=rb_data, calib_data=calib_data,
            transformer=transformer, frame_id=0
        )

    # =========================================================================
    # STEP 1.5: Load 2D keypoints if provided
    # =========================================================================
    keypoints_dict = None
    if keypoints_json_path is not None:
        print("\n" + "=" * 70)
        print("LOADING 2D KEYPOINTS")
        print("=" * 70)
        try:
            keypoints_dict = load_keypoints_from_json(keypoints_json_path)
            print(f"[OK] Loaded keypoints for {len(keypoints_dict)} frames")
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
        print("INITIALIZING METRICS COMPUTATION")
        print("=" * 70)
        metrics = Metrics(
            calib_data=calib_data, transformer=transformer,
            error_unit=error_unit, enable_realtime_plot=enable_realtime_plot
        )
        print(f"[OK] Metrics initialized (reference_frame={reference_frame})")
    elif compute_metrics and keypoints_dict is None:
        print("\n[!] WARNING: compute_metrics=True but no keypoints provided. Disabled.")

    # =========================================================================
    # STEP 1.7: Initialize Uncertainty Analyzer and Plotter
    # =========================================================================
    uncertainty_analyzer: Optional[UncertaintyAnalyzer] = None
    uncertainty_plotter:  Optional[UncertaintyPlotter]  = None

    if enable_uncertainty_analysis:
        print("\n" + "=" * 70)
        print("INITIALIZING UNCERTAINTY ANALYSIS")
        print("=" * 70)

        uncertainty_analyzer = UncertaintyAnalyzer(
            transformer=transformer, calib_data=calib_data, verbose=False
        )

        try:
            # Bootstrap: run full pipeline on frame 0 so the report is
            # populated before the main loop starts.
            initial_report = uncertainty_analyzer.analyze_full_pipeline(
                rb_data=rb_data,
                marker_positions_m=marker_positions_m,
                matching=matching,
                frame_id=0,
                keypoints_2d=None,
                R_cor=None,
                run_sensitivity=(uncertainty_sensitivity_interval > 0),
            )
            print("[OK] Uncertainty analyzer initialized")
            print(f"  Stage 1 SVD error:     "
                  f"{initial_report.model_to_optitrack.mean_error_3d_m * 1000:.3f} mm")
            print(f"  Stage 2 tracking jitter: "
                  f"{initial_report.optitrack_to_camera.translation_error_m * 1000:.3f} mm")

            if show_uncertainty_plot:
                plot_w = 620
                plot_h = 420
                uncertainty_plotter = UncertaintyPlotter(
                    history_len=300,
                    plot_width=plot_w,
                    plot_height=plot_h,
                )
                # Seed with frame-0 data
                rec0 = frame_record_from_analyzer(0, uncertainty_analyzer)
                uncertainty_plotter.update(rec0)
                print(f"  [OK] UncertaintyPlotter ready "
                      f"({plot_w}x{plot_h}, history=300 frames)")
                print(f"  Lightweight update every {uncertainty_update_interval} frames")
                if uncertainty_sensitivity_interval > 0:
                    print(f"  Sensitivity analysis every "
                          f"{uncertainty_sensitivity_interval} frames")
                else:
                    print("  Sensitivity analysis: DISABLED (using jitter estimates)")

        except Exception as e:
            print(f"[!] Error initializing uncertainty analysis: {e}")
            import traceback
            traceback.print_exc()
            enable_uncertainty_analysis = False

    # =========================================================================
    # STEP 1.8: Temporal filter
    # =========================================================================
    temporal_filter = None
    if enable_temporal_filter:
        temporal_filter = TemporalPointFilter(
            window_size=temporal_window_size, n_points=len(valid_indices)
        )
        print(f"[OK] Temporal filter (window={temporal_window_size})")

    # =========================================================================
    # STEP 2: Notch detector
    # =========================================================================
    notch_computer = None
    initial_theta = None

    if vectors_log_path is not None:
        initial_theta = parse_vectors_log(vectors_log_path)

    if use_notch:
        notch_computer = NotchAngleComputer(
            notch_model="pose", circle_method="hough", verbose=True
        )
        notch_computer.load_models(
            notch_model_path=str(get_default_weights_path()), device="auto"
        )
        print("[OK] Notch detector initialized")

    camera_center_x = calib_data.camera_model.get_center()[0]
    camera_center_y = calib_data.camera_model.get_center()[1]

    # =========================================================================
    # STEP 3: Open video
    # =========================================================================
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    input_fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # =========================================================================
    # STEP 3.5: Video writer
    # =========================================================================
    video_writer = None
    if save_video:
        if output_video_path is None:
            output_video_path = video_path.parent / f"{video_path.stem}_output.mp4"
        if output_fps is None:
            output_fps = input_fps if input_fps > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*output_codec)
        video_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, output_fps, (frame_width, frame_height)
        )
        if video_writer.isOpened():
            print(f"[OK] Saving to: {output_video_path}  "
                  f"({frame_width}x{frame_height} @ {output_fps}fps)")
        else:
            print("[!] Could not open video writer")
            video_writer = None
            save_video = False

    print("\nStarting playback. 'q' = quit  |  SPACE = pause/resume")

    # =========================================================================
    # Per-frame cache — drawn EVERY frame (not only on update intervals)
    # =========================================================================
    cached_stage1_mm:          Optional[float]          = None
    cached_stage2_mm:          Optional[float]          = None
    cached_stage3_px:          Optional[float]          = None
    cached_uncertainty_plot_img: Optional[np.ndarray]   = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_id = 0
    paused   = False

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

        # --- Theta / notch ---
        if use_notch:
            results = notch_computer.run(images=[display_frame], angle_unit="degrees")
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'circle_center') and hasattr(result, 'circle_radius'):
                    circle_center = result.circle_center
                    circle_radius = result.circle_radius
                else:
                    circle_center = circle_radius = None
                if (initial_theta is None and result.visibility == 1
                        and result.success and result.angle is not None):
                    initial_theta = result.angle
                    print(f"Initial theta from first detection: {initial_theta:.2f} deg")
                if result.visibility == 1 and initial_theta is not None:
                    theta = initial_theta - result.angle
                    notch_visible = True
                else:
                    theta = 0.0
                    notch_visible = False
                    msg = ("Waiting for initial notch detection"
                           if initial_theta is None else "Cannot detect notch")
                    cv2.putText(display_frame, msg, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            theta = 0.0
            notch_visible = True
            circle_center = circle_radius = None

        R_cor = rotation_correction_ellipse_ccw(
            theta,
            camera_center_x, camera_center_y,
            calib_data.camera_model.pp_ellipse_params[2],
            calib_data.camera_model.pp_ellipse_params[3],
        )

        # --- Draw pyramid points ---
        reprojected_points_2d = None
        should_draw = not (use_notch and not notch_visible)
        if should_draw:
            reprojected_points_2d, ransac_mask = draw_pyramid_points_and_get_coords(
                display_frame, frame_id, rb_data, calib_data,
                points_optitrack_filtered, R_cor=R_cor,
                enable_ransac=enable_ransac,
                ransac_threshold=ransac_threshold,
                temporal_filter=temporal_filter if enable_temporal_filter else None,
                circle_center=circle_center if use_notch else None,
                circle_radius=circle_radius if use_notch else None,
            )

        # --- 2D keypoints ---
        keypoints_2d = None
        if keypoints_dict is not None and frame_id in keypoints_dict:
            keypoints = keypoints_dict[frame_id]
            keypoints_2d = np.array(
                [[kp['x'], kp['y']] for kp in keypoints if kp.get('visibility', 2) > 0]
            )
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
                        reference_frame=reference_frame,
                    )
                    metrics.draw_errors_on_frame(
                        frame=display_frame, frame_errors=frame_errors,
                        keypoints=keypoints,
                        reprojected_points_2d=reprojected_points_2d,
                        show_per_point=show_per_point_errors,
                        show_frame_stats=show_frame_stats,
                        show_cumulative_stats=show_cumulative_stats,
                    )
                    if enable_realtime_plot:
                        metrics.update_realtime_plot()
                        if show_plot_in_video:
                            plot_img = metrics.get_plot_as_image(
                                width=display_frame.shape[1] // 2,
                                height=display_frame.shape[0] // 2,
                            )
                            if plot_img is not None:
                                h, w = plot_img.shape[:2]
                                display_frame[0:h,
                                              display_frame.shape[1] - w:] = plot_img
                except Exception as e:
                    cv2.putText(display_frame, f"Metrics err: {str(e)[:30]}",
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # =====================================================================
        # UNCERTAINTY ANALYSIS — lightweight update every N frames
        # =====================================================================
        if enable_uncertainty_analysis and uncertainty_analyzer is not None:
            if frame_id % uncertainty_update_interval == 0:
                try:
                    # Decide whether to run the expensive sensitivity pass
                    run_sens = (
                        uncertainty_sensitivity_interval > 0
                        and frame_id % uncertainty_sensitivity_interval == 0
                    )

                    report = uncertainty_analyzer.analyze_full_pipeline(
                        rb_data=rb_data,
                        marker_positions_m=marker_positions_m,
                        matching=matching,
                        frame_id=frame_id,
                        keypoints_2d=keypoints_2d,
                        R_cor=R_cor,
                        run_sensitivity=run_sens,
                    )

                    # ---- Update cached text values ----
                    if report.model_to_optitrack is not None:
                        cached_stage1_mm = \
                            report.model_to_optitrack.mean_error_3d_m * 1000
                    if report.optitrack_to_camera is not None:
                        cached_stage2_mm = \
                            report.optitrack_to_camera.translation_error_m * 1000
                    if (report.camera_projection is not None
                            and report.camera_projection.mean_reprojection_error_px
                            is not None):
                        cached_stage3_px = \
                            report.camera_projection.mean_reprojection_error_px

                    # ---- Push FrameRecord into plotter ----
                    if show_uncertainty_plot and uncertainty_plotter is not None:
                        rec = frame_record_from_analyzer(frame_id, uncertainty_analyzer)
                        uncertainty_plotter.update(rec)
                        try:
                            cached_uncertainty_plot_img = \
                                uncertainty_plotter.get_plot_image()
                        except Exception as plot_err:
                            if frame_id % 30 == 0:
                                print(f"  Plot render error: {plot_err}")
                            cached_uncertainty_plot_img = None

                except Exception as e:
                    if frame_id % 30 == 0:
                        print(f"Uncertainty error @ frame {frame_id}: {e}")
                        import traceback
                        traceback.print_exc()

        # =====================================================================
        # OVERLAY cached uncertainty plot EVERY frame
        # =====================================================================
        if enable_uncertainty_analysis and cached_uncertainty_plot_img is not None:
            plot_h, plot_w = cached_uncertainty_plot_img.shape[:2]
            fh, fw = display_frame.shape[:2]

            if uncertainty_plot_position == "top_right":
                y1, y2, x1, x2 = 0, plot_h, fw - plot_w, fw
            elif uncertainty_plot_position == "top_left":
                y1, y2, x1, x2 = 0, plot_h, 0, plot_w
            elif uncertainty_plot_position == "bottom_left":
                y1, y2, x1, x2 = fh - plot_h, fh, 0, plot_w
            else:  # bottom_right (default)
                y1, y2, x1, x2 = fh - plot_h, fh, fw - plot_w, fw

            if y1 >= 0 and y2 <= fh and x1 >= 0 and x2 <= fw:
                display_frame[y1:y2, x1:x2] = cached_uncertainty_plot_img

        # =====================================================================
        # Overlay cached uncertainty TEXT every frame  (ASCII symbols)
        # =====================================================================
        if enable_uncertainty_analysis:
            y_off = 100
            if cached_stage1_mm is not None:
                q = "[OK]" if cached_stage1_mm < 2 else \
                    "[!]" if cached_stage1_mm < 5 else "[X]"
                cv2.putText(display_frame,
                            f"{q} SVD fit:  {cached_stage1_mm:.2f} mm",
                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)
                y_off += 20
            if cached_stage2_mm is not None:
                q = "[OK]" if cached_stage2_mm < 1 else \
                    "[!]" if cached_stage2_mm < 2 else "[X]"
                cv2.putText(display_frame,
                            f"{q} Tracking: {cached_stage2_mm:.2f} mm",
                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)
                y_off += 20
            if cached_stage3_px is not None:
                q = "[OK]" if cached_stage3_px < 2 else \
                    "[!]" if cached_stage3_px < 5 else "[X]"
                cv2.putText(display_frame,
                            f"{q} Reproj:   {cached_stage3_px:.2f} px",
                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)

        # --- Theta overlay ---
        cv2.putText(display_frame,
                    f"Theta: {theta:.2f} deg / {np.deg2rad(theta):.2f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if use_notch and initial_theta is not None:
            src = "vectors.log" if vectors_log_path is not None else "detection"
            cv2.putText(display_frame,
                        f"Init theta: {initial_theta:.2f}deg ({src})",
                        (10, display_frame.shape[0] - (130 if metrics is not None else 80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if paused:
            cv2.putText(display_frame,
                        "PAUSED - Press SPACEBAR to resume",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- Window title ---
        win_title = "Pyramid Points Overlay (0-17)"
        if metrics is not None:
            win_title += " + Metrics"
        if enable_uncertainty_analysis:
            win_title += " + Uncertainty"
        cv2.imshow(win_title, display_frame)

        if save_video and video_writer is not None and not paused:
            video_writer.write(display_frame)
            if frame_id % 30 == 0:
                pct = (frame_id / total_frames * 100) if total_frames > 0 else 0
                print(f"Saving: frame {frame_id}/{total_frames} ({pct:.1f}%)")

        key = cv2.waitKey(16) & 0xFF
        if key == ord('q'):
            print("Playback stopped by user.")
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'} at frame {frame_id}")

        if not paused:
            frame_id += 1

    # --- Cleanup ---
    cap.release()
    if video_writer is not None:
        video_writer.release()
        if save_video and output_video_path is not None and output_video_path.exists():
            size_mb = output_video_path.stat().st_size / (1024 * 1024)
            print(f"[OK] Video saved: {output_video_path}  ({size_mb:.2f} MB)")
    cv2.destroyAllWindows()
    if uncertainty_plotter is not None:
        uncertainty_plotter.close()
    if metrics is not None:
        print("\n" + "=" * 70)
        print("FINALIZING METRICS")
        print("=" * 70)
        metrics.print_summary()
        if metrics_output_path is not None:
            metrics.save_statistics(metrics_output_path)
        if enable_realtime_plot:
            print("Close the plot window to continue...")
            metrics.close_plot()


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
        circle_radius: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Draw pyramid points on the video frame AND return the 2D coordinates.
    Applies RANSAC filtering and temporal averaging to obj_pts.
    Only draws points inside the detected circle (if provided).
    """
    is_lens_visible    = rb_data["Lens_RB"][frame_id].data.is_visible
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
        T_World_Lens    = rb_data["Lens_RB"][frame_id].get_transform()
        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
        RT = np.linalg.inv(T_World_Lens @ calib_data.RT)

        n_points = points_optitrack_m.shape[0]
        points_hom   = np.hstack([points_optitrack_m, np.ones((n_points, 1))])
        obj_pts      = (T_World_Pyramid @ points_hom.T).T[:, :3]

        proj_marker_2d = cv2.projectPoints(
            obj_pts, cv2.Rodrigues(RT[:3, :3])[0], RT[:3, 3],
            calib_data.K, calib_data.dist_coeffs
        )[0]
        homog_2d = np.hstack([proj_marker_2d.reshape(-1, 2),
                               np.ones((n_points, 1))]).T
        corrected_2d = (R_cor @ homog_2d)[:2, :].T

        # Valid mask
        if circle_center is not None and circle_radius is not None:
            valid_mask = np.array([
                np.sqrt((corrected_2d[i, 0] - circle_center[0])**2 +
                        (corrected_2d[i, 1] - circle_center[1])**2) <= circle_radius
                for i in range(n_points)
            ])
        elif enable_ransac and n_points >= 4:
            try:
                centroid  = np.mean(obj_pts, axis=0)
                distances = np.linalg.norm(obj_pts - centroid, axis=1)
                mad       = np.median(np.abs(distances - np.median(distances)))
                valid_mask = distances < ransac_threshold + 3 * mad
                n_in = np.sum(valid_mask)
                if n_points - n_in > 0:
                    cv2.putText(frame, f"RANSAC: {n_in}/{n_points} inliers",
                                (10, frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            except Exception:
                valid_mask = np.ones(n_points, dtype=bool)
        else:
            valid_mask = np.ones(n_points, dtype=bool)

        # Temporal filtering
        if temporal_filter is not None:
            obj_pts = temporal_filter.update(obj_pts, frame_id, valid_mask)
            proj_marker_2d = cv2.projectPoints(
                obj_pts, cv2.Rodrigues(RT[:3, :3])[0], RT[:3, 3],
                calib_data.K, calib_data.dist_coeffs
            )[0]
            homog_2d = np.hstack([proj_marker_2d.reshape(-1, 2),
                                   np.ones((n_points, 1))]).T
            corrected_2d = (R_cor @ homog_2d)[:2, :].T

        # Draw
        pts_in_circle = 0
        inliers_drawn = 0
        for i in range(n_points):
            if not valid_mask[i]:
                continue
            x, y = corrected_2d[i]
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < frame.shape[1] and 0 <= yi < frame.shape[0]):
                continue
            if circle_center is not None and circle_radius is not None:
                d = np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2)
                if d > circle_radius:
                    continue
                pts_in_circle += 1
            else:
                pts_in_circle += 1
            cv2.circle(frame, (xi, yi), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (xi + 8, yi - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            inliers_drawn += 1

        # Info text
        if circle_center is not None and circle_radius is not None:
            cv2.circle(frame,
                       (int(circle_center[0]), int(circle_center[1])),
                       int(circle_radius), (255, 255, 0), 3)
            cv2.putText(frame,
                        f"Points inside circle: {pts_in_circle} / {np.sum(valid_mask)}",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame,
                        f"Pyramid points: {inliers_drawn}/{np.sum(valid_mask)} valid "
                        f"(total {n_points})",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return corrected_2d, valid_mask

    except Exception as e:
        print(f"Error in draw_pyramid_points_and_get_coords @ frame {frame_id}: {e}")
        import traceback
        traceback.print_exc()
        cv2.putText(frame, f"Error: {str(e)[:50]}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return None, None


def draw_pyramid_points(
        frame: np.ndarray, frame_id: int, rb_data: dict[str, Any], calib_data,
        points_optitrack_m: np.ndarray, R_cor: np.ndarray,
        enable_ransac: bool = True, ransac_threshold: float = 0.01,
        temporal_filter: Optional[TemporalPointFilter] = None,
        circle_center: Optional[Tuple[float, float]] = None,
        circle_radius: Optional[float] = None,
) -> None:
    """Backward-compatible wrapper around draw_pyramid_points_and_get_coords."""
    draw_pyramid_points_and_get_coords(
        frame=frame, frame_id=frame_id, rb_data=rb_data,
        calib_data=calib_data, points_optitrack_m=points_optitrack_m,
        R_cor=R_cor, enable_ransac=enable_ransac,
        ransac_threshold=ransac_threshold, temporal_filter=temporal_filter,
        circle_center=circle_center, circle_radius=circle_radius,
    )


def display_calib(
        video_path: Path, rb_data: dict[str, Any], calib_data: CalibData,
        use_notch: bool = False, pen_mode: bool = False, pyramid_mode: bool = False,
) -> None:
    """Opens and displays a video file with marker overlays."""
    cap = cv2.VideoCapture(str(video_path))

    notch_computer = None
    initial_theta  = None
    if use_notch:
        notch_computer = NotchAngleComputer(
            notch_model="pose", circle_method="hough", verbose=True
        )
        notch_computer.load_models(
            notch_model_path=str(get_default_weights_path()), device="auto"
        )

    camera_center_x = calib_data.camera_model.get_center()[0]
    camera_center_y = calib_data.camera_model.get_center()[1]

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    print("Starting video playback. Press 'q' to quit.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_notch:
            results = notch_computer.run(images=[frame], angle_unit="degrees")
            if results and len(results) > 0:
                result = results[0]
                if (initial_theta is None and result.visibility == 1
                        and result.success and result.angle is not None):
                    initial_theta = result.angle
                if result.visibility == 1 and initial_theta is not None:
                    theta = initial_theta - result.angle
                else:
                    theta = 0.0
                    msg = ("Waiting for initial notch detection"
                           if initial_theta is None else "Cannot detect notch")
                    cv2.putText(frame, msg, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            theta = 0.0

        R_cor = rotation_correction_ellipse_ccw(
            theta, camera_center_x, camera_center_y,
            calib_data.camera_model.pp_ellipse_params[2],
            calib_data.camera_model.pp_ellipse_params[3],
        )

        if not (use_notch and theta == 0.0 and initial_theta is None):
            draw_marker(frame, frame_id, rb_data, calib_data, R_cor,
                        pen_mode=pen_mode, theta_deg=theta)

        cv2.putText(frame,
                    f"Theta: {theta:.2f} deg / {np.deg2rad(theta):.2f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Video Playback", frame)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


def draw_marker(
        frame: np.ndarray, frame_id: int, rb_data: dict[str, Any],
        calib_data: CalibData, R_cor: np.ndarray,
        pen_mode: bool = False, theta_deg: float = 0.0,
) -> None:
    """Draw calibration markers on the frame."""
    T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
    RT = np.linalg.inv(T_World_Lens @ calib_data.RT)

    if pen_mode:
        obj_pts = np.array(rb_data["Pen_RB"][frame_id].data.position).reshape(1, 3)
    else:
        obj_pts = np.vstack([
            np.array(v)
            for v in rb_data["Calib_RB"][frame_id].data.marker_positions.values()
        ])

    proj_2d = cv2.projectPoints(
        obj_pts, cv2.Rodrigues(RT[:3, :3])[0], RT[:3, 3],
        calib_data.K, calib_data.dist_coeffs
    )[0]
    homog = np.hstack([proj_2d.reshape(-1, 2), np.ones((proj_2d.shape[0], 1))]).T
    corrected = R_cor @ homog

    is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
    is_rb_visible   = (rb_data["Pen_RB"][frame_id].data.is_visible if pen_mode
                       else rb_data["Calib_RB"][frame_id].data.is_visible)

    if is_lens_visible and is_rb_visible:
        try:
            for i in range(corrected.shape[1]):
                pt = tuple(np.round(corrected[:2, i]).astype(int))
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        except Exception as e:
            print(f"Error drawing marker: {e}")