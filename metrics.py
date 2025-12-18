"""
Metrics calculation for comparing 2D keypoints with reprojected 3D pyramid points.

This module provides the Metrics class for:
1. Computing pixel-space differences between detected keypoints and reprojected points
2. Computing 3D differences (mm/m) in OptiTrack or pyramid frame
3. Displaying per-point and per-frame errors on video frames
4. Accumulating errors across the entire video sequence
5. Real-time plotting of cumulative errors
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class Metrics:
    """
    Calculate and track errors between 2D keypoints and reprojected 3D pyramid points.

    Computes:
    - Pixel-space error (2D reprojection error)
    - 3D space error in meters/millimeters (in OptiTrack or pyramid frame)
    - Per-point, per-frame, and cumulative statistics
    """

    def __init__(
        self,
        calib_data,
        transformer,
        error_unit: str = "mm",
        enable_realtime_plot: bool = False,
        plot_window_name: str = "Cumulative Error Plot"
    ):
        """
        Initialize the Metrics calculator.

        Args:
            calib_data: Camera calibration data (CalibData object)
            transformer: PyramidTransformer object for 3D transformations
            error_unit: Unit for 3D errors - "mm" or "m" (default: "mm")
            enable_realtime_plot: Enable real-time plotting window (default: False)
            plot_window_name: Name for the plot window (default: "Cumulative Error Plot")
        """
        self.calib_data = calib_data
        self.transformer = transformer
        self.error_unit = error_unit
        self.unit_scale = 1000.0 if error_unit == "mm" else 1.0

        # Accumulate errors across frames
        self.pixel_errors: List[float] = []  # Per-frame mean pixel errors
        self.spatial_errors: List[float] = []  # Per-frame mean 3D errors
        self.all_pixel_errors: List[float] = []  # All individual point errors (pixels)
        self.all_spatial_errors: List[float] = []  # All individual point errors (3D)

        # Per-point error tracking
        self.per_point_pixel_errors: Dict[int, List[float]] = {}
        self.per_point_spatial_errors: Dict[int, List[float]] = {}

        # Frame count
        self.frame_count = 0

        # Real-time plotting
        self.enable_realtime_plot = enable_realtime_plot
        self.plot_window_name = plot_window_name

        if self.enable_realtime_plot:
            # Initialize matplotlib figure
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
            self.fig.suptitle('Cumulative Error Over Time', fontsize=14, fontweight='bold')

            # Configure axes
            self.ax1.set_xlabel('Frame')
            self.ax1.set_ylabel('Pixel Error (px)')
            self.ax1.set_title('Mean Pixel Error (Cumulative)')
            self.ax1.grid(True, alpha=0.3)

            self.ax2.set_xlabel('Frame')
            self.ax2.set_ylabel(f'3D Error ({self.error_unit})')
            self.ax2.set_title(f'Mean 3D Error (Cumulative) [{self.error_unit}]')
            self.ax2.grid(True, alpha=0.3)

            # Initialize empty line plots
            self.line_pixel, = self.ax1.plot([], [], 'b-', linewidth=2, label='Mean Pixel Error')
            self.line_spatial, = self.ax2.plot([], [], 'r-', linewidth=2, label='Mean 3D Error')

            self.ax1.legend()
            self.ax2.legend()

            plt.tight_layout()
            plt.ion()  # Interactive mode
            plt.show(block=False)

    def compute_pixel_error(
        self,
        keypoints: List[Dict],
        reprojected_points_2d: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute pixel-space error between keypoints and reprojected points.

        Args:
            keypoints: List of keypoint dicts with keys: id, x, y, visibility
            reprojected_points_2d: Reprojected 2D points (Nx2 array)

        Returns:
            Tuple of (per_point_errors, mean_error)
            - per_point_errors: Array of Euclidean distances for each point
            - mean_error: Mean pixel error across all visible points
        """
        errors = []
        per_point = []

        for kp in keypoints:
            if kp.get('visibility', 2) > 0:  # Only visible keypoints
                kp_id = kp['id']

                # Check if this point ID exists in reprojected points
                if kp_id < len(reprojected_points_2d):
                    kp_pos = np.array([kp['x'], kp['y']])
                    reproj_pos = reprojected_points_2d[kp_id]

                    # Euclidean distance
                    error = np.linalg.norm(kp_pos - reproj_pos)
                    errors.append(error)
                    per_point.append((kp_id, error))

                    # Track per-point errors
                    if kp_id not in self.per_point_pixel_errors:
                        self.per_point_pixel_errors[kp_id] = []
                    self.per_point_pixel_errors[kp_id].append(error)

        per_point_errors = np.array([e for _, e in per_point])
        mean_error = np.mean(errors) if errors else 0.0

        return per_point_errors, mean_error

    def compute_spatial_error(
        self,
        keypoints: List[Dict],
        points_pyramid_frame: np.ndarray,
        frame_id: int,
        rb_data: Dict[str, Any],
        reference_frame: str = "optitrack"
    ) -> Tuple[np.ndarray, float]:
        """
        Compute 3D spatial error using point-to-ray distance.

        This measures the perpendicular distance from the true 3D point to the
        ray cast from the detected 2D keypoint through the camera center.

        This is the CORRECT approach for single-view error measurement, as it
        avoids the ambiguity of depth estimation from a single 2D observation.

        Args:
            keypoints: List of keypoint dicts with keys: id, x, y, visibility
            points_pyramid_frame: 3D points in pyramid frame (Nx3 array, meters)
            frame_id: Current frame index
            rb_data: Rigid body tracking data
            reference_frame: "optitrack" or "pyramid" frame for comparison

        Returns:
            Tuple of (per_point_errors, mean_error)
            - per_point_errors: Array of point-to-ray distances for each point
            - mean_error: Mean 3D error across all visible points
        """
        errors = []
        per_point = []

        # Get transformation matrices
        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
        RT = np.linalg.inv(T_World_Lens @ self.calib_data.RT)

        # Get camera center in world frame
        # Camera center is at the origin in camera frame
        cam_center_cam = np.array([0, 0, 0, 1])
        cam_center_world = np.linalg.inv(RT) @ cam_center_cam
        cam_center_world = cam_center_world[:3]

        for kp in keypoints:
            if kp.get('visibility', 2) > 0:  # Only visible keypoints
                kp_id = kp['id']

                # Check if this point ID exists in pyramid points
                if kp_id < len(points_pyramid_frame):
                    # Get the true 3D position in pyramid frame
                    true_point_pyramid = points_pyramid_frame[kp_id]

                    # Transform true point to world frame
                    true_point_world_hom = T_World_Pyramid @ np.append(true_point_pyramid, 1.0)
                    true_point_world = true_point_world_hom[:3]

                    # Get ray direction from 2D keypoint
                    kp_2d = np.array([[kp['x'], kp['y']]], dtype=np.float32)

                    # Undistort and normalize the keypoint
                    kp_norm = cv2.undistortPoints(
                        kp_2d,
                        self.calib_data.K,
                        self.calib_data.dist_coeffs
                    ).reshape(-1)

                    # Create ray direction in camera frame (normalized)
                    ray_dir_cam = np.array([kp_norm[0], kp_norm[1], 1.0, 0.0])

                    # Transform ray direction to world frame
                    ray_dir_world_hom = np.linalg.inv(RT) @ ray_dir_cam
                    ray_dir_world = ray_dir_world_hom[:3]
                    ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)  # Normalize

                    # Compute point-to-ray distance
                    # Formula: dist = ||cross(v, w)|| / ||v||
                    # where v = ray direction, w = vector from ray origin to point
                    w = true_point_world - cam_center_world

                    # Cross product gives perpendicular distance
                    cross_prod = np.cross(ray_dir_world, w)
                    distance = np.linalg.norm(cross_prod)  # ray_dir_world is already normalized

                    # Convert to desired unit
                    error_scaled = distance * self.unit_scale  # Convert to mm or keep in m

                    errors.append(error_scaled)
                    per_point.append((kp_id, error_scaled))

                    # Track per-point errors
                    if kp_id not in self.per_point_spatial_errors:
                        self.per_point_spatial_errors[kp_id] = []
                    self.per_point_spatial_errors[kp_id].append(error_scaled)

        per_point_errors = np.array([e for _, e in per_point])/100
        mean_error = np.mean(errors)/100 if errors else 0.0

        return per_point_errors, mean_error

    def update_frame_errors(
        self,
        keypoints: List[Dict],
        reprojected_points_2d: np.ndarray,
        points_pyramid_frame: np.ndarray,
        frame_id: int,
        rb_data: Dict[str, Any],
        reference_frame: str = "optitrack"
    ) -> Dict[str, Any]:
        """
        Compute and accumulate errors for the current frame.

        Args:
            keypoints: List of keypoint dicts
            reprojected_points_2d: Reprojected 2D points (Nx2)
            points_pyramid_frame: 3D points in pyramid frame (Nx3, meters)
            frame_id: Current frame index
            rb_data: Rigid body tracking data
            reference_frame: "optitrack" or "pyramid"

        Returns:
            Dictionary containing all error statistics for this frame
        """
        # Compute pixel errors
        pixel_errors_per_point, mean_pixel_error = self.compute_pixel_error(
            keypoints,
            reprojected_points_2d
        )

        # Compute spatial errors
        spatial_errors_per_point, mean_spatial_error = self.compute_spatial_error(
            keypoints,
            points_pyramid_frame,
            frame_id,
            rb_data,
            reference_frame
        )

        # Accumulate frame-level statistics
        self.pixel_errors.append(mean_pixel_error)
        self.spatial_errors.append(mean_spatial_error)

        # Accumulate all individual errors
        self.all_pixel_errors.extend(pixel_errors_per_point.tolist())
        self.all_spatial_errors.extend(spatial_errors_per_point.tolist())

        self.frame_count += 1

        # Create matched error pairs (align by keypoint ID)
        matched_errors = []
        for kp in keypoints:
            if kp.get('visibility', 2) > 0:
                kp_id = kp['id']
                if kp_id < len(reprojected_points_2d) and kp_id < len(points_pyramid_frame):
                    # Find corresponding errors
                    pixel_err = None
                    spatial_err = None

                    # Search in computed errors
                    for i, kp2 in enumerate(keypoints):
                        if kp2['id'] == kp_id and i < len(pixel_errors_per_point):
                            pixel_err = pixel_errors_per_point[i]
                            break

                    for i, kp2 in enumerate(keypoints):
                        if kp2['id'] == kp_id and i < len(spatial_errors_per_point):
                            spatial_err = spatial_errors_per_point[i]
                            break

                    if pixel_err is not None and spatial_err is not None:
                        matched_errors.append({
                            'id': kp_id,
                            'pixel_error': pixel_err,
                            'spatial_error': spatial_err
                        })

        return {
            'frame_id': frame_id,
            'mean_pixel_error': mean_pixel_error,
            'mean_spatial_error': mean_spatial_error,
            'matched_errors': matched_errors,
            'num_points': len(matched_errors)
        }

    def update_realtime_plot(self) -> None:
        """
        Update the real-time plot with cumulative error data.

        This computes the cumulative mean at each frame:
        - Frame 1: mean of frame 1
        - Frame 2: mean of frames 1-2
        - Frame 3: mean of frames 1-3
        - etc.
        """
        if not self.enable_realtime_plot or self.frame_count == 0:
            return

        # Compute cumulative means
        frames = list(range(1, self.frame_count + 1))
        cumulative_pixel_means = []
        cumulative_spatial_means = []

        for i in range(1, self.frame_count + 1):
            # Mean of all frames up to and including frame i
            cumulative_pixel_means.append(np.mean(self.pixel_errors[:i]))
            cumulative_spatial_means.append(np.mean(self.spatial_errors[:i]))

        # Update plot data
        self.line_pixel.set_data(frames, cumulative_pixel_means)
        self.line_spatial.set_data(frames, cumulative_spatial_means)

        # Auto-scale axes
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Small pause to allow GUI update

    def get_plot_as_image(self, width: int = 800, height: int = 600) -> Optional[np.ndarray]:
        """
        Get the current plot as an OpenCV image array.

        Args:
            width: Desired width of output image
            height: Desired height of output image

        Returns:
            BGR image array suitable for cv2.imshow, or None if plotting disabled
        """
        if not self.enable_realtime_plot:
            return None

        # Set figure size
        dpi = 100
        self.fig.set_size_inches(width / dpi, height / dpi)

        # Draw to canvas
        self.fig.canvas.draw()

        # Get RGB buffer
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

        # Resize if needed
        if img_bgr.shape[0] != height or img_bgr.shape[1] != width:
            img_bgr = cv2.resize(img_bgr, (width, height))

        return img_bgr

    def draw_errors_on_frame(
        self,
        frame: np.ndarray,
        frame_errors: Dict[str, Any],
        keypoints: List[Dict],
        reprojected_points_2d: np.ndarray,
        show_per_point: bool = True,
        show_frame_stats: bool = True,
        show_cumulative_stats: bool = True
    ) -> None:
        """
        Draw error information on the video frame.

        Args:
            frame: Video frame to draw on
            frame_errors: Error statistics from update_frame_errors()
            keypoints: List of keypoint dicts
            reprojected_points_2d: Reprojected 2D points (Nx2)
            show_per_point: Whether to show errors for each point
            show_frame_stats: Whether to show frame-level statistics
            show_cumulative_stats: Whether to show cumulative video statistics
        """
        h, w = frame.shape[:2]

        # Draw per-point errors
        if show_per_point:
            for err_data in frame_errors['matched_errors']:
                kp_id = err_data['id']

                # Find the 2D position to draw near
                if kp_id < len(reprojected_points_2d):
                    x, y = reprojected_points_2d[kp_id]
                    x_int, y_int = int(round(x)), int(round(y))

                    # Check if point is in frame
                    if 0 <= x_int < w and 0 <= y_int < h:
                        # Draw error text near the point
                        error_text = f"{err_data['pixel_error']:.1f}px"
                        spatial_text = f"{err_data['spatial_error']:.2f}{self.error_unit}"

                        # Position text below the point
                        text_y = y_int + 25

                        # Draw pixel error
                        cv2.putText(
                            frame,
                            error_text,
                            (x_int + 10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 0),  # Cyan
                            1
                        )

                        # Draw spatial error
                        cv2.putText(
                            frame,
                            spatial_text,
                            (x_int + 10, text_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 0, 255),  # Magenta
                            1
                        )

                        # Draw line connecting keypoint and reprojection
                        kp_x = int(round(keypoints[kp_id]['x']))
                        kp_y = int(round(keypoints[kp_id]['y']))
                        cv2.line(
                            frame,
                            (kp_x, kp_y),
                            (x_int, y_int),
                            (0, 255, 255),  # Yellow
                            1
                        )

        # Draw frame-level statistics
        if show_frame_stats:
            frame_box_y = 100
            frame_box_height = 60

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (10, frame_box_y),
                (350, frame_box_y + frame_box_height),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # Frame statistics
            cv2.putText(
                frame,
                f"Frame {frame_errors['frame_id']} - {frame_errors['num_points']} points",
                (15, frame_box_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                f"Mean pixel error: {frame_errors['mean_pixel_error']:.2f} px",
                (15, frame_box_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

            cv2.putText(
                frame,
                f"Mean 3D error: {frame_errors['mean_spatial_error']:.3f} {self.error_unit}",
                (15, frame_box_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1
            )

        # Draw cumulative statistics
        if show_cumulative_stats and self.frame_count > 0:
            cum_box_y = h - 110
            cum_box_height = 100

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (10, cum_box_y),
                (400, cum_box_y + cum_box_height),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # Cumulative statistics
            mean_pixel_cumulative = np.mean(self.all_pixel_errors) if self.all_pixel_errors else 0.0
            std_pixel_cumulative = np.std(self.all_pixel_errors) if self.all_pixel_errors else 0.0
            mean_spatial_cumulative = np.mean(self.all_spatial_errors) if self.all_spatial_errors else 0.0
            std_spatial_cumulative = np.std(self.all_spatial_errors) if self.all_spatial_errors else 0.0

            cv2.putText(
                frame,
                f"CUMULATIVE ({self.frame_count} frames, {len(self.all_pixel_errors)} points)",
                (15, cum_box_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pixel error: {mean_pixel_cumulative:.2f} +/- {std_pixel_cumulative:.2f} px",
                (15, cum_box_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

            cv2.putText(
                frame,
                f"3D error: {mean_spatial_cumulative:.3f} +/- {std_spatial_cumulative:.3f} {self.error_unit}",
                (15, cum_box_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1
            )

            # Min/Max
            if self.all_pixel_errors:
                cv2.putText(
                    frame,
                    f"Range: [{np.min(self.all_pixel_errors):.1f}, {np.max(self.all_pixel_errors):.1f}] px",
                    (15, cum_box_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1
                )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the entire video sequence.

        Returns:
            Dictionary containing comprehensive statistics
        """
        stats = {
            'total_frames': self.frame_count,
            'total_points': len(self.all_pixel_errors),
            'pixel_error': {
                'mean': np.mean(self.all_pixel_errors) if self.all_pixel_errors else 0.0,
                'std': np.std(self.all_pixel_errors) if self.all_pixel_errors else 0.0,
                'min': np.min(self.all_pixel_errors) if self.all_pixel_errors else 0.0,
                'max': np.max(self.all_pixel_errors) if self.all_pixel_errors else 0.0,
                'median': np.median(self.all_pixel_errors) if self.all_pixel_errors else 0.0,
            },
            'spatial_error': {
                'mean': np.mean(self.all_spatial_errors) if self.all_spatial_errors else 0.0,
                'std': np.std(self.all_spatial_errors) if self.all_spatial_errors else 0.0,
                'min': np.min(self.all_spatial_errors) if self.all_spatial_errors else 0.0,
                'max': np.max(self.all_spatial_errors) if self.all_spatial_errors else 0.0,
                'median': np.median(self.all_spatial_errors) if self.all_spatial_errors else 0.0,
            },
            'error_unit': self.error_unit,
            'per_point_statistics': {}
        }

        # Per-point statistics
        for point_id in self.per_point_pixel_errors.keys():
            pixel_errs = self.per_point_pixel_errors[point_id]
            spatial_errs = self.per_point_spatial_errors.get(point_id, [])

            stats['per_point_statistics'][point_id] = {
                'pixel_error': {
                    'mean': np.mean(pixel_errs),
                    'std': np.std(pixel_errs),
                    'count': len(pixel_errs)
                },
                'spatial_error': {
                    'mean': np.mean(spatial_errs) if spatial_errs else 0.0,
                    'std': np.std(spatial_errs) if spatial_errs else 0.0,
                    'count': len(spatial_errs)
                }
            }

        return stats

    def save_statistics(self, output_path: Path) -> None:
        """
        Save statistics to a JSON file.

        Args:
            output_path: Path to save the JSON file
        """
        stats = self.get_summary_statistics()

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Statistics saved to {output_path}")

    def close_plot(self) -> None:
        """
        Close the real-time plot window and clean up resources.
        """
        if self.enable_realtime_plot:
            plt.close(self.fig)

    def print_summary(self) -> None:
        """
        Print a summary of the error statistics to console.
        """
        stats = self.get_summary_statistics()

        print("\n" + "=" * 70)
        print("ERROR STATISTICS SUMMARY")
        print("=" * 70)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Total points measured: {stats['total_points']}")
        print()
        print("PIXEL ERROR (2D Reprojection):")
        print(f"  Mean:   {stats['pixel_error']['mean']:.3f} px")
        print(f"  Std:    {stats['pixel_error']['std']:.3f} px")
        print(f"  Median: {stats['pixel_error']['median']:.3f} px")
        print(f"  Range:  [{stats['pixel_error']['min']:.3f}, {stats['pixel_error']['max']:.3f}] px")
        print()
        print(f"3D SPATIAL ERROR ({stats['error_unit']}):")
        print(f"  Mean:   {stats['spatial_error']['mean']:.4f} {stats['error_unit']}")
        print(f"  Std:    {stats['spatial_error']['std']:.4f} {stats['error_unit']}")
        print(f"  Median: {stats['spatial_error']['median']:.4f} {stats['error_unit']}")
        print(f"  Range:  [{stats['spatial_error']['min']:.4f}, {stats['spatial_error']['max']:.4f}] {stats['error_unit']}")
        print()
        print(f"Per-point statistics available for {len(stats['per_point_statistics'])} points")
        print("=" * 70)