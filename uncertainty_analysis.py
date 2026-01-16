"""
Uncertainty and Accuracy Analysis Pipeline for Frame Transformations

This module analyzes error propagation through the transformation chain:
1. 3D Model → OptiTrack frame (SVD fitting error)
2. OptiTrack → Camera frame (extrinsics uncertainty)
3. Camera projection (intrinsics uncertainty)
4. Overall reprojection error

Each stage quantifies:
- Geometric errors (translations, rotations)
- Statistical measures (mean, std, max errors)
- Error propagation to final 2D projection
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import cv2

from pyramid_transformer import PyramidTransformer
from calib_data import CalibData


@dataclass
class TransformationUncertainty:
    """Stores uncertainty metrics for a single transformation stage."""
    stage_name: str
    
    # Geometric errors
    translation_error_m: Optional[float] = None
    rotation_error_deg: Optional[float] = None
    
    # Point-wise errors
    point_errors_3d_m: Optional[npt.NDArray[np.float64]] = None  # N points
    mean_error_3d_m: Optional[float] = None
    std_error_3d_m: Optional[float] = None
    max_error_3d_m: Optional[float] = None
    
    # 2D projection errors (if applicable)
    reprojection_errors_px: Optional[npt.NDArray[np.float64]] = None  # N points
    mean_reprojection_error_px: Optional[float] = None
    std_reprojection_error_px: Optional[float] = None
    max_reprojection_error_px: Optional[float] = None
    
    # Additional metrics
    condition_number: Optional[float] = None
    residual_norm: Optional[float] = None
    
    # Metadata
    num_points: int = 0
    notes: str = ""


@dataclass
class UncertaintyReport:
    """Complete uncertainty analysis report for the transformation pipeline."""
    
    # Stage-wise uncertainties
    model_to_optitrack: Optional[TransformationUncertainty] = None
    optitrack_to_camera: Optional[TransformationUncertainty] = None
    camera_projection: Optional[TransformationUncertainty] = None
    
    # End-to-end metrics
    total_reprojection_error_px: Optional[float] = None
    error_propagation_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Per-frame analysis (if applicable)
    frame_uncertainties: Dict[int, Dict[str, TransformationUncertainty]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON serialization."""
        result = {}
        
        for stage_name, stage_data in [
            ("model_to_optitrack", self.model_to_optitrack),
            ("optitrack_to_camera", self.optitrack_to_camera),
            ("camera_projection", self.camera_projection)
        ]:
            if stage_data is not None:
                result[stage_name] = {
                    "translation_error_m": stage_data.translation_error_m,
                    "rotation_error_deg": stage_data.rotation_error_deg,
                    "mean_error_3d_m": stage_data.mean_error_3d_m,
                    "std_error_3d_m": stage_data.std_error_3d_m,
                    "max_error_3d_m": stage_data.max_error_3d_m,
                    "mean_reprojection_error_px": stage_data.mean_reprojection_error_px,
                    "std_reprojection_error_px": stage_data.std_reprojection_error_px,
                    "max_reprojection_error_px": stage_data.max_reprojection_error_px,
                    "condition_number": stage_data.condition_number,
                    "residual_norm": stage_data.residual_norm,
                    "num_points": stage_data.num_points,
                    "notes": stage_data.notes
                }
        
        result["total_reprojection_error_px"] = self.total_reprojection_error_px
        result["error_propagation_analysis"] = self.error_propagation_analysis
        
        return result
    
    def save_json(self, filepath: Path) -> None:
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Uncertainty report saved to: {filepath}")


class UncertaintyAnalyzer:
    """
    Analyzes uncertainty at each stage of the transformation pipeline.
    """
    
    def __init__(
        self,
        transformer: PyramidTransformer,
        calib_data: CalibData,
        verbose: bool = True
    ):
        """
        Initialize the uncertainty analyzer.
        
        Args:
            transformer: PyramidTransformer with computed transformations
            calib_data: Camera calibration data
            verbose: Whether to print detailed information
        """
        self.transformer = transformer
        self.calib_data = calib_data
        self.verbose = verbose
        
        self.report = UncertaintyReport()
    
    def analyze_model_to_optitrack(
        self,
        marker_positions_m: Dict[str, npt.NDArray[np.float64]],
        matching: Dict[str, int]
    ) -> TransformationUncertainty:
        """
        Analyze uncertainty in the 3D model → OptiTrack transformation.
        
        This stage uses SVD to fit the pyramid's marker constellation to OptiTrack markers.
        
        Args:
            marker_positions_m: OptiTrack marker positions {marker_name: [x, y, z]}
            matching: Dictionary mapping marker names to pyramid point IDs
            
        Returns:
            TransformationUncertainty object with SVD fitting errors
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STAGE 1: 3D Model → OptiTrack Frame")
            print("="*70)
        
        uncertainty = TransformationUncertainty(
            stage_name="model_to_optitrack",
            num_points=len(matching)
        )
        
        # Get the constellation points in pyramid frame
        constellation_ids = list(matching.values())
        constellation_points_pyramid = self.transformer.points_m[constellation_ids]
        
        # Transform to pyramid frame (world → pyramid)
        R_pyramid = self.transformer.R_pyramid
        pyramid_origin = self.transformer.pyramid_origin_m
        constellation_pyramid_frame = (R_pyramid.T @ (constellation_points_pyramid - pyramid_origin).T).T
        
        # Get corresponding OptiTrack positions
        optitrack_positions = np.array([marker_positions_m[name] for name in matching.keys()])
        
        # Compute the fitted OptiTrack positions using the computed transformation
        constellation_optitrack_fitted = self.transformer.transform_pyramid_to_optitrack(
            constellation_pyramid_frame
        )
        
        # Calculate 3D errors
        point_errors_3d = np.linalg.norm(optitrack_positions - constellation_optitrack_fitted, axis=1)
        
        uncertainty.point_errors_3d_m = point_errors_3d
        uncertainty.mean_error_3d_m = float(np.mean(point_errors_3d))
        uncertainty.std_error_3d_m = float(np.std(point_errors_3d))
        uncertainty.max_error_3d_m = float(np.max(point_errors_3d))
        
        # Compute SVD condition number (indicates numerical stability)
        # Center the points
        centered_pyramid = constellation_pyramid_frame - np.mean(constellation_pyramid_frame, axis=0)
        centered_optitrack = optitrack_positions - np.mean(optitrack_positions, axis=0)
        
        # SVD of covariance matrix
        H = centered_pyramid.T @ centered_optitrack
        U, S, Vt = np.linalg.svd(H)
        
        uncertainty.condition_number = float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf')
        uncertainty.residual_norm = float(np.linalg.norm(optitrack_positions - constellation_optitrack_fitted))
        
        uncertainty.notes = f"SVD-based rigid transformation. {len(matching)} markers used for fitting."
        
        if self.verbose:
            print(f"  SVD Fitting Quality:")
            print(f"    Mean 3D error:      {uncertainty.mean_error_3d_m*1000:.3f} mm")
            print(f"    Std 3D error:       {uncertainty.std_error_3d_m*1000:.3f} mm")
            print(f"    Max 3D error:       {uncertainty.max_error_3d_m*1000:.3f} mm")
            print(f"    Condition number:   {uncertainty.condition_number:.2f}")
            print(f"    Residual norm:      {uncertainty.residual_norm*1000:.3f} mm")
        
        self.report.model_to_optitrack = uncertainty
        return uncertainty
    
    def analyze_optitrack_to_camera(
        self,
        rb_data: Dict[str, Any],
        frame_id: int,
        test_points_pyramid: npt.NDArray[np.float64]
    ) -> TransformationUncertainty:
        """
        Analyze uncertainty in OptiTrack → Camera frame transformation (extrinsics).
        
        This includes:
        - OptiTrack tracking noise
        - Extrinsic calibration accuracy (RT matrix)
        
        Args:
            rb_data: Rigid body tracking data
            frame_id: Frame to analyze
            test_points_pyramid: Points in pyramid frame to transform (Nx3)
            
        Returns:
            TransformationUncertainty object with extrinsics errors
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STAGE 2: OptiTrack → Camera Frame (Extrinsics)")
            print("="*70)
        
        uncertainty = TransformationUncertainty(
            stage_name="optitrack_to_camera",
            num_points=len(test_points_pyramid)
        )
        
        # Get transformations
        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
        
        # Transform points: pyramid → optitrack → world → camera
        points_optitrack = self.transformer.transform_pyramid_to_optitrack(test_points_pyramid)
        
        # To world frame
        n_points = points_optitrack.shape[0]
        points_hom = np.hstack([points_optitrack, np.ones((n_points, 1))])
        points_world = (T_World_Pyramid @ points_hom.T).T[:, :3]
        
        # To camera frame
        RT = np.linalg.inv(T_World_Lens @ self.calib_data.RT)
        points_camera = (RT[:3, :3] @ points_world.T + RT[:3, 3:4]).T
        
        # Analyze OptiTrack tracking uncertainty
        # Get marker positions and their typical noise
        pyramid_markers = rb_data["Pyramid_RB"][frame_id].data.marker_positions
        if pyramid_markers:
            marker_array = np.array(list(pyramid_markers.values()))
            # Estimate tracking noise from marker spread (heuristic)
            marker_centroid = np.mean(marker_array, axis=0)
            marker_deviations = np.linalg.norm(marker_array - marker_centroid, axis=1)
            tracking_noise_estimate = np.std(marker_deviations)
        else:
            tracking_noise_estimate = 0.001  # 1mm default estimate
        
        uncertainty.translation_error_m = tracking_noise_estimate
        uncertainty.notes = f"OptiTrack tracking noise estimate: {tracking_noise_estimate*1000:.3f} mm. "
        uncertainty.notes += f"Extrinsics from Lens_RB and RT matrix."
        
        # Estimate rotation uncertainty from quaternion noise (typical ~0.5 degrees)
        uncertainty.rotation_error_deg = 0.5
        
        if self.verbose:
            print(f"  Extrinsics Uncertainty:")
            print(f"    OptiTrack tracking noise:  {tracking_noise_estimate*1000:.3f} mm")
            print(f"    Estimated rotation noise:  {uncertainty.rotation_error_deg:.2f}°")
            print(f"    RT matrix condition:       {np.linalg.cond(RT):.2f}")
        
        self.report.optitrack_to_camera = uncertainty
        return uncertainty
    
    def analyze_camera_projection(
        self,
        points_3d_camera: npt.NDArray[np.float64],
        points_2d_observed: Optional[npt.NDArray[np.float64]] = None,
        R_cor: Optional[npt.NDArray[np.float64]] = None
    ) -> TransformationUncertainty:
        """
        Analyze uncertainty in camera projection (intrinsics).
        
        This includes:
        - Focal length uncertainty
        - Principal point uncertainty
        - Distortion model errors
        
        Args:
            points_3d_camera: Points in camera frame (Nx3)
            points_2d_observed: Observed 2D points for comparison (Nx2), optional
            R_cor: Rotation correction matrix (3x3), optional
            
        Returns:
            TransformationUncertainty object with projection errors
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STAGE 3: Camera Projection (Intrinsics)")
            print("="*70)
        
        uncertainty = TransformationUncertainty(
            stage_name="camera_projection",
            num_points=len(points_3d_camera)
        )
        
        # Project points
        rvec = np.zeros(3)  # Identity rotation (already in camera frame)
        tvec = np.zeros(3)  # No translation
        
        points_2d_projected, _ = cv2.projectPoints(
            points_3d_camera,
            rvec,
            tvec,
            self.calib_data.K,
            self.calib_data.dist_coeffs
        )
        points_2d_projected = points_2d_projected.reshape(-1, 2)
        
        # Apply rotation correction if provided
        if R_cor is not None:
            points_2d_hom = np.hstack([points_2d_projected, np.ones((len(points_2d_projected), 1))]).T
            points_2d_corrected = (R_cor @ points_2d_hom)[:2, :].T
            points_2d_projected = points_2d_corrected
        
        # Calculate reprojection errors if ground truth provided
        if points_2d_observed is not None:
            reprojection_errors = np.linalg.norm(points_2d_projected - points_2d_observed, axis=1)
            
            uncertainty.reprojection_errors_px = reprojection_errors
            uncertainty.mean_reprojection_error_px = float(np.mean(reprojection_errors))
            uncertainty.std_reprojection_error_px = float(np.std(reprojection_errors))
            uncertainty.max_reprojection_error_px = float(np.max(reprojection_errors))
        
        # Analyze intrinsics uncertainty
        # Condition number of K matrix
        uncertainty.condition_number = float(np.linalg.cond(self.calib_data.K))
        
        # Estimate focal length uncertainty (typical 0.5-1% for good calibration)
        fx = self.calib_data.K[0, 0]
        fy = self.calib_data.K[1, 1]
        focal_length_uncertainty_pct = 0.5  # Assume 0.5% uncertainty
        
        uncertainty.notes = f"Focal length: fx={fx:.1f}, fy={fy:.1f}. "
        uncertainty.notes += f"Estimated focal length uncertainty: {focal_length_uncertainty_pct}%. "
        uncertainty.notes += f"Distortion coeffs: {self.calib_data.dist_coeffs.ravel()}"
        
        if self.verbose:
            print(f"  Intrinsics Uncertainty:")
            print(f"    Focal length (fx, fy):     ({fx:.1f}, {fy:.1f}) px")
            print(f"    Principal point (cx, cy):  {self.calib_data.camera_model.get_center()}")
            print(f"    K matrix condition number: {uncertainty.condition_number:.2f}")
            print(f"    Distortion coefficients:   {self.calib_data.dist_coeffs.ravel()}")
            
            if points_2d_observed is not None:
                print(f"  Reprojection Errors:")
                print(f"    Mean error:  {uncertainty.mean_reprojection_error_px:.2f} px")
                print(f"    Std error:   {uncertainty.std_reprojection_error_px:.2f} px")
                print(f"    Max error:   {uncertainty.max_reprojection_error_px:.2f} px")
        
        self.report.camera_projection = uncertainty
        return uncertainty
    
    def analyze_full_pipeline(
        self,
        rb_data: Dict[str, Any],
        marker_positions_m: Dict[str, npt.NDArray[np.float64]],
        matching: Dict[str, int],
        frame_id: int = 0,
        keypoints_2d: Optional[npt.NDArray[np.float64]] = None,
        R_cor: Optional[npt.NDArray[np.float64]] = None
    ) -> UncertaintyReport:
        """
        Perform complete uncertainty analysis through the entire pipeline.
        
        Args:
            rb_data: Rigid body tracking data
            marker_positions_m: OptiTrack marker positions for SVD fitting
            matching: Marker name to pyramid point ID mapping
            frame_id: Frame to analyze
            keypoints_2d: Observed 2D keypoints (Nx2), optional
            R_cor: Rotation correction matrix, optional
            
        Returns:
            Complete uncertainty report
        """
        if self.verbose:
            print("\n" + "="*70)
            print("FULL PIPELINE UNCERTAINTY ANALYSIS")
            print("="*70)
        
        # Stage 1: Model → OptiTrack
        self.analyze_model_to_optitrack(marker_positions_m, matching)
        
        # Get all pyramid points (0-17)
        points_pyramid = self.transformer.get_pyramid_points_in_pyramid_frame()
        
        # Stage 2: OptiTrack → Camera
        self.analyze_optitrack_to_camera(rb_data, frame_id, points_pyramid)
        
        # Stage 3: Camera Projection
        # Transform points through full pipeline for projection analysis
        points_optitrack = self.transformer.transform_pyramid_to_optitrack(points_pyramid)
        
        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        RT = np.linalg.inv(T_World_Lens @ self.calib_data.RT)
        
        # To world frame
        n_points = points_optitrack.shape[0]
        points_hom = np.hstack([points_optitrack, np.ones((n_points, 1))])
        points_world = (T_World_Pyramid @ points_hom.T).T[:, :3]
        
        # To camera frame
        points_camera = (RT[:3, :3] @ points_world.T + RT[:3, 3:4]).T
        
        self.analyze_camera_projection(points_camera, keypoints_2d, R_cor)
        
        # Calculate total end-to-end error if 2D observations provided
        if keypoints_2d is not None and self.report.camera_projection.mean_reprojection_error_px is not None:
            self.report.total_reprojection_error_px = self.report.camera_projection.mean_reprojection_error_px
        
        # Error propagation analysis
        self._analyze_error_propagation()
        
        return self.report
    
    def _analyze_error_propagation(self) -> None:
        """
        Analyze how errors propagate through the pipeline.
        
        Uses first-order error propagation (linearization).
        """
        if self.verbose:
            print("\n" + "="*70)
            print("ERROR PROPAGATION ANALYSIS")
            print("="*70)
        
        propagation = {}
        
        # Stage 1 contribution
        if self.report.model_to_optitrack is not None:
            stage1_error_mm = self.report.model_to_optitrack.mean_error_3d_m * 1000
            propagation["stage1_3d_error_mm"] = stage1_error_mm
            
            if self.verbose:
                print(f"  Stage 1 (Model→OptiTrack): {stage1_error_mm:.3f} mm")
        
        # Stage 2 contribution
        if self.report.optitrack_to_camera is not None:
            stage2_error_mm = self.report.optitrack_to_camera.translation_error_m * 1000
            propagation["stage2_3d_error_mm"] = stage2_error_mm
            
            if self.verbose:
                print(f"  Stage 2 (OptiTrack→Camera): {stage2_error_mm:.3f} mm")
        
        # Combined 3D error (RSS - root sum square)
        if "stage1_3d_error_mm" in propagation and "stage2_3d_error_mm" in propagation:
            combined_3d_error = np.sqrt(
                propagation["stage1_3d_error_mm"]**2 + 
                propagation["stage2_3d_error_mm"]**2
            )
            propagation["combined_3d_error_mm"] = combined_3d_error
            
            if self.verbose:
                print(f"  Combined 3D error (RSS):    {combined_3d_error:.3f} mm")
        
        # Stage 3 contribution (projection)
        if self.report.camera_projection is not None:
            if self.report.camera_projection.mean_reprojection_error_px is not None:
                propagation["projection_error_px"] = self.report.camera_projection.mean_reprojection_error_px
                
                if self.verbose:
                    print(f"  Stage 3 (Projection):       {propagation['projection_error_px']:.2f} px")
        
        # Estimate 3D→2D propagation factor (depends on depth and focal length)
        # Typical: 1mm @ 1m depth ≈ 1px error for f=1000px
        if "combined_3d_error_mm" in propagation:
            fx = self.calib_data.K[0, 0]
            # Assume typical depth of 1m for estimation
            estimated_depth_m = 1.0
            expected_2d_error_from_3d = (propagation["combined_3d_error_mm"] / 1000) * (fx / estimated_depth_m)
            propagation["expected_2d_error_from_3d_px"] = expected_2d_error_from_3d
            
            if self.verbose:
                print(f"  Expected 2D error from 3D:  {expected_2d_error_from_3d:.2f} px (at {estimated_depth_m}m)")
        
        self.report.error_propagation_analysis = propagation
    
    def visualize_uncertainties(
        self,
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Optional[Figure]:
        """
        Create visualization of uncertainties at each stage.
        
        Args:
            save_path: Path to save figure, optional
            show: Whether to display the figure
            
        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Uncertainty Analysis Across Transformation Pipeline", fontsize=16, fontweight='bold')
        
        # Plot 1: 3D errors by stage
        ax1 = axes[0, 0]
        stages = []
        errors_3d_mm = []
        error_bars = []
        
        if self.report.model_to_optitrack is not None:
            stages.append("Model→\nOptiTrack")
            errors_3d_mm.append(self.report.model_to_optitrack.mean_error_3d_m * 1000)
            error_bars.append(self.report.model_to_optitrack.std_error_3d_m * 1000)
        
        if self.report.optitrack_to_camera is not None:
            stages.append("OptiTrack→\nCamera")
            errors_3d_mm.append(self.report.optitrack_to_camera.translation_error_m * 1000)
            error_bars.append(0)  # No std available for this estimate
        
        if stages:
            x_pos = np.arange(len(stages))
            ax1.bar(x_pos, errors_3d_mm, yerr=error_bars, capsize=5, 
                   color=['#2E86AB', '#A23B72'], alpha=0.7, edgecolor='black')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(stages)
            ax1.set_ylabel("3D Error (mm)", fontsize=12)
            ax1.set_title("3D Geometric Errors", fontsize=13, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Per-point 3D errors (Stage 1)
        ax2 = axes[0, 1]
        if self.report.model_to_optitrack is not None and self.report.model_to_optitrack.point_errors_3d_m is not None:
            errors = self.report.model_to_optitrack.point_errors_3d_m * 1000
            point_ids = np.arange(len(errors))
            ax2.scatter(point_ids, errors, c=errors, cmap='YlOrRd', s=100, edgecolor='black', linewidth=1)
            ax2.axhline(np.mean(errors), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f} mm')
            ax2.set_xlabel("Point ID", fontsize=12)
            ax2.set_ylabel("3D Error (mm)", fontsize=12)
            ax2.set_title("Per-Point SVD Fitting Errors", fontsize=13, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # Plot 3: 2D reprojection errors
        ax3 = axes[1, 0]
        if self.report.camera_projection is not None and self.report.camera_projection.reprojection_errors_px is not None:
            errors = self.report.camera_projection.reprojection_errors_px
            point_ids = np.arange(len(errors))
            ax3.scatter(point_ids, errors, c=errors, cmap='viridis', s=100, edgecolor='black', linewidth=1)
            ax3.axhline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f} px')
            ax3.set_xlabel("Point ID", fontsize=12)
            ax3.set_ylabel("Reprojection Error (px)", fontsize=12)
            ax3.set_title("2D Reprojection Errors", fontsize=13, fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
        
        # Plot 4: Error propagation summary
        ax4 = axes[1, 1]
        if self.report.error_propagation_analysis:
            labels = []
            values = []
            colors = []
            
            if "stage1_3d_error_mm" in self.report.error_propagation_analysis:
                labels.append("Stage 1:\n3D Error")
                values.append(self.report.error_propagation_analysis["stage1_3d_error_mm"])
                colors.append('#2E86AB')
            
            if "stage2_3d_error_mm" in self.report.error_propagation_analysis:
                labels.append("Stage 2:\n3D Error")
                values.append(self.report.error_propagation_analysis["stage2_3d_error_mm"])
                colors.append('#A23B72')
            
            if "combined_3d_error_mm" in self.report.error_propagation_analysis:
                labels.append("Combined\n3D Error")
                values.append(self.report.error_propagation_analysis["combined_3d_error_mm"])
                colors.append('#F18F01')
            
            if "projection_error_px" in self.report.error_propagation_analysis:
                # Scale to same units for comparison (convert px to mm equivalent)
                labels.append("Final 2D\nError (px)")
                values.append(self.report.error_propagation_analysis["projection_error_px"])
                colors.append('#C73E1D')
            
            if labels:
                x_pos = np.arange(len(labels))
                bars = ax4.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(labels, fontsize=10)
                ax4.set_ylabel("Error Magnitude", fontsize=12)
                ax4.set_title("Error Propagation Summary", fontsize=13, fontweight='bold')
                ax4.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.2f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Uncertainty visualization saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of all uncertainties."""
        print("\n" + "="*70)
        print("UNCERTAINTY ANALYSIS SUMMARY")
        print("="*70)
        
        # Stage 1
        if self.report.model_to_optitrack is not None:
            u = self.report.model_to_optitrack
            print("\n📊 Stage 1: 3D Model → OptiTrack Frame")
            print(f"  Method: SVD rigid transformation")
            print(f"  Points used: {u.num_points}")
            print(f"  Mean 3D error:       {u.mean_error_3d_m*1000:.3f} ± {u.std_error_3d_m*1000:.3f} mm")
            print(f"  Max 3D error:        {u.max_error_3d_m*1000:.3f} mm")
            print(f"  Condition number:    {u.condition_number:.2f}")
            print(f"  Quality: {'✓ Excellent' if u.mean_error_3d_m < 0.002 else '⚠ Acceptable' if u.mean_error_3d_m < 0.005 else '✗ Poor'}")
        
        # Stage 2
        if self.report.optitrack_to_camera is not None:
            u = self.report.optitrack_to_camera
            print("\n📊 Stage 2: OptiTrack → Camera Frame (Extrinsics)")
            print(f"  Translation noise:   {u.translation_error_m*1000:.3f} mm")
            print(f"  Rotation noise:      {u.rotation_error_deg:.2f}°")
            print(f"  Quality: {'✓ Good tracking' if u.translation_error_m < 0.002 else '⚠ Moderate noise'}")
        
        # Stage 3
        if self.report.camera_projection is not None:
            u = self.report.camera_projection
            print("\n📊 Stage 3: Camera Projection (Intrinsics)")
            if u.mean_reprojection_error_px is not None:
                print(f"  Mean reprojection:   {u.mean_reprojection_error_px:.2f} ± {u.std_reprojection_error_px:.2f} px")
                print(f"  Max reprojection:    {u.max_reprojection_error_px:.2f} px")
                print(f"  Quality: {'✓ Excellent' if u.mean_reprojection_error_px < 2 else '⚠ Acceptable' if u.mean_reprojection_error_px < 5 else '✗ Poor'}")
            print(f"  K condition number:  {u.condition_number:.2f}")
        
        # Overall
        if self.report.error_propagation_analysis:
            print("\n📊 Error Propagation Analysis")
            for key, value in self.report.error_propagation_analysis.items():
                print(f"  {key}: {value:.3f}")
        
        print("\n" + "="*70)
