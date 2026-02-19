"""
Uncertainty and Accuracy Analysis Pipeline for Frame Transformations

This module analyzes error propagation through the transformation chain:
1. 3D Model -> OptiTrack frame (SVD fitting error)
2. OptiTrack -> Camera frame (extrinsics: translation + rotation + time sync)
3. Camera projection (intrinsics uncertainty)
4. Overall reprojection error

SENSITIVITY ANALYSIS:
    Instead of heuristic noise estimates, we use perturbation-based analysis.
    For a given frame, we compute the nominal 2D projection, then independently
    perturb each error source and measure the resulting delta in reprojection:

    - delta_translation: Perturb the rigid body position by +/- epsilon in x, y, z
    - delta_orientation: Perturb the rigid body quaternion by small-angle rotations
    - delta_time: Shift the OptiTrack frame index by +/- dt frames (temporal sync error)

VISUALIZATION (UncertaintyPlotter):
    4-panel real-time diagnostic figure with physically meaningful titles:

    Plot 1 (top-left):  INTRINSICS IMPACT
        Sensitivity of reprojection to each camera calibration parameter:
        focal length (fx/fy), principal point (cx/cy), distortion (k1..k3).
        Units: px per unit parameter perturbation.

    Plot 2 (top-right): EXTRINSICS — TRANSLATION SENSITIVITY
        Pixels of reprojection displacement per 1 mm of rigid-body position
        shift, broken down by X / Y / Z axis plus RSS.

    Plot 3 (bottom-left): EXTRINSICS — ROTATION + TIME-SYNC SENSITIVITY
        Pixels per degree of rigid-body rotation (roll / pitch / yaw) and
        pixels per frame of time-sync offset.

    Plot 4 (bottom-right): ERROR BUDGET
        Stacked-area contributions (SVD fitting, translation noise, rotation
        noise, time-sync) forming a predicted total, overlaid with the actual
        measured reprojection error.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from pyramid_transformer import PyramidTransformer
from calib_data import CalibData


# =============================================================================
# Thresholds
# =============================================================================

GOOD_PX   = 2.0
ACCEPT_PX = 5.0


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class TransformationUncertainty:
    """Stores uncertainty metrics for a single transformation stage."""
    stage_name: str

    translation_error_m: Optional[float] = None
    rotation_error_deg: Optional[float] = None

    point_errors_3d_m: Optional[npt.NDArray[np.float64]] = None
    mean_error_3d_m: Optional[float] = None
    std_error_3d_m: Optional[float] = None
    max_error_3d_m: Optional[float] = None

    reprojection_errors_px: Optional[npt.NDArray[np.float64]] = None
    mean_reprojection_error_px: Optional[float] = None
    std_reprojection_error_px: Optional[float] = None
    max_reprojection_error_px: Optional[float] = None

    condition_number: Optional[float] = None
    residual_norm: Optional[float] = None
    num_points: int = 0
    notes: str = ""


@dataclass
class SensitivityResult:
    """Result of a single perturbation-based sensitivity analysis."""
    perturbation_name: str
    perturbation_unit: str
    perturbation_magnitude: float

    displacement_px: Optional[npt.NDArray[np.float64]] = None
    mean_displacement_px: float = 0.0
    max_displacement_px: float = 0.0
    sensitivity: float = 0.0
    sensitivity_unit: str = ""

    def __repr__(self):
        return (f"Sensitivity({self.perturbation_name}: "
                f"{self.sensitivity:.3f} {self.sensitivity_unit}, "
                f"mean_disp={self.mean_displacement_px:.2f}px)")


@dataclass
class ErrorBudget:
    """Error budget showing each error source's contribution to total reprojection error."""
    translation_contribution_px: float = 0.0
    orientation_contribution_px: float = 0.0
    time_sync_contribution_px: float = 0.0
    svd_fitting_contribution_px: float = 0.0

    assumed_translation_noise_mm: float = 0.0
    assumed_orientation_noise_deg: float = 0.0
    assumed_time_sync_noise_frames: float = 0.0

    combined_predicted_px: float = 0.0
    actual_reprojection_px: Optional[float] = None

    sensitivities: Dict[str, SensitivityResult] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        result = {
            "translation_contribution_px": self.translation_contribution_px,
            "orientation_contribution_px": self.orientation_contribution_px,
            "time_sync_contribution_px": self.time_sync_contribution_px,
            "svd_fitting_contribution_px": self.svd_fitting_contribution_px,
            "combined_predicted_px": self.combined_predicted_px,
            "actual_reprojection_px": self.actual_reprojection_px,
            "assumed_noise": {
                "translation_mm": self.assumed_translation_noise_mm,
                "orientation_deg": self.assumed_orientation_noise_deg,
                "time_sync_frames": self.assumed_time_sync_noise_frames,
            },
            "sensitivities": {}
        }
        for name, s in self.sensitivities.items():
            result["sensitivities"][name] = {
                "sensitivity": s.sensitivity,
                "sensitivity_unit": s.sensitivity_unit,
                "mean_displacement_px": s.mean_displacement_px,
                "perturbation_magnitude": s.perturbation_magnitude,
                "perturbation_unit": s.perturbation_unit,
            }
        return result


@dataclass
class UncertaintyReport:
    """Complete uncertainty analysis report for the transformation pipeline."""
    model_to_optitrack: Optional[TransformationUncertainty] = None
    optitrack_to_camera: Optional[TransformationUncertainty] = None
    camera_projection: Optional[TransformationUncertainty] = None

    error_budget: Optional[ErrorBudget] = None

    total_reprojection_error_px: Optional[float] = None
    error_propagation_analysis: Dict[str, Any] = field(default_factory=dict)

    frame_uncertainties: Dict[int, Dict[str, TransformationUncertainty]] = \
        field(default_factory=dict)

    # Cached 3D points in camera frame — used by UncertaintyPlotter for
    # intrinsics sensitivity (populated by analyze_full_pipeline).
    points_3d_camera_last: Optional[npt.NDArray[np.float64]] = None

    # Cached rvec / tvec used for the last projection (identity when points
    # are already in camera frame, which is the case here).
    rvec_last: Optional[npt.NDArray[np.float64]] = None
    tvec_last: Optional[npt.NDArray[np.float64]] = None

    def to_dict(self) -> Dict:
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
        if self.error_budget is not None:
            result["error_budget"] = self.error_budget.to_dict()
        return result

    def save_json(self, filepath: Path) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[OK] Uncertainty report saved to: {filepath}")


# =============================================================================
# Helper: small-angle quaternion perturbation
# =============================================================================

def _quaternion_from_transform(T: np.ndarray) -> np.ndarray:
    """Extract quaternion (w, x, y, z) from a 4x4 homogeneous transform."""
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _small_angle_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues formula for rotation matrix from axis-angle."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def _perturb_transform_translation(T: np.ndarray, delta: np.ndarray) -> np.ndarray:
    T_pert = T.copy()
    T_pert[:3, 3] += delta
    return T_pert


def _perturb_transform_rotation(T: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    T_pert = T.copy()
    dR = _small_angle_rotation_matrix(axis, angle_rad)
    T_pert[:3, :3] = dR @ T[:3, :3]
    return T_pert


# =============================================================================
# Main analyzer class
# =============================================================================

class UncertaintyAnalyzer:
    """
    Analyzes uncertainty at each stage of the transformation pipeline.

    Includes perturbation-based sensitivity analysis for:
    - delta_translation (px/mm)
    - delta_orientation (px/deg)
    - delta_time (px/frame of sync offset)
    """

    def __init__(
        self,
        transformer: PyramidTransformer,
        calib_data: CalibData,
        verbose: bool = True
    ):
        self.transformer = transformer
        self.calib_data = calib_data
        self.verbose = verbose
        self.report = UncertaintyReport()

    # =========================================================================
    # Core projection function
    # =========================================================================

    def _project_points(
        self,
        points_optitrack: np.ndarray,
        T_World_Pyramid: np.ndarray,
        T_World_Lens: np.ndarray,
        R_cor: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Full projection pipeline: OptiTrack -> World -> Camera -> 2D pixels."""
        n = points_optitrack.shape[0]
        RT = np.linalg.inv(T_World_Lens @ self.calib_data.RT)

        pts_hom = np.hstack([points_optitrack, np.ones((n, 1))])
        pts_world = (T_World_Pyramid @ pts_hom.T).T[:, :3]

        proj_2d = cv2.projectPoints(
            pts_world,
            cv2.Rodrigues(RT[:3, :3])[0],
            RT[:3, 3],
            self.calib_data.K,
            self.calib_data.dist_coeffs
        )[0].reshape(-1, 2)

        if R_cor is not None:
            pts_hom_2d = np.hstack([proj_2d, np.ones((n, 1))]).T
            proj_2d = (R_cor @ pts_hom_2d)[:2, :].T

        return proj_2d

    # =========================================================================
    # Stage 1: Model -> OptiTrack (SVD)
    # =========================================================================

    def analyze_model_to_optitrack(
        self,
        marker_positions_m: Dict[str, npt.NDArray[np.float64]],
        matching: Dict[str, int]
    ) -> TransformationUncertainty:
        """Analyze SVD fitting quality."""
        if self.verbose:
            print("\n" + "="*70)
            print("STAGE 1: 3D Model -> OptiTrack Frame (SVD)")
            print("="*70)

        uncertainty = TransformationUncertainty(
            stage_name="model_to_optitrack",
            num_points=len(matching)
        )

        constellation_ids = list(matching.values())
        constellation_points_pyramid = self.transformer.points_m[constellation_ids]
        R_pyramid = self.transformer.R_pyramid
        pyramid_origin = self.transformer.pyramid_origin_m
        constellation_pyramid_frame = (R_pyramid.T @ (constellation_points_pyramid - pyramid_origin).T).T
        optitrack_positions = np.array([marker_positions_m[name] for name in matching.keys()])
        constellation_optitrack_fitted = self.transformer.transform_pyramid_to_optitrack(
            constellation_pyramid_frame
        )

        point_errors_3d = np.linalg.norm(optitrack_positions - constellation_optitrack_fitted, axis=1)
        uncertainty.point_errors_3d_m = point_errors_3d
        uncertainty.mean_error_3d_m = float(np.mean(point_errors_3d))
        uncertainty.std_error_3d_m = float(np.std(point_errors_3d))
        uncertainty.max_error_3d_m = float(np.max(point_errors_3d))

        centered_pyramid = constellation_pyramid_frame - np.mean(constellation_pyramid_frame, axis=0)
        centered_optitrack = optitrack_positions - np.mean(optitrack_positions, axis=0)
        H = centered_pyramid.T @ centered_optitrack
        U, S, Vt = np.linalg.svd(H)
        uncertainty.condition_number = float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf')
        uncertainty.residual_norm = float(np.linalg.norm(optitrack_positions - constellation_optitrack_fitted))
        uncertainty.notes = f"SVD rigid transformation. {len(matching)} markers used."

        if self.verbose:
            print(f"  Mean 3D error:    {uncertainty.mean_error_3d_m*1000:.3f} mm")
            print(f"  Std 3D error:     {uncertainty.std_error_3d_m*1000:.3f} mm")
            print(f"  Max 3D error:     {uncertainty.max_error_3d_m*1000:.3f} mm")
            print(f"  Condition number: {uncertainty.condition_number:.2f}")

        self.report.model_to_optitrack = uncertainty
        return uncertainty

    # =========================================================================
    # Stage 2: OptiTrack -> Camera
    # =========================================================================

    def analyze_optitrack_to_camera(
        self,
        rb_data: Dict[str, Any],
        frame_id: int,
        test_points_pyramid: npt.NDArray[np.float64]
    ) -> TransformationUncertainty:
        """Analyze OptiTrack -> Camera extrinsics uncertainty via frame-to-frame jitter."""
        if self.verbose:
            print("\n" + "="*70)
            print("STAGE 2: OptiTrack -> Camera Frame (Extrinsics)")
            print("="*70)

        uncertainty = TransformationUncertainty(
            stage_name="optitrack_to_camera",
            num_points=len(test_points_pyramid)
        )

        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        RT = np.linalg.inv(T_World_Lens @ self.calib_data.RT)

        translation_noise_m = self._estimate_tracking_noise_from_jitter(
            rb_data, frame_id, rb_key="Pyramid_RB", window=5
        )
        rotation_noise_deg = self._estimate_rotation_noise_from_jitter(
            rb_data, frame_id, rb_key="Pyramid_RB", window=5
        )

        uncertainty.translation_error_m = translation_noise_m
        uncertainty.rotation_error_deg = rotation_noise_deg
        uncertainty.condition_number = float(np.linalg.cond(RT))
        uncertainty.notes = (
            f"Translation noise (frame-to-frame jitter): {translation_noise_m*1000:.3f} mm. "
            f"Rotation noise: {rotation_noise_deg:.4f} deg. "
            f"RT condition: {uncertainty.condition_number:.2f}."
        )

        if self.verbose:
            print(f"  Translation noise (jitter): {translation_noise_m*1000:.3f} mm")
            print(f"  Rotation noise (jitter):    {rotation_noise_deg:.4f} deg")
            print(f"  RT matrix condition:        {uncertainty.condition_number:.2f}")

        self.report.optitrack_to_camera = uncertainty
        return uncertainty

    def _estimate_tracking_noise_from_jitter(
        self, rb_data: Dict, frame_id: int, rb_key: str, window: int = 5
    ) -> float:
        positions = []
        rb_frames = rb_data[rb_key]
        n_frames = len(rb_frames)
        for fid in range(max(0, frame_id - window), min(n_frames, frame_id + window + 1)):
            if rb_frames[fid].data.is_visible:
                T = rb_frames[fid].get_transform()
                positions.append(T[:3, 3].copy())
        if len(positions) < 3:
            return 0.001
        positions = np.array(positions)
        deltas = np.diff(positions, axis=0)
        noise_per_axis = np.std(deltas, axis=0)
        return float(np.linalg.norm(noise_per_axis))

    def _estimate_rotation_noise_from_jitter(
        self, rb_data: Dict, frame_id: int, rb_key: str, window: int = 5
    ) -> float:
        rotations = []
        rb_frames = rb_data[rb_key]
        n_frames = len(rb_frames)
        for fid in range(max(0, frame_id - window), min(n_frames, frame_id + window + 1)):
            if rb_frames[fid].data.is_visible:
                T = rb_frames[fid].get_transform()
                rotations.append(T[:3, :3].copy())
        if len(rotations) < 3:
            return 0.1
        angles_rad = []
        for i in range(len(rotations) - 1):
            dR = rotations[i + 1] @ rotations[i].T
            trace_val = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
            angles_rad.append(np.arccos(trace_val))
        return float(np.degrees(np.std(angles_rad)))

    # =========================================================================
    # Stage 3: Camera Projection
    # =========================================================================

    def analyze_camera_projection(
        self,
        points_3d_camera: npt.NDArray[np.float64],
        points_2d_observed: Optional[npt.NDArray[np.float64]] = None,
        R_cor: Optional[npt.NDArray[np.float64]] = None
    ) -> TransformationUncertainty:
        """Analyze camera projection uncertainty (intrinsics)."""
        if self.verbose:
            print("\n" + "="*70)
            print("STAGE 3: Camera Projection (Intrinsics)")
            print("="*70)

        uncertainty = TransformationUncertainty(
            stage_name="camera_projection",
            num_points=len(points_3d_camera)
        )

        rvec = np.zeros(3)
        tvec = np.zeros(3)
        points_2d_projected, _ = cv2.projectPoints(
            points_3d_camera, rvec, tvec,
            self.calib_data.K, self.calib_data.dist_coeffs
        )
        points_2d_projected = points_2d_projected.reshape(-1, 2)

        if R_cor is not None:
            pts_hom = np.hstack([points_2d_projected,
                                  np.ones((len(points_2d_projected), 1))]).T
            points_2d_projected = (R_cor @ pts_hom)[:2, :].T

        if points_2d_observed is not None:
            n_compare = min(len(points_2d_projected), len(points_2d_observed))
            if n_compare > 0:
                reproj_err = np.linalg.norm(
                    points_2d_projected[:n_compare] - points_2d_observed[:n_compare], axis=1
                )
                uncertainty.reprojection_errors_px = reproj_err
                uncertainty.mean_reprojection_error_px = float(np.mean(reproj_err))
                uncertainty.std_reprojection_error_px = float(np.std(reproj_err))
                uncertainty.max_reprojection_error_px = float(np.max(reproj_err))

        uncertainty.condition_number = float(np.linalg.cond(self.calib_data.K))
        fx = self.calib_data.K[0, 0]
        fy = self.calib_data.K[1, 1]
        uncertainty.notes = f"Focal length: fx={fx:.1f}, fy={fy:.1f}."

        if self.verbose:
            print(f"  Focal length (fx, fy): ({fx:.1f}, {fy:.1f}) px")
            print(f"  K condition number:    {uncertainty.condition_number:.2f}")
            if uncertainty.mean_reprojection_error_px is not None:
                print(f"  Mean reproj error:     {uncertainty.mean_reprojection_error_px:.2f} px")
                print(f"  Max reproj error:      {uncertainty.max_reprojection_error_px:.2f} px")

        self.report.camera_projection = uncertainty
        return uncertainty

    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================

    def analyze_sensitivity(
        self,
        rb_data: Dict[str, Any],
        frame_id: int,
        points_optitrack: np.ndarray,
        R_cor: Optional[np.ndarray] = None,
        translation_perturbation_mm: float = 1.0,
        rotation_perturbation_deg: float = 0.1,
        time_perturbation_frames: int = 1,
    ) -> ErrorBudget:
        """
        Perturbation-based sensitivity analysis.

        Computes the nominal projection for the given frame, then independently
        perturbs translation, orientation and time-sync to measure the Jacobian
        of reprojection error w.r.t. each error source.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("PERTURBATION-BASED SENSITIVITY ANALYSIS")
            print("="*70)

        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()

        pts_2d_nominal = self._project_points(
            points_optitrack, T_World_Pyramid, T_World_Lens, R_cor
        )

        budget = ErrorBudget()
        budget.sensitivities = {}

        # --- 1. TRANSLATION ---
        eps_t = translation_perturbation_mm / 1000.0
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            delta = np.zeros(3)
            delta[axis_idx] = eps_t
            T_pert = _perturb_transform_translation(T_World_Pyramid, delta)
            pts_2d_pert = self._project_points(points_optitrack, T_pert, T_World_Lens, R_cor)
            disp = np.linalg.norm(pts_2d_pert - pts_2d_nominal, axis=1)
            mean_disp = float(np.mean(disp))
            sensitivity = mean_disp / translation_perturbation_mm

            budget.sensitivities[f"translation_{axis_name}"] = SensitivityResult(
                perturbation_name=f"translation_{axis_name}",
                perturbation_unit="mm",
                perturbation_magnitude=translation_perturbation_mm,
                displacement_px=disp,
                mean_displacement_px=mean_disp,
                max_displacement_px=float(np.max(disp)),
                sensitivity=sensitivity,
                sensitivity_unit="px/mm"
            )
            if self.verbose:
                print(f"  delta_translation_{axis_name} ({translation_perturbation_mm}mm): "
                      f"mean={mean_disp:.3f}px, sensitivity={sensitivity:.3f} px/mm")

        trans_sensitivities = [
            budget.sensitivities[f"translation_{a}"].sensitivity for a in ["x", "y", "z"]
        ]
        combined_trans_sensitivity = np.linalg.norm(trans_sensitivities)

        # --- 2. ROTATION ---
        eps_r = np.deg2rad(rotation_perturbation_deg)
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        axis_names = ["roll", "pitch", "yaw"]
        for axis, axis_name in zip(axes, axis_names):
            T_pert = _perturb_transform_rotation(T_World_Pyramid, axis, eps_r)
            pts_2d_pert = self._project_points(points_optitrack, T_pert, T_World_Lens, R_cor)
            disp = np.linalg.norm(pts_2d_pert - pts_2d_nominal, axis=1)
            mean_disp = float(np.mean(disp))
            sensitivity = mean_disp / rotation_perturbation_deg

            budget.sensitivities[f"rotation_{axis_name}"] = SensitivityResult(
                perturbation_name=f"rotation_{axis_name}",
                perturbation_unit="deg",
                perturbation_magnitude=rotation_perturbation_deg,
                displacement_px=disp,
                mean_displacement_px=mean_disp,
                max_displacement_px=float(np.max(disp)),
                sensitivity=sensitivity,
                sensitivity_unit="px/deg"
            )
            if self.verbose:
                print(f"  delta_rotation_{axis_name} ({rotation_perturbation_deg}deg): "
                      f"mean={mean_disp:.3f}px, sensitivity={sensitivity:.3f} px/deg")

        rot_sensitivities = [
            budget.sensitivities[f"rotation_{a}"].sensitivity for a in axis_names
        ]
        combined_rot_sensitivity = np.linalg.norm(rot_sensitivities)

        # --- 3. TIME SYNC ---
        n_frames = len(rb_data["Pyramid_RB"])
        dt = time_perturbation_frames
        time_displacements = []
        for direction, label in [(dt, "forward"), (-dt, "backward")]:
            shifted_fid = frame_id + direction
            if (0 <= shifted_fid < n_frames and
                    rb_data["Pyramid_RB"][shifted_fid].data.is_visible):
                T_shifted = rb_data["Pyramid_RB"][shifted_fid].get_transform()
                pts_2d_shifted = self._project_points(
                    points_optitrack, T_shifted, T_World_Lens, R_cor
                )
                disp = np.linalg.norm(pts_2d_shifted - pts_2d_nominal, axis=1)
                time_displacements.append(disp)
                if self.verbose:
                    print(f"  delta_time_{label} ({direction:+d} frames): "
                          f"mean={np.mean(disp):.3f}px")

        if len(time_displacements) > 0:
            avg_disp = np.mean(time_displacements, axis=0)
            mean_disp = float(np.mean(avg_disp))
            sensitivity = mean_disp / abs(dt)
            budget.sensitivities["time_sync"] = SensitivityResult(
                perturbation_name="time_sync",
                perturbation_unit="frames",
                perturbation_magnitude=float(abs(dt)),
                displacement_px=avg_disp,
                mean_displacement_px=mean_disp,
                max_displacement_px=float(np.max(avg_disp)),
                sensitivity=sensitivity,
                sensitivity_unit="px/frame"
            )
            if self.verbose:
                print(f"  Time sync sensitivity: {sensitivity:.3f} px/frame")

        # --- 4. BUILD ERROR BUDGET ---
        if self.report.optitrack_to_camera is not None:
            trans_noise_mm = self.report.optitrack_to_camera.translation_error_m * 1000
            rot_noise_deg = self.report.optitrack_to_camera.rotation_error_deg or 0.0
        else:
            trans_noise_mm = 0.5
            rot_noise_deg = 0.05
        time_noise_frames = 0.5

        budget.assumed_translation_noise_mm = trans_noise_mm
        budget.assumed_orientation_noise_deg = rot_noise_deg
        budget.assumed_time_sync_noise_frames = time_noise_frames

        budget.translation_contribution_px = combined_trans_sensitivity * trans_noise_mm
        budget.orientation_contribution_px = combined_rot_sensitivity * rot_noise_deg

        if "time_sync" in budget.sensitivities:
            budget.time_sync_contribution_px = (
                budget.sensitivities["time_sync"].sensitivity * time_noise_frames
            )

        if self.report.model_to_optitrack is not None:
            svd_error_mm = self.report.model_to_optitrack.mean_error_3d_m * 1000
            budget.svd_fitting_contribution_px = combined_trans_sensitivity * svd_error_mm

        budget.combined_predicted_px = np.sqrt(
            budget.translation_contribution_px**2 +
            budget.orientation_contribution_px**2 +
            budget.time_sync_contribution_px**2 +
            budget.svd_fitting_contribution_px**2
        )

        if (self.report.camera_projection is not None and
                self.report.camera_projection.mean_reprojection_error_px is not None):
            budget.actual_reprojection_px = self.report.camera_projection.mean_reprojection_error_px

        if self.verbose:
            print(f"\n  --- ERROR BUDGET ---")
            print(f"  Translation ({trans_noise_mm:.2f}mm noise): "
                  f"{budget.translation_contribution_px:.2f} px")
            print(f"  Orientation ({rot_noise_deg:.4f}deg noise): "
                  f"{budget.orientation_contribution_px:.2f} px")
            print(f"  Time sync ({time_noise_frames:.1f} frames noise): "
                  f"{budget.time_sync_contribution_px:.2f} px")
            print(f"  SVD fitting: {budget.svd_fitting_contribution_px:.2f} px")
            print(f"  Combined (RSS): {budget.combined_predicted_px:.2f} px")
            if budget.actual_reprojection_px is not None:
                print(f"  Actual measured: {budget.actual_reprojection_px:.2f} px")

        self.report.error_budget = budget
        return budget

    # =========================================================================
    # Full pipeline
    # =========================================================================

    def analyze_full_pipeline(
        self,
        rb_data: Dict[str, Any],
        marker_positions_m: Dict[str, npt.NDArray[np.float64]],
        matching: Dict[str, int],
        frame_id: int = 0,
        keypoints_2d: Optional[npt.NDArray[np.float64]] = None,
        R_cor: Optional[npt.NDArray[np.float64]] = None,
        run_sensitivity: bool = False,
        translation_perturbation_mm: float = 1.0,
        rotation_perturbation_deg: float = 0.1,
        time_perturbation_frames: int = 1,
    ) -> UncertaintyReport:
        """Perform complete uncertainty analysis through the entire pipeline."""
        if self.verbose:
            print("\n" + "="*70)
            print("FULL PIPELINE UNCERTAINTY ANALYSIS")
            print("="*70)

        self.analyze_model_to_optitrack(marker_positions_m, matching)

        points_pyramid = self.transformer.get_pyramid_points_in_pyramid_frame()
        self.analyze_optitrack_to_camera(rb_data, frame_id, points_pyramid)

        points_optitrack = self.transformer.transform_pyramid_to_optitrack(points_pyramid)
        T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
        T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        RT = np.linalg.inv(T_World_Lens @ self.calib_data.RT)

        n_points = points_optitrack.shape[0]
        pts_hom = np.hstack([points_optitrack, np.ones((n_points, 1))])
        pts_world = (T_World_Pyramid @ pts_hom.T).T[:, :3]
        pts_camera = (RT[:3, :3] @ pts_world.T + RT[:3, 3:4]).T

        # Cache camera-frame points for UncertaintyPlotter intrinsics analysis
        self.report.points_3d_camera_last = pts_camera.copy()
        self.report.rvec_last = np.zeros(3)
        self.report.tvec_last = np.zeros(3)

        self.analyze_camera_projection(pts_camera, keypoints_2d, R_cor)

        if (keypoints_2d is not None and
                self.report.camera_projection.mean_reprojection_error_px is not None):
            self.report.total_reprojection_error_px = \
                self.report.camera_projection.mean_reprojection_error_px

        if run_sensitivity:
            self.analyze_sensitivity(
                rb_data=rb_data,
                frame_id=frame_id,
                points_optitrack=points_optitrack,
                R_cor=R_cor,
                translation_perturbation_mm=translation_perturbation_mm,
                rotation_perturbation_deg=rotation_perturbation_deg,
                time_perturbation_frames=time_perturbation_frames,
            )

        return self.report

    # =========================================================================
    # Legacy visualization (kept for compatibility)
    # =========================================================================

    def visualize_error_budget(
        self,
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Optional[Figure]:
        if self.report.error_budget is None:
            print("No error budget available. Run analyze_sensitivity() first.")
            return None
        budget = self.report.error_budget
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Error Budget: Per-Source Contributions to Reprojection Error",
                      fontsize=14, fontweight='bold')
        ax1 = axes[0]
        sources, contributions, colors = [], [], []
        if budget.svd_fitting_contribution_px > 0:
            sources.append(f"SVD fitting")
            contributions.append(budget.svd_fitting_contribution_px)
            colors.append('#2E86AB')
        sources.append(f"Translation\n({budget.assumed_translation_noise_mm:.2f}mm)")
        contributions.append(budget.translation_contribution_px)
        colors.append('#A23B72')
        sources.append(f"Orientation\n({budget.assumed_orientation_noise_deg:.4f}deg)")
        contributions.append(budget.orientation_contribution_px)
        colors.append('#F18F01')
        sources.append(f"Time sync\n({budget.assumed_time_sync_noise_frames:.1f}fr)")
        contributions.append(budget.time_sync_contribution_px)
        colors.append('#C73E1D')
        x_pos = np.arange(len(sources))
        bars = ax1.bar(x_pos, contributions, color=colors, alpha=0.8, edgecolor='black')
        ax1.axhline(budget.combined_predicted_px, color='black', linestyle='--', linewidth=2,
                     label=f'Combined RSS: {budget.combined_predicted_px:.2f}px')
        if budget.actual_reprojection_px is not None:
            ax1.axhline(budget.actual_reprojection_px, color='red', linestyle='-', linewidth=2,
                         label=f'Actual: {budget.actual_reprojection_px:.2f}px')
        for bar, val in zip(bars, contributions):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{val:.2f}px', ha='center', va='bottom', fontsize=9)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(sources, fontsize=9)
        ax1.set_ylabel("Contribution (px)")
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def print_summary(self) -> None:
        print("\n" + "="*70)
        print("UNCERTAINTY ANALYSIS SUMMARY")
        print("="*70)
        if self.report.model_to_optitrack is not None:
            u = self.report.model_to_optitrack
            q = '[OK]' if u.mean_error_3d_m < 0.002 else '[!]' if u.mean_error_3d_m < 0.005 else '[X]'
            print(f"\n  Stage 1: SVD Fitting")
            print(f"  Mean 3D error: {u.mean_error_3d_m*1000:.3f} +/- {u.std_error_3d_m*1000:.3f} mm")
            print(f"  Max 3D error:  {u.max_error_3d_m*1000:.3f} mm  {q}")
        if self.report.optitrack_to_camera is not None:
            u = self.report.optitrack_to_camera
            print(f"\n  Stage 2: OptiTrack -> Camera")
            print(f"  Translation jitter: {u.translation_error_m*1000:.3f} mm")
            print(f"  Rotation jitter:    {u.rotation_error_deg:.4f} deg")
        if self.report.camera_projection is not None:
            u = self.report.camera_projection
            print(f"\n  Stage 3: Camera Projection")
            if u.mean_reprojection_error_px is not None:
                q = '[OK]' if u.mean_reprojection_error_px < 2 else '[!]' if u.mean_reprojection_error_px < 5 else '[X]'
                print(f"  Mean reproj error: {u.mean_reprojection_error_px:.2f} +/- "
                      f"{u.std_reprojection_error_px:.2f} px  {q}")
        if self.report.error_budget is not None:
            b = self.report.error_budget
            print(f"\n  Error Budget:")
            print(f"    SVD:         {b.svd_fitting_contribution_px:.2f} px")
            print(f"    Translation: {b.translation_contribution_px:.2f} px")
            print(f"    Orientation: {b.orientation_contribution_px:.2f} px")
            print(f"    Time sync:   {b.time_sync_contribution_px:.2f} px")
            print(f"    RSS total:   {b.combined_predicted_px:.2f} px")
            if b.actual_reprojection_px is not None:
                print(f"    Actual:      {b.actual_reprojection_px:.2f} px")
        print("\n" + "="*70)


# =============================================================================
# Intrinsics sensitivity helper
# =============================================================================

def compute_intrinsics_sensitivity(
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    points_3d: np.ndarray,
    rvec: Optional[np.ndarray] = None,
    tvec: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Perturb each intrinsic parameter independently and measure mean reprojection
    displacement (px).

    Returns dict: parameter_name -> sensitivity (px per unit perturbation).
      fx, fy, cx, cy : px per 1-pixel change
      k1, k2, p1, p2 : px per 1e-3 change in distortion coefficient
    """
    if rvec is None:
        rvec = np.zeros(3)
    if tvec is None:
        tvec = np.zeros(3)

    dist_flat = dist_coeffs.ravel()
    n_dist = min(len(dist_flat), 5)
    dist_padded = np.zeros(5)
    dist_padded[:n_dist] = dist_flat[:n_dist]

    baseline, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_padded.reshape(-1, 1))
    baseline = baseline.reshape(-1, 2)

    results: Dict[str, float] = {}

    # Focal length and principal point (perturbation = 1px in that parameter)
    K_perturbations = [
        ("fx (+1px)",  np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)),
        ("fy (+1px)",  np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)),
        ("cx (+1px)",  np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)),
        ("cy (+1px)",  np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)),
    ]
    for name, K_delta in K_perturbations:
        K_pert = K.astype(float) + K_delta
        pts, _ = cv2.projectPoints(points_3d, rvec, tvec, K_pert,
                                    dist_padded.reshape(-1, 1))
        disp = np.mean(np.linalg.norm(pts.reshape(-1, 2) - baseline, axis=1))
        results[name] = float(disp)

    # Distortion coefficients (perturbation = 1e-3)
    dist_names = ["k1 (+1e-3)", "k2 (+1e-3)", "p1 (+1e-3)", "p2 (+1e-3)", "k3 (+1e-3)"]
    for i in range(n_dist):
        d_pert = dist_padded.copy()
        d_pert[i] += 1e-3
        pts, _ = cv2.projectPoints(points_3d, rvec, tvec, K,
                                    d_pert.reshape(-1, 1))
        disp = np.mean(np.linalg.norm(pts.reshape(-1, 2) - baseline, axis=1))
        results[dist_names[i]] = float(disp)

    return results


# =============================================================================
# FrameRecord — lightweight per-frame data snapshot for the plotter
# =============================================================================

@dataclass
class FrameRecord:
    """Per-frame data snapshot pushed into UncertaintyPlotter."""
    frame_id: int

    # Plot 1: intrinsics sensitivity  (param_name -> px/unit)
    intrinsics_sensitivity: Dict[str, float] = field(default_factory=dict)

    # Plot 2: translation sensitivity (px/mm per axis)
    trans_sens_x: float = 0.0
    trans_sens_y: float = 0.0
    trans_sens_z: float = 0.0

    # Plot 3: rotation sensitivity (px/deg) + time-sync (px/frame)
    rot_sens_roll:  float = 0.0
    rot_sens_pitch: float = 0.0
    rot_sens_yaw:   float = 0.0
    time_sync_sens: float = 0.0

    # Plot 4: error budget contributions (px)
    svd_contrib:         float = 0.0
    translation_contrib: float = 0.0
    rotation_contrib:    float = 0.0
    timesync_contrib:    float = 0.0
    combined_predicted:  float = 0.0
    actual_reproj:       Optional[float] = None


def frame_record_from_analyzer(
    frame_id: int,
    analyzer: UncertaintyAnalyzer,
) -> FrameRecord:
    """
    Build a FrameRecord from a completed UncertaintyAnalyzer report.

    Call this after analyze_full_pipeline() (with or without run_sensitivity).
    Intrinsics sensitivity is computed here using the cached pts_camera.
    """
    report = analyzer.report
    budget = report.error_budget

    # --- Intrinsics sensitivity ---
    intrinsics_sens: Dict[str, float] = {}
    if report.points_3d_camera_last is not None and len(report.points_3d_camera_last) > 0:
        try:
            intrinsics_sens = compute_intrinsics_sensitivity(
                K=analyzer.calib_data.K,
                dist_coeffs=analyzer.calib_data.dist_coeffs,
                points_3d=report.points_3d_camera_last,
                rvec=report.rvec_last,
                tvec=report.tvec_last,
            )
        except Exception:
            intrinsics_sens = {}

    def _s(key: str, default: float = 0.0) -> float:
        if budget and key in budget.sensitivities:
            return budget.sensitivities[key].sensitivity
        return default

    # Error budget: use noise-based contributions even when full sensitivity
    # analysis was not run (graceful degradation).
    if budget is not None:
        svd    = budget.svd_fitting_contribution_px
        trans  = budget.translation_contribution_px
        rot    = budget.orientation_contribution_px
        tsync  = budget.time_sync_contribution_px
        pred   = budget.combined_predicted_px
        actual = budget.actual_reprojection_px
    else:
        # Fall back to jitter-based noise estimates without axis breakdown
        svd = trans = rot = tsync = pred = 0.0
        actual = None
        if report.model_to_optitrack is not None:
            svd = report.model_to_optitrack.mean_error_3d_m * 1000 * 3.0  # rough px estimate
        if report.optitrack_to_camera is not None:
            trans = report.optitrack_to_camera.translation_error_m * 1000 * 3.0
            rot   = (report.optitrack_to_camera.rotation_error_deg or 0.0) * 10.0
        if report.camera_projection is not None:
            actual = report.camera_projection.mean_reprojection_error_px
        pred = np.sqrt(svd**2 + trans**2 + rot**2 + tsync**2)

    return FrameRecord(
        frame_id=frame_id,
        intrinsics_sensitivity=intrinsics_sens,
        trans_sens_x=_s("translation_x"),
        trans_sens_y=_s("translation_y"),
        trans_sens_z=_s("translation_z"),
        rot_sens_roll =_s("rotation_roll"),
        rot_sens_pitch=_s("rotation_pitch"),
        rot_sens_yaw  =_s("rotation_yaw"),
        time_sync_sens=_s("time_sync"),
        svd_contrib         =svd,
        translation_contrib =trans,
        rotation_contrib    =rot,
        timesync_contrib    =tsync,
        combined_predicted  =pred,
        actual_reproj       =actual,
    )


# =============================================================================
# UncertaintyPlotter — 4-panel real-time diagnostic figure
# =============================================================================

class UncertaintyPlotter:
    """
    Real-time 4-panel uncertainty diagnostic figure.

    Maintains a rolling history of FrameRecords and renders:
      Plot 1 — Intrinsics Impact on Reprojection (px per unit Δparam)
      Plot 2 — Extrinsics: Translation Sensitivity (px/mm per axis)
      Plot 3 — Extrinsics: Rotation + Time-Sync Sensitivity (px/° and px/frame)
      Plot 4 — Error Budget: Predicted vs Actual Reprojection (px, stacked)

    Usage:
        plotter = UncertaintyPlotter()
        ...
        plotter.update(frame_record_from_analyzer(frame_id, analyzer))
        img = plotter.get_plot_image()   # returns BGR numpy array for cv2
    """

    # Colour palette
    _COLORS = {
        "bg":      '#1a1a2e',
        "panel":   '#0f0f23',
        "spine":   '#444466',
        "tick":    '#aaaacc',
        "good":    '#27ae60',
        "accept":  '#f39c12',
        "bad":     '#e74c3c',
        "fx":      '#e74c3c',
        "fy":      '#e67e22',
        "cx":      '#3498db',
        "cy":      '#2980b9',
        "k1":      '#27ae60',
        "trans_x": '#e74c3c',
        "trans_y": '#3498db',
        "trans_z": '#2ecc71',
        "rss":     '#ffffff',
        "roll":    '#e74c3c',
        "pitch":   '#f39c12',
        "yaw":     '#9b59b6',
        "tsync":   '#1abc9c',
        "svd":     '#2E86AB',
        "trans":   '#A23B72',
        "rot":     '#F18F01',
        "sync":    '#C73E1D',
        "pred":    '#ffffff',
        "actual":  '#ff4444',
        "annot_fg": '#ffcc00',
        "annot_bg": '#1a1a2e',
        "annot_ec": '#ffcc00',
        "annot_ec2": '#8888aa',
    }

    def __init__(
        self,
        history_len: int = 300,
        plot_width:  int = 600,
        plot_height: int = 400,
    ):
        self.history:    List[FrameRecord] = []
        self.history_len = history_len
        self.plot_width  = plot_width
        self.plot_height = plot_height

        # Build figure once; canvas is persistent
        self.fig = plt.figure(figsize=(14, 8), facecolor=self._COLORS["bg"])
        self.fig.suptitle(
            "Real-Time Uncertainty Analysis  ·  OptiTrack → Camera Pipeline",
            fontsize=12, fontweight='bold', color='white', y=0.98
        )
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.50, wspace=0.38,
                               left=0.07, right=0.97,
                               top=0.92, bottom=0.07)
        self.axes = [
            self.fig.add_subplot(gs[0, 0]),
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[1, 0]),
            self.fig.add_subplot(gs[1, 1]),
        ]
        for ax in self.axes:
            self._style_ax(ax)

        self.canvas = FigureCanvasAgg(self.fig)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, record: FrameRecord) -> None:
        """Append a FrameRecord and trim history."""
        self.history.append(record)
        if len(self.history) > self.history_len:
            self.history = self.history[-self.history_len:]
        self._redraw()

    def get_plot_image(self) -> npt.NDArray[np.uint8]:
        """Return the current figure as a BGR numpy array (for cv2 overlay)."""
        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img = cv2.resize(img, (self.plot_width, self.plot_height))
        return img

    def close(self) -> None:
        plt.close(self.fig)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _style_ax(self, ax) -> None:
        c = self._COLORS
        ax.set_facecolor(c["panel"])
        for spine in ax.spines.values():
            spine.set_color(c["spine"])
        ax.tick_params(colors=c["tick"], labelsize=8)
        ax.yaxis.label.set_color(c["tick"])
        ax.xaxis.label.set_color(c["tick"])

    def _annotate(self, ax, text: str, color_fg: str, color_ec: str) -> None:
        c = self._COLORS
        ax.annotate(
            text,
            xy=(0.97, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=7.5, color=color_fg,
            bbox=dict(boxstyle='round,pad=0.3',
                      fc=c["annot_bg"], ec=color_ec, alpha=0.88)
        )

    def _redraw(self) -> None:
        for ax in self.axes:
            ax.cla()
            self._style_ax(ax)
        if not self.history:
            return
        frames = [r.frame_id for r in self.history]
        self._plot_intrinsics(self.axes[0], frames)
        self._plot_translation(self.axes[1], frames)
        self._plot_rotation_timesync(self.axes[2], frames)
        self._plot_error_budget(self.axes[3], frames)

    # ------------------------------------------------------------------
    # Plot 1 — Intrinsics Impact
    # ------------------------------------------------------------------

    def _plot_intrinsics(self, ax, frames: List[int]) -> None:
        c = self._COLORS
        ax.set_title(
            "Intrinsics Impact  [px per unit Δparam]",
            fontsize=9, fontweight='bold', color='white', pad=4
        )
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Sensitivity (px)", fontsize=8)

        key_series = [
            ("fx (+1px)",  c["fx"],  "fx  (px/px)"),
            ("fy (+1px)",  c["fy"],  "fy  (px/px)"),
            ("cx (+1px)",  c["cx"],  "cx  (px/px)"),
            ("cy (+1px)",  c["cy"],  "cy  (px/px)"),
            ("k1 (+1e-3)", c["k1"],  "k1  (px/1e-3)"),
        ]
        for key, color, label in key_series:
            vals = [r.intrinsics_sensitivity.get(key, np.nan) for r in self.history]
            if any(not np.isnan(v) for v in vals):
                ax.plot(frames, vals, color=color, linewidth=1.3, alpha=0.85, label=label)

        ax.axhline(GOOD_PX,   color=c["good"],   linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(ACCEPT_PX, color=c["accept"], linestyle='--', linewidth=1, alpha=0.6)
        ax.text(frames[0], GOOD_PX + 0.08, f'{GOOD_PX}px (good)',
                fontsize=7, color=c["good"])
        ax.text(frames[0], ACCEPT_PX + 0.08, f'{ACCEPT_PX}px (accept)',
                fontsize=7, color=c["accept"])

        last = self.history[-1]
        if last.intrinsics_sensitivity:
            dominant = max(last.intrinsics_sensitivity,
                           key=last.intrinsics_sensitivity.get)
            val = last.intrinsics_sensitivity[dominant]
            color_ec = c["annot_ec"] if val > ACCEPT_PX else \
                       (c["accept"] if val > GOOD_PX else c["good"])
            self._annotate(ax,
                           f"Worst: {dominant}\n{val:.3f} px/unit",
                           c["annot_fg"], color_ec)

        ax.legend(fontsize=7, loc='upper left',
                  labelcolor='white', facecolor=c["annot_bg"],
                  edgecolor=c["spine"], ncol=2)
        ax.grid(axis='y', alpha=0.15, color=c["spine"])

    # ------------------------------------------------------------------
    # Plot 2 — Translation Sensitivity
    # ------------------------------------------------------------------

    def _plot_translation(self, ax, frames: List[int]) -> None:
        c = self._COLORS
        ax.set_title(
            "Extrinsics · Translation Sensitivity  [px / mm shift in RB position]",
            fontsize=9, fontweight='bold', color='white', pad=4
        )
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Sensitivity (px/mm)", fontsize=8)

        xs = [r.trans_sens_x for r in self.history]
        ys = [r.trans_sens_y for r in self.history]
        zs = [r.trans_sens_z for r in self.history]
        rss = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(xs, ys, zs)]

        ax.plot(frames, xs,  color=c["trans_x"], linewidth=1.3, label='X  (px/mm)')
        ax.plot(frames, ys,  color=c["trans_y"], linewidth=1.3, label='Y  (px/mm)')
        ax.plot(frames, zs,  color=c["trans_z"], linewidth=1.3, label='Z (depth, px/mm)')
        ax.plot(frames, rss, color=c["rss"],     linewidth=1.8,
                linestyle=':', label='RSS combined')

        last = self.history[-1]
        last_rss = np.sqrt(last.trans_sens_x**2 + last.trans_sens_y**2 + last.trans_sens_z**2)
        mean_rss = float(np.nanmean(rss)) if rss else 0.0
        self._annotate(ax,
                       f"RSS now:  {last_rss:.3f} px/mm\nRSS mean: {mean_rss:.3f} px/mm",
                       'white', c["annot_ec2"])

        ax.legend(fontsize=7, loc='upper left',
                  labelcolor='white', facecolor=c["annot_bg"],
                  edgecolor=c["spine"])
        ax.grid(axis='y', alpha=0.15, color=c["spine"])

    # ------------------------------------------------------------------
    # Plot 3 — Rotation + Time-Sync Sensitivity
    # ------------------------------------------------------------------

    def _plot_rotation_timesync(self, ax, frames: List[int]) -> None:
        c = self._COLORS
        ax.set_title(
            "Extrinsics · Rotation (px/°)  &  Time-Sync Sensitivity (px/frame)",
            fontsize=9, fontweight='bold', color='white', pad=4
        )
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Sensitivity", fontsize=8)

        roll  = [r.rot_sens_roll  for r in self.history]
        pitch = [r.rot_sens_pitch for r in self.history]
        yaw   = [r.rot_sens_yaw   for r in self.history]
        tsync = [r.time_sync_sens for r in self.history]

        ax.plot(frames, roll,  color=c["roll"],  linewidth=1.2, label='Roll  (px/°)')
        ax.plot(frames, pitch, color=c["pitch"], linewidth=1.2, label='Pitch (px/°)')
        ax.plot(frames, yaw,   color=c["yaw"],   linewidth=1.2, label='Yaw   (px/°)')
        ax.plot(frames, tsync, color=c["tsync"], linewidth=1.5,
                linestyle='-.', label='Time-sync (px/frame)')

        last = self.history[-1]
        worst_rot_name, worst_rot_val = max(
            [("Roll",  last.rot_sens_roll),
             ("Pitch", last.rot_sens_pitch),
             ("Yaw",   last.rot_sens_yaw)],
            key=lambda x: x[1]
        )
        self._annotate(ax,
                       f"Dominant DoF: {worst_rot_name}\n"
                       f"{worst_rot_val:.3f} px/°\n"
                       f"Time-sync: {last.time_sync_sens:.3f} px/fr",
                       c["annot_fg"], c["annot_ec"])

        ax.legend(fontsize=7, loc='upper left',
                  labelcolor='white', facecolor=c["annot_bg"],
                  edgecolor=c["spine"])
        ax.grid(axis='y', alpha=0.15, color=c["spine"])

    # ------------------------------------------------------------------
    # Plot 4 — Error Budget (stacked area)
    # ------------------------------------------------------------------

    def _plot_error_budget(self, ax, frames: List[int]) -> None:
        c = self._COLORS
        ax.set_title(
            "Error Budget: Predicted Contributions  vs  Actual Reproj  (px)",
            fontsize=9, fontweight='bold', color='white', pad=4
        )
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Reprojection Error (px)", fontsize=8)

        frames_arr = np.array(frames)
        svd    = np.array([r.svd_contrib         for r in self.history])
        trans  = np.array([r.translation_contrib  for r in self.history])
        rot    = np.array([r.rotation_contrib     for r in self.history])
        tsync  = np.array([r.timesync_contrib     for r in self.history])
        pred   = np.array([r.combined_predicted   for r in self.history])
        actual = np.array([r.actual_reproj if r.actual_reproj is not None else np.nan
                            for r in self.history])

        ax.stackplot(
            frames_arr,
            svd, trans, rot, tsync,
            labels=['SVD fitting', 'Translation noise', 'Rotation noise', 'Time-sync'],
            colors=[c["svd"], c["trans"], c["rot"], c["sync"]],
            alpha=0.5
        )
        ax.plot(frames_arr, pred,   color=c["pred"],   linewidth=1.8,
                linestyle='--', label='Predicted RSS', zorder=5)
        ax.plot(frames_arr, actual, color=c["actual"], linewidth=2.0,
                label='Actual measured', zorder=6)

        ax.axhline(GOOD_PX,   color=c["good"],   linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(ACCEPT_PX, color=c["accept"], linestyle=':', linewidth=1, alpha=0.7)
        ax.text(frames_arr[0], GOOD_PX + 0.05,
                f'{GOOD_PX}px', fontsize=7, color=c["good"])
        ax.text(frames_arr[0], ACCEPT_PX + 0.05,
                f'{ACCEPT_PX}px', fontsize=7, color=c["accept"])

        last = self.history[-1]
        total_var = (last.svd_contrib**2 + last.translation_contrib**2 +
                     last.rotation_contrib**2 + last.timesync_contrib**2)
        if total_var > 1e-6:
            dominant_name, dominant_val = max(
                [("SVD",   last.svd_contrib),
                 ("Trans", last.translation_contrib),
                 ("Rot",   last.rotation_contrib),
                 ("Sync",  last.timesync_contrib)],
                key=lambda x: x[1]
            )
            pct = 100 * dominant_val**2 / total_var
            actual_str = f"{last.actual_reproj:.2f}px" if last.actual_reproj is not None else "N/A"
            self._annotate(ax,
                           f"Predicted: {last.combined_predicted:.2f}px\n"
                           f"Actual:    {actual_str}\n"
                           f"Dominant:  {dominant_name} ({pct:.0f}%)",
                           'white', c["annot_ec2"])

        ax.legend(fontsize=7, loc='upper left',
                  labelcolor='white', facecolor=c["annot_bg"],
                  edgecolor=c["spine"], ncol=2)
        ax.grid(axis='y', alpha=0.15, color=c["spine"])