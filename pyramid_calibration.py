# pyramid_calibration.py

"""
Pyramid Calibration System
===========================

Estimates the rigid transformation (rotation + translation) between:
- Pyramid local frame (from 3D scan, in mm)
- OptiTrack rigid body frame (constellation at top of pyramid, in mm)

Uses point-to-point correspondences and Kabsch algorithm (SVD-based alignment).

Translation is known from LocalReferential in JSON.
Rotation must be estimated from corresponding points.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Results from calibration."""
    rotation_matrix: np.ndarray  # 3x3 rotation from pyramid to OptiTrack
    translation_vector: np.ndarray  # 3x1 translation (mm)
    rmse: float  # Root mean square error of fit (mm)
    n_points: int  # Number of correspondence points used
    point_errors: np.ndarray  # Per-point errors (mm)


class PyramidCalibrator:
    """
    Calibrates the rigid transform between pyramid local frame and OptiTrack frame.
    
    Uses point correspondences measured in both frames:
    - Points in pyramid frame: from 3D scan data (mm)
    - Points in OptiTrack frame: from OptiTrack tracking with constellation visible (mm)
    """
    
    def __init__(self, pyramid_json_path: Path, unit: str = 'mm'):
        """
        Initialize calibrator.
        
        Args:
            pyramid_json_path: Path to ModelMire3DSLAM.json
            unit: 'mm' or 'm' - will convert internally to mm
        """
        self.pyramid_json_path = pyramid_json_path
        self.unit = unit
        self.markers_pyramid_frame_mm = None  # In mm
        self.local_referential_mm = None  # In mm
        
        self._load_pyramid_data()
    
    def _load_pyramid_data(self) -> None:
        """Load pyramid geometry from JSON and convert to mm."""
        with open(self.pyramid_json_path, 'r') as f:
            data = json.load(f)
        
        # Extract marker positions in pyramid local frame (in meters from JSON)
        markers = []
        for marker in data['Markers']:
            pos = marker['LocalPosition']
            x = pos.get('x', 0.0)
            y = pos.get('y', 0.0)
            z = pos.get('z', 0.0)
            markers.append([x, y, z])
        
        # Convert meters to mm
        self.markers_pyramid_frame_mm = np.array(markers) * 1000.0
        
        # Extract local referential (in meters from JSON)
        local_ref_pos = data['LocalReferential']['position']
        local_ref_m = np.array([
            local_ref_pos['x'],
            local_ref_pos['y'],
            local_ref_pos['z']
        ])
        
        # Convert to mm
        self.local_referential_mm = local_ref_m * 1000.0
        
        print(f"Loaded {len(self.markers_pyramid_frame_mm)} markers from pyramid scan")
        print(f"Local referential (OptiTrack origin in pyramid frame): {self.local_referential_mm} mm")
    
    def calibrate_from_correspondences(
        self,
        pyramid_points_mm: np.ndarray,
        optitrack_points_mm: np.ndarray
    ) -> CalibrationResult:
        """
        Estimate rotation from point correspondences using Kabsch algorithm.
        
        The translation is already known from LocalReferential.
        We only need to estimate rotation.
        
        Args:
            pyramid_points_mm: Nx3 array of points in pyramid frame (mm)
            optitrack_points_mm: Nx3 array of corresponding points in OptiTrack frame (mm)
            
        Returns:
            CalibrationResult with rotation matrix, translation, and error metrics
        """
        if pyramid_points_mm.shape != optitrack_points_mm.shape:
            raise ValueError(f"Point arrays must have same shape. Got {pyramid_points_mm.shape} and {optitrack_points_mm.shape}")
        
        if pyramid_points_mm.shape[0] < 3:
            raise ValueError(f"Need at least 3 point correspondences, got {pyramid_points_mm.shape[0]}")
        
        n_points = pyramid_points_mm.shape[0]
        print(f"\n{'='*70}")
        print(f"CALIBRATING ROTATION FROM {n_points} POINT CORRESPONDENCES")
        print(f"{'='*70}")
        
        # Use Kabsch algorithm to find optimal rotation
        # See: https://en.wikipedia.org/wiki/Kabsch_algorithm
        R = self._kabsch_algorithm(pyramid_points_mm, optitrack_points_mm)
        
        # Translation is known from JSON
        t = self.local_referential_mm
        
        # Compute errors
        pyramid_points_transformed = (R @ pyramid_points_mm.T).T + t
        errors = np.linalg.norm(pyramid_points_transformed - optitrack_points_mm, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        
        print(f"\nCalibration Results:")
        print(f"  RMSE: {rmse:.4f} mm")
        print(f"  Max error: {np.max(errors):.4f} mm")
        print(f"  Min error: {np.min(errors):.4f} mm")
        print(f"  Mean error: {np.mean(errors):.4f} mm")
        
        print(f"\nEstimated Rotation Matrix:")
        print(R)
        print(f"\nTranslation (from JSON): {t} mm")
        
        # Verify rotation is valid
        det = np.linalg.det(R)
        print(f"\nRotation matrix determinant: {det:.6f} (should be 1.0)")
        
        orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
        print(f"Orthogonality error: {orthogonality_error:.10f} (should be ~0)")
        
        if not (0.99 < det < 1.01):
            print("WARNING: Determinant not close to 1 - rotation may be invalid!")
        
        if orthogonality_error > 1e-6:
            print("WARNING: Rotation matrix not orthogonal - may need more/better points!")
        
        return CalibrationResult(
            rotation_matrix=R,
            translation_vector=t,
            rmse=rmse,
            n_points=n_points,
            point_errors=errors
        )
    
    def _kabsch_algorithm(
        self,
        P: np.ndarray,
        Q: np.ndarray
    ) -> np.ndarray:
        """
        Kabsch algorithm to find optimal rotation matrix.
        
        Finds rotation R such that: R @ P ≈ Q (after centering)
        
        Args:
            P: Nx3 source points (pyramid frame)
            Q: Nx3 target points (OptiTrack frame)
            
        Returns:
            3x3 rotation matrix
        """
        # Center the point sets
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        print(f"\nKabsch Algorithm:")
        print(f"  Centroid P (pyramid): {centroid_P}")
        print(f"  Centroid Q (OptiTrack): {centroid_Q}")
        
        # Compute covariance matrix
        H = P_centered.T @ Q_centered
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation
        R = Vt.T @ U.T
        
        # Handle reflection case (det(R) = -1)
        if np.linalg.det(R) < 0:
            print("  Detected reflection - correcting...")
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        print(f"  Singular values: {S}")
        
        return R
    
    def calibrate_from_tracked_markers(
        self,
        marker_indices: List[int],
        rb_data: dict,
        frame_id: int,
        T_World_Pyramid: np.ndarray
    ) -> CalibrationResult:
        """
        Calibrate using tracked marker positions.
        
        This method extracts marker positions from OptiTrack tracking data
        and uses them for calibration.
        
        Args:
            marker_indices: List of marker indices to use (e.g., [0, 1, 2, 5, 8])
            rb_data: Rigid body tracking data
            frame_id: Frame to use for calibration (pyramid should be stable)
            T_World_Pyramid: 4x4 transform from world to pyramid at this frame
            
        Returns:
            CalibrationResult
        """
        # Get marker positions in pyramid local frame
        pyramid_points_mm = self.markers_pyramid_frame_mm[marker_indices]
        
        # Transform markers to world frame (assuming identity rotation for now)
        pyramid_points_world = (T_World_Pyramid @ np.hstack([
            pyramid_points_mm / 1000.0,  # Convert to meters for transform
            np.ones((len(pyramid_points_mm), 1))
        ]).T).T[:, :3] * 1000.0  # Back to mm
        
        # TODO: Extract corresponding points from OptiTrack marker data
        # This requires access to individual marker positions from rigid body
        # For now, this is a placeholder
        
        raise NotImplementedError(
            "Automatic marker extraction not implemented. "
            "Use calibrate_from_correspondences() with manually measured points."
        )
    
    def calibrate_interactive(
        self,
        rb_data: dict,
        video_path: Path,
        calib_data
    ) -> CalibrationResult:
        """
        Interactive calibration by clicking corresponding points.
        
        Shows video with pyramid overlay and lets user click corresponding points.
        
        Args:
            rb_data: Rigid body tracking data
            video_path: Path to video
            calib_data: Camera calibration
            
        Returns:
            CalibrationResult
        """
        import cv2
        
        print("\n" + "="*70)
        print("INTERACTIVE CALIBRATION")
        print("="*70)
        print("\nInstructions:")
        print("1. Video will show pyramid markers")
        print("2. Click on a marker in the video")
        print("3. Enter the corresponding marker number from scan")
        print("4. Repeat for at least 3 markers")
        print("5. Press 'c' when done to calculate")
        print("6. Press 'q' to quit")
        
        # TODO: Implement interactive point selection
        raise NotImplementedError("Interactive calibration not yet implemented")
    
    def save_calibration(self, result: CalibrationResult, output_path: Path) -> None:
        """
        Save calibration result to file.
        
        Args:
            result: CalibrationResult to save
            output_path: Path to save calibration
        """
        calib_data = {
            'rotation_matrix': result.rotation_matrix.tolist(),
            'translation_vector_mm': result.translation_vector.tolist(),
            'rmse_mm': float(result.rmse),
            'n_points': int(result.n_points),
            'point_errors_mm': result.point_errors.tolist(),
            'unit': 'mm'
        }
        
        with open(output_path, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"\n✓ Calibration saved to: {output_path}")
    
    def load_calibration(self, calib_path: Path) -> CalibrationResult:
        """
        Load calibration from file.
        
        Args:
            calib_path: Path to calibration file
            
        Returns:
            CalibrationResult
        """
        with open(calib_path, 'r') as f:
            data = json.load(f)
        
        return CalibrationResult(
            rotation_matrix=np.array(data['rotation_matrix']),
            translation_vector=np.array(data['translation_vector_mm']),
            rmse=data['rmse_mm'],
            n_points=data['n_points'],
            point_errors=np.array(data['point_errors_mm'])
        )
    
    def get_rotation_euler_angles(self, R: np.ndarray) -> Dict[str, float]:
        """
        Convert rotation matrix to Euler angles (XYZ convention).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Dict with roll, pitch, yaw in degrees
        """
        from scipy.spatial.transform import Rotation
        
        rot = Rotation.from_matrix(R)
        euler = rot.as_euler('xyz', degrees=True)
        
        return {
            'roll_x': euler[0],
            'pitch_y': euler[1],
            'yaw_z': euler[2]
        }
    
    def print_rotation_analysis(self, R: np.ndarray) -> None:
        """
        Print detailed analysis of rotation matrix.
        
        Args:
            R: 3x3 rotation matrix
        """
        print("\n" + "="*70)
        print("ROTATION ANALYSIS")
        print("="*70)
        
        # Euler angles
        angles = self.get_rotation_euler_angles(R)
        print(f"\nEuler Angles (XYZ convention):")
        print(f"  Roll (X):  {angles['roll_x']:8.3f}°")
        print(f"  Pitch (Y): {angles['pitch_y']:8.3f}°")
        print(f"  Yaw (Z):   {angles['yaw_z']:8.3f}°")
        
        # Axis-angle
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(R)
        rotvec = rot.as_rotvec()
        angle = np.linalg.norm(rotvec)
        if angle > 1e-6:
            axis = rotvec / angle
            print(f"\nAxis-Angle Representation:")
            print(f"  Angle: {np.degrees(angle):.3f}°")
            print(f"  Axis:  [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
        else:
            print(f"\nAxis-Angle: Nearly identity (angle: {np.degrees(angle):.6f}°)")
        
        # Check if close to common rotations
        print(f"\nComparison to Common Rotations:")
        
        common_rotations = {
            "Identity": np.eye(3),
            "90° around Z": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            "-90° around Z": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            "180° around Z": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "90° around X": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            "90° around Y": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        }
        
        for name, R_common in common_rotations.items():
            diff = np.linalg.norm(R - R_common, 'fro')
            if diff < 0.1:
                print(f"  {name:15s}: diff = {diff:.6f} ← CLOSE MATCH")
            else:
                print(f"  {name:15s}: diff = {diff:.6f}")


def example_calibration_manual():
    """
    Example: Manual calibration with measured point correspondences.
    """
    print("="*70)
    print("EXAMPLE: MANUAL CALIBRATION")
    print("="*70)
    
    # Initialize calibrator
    json_path = Path("ModelMire3DSLAM.json")
    calibrator = PyramidCalibrator(json_path)
    
    # Example: Suppose you measured 4 marker positions
    # You identified these markers in the scan and measured them with OptiTrack
    
    # Marker indices from scan (e.g., markers 0, 1, 5, 10)
    marker_indices = [0, 1, 5, 10]
    
    # Their positions in pyramid frame (from scan, in mm)
    pyramid_points_mm = calibrator.markers_pyramid_frame_mm[marker_indices]
    
    print(f"\nMarker positions in pyramid frame (mm):")
    for i, idx in enumerate(marker_indices):
        print(f"  Marker {idx}: {pyramid_points_mm[i]}")
    
    # MANUAL INPUT REQUIRED: Measure these same markers with OptiTrack
    # When pyramid is at known position, record marker positions in OptiTrack frame
    # This is what you need to provide:
    optitrack_points_mm = np.array([
        # Example values - REPLACE WITH YOUR MEASUREMENTS
        [-10.0, -50.0, 120.0],  # Marker 0 in OptiTrack frame
        [-9.5, -55.0, 125.0],    # Marker 1 in OptiTrack frame
        [15.0, -45.0, 122.0],    # Marker 5 in OptiTrack frame
        [20.0, -30.0, 118.0],    # Marker 10 in OptiTrack frame
    ])
    
    print(f"\n⚠️  REPLACE with your actual OptiTrack measurements!")
    print(f"Marker positions in OptiTrack frame (mm):")
    for i, idx in enumerate(marker_indices):
        print(f"  Marker {idx}: {optitrack_points_mm[i]}")
    
    # Calibrate
    result = calibrator.calibrate_from_correspondences(
        pyramid_points_mm,
        optitrack_points_mm
    )
    
    # Analyze rotation
    calibrator.print_rotation_analysis(result.rotation_matrix)
    
    # Save calibration
    output_path = Path("pyramid_calibration.json")
    calibrator.save_calibration(result, output_path)
    
    return result


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PYRAMID CALIBRATION SYSTEM")
    print("="*70)
    print("\nThis script estimates the rotation between:")
    print("  - Pyramid local frame (from 3D scan)")
    print("  - OptiTrack rigid body frame (constellation)")
    print("\nTranslation is known from LocalReferential in JSON.")
    print("Rotation is estimated from point correspondences using Kabsch algorithm.")
    print("="*70)
    
    # Run example (you need to provide actual measurements)
    try:
        result = example_calibration_manual()
    except FileNotFoundError:
        print("\n⚠️  ModelMire3DSLAM.json not found in current directory")
        print("Place the JSON file here and provide actual OptiTrack measurements")
