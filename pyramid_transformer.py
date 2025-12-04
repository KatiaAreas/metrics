# pyramid_manager.py

"""
PyramidManager - Integrated with Calibration System

Manages pyramid geometry and transformations with support for:
- Loading from JSON (meters converted to mm)
- Manual rotation specification
- Automatic calibration from point correspondences
- Saving/loading calibration files
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CalibrationInfo:
    """Calibration metadata."""
    method: str  # 'manual', 'kabsch', 'loaded'
    rmse_mm: Optional[float] = None
    n_points: Optional[int] = None
    calibration_file: Optional[Path] = None


class PyramidManager:
    """
    Manages pyramid geometry and coordinate transformations.
    
    Supports:
    - Translation from JSON LocalReferential
    - Rotation from manual specification or calibration
    - Kabsch algorithm for automatic calibration
    - All units in mm (millimeters)
    """
    
    def __init__(
        self,
        pyramid_json_path: Path,
        rotation_matrix: Optional[np.ndarray] = None,
        calibration_file: Optional[Path] = None
    ):
        """
        Initialize PyramidManager.
        
        Args:
            pyramid_json_path: Path to ModelMire3DSLAM.json
            rotation_matrix: Optional 3x3 rotation matrix (manual specification)
            calibration_file: Optional path to calibration JSON file
            
        Priority: calibration_file > rotation_matrix > identity
        """
        self.pyramid_json_path = pyramid_json_path
        self.markers_pyramid_frame_mm = None  # In mm
        self.local_referential_mm = None  # In mm
        self.T_optitrack_pyramid = None  # 4x4 transform (mm units)
        self.rotation_matrix = None  # 3x3 rotation
        self.calibration_info = None
        
        # Load pyramid geometry
        self._load_pyramid_data()
        
        # Set rotation
        if calibration_file is not None and calibration_file.exists():
            self._load_calibration(calibration_file)
        elif rotation_matrix is not None:
            self.set_rotation(rotation_matrix)
            self.calibration_info = CalibrationInfo(method='manual')
        else:
            self.rotation_matrix = np.eye(3)
            self.calibration_info = CalibrationInfo(method='identity')
        
        # Compute transformation matrix
        self._compute_transform()
    
    def _load_pyramid_data(self) -> None:
        """Load pyramid geometry from JSON and convert to mm."""
        with open(self.pyramid_json_path, 'r') as f:
            data = json.load(f)
        
        # Extract marker positions (in meters from JSON)
        markers = []
        for marker in data['Markers']:
            pos = marker['LocalPosition']
            x = pos.get('x', 0.0)
            y = pos.get('y', 0.0)
            z = pos.get('z', 0.0)
            markers.append([x, y, z])
        
        # Convert to mm
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
        
        print(f"✓ Loaded {len(self.markers_pyramid_frame_mm)} markers")
        print(f"✓ Local referential (OptiTrack origin): {self.local_referential_mm} mm")
    
    def _compute_transform(self) -> None:
        """
        Compute 4x4 transformation matrix from pyramid to OptiTrack frame.
        
        T = [R | t]  where:
            [0 | 1]
        - R: 3x3 rotation matrix
        - t: 3x1 translation vector (mm)
        """
        self.T_optitrack_pyramid = np.eye(4)
        self.T_optitrack_pyramid[0:3, 0:3] = self.rotation_matrix
        self.T_optitrack_pyramid[0:3, 3] = self.local_referential_mm
    
    def set_rotation(self, rotation_matrix: np.ndarray) -> None:
        """
        Set rotation matrix manually.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {rotation_matrix.shape}")
        
        # Verify it's a valid rotation (orthogonal, det=1)
        det = np.linalg.det(rotation_matrix)
        ortho_error = np.linalg.norm(rotation_matrix @ rotation_matrix.T - np.eye(3))
        
        if not (0.99 < det < 1.01):
            print(f"⚠️  Warning: Determinant = {det:.6f}, expected 1.0")
        
        if ortho_error > 1e-6:
            print(f"⚠️  Warning: Orthogonality error = {ortho_error:.10f}")
        
        self.rotation_matrix = rotation_matrix
        self._compute_transform()
        
        if self.calibration_info and self.calibration_info.method == 'identity':
            self.calibration_info = CalibrationInfo(method='manual')
    
    def set_rotation_from_angle(self, angle_degrees: float, axis: str = 'z') -> None:
        """
        Set rotation from angle and axis.
        
        Args:
            angle_degrees: Rotation angle in degrees
            axis: 'x', 'y', or 'z'
        """
        R = create_rotation_matrix(angle_degrees, axis)
        self.set_rotation(R)
    
    def calibrate_from_correspondences(
        self,
        marker_indices: List[int],
        optitrack_points_mm: np.ndarray
    ) -> Dict:
        """
        Calibrate rotation using point correspondences (Kabsch algorithm).
        
        Args:
            marker_indices: List of marker indices (e.g., [0, 1, 5, 10])
            optitrack_points_mm: Nx3 array of corresponding points in OptiTrack frame (mm)
            
        Returns:
            Dict with calibration results (rmse, rotation_matrix, etc.)
        """
        if len(marker_indices) != optitrack_points_mm.shape[0]:
            raise ValueError(
                f"Number of indices ({len(marker_indices)}) must match "
                f"number of points ({optitrack_points_mm.shape[0]})"
            )
        
        if len(marker_indices) < 3:
            raise ValueError("Need at least 3 point correspondences")
        
        # Get corresponding pyramid points
        pyramid_points_mm = self.markers_pyramid_frame_mm[marker_indices]
        
        print(f"\n{'='*70}")
        print(f"CALIBRATING FROM {len(marker_indices)} POINT CORRESPONDENCES")
        print(f"{'='*70}")
        
        print(f"\nMarker indices: {marker_indices}")
        print(f"\nPyramid points (mm):")
        for i, idx in enumerate(marker_indices):
            print(f"  [{idx:2d}] {pyramid_points_mm[i]}")
        
        print(f"\nOptiTrack points (mm):")
        for i, idx in enumerate(marker_indices):
            print(f"  [{idx:2d}] {optitrack_points_mm[i]}")
        
        # Estimate rotation using Kabsch algorithm
        R = kabsch_algorithm(pyramid_points_mm, optitrack_points_mm)
        
        # Compute errors
        pyramid_transformed = (R @ pyramid_points_mm.T).T + self.local_referential_mm
        errors = np.linalg.norm(pyramid_transformed - optitrack_points_mm, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        
        print(f"\n{'='*70}")
        print(f"CALIBRATION RESULTS")
        print(f"{'='*70}")
        print(f"RMSE:       {rmse:.4f} mm")
        print(f"Max error:  {np.max(errors):.4f} mm")
        print(f"Min error:  {np.min(errors):.4f} mm")
        print(f"Mean error: {np.mean(errors):.4f} mm")
        
        print(f"\nPer-point errors (mm):")
        for i, (idx, err) in enumerate(zip(marker_indices, errors)):
            print(f"  Marker {idx:2d}: {err:.4f} mm")
        
        print(f"\nEstimated Rotation Matrix:")
        print(R)
        
        # Verify rotation validity
        det = np.linalg.det(R)
        ortho_error = np.linalg.norm(R @ R.T - np.eye(3))
        print(f"\nDeterminant: {det:.8f} (should be 1.0)")
        print(f"Orthogonality error: {ortho_error:.10f} (should be ~0)")
        
        if rmse < 1.0:
            print(f"\n✓ Excellent calibration! (RMSE < 1 mm)")
        elif rmse < 5.0:
            print(f"\n✓ Good calibration (RMSE < 5 mm)")
        elif rmse < 10.0:
            print(f"\n⚠️  Acceptable calibration (RMSE < 10 mm)")
        else:
            print(f"\n❌ Poor calibration (RMSE > 10 mm) - check measurements!")
        
        # Apply rotation
        self.set_rotation(R)
        self.calibration_info = CalibrationInfo(
            method='kabsch',
            rmse_mm=rmse,
            n_points=len(marker_indices)
        )
        
        return {
            'rotation_matrix': R,
            'translation_mm': self.local_referential_mm,
            'rmse_mm': rmse,
            'errors_mm': errors,
            'n_points': len(marker_indices),
            'marker_indices': marker_indices
        }
    
    def save_calibration(self, output_path: Path, extra_info: Optional[Dict] = None) -> None:
        """
        Save calibration to JSON file.
        
        Args:
            output_path: Path to save calibration
            extra_info: Optional extra information to save
        """
        calib_data = {
            'rotation_matrix': self.rotation_matrix.tolist(),
            'translation_mm': self.local_referential_mm.tolist(),
            'method': self.calibration_info.method if self.calibration_info else 'unknown',
            'unit': 'mm'
        }
        
        if self.calibration_info:
            if self.calibration_info.rmse_mm is not None:
                calib_data['rmse_mm'] = float(self.calibration_info.rmse_mm)
            if self.calibration_info.n_points is not None:
                calib_data['n_points'] = int(self.calibration_info.n_points)
        
        if extra_info:
            calib_data['extra'] = extra_info
        
        with open(output_path, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"\n✓ Calibration saved to: {output_path}")
        self.calibration_info.calibration_file = output_path
    
    def _load_calibration(self, calib_path: Path) -> None:
        """Load calibration from JSON file."""
        with open(calib_path, 'r') as f:
            data = json.load(f)
        
        self.rotation_matrix = np.array(data['rotation_matrix'])
        
        # Translation from file or from JSON (file takes precedence for verification)
        if 'translation_mm' in data:
            file_translation = np.array(data['translation_mm'])
            if not np.allclose(file_translation, self.local_referential_mm, atol=0.1):
                print(f"⚠️  Warning: Translation mismatch!")
                print(f"   From calibration file: {file_translation}")
                print(f"   From pyramid JSON: {self.local_referential_mm}")
        
        self.calibration_info = CalibrationInfo(
            method=data.get('method', 'loaded'),
            rmse_mm=data.get('rmse_mm'),
            n_points=data.get('n_points'),
            calibration_file=calib_path
        )
        
        print(f"✓ Loaded calibration from: {calib_path}")
        if self.calibration_info.rmse_mm:
            print(f"  RMSE: {self.calibration_info.rmse_mm:.4f} mm")
    
    # === Core transformation methods (unchanged) ===
    
    def get_pyramid_summits_in_pyramid_frame(self) -> np.ndarray:
        """Get all marker positions in pyramid local frame (mm)."""
        return self.markers_pyramid_frame_mm.copy()
    
    def get_summits_in_optitrack_frame(self) -> np.ndarray:
        """
        Get all markers in OptiTrack rigid body frame (mm).
        
        This is the main function for display.
        """
        return self.transform_pyramid_to_optitrack(self.markers_pyramid_frame_mm)
    
    def transform_pyramid_to_optitrack(self, points_pyramid_mm: np.ndarray) -> np.ndarray:
        """
        Transform points from pyramid frame to OptiTrack frame (both in mm).
        
        Args:
            points_pyramid_mm: Nx3 array in pyramid frame (mm)
            
        Returns:
            Nx3 array in OptiTrack frame (mm)
        """
        n_points = points_pyramid_mm.shape[0]
        points_hom = np.hstack([points_pyramid_mm, np.ones((n_points, 1))])
        points_optitrack = (self.T_optitrack_pyramid @ points_hom.T).T[:, 0:3]
        return points_optitrack
    
    def get_specific_summits(self, indices: List[int]) -> np.ndarray:
        """Get specific markers by index (mm)."""
        return self.markers_pyramid_frame_mm[indices].copy()
    
    def get_transform_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix (mm)."""
        return self.T_optitrack_pyramid.copy()
    
    def get_transform_matrix_meters(self) -> np.ndarray:
        """Get 4x4 transformation matrix in meters (for compatibility with OptiTrack)."""
        T_m = self.T_optitrack_pyramid.copy()
        T_m[0:3, 3] /= 1000.0  # Convert translation to meters
        return T_m
    
    def get_summits_in_optitrack_frame_meters(self) -> np.ndarray:
        """Get all markers in OptiTrack frame in meters (for compatibility)."""
        return self.get_summits_in_optitrack_frame() / 1000.0
    
    def print_info(self) -> None:
        """Print PyramidManager information."""
        print(f"\n{'='*70}")
        print("PYRAMID MANAGER INFO")
        print(f"{'='*70}")
        print(f"Markers: {len(self.markers_pyramid_frame_mm)}")
        print(f"Units: mm (millimeters)")
        print(f"Local referential: {self.local_referential_mm} mm")
        
        has_rotation = not np.allclose(self.rotation_matrix, np.eye(3))
        print(f"Rotation: {'Yes' if has_rotation else 'Identity (no rotation)'}")
        
        if self.calibration_info:
            print(f"\nCalibration method: {self.calibration_info.method}")
            if self.calibration_info.rmse_mm:
                print(f"Calibration RMSE: {self.calibration_info.rmse_mm:.4f} mm")
            if self.calibration_info.n_points:
                print(f"Calibration points: {self.calibration_info.n_points}")
            if self.calibration_info.calibration_file:
                print(f"Calibration file: {self.calibration_info.calibration_file}")
        
        if has_rotation:
            print(f"\nRotation matrix:")
            print(self.rotation_matrix)
            
            # Show Euler angles
            angles = get_euler_angles(self.rotation_matrix)
            print(f"\nEuler angles (XYZ):")
            print(f"  Roll (X):  {angles['roll_x']:7.2f}°")
            print(f"  Pitch (Y): {angles['pitch_y']:7.2f}°")
            print(f"  Yaw (Z):   {angles['yaw_z']:7.2f}°")
    
    def __repr__(self) -> str:
        has_rotation = not np.allclose(self.rotation_matrix, np.eye(3))
        method = self.calibration_info.method if self.calibration_info else 'unknown'
        return (f"PyramidManager(markers={len(self.markers_pyramid_frame_mm)}, "
                f"has_rotation={has_rotation}, method={method})")


# === Helper functions ===

def create_rotation_matrix(angle_degrees: float, axis: str = 'z') -> np.ndarray:
    """
    Create rotation matrix from angle and axis.
    
    Args:
        angle_degrees: Rotation angle in degrees
        axis: 'x', 'y', or 'z'
        
    Returns:
        3x3 rotation matrix
    """
    angle_rad = np.deg2rad(angle_degrees)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    
    if axis.lower() == 'z':
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'x':
        return np.array([
            [1,  0, 0],
            [0,  c, -s],
            [0,  s,  c]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")


def kabsch_algorithm(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm: find optimal rotation from P to Q.
    
    Finds R such that: R @ P ≈ Q (after centering)
    
    Args:
        P: Nx3 source points
        Q: Nx3 target points
        
    Returns:
        3x3 rotation matrix
    """
    # Center point sets
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    print(f"\nKabsch algorithm:")
    print(f"  Centroid P: {centroid_P}")
    print(f"  Centroid Q: {centroid_Q}")
    
    # Covariance matrix
    H = P_centered.T @ Q_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation
    R = Vt.T @ U.T
    
    # Handle reflection (det(R) = -1)
    if np.linalg.det(R) < 0:
        print("  Detected reflection - correcting...")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    print(f"  Singular values: {S}")
    
    return R


def get_euler_angles(R: np.ndarray) -> Dict[str, float]:
    """Convert rotation matrix to Euler angles (XYZ, degrees)."""
    try:
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(R)
        euler = rot.as_euler('xyz', degrees=True)
        return {'roll_x': euler[0], 'pitch_y': euler[1], 'yaw_z': euler[2]}
    except ImportError:
        # Fallback if scipy not available
        return {'roll_x': 0.0, 'pitch_y': 0.0, 'yaw_z': 0.0}


# === Convenience function ===

def load_pyramid_summits(
    pyramid_json_path: Path,
    rotation_matrix: Optional[np.ndarray] = None,
    calibration_file: Optional[Path] = None,
    unit: str = 'mm'
) -> np.ndarray:
    """
    Convenience function to load pyramid summits.
    
    Args:
        pyramid_json_path: Path to JSON
        rotation_matrix: Optional rotation matrix
        calibration_file: Optional calibration file
        unit: 'mm' or 'm'
        
    Returns:
        Nx3 array of summits in specified unit
    """
    manager = PyramidManager(pyramid_json_path, rotation_matrix, calibration_file)
    
    if unit == 'mm':
        return manager.get_summits_in_optitrack_frame()
    elif unit == 'm':
        return manager.get_summits_in_optitrack_frame_meters()
    else:
        raise ValueError(f"Invalid unit: {unit}. Must be 'mm' or 'm'")


if __name__ == "__main__":
    print("PyramidManager with Integrated Calibration System")
    print("="*70)
    print("\nSee pyramid_calibration.py for calibration examples")
    print("See calibration_example.py for complete workflow")