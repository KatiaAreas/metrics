"""
Pyramid Transformer: Transform between pyramid frame and OptiTrack frame.

Frame definitions:
- Pyramid frame: Origin at point 0, Z axis (0→1), X axis (perpendicular to Z, in plane 0-1-4), Y = Z × X
- Constellation frame: Barycenter of points 18-21, Y toward point 20, Z perpendicular to plane
- OptiTrack frame: From rigid body orientation (given as input)

Transformation chain:
  R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation

Units: All positions in METERS throughout (JSON data assumed to be in meters)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class ConstellationMatch:
    """Match between 3D model points and OptiTrack markers."""
    json_point: int  # Point index in JSON (18-21)
    marker_name: str  # OptiTrack marker name (Marker 001-004)
    distance_error: float  # Geometric matching error


def compute_rotation_from_point_correspondence(
    source_points: np.ndarray,
    target_points: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Compute rotation matrix using SVD (Kabsch algorithm).

    Args:
        source_points: N×3 array of points in source frame
        target_points: N×3 array of corresponding points in target frame

    Returns:
        R: 3×3 rotation matrix from source to target frame
        rmse: Root mean square error after transformation
    """
    # Center the points
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    # Compute covariance matrix
    H = centered_source.T @ centered_target

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Ensure right-handed coordinate system (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute RMSE
    transformed = (R @ centered_source.T).T + centroid_target
    rmse = np.sqrt(np.mean(np.sum((transformed - target_points)**2, axis=1)))

    return R, rmse


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        quaternion: [x, y, z, w] quaternion

    Returns:
        3×3 rotation matrix
    """
    try:
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(quaternion)
        return rot.as_matrix()
    except ImportError:
        # Manual implementation
        x, y, z, w = quaternion
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])


def transform_world_to_local_frame(
    positions_world: Dict[str, np.ndarray],
    rb_position: np.ndarray,
    rb_quaternion: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Transform positions from world frame to rigid body local frame.

    Args:
        positions_world: Dict of positions in world frame (meters)
        rb_position: Rigid body position in world frame (meters)
        rb_quaternion: Rigid body orientation [x, y, z, w]

    Returns:
        Dict of positions in local frame (meters)
    """
    # Get rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(rb_quaternion)

    # Transform: p_local = R^T × (p_world - t_world)
    result = {}
    for name, pos_world in positions_world.items():
        pos_local = R.T @ (pos_world - rb_position)
        result[name] = pos_local

    return result


def extract_marker_positions_from_rb_data(
    rb_data,
    frame_id: int = 0,
    rigid_body_name: str = "Pyramid_RB"
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Extract marker positions and rigid body pose from rb_data.

    Args:
        rb_data: OptiTrack rigid body data
        frame_id: Frame index
        rigid_body_name: Name of rigid body

    Returns:
        marker_positions_local: Dict of marker positions in constellation local frame (meters)
        rb_position: Rigid body position in world frame (meters)
        rb_quaternion: Rigid body orientation [x, y, z, w]
    """
    pyramid_rb = rb_data[rigid_body_name][frame_id]

    # Extract marker positions (in world frame, meters)
    marker_positions_world = {}
    for marker_name in ['Marker 001', 'Marker 002', 'Marker 003', 'Marker 004']:
        marker = pyramid_rb.data.marker_positions[marker_name]
        if hasattr(marker, '_x'):
            pos = np.array([marker._x, marker._y, marker._z])
        else:
            pos = np.array(marker)
        marker_positions_world[marker_name] = pos

    # Extract rigid body pose
    rb_pos = pyramid_rb.data.position
    rb_position = np.array([rb_pos._x, rb_pos._y, rb_pos._z])

    rb_ori = pyramid_rb.data.orientation
    rb_quaternion = np.array([rb_ori._x, rb_ori._y, rb_ori._z, rb_ori._w])

    # Convert marker positions to constellation local frame
    marker_positions_local = transform_world_to_local_frame(
        marker_positions_world,
        rb_position,
        rb_quaternion
    )

    return marker_positions_local, rb_position, rb_quaternion


class PyramidTransformer:
    """
    Transform points between pyramid frame and OptiTrack frame.

    Transformation: R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation

    All units are in METERS.
    """

    def __init__(self, json_path: Path):
        """
        Initialize transformer.

        Args:
            json_path: Path to 3D model JSON file (assumed to contain positions in meters)
        """
        self.json_path = json_path
        self.points_m = None  # All points 0-21 in meters
        self.pyramid_origin_m = None  # Point 0 in meters
        self.R_pyramid = None  # Rotation matrix of pyramid frame (world frame)
        self.constellation_indices = [18, 19, 20, 21]  # Points forming constellation
        self.constellation_barycenter_m = None  # Barycenter in meters
        self.R_constellation = None  # Rotation matrix of constellation frame (world frame)
        self.marker_match = None  # Dict mapping marker names to point indices
        self.referential_point_m = None  # Reference point from JSON (if available)

        # Load and process JSON
        self._load_json()
        self._compute_pyramid_frame()
        self._compute_constellation_frame()

        # Compute relative rotation
        self.compute_relative_rotation()

    def _load_json(self) -> None:
        """Load 3D model JSON and extract points (in meters)."""
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # Extract points 0-21
        markers = []
        for i, marker in enumerate(data['Markers']):
            if i > 21:
                break
            pos = marker['LocalPosition']
            x = pos.get('x', 0.0)
            y = pos.get('y', 0.0)
            z = pos.get('z', 0.0)
            markers.append([x, y, z])

        # Store in meters (assuming JSON data is in meters)
        self.points_m = np.array(markers)

        # Extract referential point (LocalReferential position) if available
        if 'LocalReferential' in data:
            ref_pos = data['LocalReferential']['position']
            ref_x = ref_pos.get('x', 0.0)
            ref_y = ref_pos.get('y', 0.0)
            ref_z = ref_pos.get('z', 0.0)
            self.referential_point_m = np.array([ref_x, ref_y, ref_z])
            print(f"✓ Loaded referential point: {self.referential_point_m} m")
        else:
            self.referential_point_m = None
            print("⚠️  No LocalReferential found in JSON")

        print(f"✓ Loaded {len(self.points_m)} points from JSON (units: meters)")

    def _compute_pyramid_frame(self) -> None:
        """
        Compute pyramid frame definition.

        Origin: Point 0
        Z axis: Direction from point 0 to point 1
        X axis: Perpendicular to Z, in plane formed by points 0, 1, 4
        Y axis: Z × X (right-handed orthonormal frame)
        """
        # Origin
        self.pyramid_origin_m = self.points_m[0].copy()

        # =========================================================================
        # Z axis: 0 → 1
        # =========================================================================
        z_vec = self.points_m[1] - self.points_m[0]
        z_axis = z_vec / np.linalg.norm(z_vec)

        # =========================================================================
        # X axis: Perpendicular to Z, in plane 0-1-4
        # =========================================================================
        # Vector from 0 to 4
        vec_0_to_4 = self.points_m[4] - self.points_m[0]

        # Project this vector onto the plane perpendicular to Z
        # Projection formula: v_proj = v - (v · n)n
        vec_0_to_4_proj = vec_0_to_4 - np.dot(vec_0_to_4, z_axis) * z_axis
        x_axis = vec_0_to_4_proj / np.linalg.norm(vec_0_to_4_proj)

        # =========================================================================
        # Y axis: Z × X (right-handed orthonormal frame)
        # =========================================================================
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Recompute X to ensure perfect orthonormality (X = Y × Z)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Build rotation matrix [X | Y | Z]
        self.R_pyramid = np.column_stack([x_axis, y_axis, z_axis])

        # Verify orthonormality
        det = np.linalg.det(self.R_pyramid)
        orthogonality = np.linalg.norm(self.R_pyramid @ self.R_pyramid.T - np.eye(3))

        print(f"\n✓ Pyramid frame computed:")
        print(f"  Origin (point 0): {self.pyramid_origin_m} m")
        print(f"  Z axis (0→1): {z_axis}")
        print(f"  X axis (perpendicular to Z, in plane 0-1-4): {x_axis}")
        print(f"  Y axis (Z×X, right-handed): {y_axis}")
        print(f"\n  Verification:")
        print(f"    Determinant: {det:.6f} (should be 1.0)")
        print(f"    Orthogonality error: {orthogonality:.2e} (should be ~0)")

        # Additional verification
        dot_xz = np.dot(x_axis, z_axis)
        print(f"    X·Z = {dot_xz:.2e} (should be ~0)")

        # Check that X lies in plane 0-1-4
        vec_01 = self.points_m[1] - self.points_m[0]
        vec_04 = self.points_m[4] - self.points_m[0]
        plane_normal = np.cross(vec_01, vec_04)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        dot_x_plane = np.dot(x_axis, plane_normal)
        print(f"    X·(plane normal) = {dot_x_plane:.2e} (should be ~0 if X is in plane)")

    def _compute_constellation_frame(self) -> None:
        """
        Compute constellation frame definition.

        Origin: Barycenter of points 18, 19, 20, 21 (or referential point if available)
        Y axis: Direction toward point 20 (projected on plane of 4 points)
        Z axis: Normal to plane of 4 points
        X axis: Y × Z (right-handed, on plane)
        """
        # Get constellation points
        constellation_points = self.points_m[self.constellation_indices]

        # Barycenter (origin)
        # Use referential point if available, otherwise compute barycenter
        if self.referential_point_m is not None:
            self.constellation_barycenter_m = self.referential_point_m
            print(f"\n✓ Using referential point as constellation origin: {self.constellation_barycenter_m} m")
        else:
            self.constellation_barycenter_m = np.mean(constellation_points, axis=0)
            print(f"\n✓ Using computed barycenter as constellation origin: {self.constellation_barycenter_m} m")

        # Compute plane normal using cross product of two vectors in plane
        # Use points 18, 19, 20 to define plane
        v1 = constellation_points[1] - constellation_points[0]  # 19 - 18
        v2 = constellation_points[2] - constellation_points[0]  # 20 - 18

        z_axis = np.cross(v1, v2)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Y axis: Direction from barycenter to point 20
        y_vec = self.points_m[20] - self.constellation_barycenter_m
        # Project onto plane (remove Z component)
        y_vec = y_vec - np.dot(y_vec, z_axis) * z_axis
        y_axis = y_vec / np.linalg.norm(y_vec)

        # X axis: Y × Z (on plane, right-handed)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Build rotation matrix [X | Y | Z]
        self.R_constellation = np.column_stack([x_axis, y_axis, z_axis])

        print(f"✓ Constellation frame computed:")
        print(f"  Origin: {self.constellation_barycenter_m} m")
        print(f"  X axis (on plane): {x_axis}")
        print(f"  Y axis (toward point 20): {y_axis}")
        print(f"  Z axis (normal to plane): {z_axis}")

    def compute_relative_rotation(self) -> np.ndarray:
        """
        Compute relative rotation from pyramid frame to constellation frame.

        R_pyramid_to_constellation transforms vectors from pyramid frame to constellation frame:
          v_constellation = R_pyramid_to_constellation @ v_pyramid

        This is computed as:
          R_pyramid_to_constellation = R_constellation^T @ R_pyramid

        Returns:
            3×3 rotation matrix from pyramid to constellation frame
        """
        # Relative rotation from pyramid to constellation
        self.R_pyramid_to_constellation = self.R_constellation.T @ self.R_pyramid

        # Verify it's a valid rotation matrix
        det = np.linalg.det(self.R_pyramid_to_constellation)
        orthogonality = np.linalg.norm(
            self.R_pyramid_to_constellation @ self.R_pyramid_to_constellation.T - np.eye(3)
        )

        print(f"\n✓ Relative rotation R_pyramid_to_constellation computed:")
        print(f"  Determinant: {det:.6f} (should be 1.0)")
        print(f"  Orthogonality error: {orthogonality:.2e} (should be ~0)")
        print(f"  Matrix:\n{self.R_pyramid_to_constellation}")

        return self.R_pyramid_to_constellation

    def match_constellation_markers(
        self,
        marker_positions_m: Dict[str, np.ndarray],
        initial_guess: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Match OptiTrack markers to 3D model constellation points using geometry.

        Uses brute force to find the matching that minimizes geometric distortion.

        Args:
            marker_positions_m: Dict of marker names to positions in meters
                               (in constellation local frame)
            initial_guess: Optional initial matching to try first
                          {'Marker 001': 21, 'Marker 002': 20, ...}

        Returns:
            Best matching dict: marker_name → point_index
        """
        marker_names = ['Marker 001', 'Marker 002', 'Marker 003', 'Marker 004']
        point_indices = [18, 19, 20, 21]

        # Get constellation points in world frame
        constellation_points_world = self.points_m[point_indices]

        # Transform to constellation frame (centered at barycenter)
        constellation_points_local = (self.R_constellation.T @
                                     (constellation_points_world - self.constellation_barycenter_m).T).T

        # Get marker positions as array (already in constellation frame)
        marker_positions = np.array([marker_positions_m[name] for name in marker_names])

        # Try initial guess first if provided
        if initial_guess:
            print(f"\n✓ Testing initial guess:")
            for marker, point in initial_guess.items():
                print(f"  {marker} → Point {point}")

            # Check if it's valid
            guess_indices = [initial_guess[name] for name in marker_names]
            if set(guess_indices) == set(point_indices):
                error = self._compute_matching_error(
                    marker_positions,
                    constellation_points_local,
                    [point_indices.index(idx) for idx in guess_indices]
                )
                print(f"  Error: {error:.6f} m²")

                # If error is reasonable, use it
                if error < 0.01:  # 0.1m RMS error = 0.01m² for 4 points
                    print(f"  ✓ Initial guess accepted (error < 0.01 m²)")
                    self.marker_match = initial_guess
                    return initial_guess

        # Brute force: try all permutations
        print(f"\n⚙ Brute force matching {len(marker_names)}! = 24 permutations...")

        import itertools
        best_error = float('inf')
        best_permutation = None

        for perm in itertools.permutations(range(4)):
            error = self._compute_matching_error(
                marker_positions,
                constellation_points_local,
                list(perm)
            )

            if error < best_error:
                best_error = error
                best_permutation = perm

        # Build result
        result = {
            marker_names[i]: point_indices[best_permutation[i]]
            for i in range(4)
        }

        print(f"\n✓ Best matching found (error: {best_error:.6f} m²):")
        for marker, point in result.items():
            print(f"  {marker} → Point {point}")

        self.marker_match = result
        return result

    def _compute_matching_error(
        self,
        markers: np.ndarray,
        points: np.ndarray,
        permutation: List[int]
    ) -> float:
        """
        Compute geometric error for a given marker-to-point matching.

        Computes sum of squared distance differences between all pairs.

        Args:
            markers: 4×3 array of marker positions
            points: 4×3 array of point positions
            permutation: [i,j,k,l] meaning marker[0]→point[i], marker[1]→point[j], etc.

        Returns:
            Total squared distance error
        """
        # Reorder points according to permutation
        reordered_points = points[permutation]

        # Compute pairwise distances for markers
        marker_dists = np.array([
            [np.linalg.norm(markers[i] - markers[j]) for j in range(4)]
            for i in range(4)
        ])

        # Compute pairwise distances for points
        point_dists = np.array([
            [np.linalg.norm(reordered_points[i] - reordered_points[j]) for j in range(4)]
            for i in range(4)
        ])

        # Sum of squared differences
        error = np.sum((marker_dists - point_dists) ** 2)

        return error

    def set_optitrack_rotation(self, R_constellation_to_optitrack: np.ndarray) -> None:
        """
        Set the rotation from constellation frame to OptiTrack frame.

        This is given as input (3×3 matrix).

        Then compute: R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation

        Args:
            R_constellation_to_optitrack: 3×3 rotation matrix that transforms vectors
                                          from constellation frame to OptiTrack frame
        """
        self.R_constellation_to_optitrack = R_constellation_to_optitrack

        # Compute full pyramid-to-OptiTrack rotation
        self.R_pyramid_to_optitrack = R_constellation_to_optitrack @ self.R_pyramid_to_constellation

        # Compute full transform (rotation + translation)
        self._compute_full_transform()

        print(f"\n✓ OptiTrack rotation set:")
        print(f"  R_constellation_to_optitrack (given as input):")
        print(f"{R_constellation_to_optitrack}")

        print(f"\n✓ Full pyramid-to-OptiTrack rotation computed:")
        print(f"  R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation")
        print(f"{self.R_pyramid_to_optitrack}")

    def _compute_full_transform(self) -> None:
        """
        Compute full 4×4 transformation matrix from pyramid frame to OptiTrack frame.

        Includes rotation and translation.

        The transformation is:
          p_optitrack = R_pyramid_to_optitrack @ p_pyramid + t_pyramid_to_optitrack

        Where:
          t_pyramid_to_optitrack = (constellation_barycenter in OptiTrack) -
                                   R_pyramid_to_optitrack @ (pyramid_origin in pyramid frame)

        Since pyramid origin is at (0,0,0) in pyramid frame:
          t_pyramid_to_optitrack = constellation_barycenter_in_optitrack
        """
        # Transform constellation barycenter from world frame to constellation frame
        barycenter_in_constellation = self.R_constellation.T @ (
            self.constellation_barycenter_m - self.constellation_barycenter_m
        )  # This is [0, 0, 0]

        # Transform pyramid origin from world frame to pyramid frame
        pyramid_origin_in_pyramid = self.R_pyramid.T @ (
            self.pyramid_origin_m - self.pyramid_origin_m
        )  # This is also [0, 0, 0]

        # The translation is: where does pyramid origin end up in OptiTrack frame?
        # First: pyramid origin in world frame → constellation frame
        pyramid_origin_in_constellation = self.R_constellation.T @ (
            self.pyramid_origin_m - self.constellation_barycenter_m
        )

        # Second: constellation frame → OptiTrack frame
        t_in_optitrack = self.R_constellation_to_optitrack @ (
            -self.R_pyramid_to_constellation @ pyramid_origin_in_pyramid +
            pyramid_origin_in_constellation
        )

        # Simpler approach: transform pyramid origin through both steps
        # Pyramid origin in world → constellation frame → rotate to OptiTrack
        pyramid_origin_in_constellation_frame = self.R_constellation.T @ (
            self.pyramid_origin_m - self.constellation_barycenter_m
        )
        t_in_optitrack = self.R_constellation_to_optitrack @ pyramid_origin_in_constellation_frame

        # Actually, the correct translation is:
        # Where does the constellation barycenter end up minus where pyramid origin ends up
        # But in OptiTrack frame, constellation is at origin, so:
        t_in_optitrack = -self.R_pyramid_to_optitrack @ np.zeros(3)

        # Correct computation:
        # p_world = R_pyramid @ p_pyramid + pyramid_origin
        # p_constellation = R_constellation^T @ (p_world - constellation_barycenter)
        # p_optitrack = R_constellation_to_optitrack @ p_constellation
        #
        # Combining:
        # p_optitrack = R_constellation_to_optitrack @ R_constellation^T @
        #               (R_pyramid @ p_pyramid + pyramid_origin - constellation_barycenter)
        #             = R_pyramid_to_optitrack @ p_pyramid +
        #               R_constellation_to_optitrack @ R_constellation^T @ (pyramid_origin - constellation_barycenter)

        t_in_optitrack = self.R_constellation_to_optitrack @ (
            self.R_constellation.T @ (self.pyramid_origin_m - self.constellation_barycenter_m)
        )

        # Build 4×4 transformation matrix
        self.T_pyramid_to_optitrack = np.eye(4)
        self.T_pyramid_to_optitrack[0:3, 0:3] = self.R_pyramid_to_optitrack
        self.T_pyramid_to_optitrack[0:3, 3] = t_in_optitrack

        print(f"\n✓ Full 4×4 transformation T_pyramid_to_optitrack:")
        print(f"{self.T_pyramid_to_optitrack}")

    def transform_pyramid_to_optitrack(self, points_pyramid_m: np.ndarray) -> np.ndarray:
        """
        Transform points from pyramid frame to OptiTrack frame.

        Args:
            points_pyramid_m: N×3 array of points in pyramid frame (meters)

        Returns:
            N×3 array of points in OptiTrack frame (meters)
        """
        if not hasattr(self, 'T_pyramid_to_optitrack'):
            raise ValueError("Must call set_optitrack_rotation() first!")

        n_points = points_pyramid_m.shape[0]
        points_hom = np.hstack([points_pyramid_m, np.ones((n_points, 1))])
        points_optitrack = (self.T_pyramid_to_optitrack @ points_hom.T).T[:, 0:3]

        return points_optitrack

    def get_constellation_points_in_optitrack_frame(self) -> np.ndarray:
        """
        Get constellation points (18-21) in OptiTrack frame.

        Returns:
            4×3 array of constellation points in OptiTrack frame (meters)
        """
        # Transform constellation points from world to pyramid frame first
        constellation_points_world = self.points_m[self.constellation_indices]

        # World → Pyramid frame
        constellation_points_pyramid = (self.R_pyramid.T @
                                       (constellation_points_world - self.pyramid_origin_m).T).T

        # Pyramid frame → OptiTrack frame
        return self.transform_pyramid_to_optitrack(constellation_points_pyramid)

    def compute_optitrack_rotation_from_markers(
        self,
        marker_positions_m: Dict[str, np.ndarray],
        matching: Dict[str, int]
    ) -> np.ndarray:
        """
        Compute R_constellation_to_optitrack using SVD on known point correspondences.

        Args:
            marker_positions_m: Dict of marker positions in OptiTrack constellation frame (meters)
            matching: Dict mapping marker names to point indices
                     e.g., {'Marker 002': 20, 'Marker 001': 21, ...}

        Returns:
            R_constellation_to_optitrack: 3×3 rotation matrix
        """
        print(f"\n{'=' * 70}")
        print("COMPUTING ROTATION FROM POINT CORRESPONDENCE (SVD METHOD)")
        print(f"{'=' * 70}")

        # Get constellation points in world frame
        constellation_indices = [18, 19, 20, 21]
        constellation_points_world = self.points_m[constellation_indices]

        # Transform constellation points to constellation frame
        points_constellation_frame = (self.R_constellation.T @
                                     (constellation_points_world - self.constellation_barycenter_m).T).T

        # Get marker positions in same order as constellation points
        marker_names = ['Marker 001', 'Marker 002', 'Marker 003', 'Marker 004']

        # Build corresponding point arrays
        points_source = []  # In constellation frame (from JSON model)
        points_target = []  # In OptiTrack frame (from markers)

        for marker_name in marker_names:
            if marker_name not in matching:
                raise ValueError(f"Marker {marker_name} not in matching")

            point_idx = matching[marker_name]
            if point_idx not in constellation_indices:
                raise ValueError(f"Point {point_idx} is not a constellation point")

            # Get point in constellation frame
            array_idx = constellation_indices.index(point_idx)
            point_constellation = points_constellation_frame[array_idx]
            points_source.append(point_constellation)

            # Get marker position in OptiTrack frame
            marker_position = marker_positions_m[marker_name]
            points_target.append(marker_position)

        points_source = np.array(points_source)
        points_target = np.array(points_target)

        print("\nPoint correspondence:")
        for i, marker_name in enumerate(marker_names):
            point_idx = matching[marker_name]
            print(f"  {marker_name} (Point {point_idx}):")
            print(f"    Constellation frame: {points_source[i]}")
            print(f"    OptiTrack frame:     {points_target[i]}")

        # Compute rotation using SVD
        R_constellation_to_optitrack, rmse = compute_rotation_from_point_correspondence(
            points_source,
            points_target
        )

        print(f"\n✓ R_constellation_to_optitrack computed:")
        print(f"  RMSE: {rmse:.6f} m")
        print(f"{R_constellation_to_optitrack}")

        # Set the rotation and compute full transform
        self.set_optitrack_rotation(R_constellation_to_optitrack)

        return R_constellation_to_optitrack

    def print_info(self) -> None:
        """Print transformer information."""
        print(f"\n{'='*70}")
        print("PYRAMID TRANSFORMER INFO")
        print(f"{'='*70}")
        print(f"JSON file: {self.json_path}")
        print(f"Total points: {len(self.points_m)}")
        print(f"Units: meters")
        print(f"Pyramid origin (point 0): {self.pyramid_origin_m} m")
        print(f"Constellation origin: {self.constellation_barycenter_m} m")
        print(f"Constellation points: {self.constellation_indices}")

        if hasattr(self, 'R_pyramid_to_constellation'):
            print(f"\n✓ Relative rotation R_pyramid_to_constellation computed")

        if self.marker_match:
            print(f"\nMarker matching:")
            for marker, point in self.marker_match.items():
                print(f"  {marker} → Point {point}")

        if hasattr(self, 'R_constellation_to_optitrack'):
            print(f"\n✓ OptiTrack rotation configured")
            print(f"✓ Full pyramid-to-OptiTrack transform ready")
        else:
            print(f"\n⚠ OptiTrack rotation not set (call set_optitrack_rotation)")

        print(f"{'='*70}\n")

    def plot_constellation_frame(self, save_path: Optional[Path] = None) -> None:
        """
        Plot the constellation frame with points 18, 19, 20, 21 and the plane.

        Shows:
        - 4 constellation points
        - Barycenter (origin)
        - X, Y, Z axes
        - Plane formed by the 4 points

        Args:
            save_path: Optional path to save the figure
        """
        # Get constellation points
        points = self.points_m[self.constellation_indices]
        barycenter = self.constellation_barycenter_m

        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='blue', s=100, marker='o', label='Constellation points')

        # Label points
        for i, idx in enumerate(self.constellation_indices):
            ax.text(points[i, 0], points[i, 1], points[i, 2],
                   f'  P{idx}', fontsize=12)

        # Plot barycenter
        ax.scatter([barycenter[0]], [barycenter[1]], [barycenter[2]],
                  c='red', s=200, marker='*', label='Origin')

        # Plot axes
        axis_length = 0.05  # meters

        # X axis (red)
        x_axis = self.R_constellation[:, 0]
        ax.quiver(barycenter[0], barycenter[1], barycenter[2],
                 x_axis[0] * axis_length, x_axis[1] * axis_length, x_axis[2] * axis_length,
                 color='red', arrow_length_ratio=0.2, linewidth=2, label='X axis')

        # Y axis (green)
        y_axis = self.R_constellation[:, 1]
        ax.quiver(barycenter[0], barycenter[1], barycenter[2],
                 y_axis[0] * axis_length, y_axis[1] * axis_length, y_axis[2] * axis_length,
                 color='green', arrow_length_ratio=0.2, linewidth=2, label='Y axis (→P20)')

        # Z axis (blue)
        z_axis = self.R_constellation[:, 2]
        ax.quiver(barycenter[0], barycenter[1], barycenter[2],
                 z_axis[0] * axis_length, z_axis[1] * axis_length, z_axis[2] * axis_length,
                 color='blue', arrow_length_ratio=0.2, linewidth=2, label='Z axis (⊥plane)')

        # Plot plane
        xx, yy = np.meshgrid(np.linspace(-0.05, 0.05, 10), np.linspace(-0.05, 0.05, 10))
        zz = np.zeros_like(xx)

        # Transform to world coordinates
        plane_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        plane_world = barycenter + (self.R_constellation @ plane_points.T).T
        xx_world = plane_world[:, 0].reshape(xx.shape)
        yy_world = plane_world[:, 1].reshape(yy.shape)
        zz_world = plane_world[:, 2].reshape(zz.shape)

        ax.plot_surface(xx_world, yy_world, zz_world, alpha=0.2, color='cyan')

        # Set labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Constellation Frame\n(Points 18, 19, 20, 21)', fontsize=14, fontweight='bold')
        ax.legend()

        # Equal aspect ratio
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Figure saved to: {save_path}")

        plt.show()
        print(f"\n✓ Constellation frame plotted")

    def plot_distance_ranking(self, save_path: Optional[Path] = None) -> None:
        """
        Plot the ranking of distances from each point to the referential point.

        Args:
            save_path: Optional path to save the figure
        """
        if self.referential_point_m is None:
            print("⚠️  Referential point not loaded. Cannot plot distances.")
            return

        # Calculate distances
        distances = np.linalg.norm(self.points_m - self.referential_point_m, axis=1)
        point_indices = np.arange(len(self.points_m))
        sorted_idx = np.argsort(distances)
        sorted_distances = distances[sorted_idx]
        sorted_points = point_indices[sorted_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_distances)))

        # Plot bars
        bars = ax.bar(range(len(sorted_distances)), sorted_distances, color=colors)

        # Customize x-axis
        ax.set_xticks(range(len(sorted_points)))
        ax.set_xticklabels([f'P{idx}' for idx in sorted_points], rotation=45, ha='right')

        # Add distance values
        for i, (bar, dist) in enumerate(zip(bars, sorted_distances)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{dist:.3f}',
                    ha='center', va='bottom', fontsize=9)

        # Labels and title
        ax.set_xlabel('Point ID (sorted by distance)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance to Referential (m)', fontsize=12, fontweight='bold')
        ax.set_title('Distance Ranking: Points to Referential Point\n(Closest to Furthest)',
                     fontsize=14, fontweight='bold')

        # Add grid
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add median line
        median_dist = np.median(sorted_distances)
        ax.axhline(y=median_dist, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {median_dist:.3f} m', alpha=0.7)

        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Figure saved to: {save_path}")

        plt.show()

        # Print statistics
        print(f"\n✓ Distance ranking plotted")
        print(f"  Closest point: P{sorted_points[0]} at {sorted_distances[0]:.4f} m")
        print(f"  Furthest point: P{sorted_points[-1]} at {sorted_distances[-1]:.4f} m")
        print(f"  Mean distance: {np.mean(distances):.4f} m")
        print(f"  Median distance: {median_dist:.4f} m")

    def plot_distance_ranking_with_3d(self, save_path: Optional[Path] = None) -> None:
        """
        Plot distance ranking alongside 3D visualization.

        Args:
            save_path: Optional path to save the figure
        """
        if self.referential_point_m is None:
            print("⚠️  Referential point not loaded. Cannot plot distances.")
            return

        # Calculate distances
        distances = np.linalg.norm(self.points_m - self.referential_point_m, axis=1)
        point_indices = np.arange(len(self.points_m))
        sorted_idx = np.argsort(distances)
        sorted_distances = distances[sorted_idx]
        sorted_points = point_indices[sorted_idx]

        # Create figure with two subplots
        fig = plt.figure(figsize=(18, 8))

        # --- LEFT: 3D scatter plot ---
        ax1 = fig.add_subplot(121, projection='3d')

        # Plot all points colored by distance
        scatter = ax1.scatter(self.points_m[:, 0],
                              self.points_m[:, 1],
                              self.points_m[:, 2],
                              c=distances, cmap='viridis', s=100,
                              edgecolors='black', linewidth=1)

        # Plot referential point
        ax1.scatter([self.referential_point_m[0]],
                    [self.referential_point_m[1]],
                    [self.referential_point_m[2]],
                    c='red', s=300, marker='*',
                    edgecolors='darkred', linewidth=2,
                    label='Referential')

        # Label points
        for i in range(len(self.points_m)):
            ax1.text(self.points_m[i, 0],
                     self.points_m[i, 1],
                     self.points_m[i, 2],
                     f'  {i}', fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
        cbar.set_label('Distance to Referential (m)', fontsize=10)

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D View: Points Colored by Distance', fontsize=12, fontweight='bold')
        ax1.legend()

        # --- RIGHT: Distance ranking bar chart ---
        ax2 = fig.add_subplot(122)

        colors_bars = plt.cm.viridis(np.linspace(0, 1, len(sorted_distances)))
        bars = ax2.bar(range(len(sorted_distances)), sorted_distances, color=colors_bars)

        ax2.set_xticks(range(len(sorted_points)))
        ax2.set_xticklabels([f'P{idx}' for idx in sorted_points], rotation=45, ha='right')

        # Add distance values
        for i, (bar, dist) in enumerate(zip(bars, sorted_distances)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{dist:.3f}',
                     ha='center', va='bottom', fontsize=8)

        ax2.set_xlabel('Point ID (sorted by distance)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Distance to Referential (m)', fontsize=11, fontweight='bold')
        ax2.set_title('Distance Ranking', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)

        # Add median line
        median_dist = np.median(sorted_distances)
        ax2.axhline(y=median_dist, color='red', linestyle='--', linewidth=2,
                    label=f'Median: {median_dist:.3f} m', alpha=0.7)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Figure saved to: {save_path}")

        plt.show()
        print(f"\n✓ Distance ranking with 3D view plotted")
        return sorted_points, distances, point_indices, sorted_idx, sorted_distances

    def get_pyramid_points_in_pyramid_frame(self) -> np.ndarray:
        """
        Get all pyramid points transformed to the pyramid frame.

        Returns:
            N×3 array of points in pyramid frame (meters)
        """
        return (self.R_pyramid.T @ (self.points_m - self.pyramid_origin_m).T).T


def plot_svd_fit_quality(
    transformer: PyramidTransformer,
    marker_positions_m: Dict[str, np.ndarray],
    matching: Dict[str, int],
    save_path: Optional[Path] = None
):
    """
    Plot to visualize the quality of the SVD rotation fit.

    Args:
        transformer: PyramidTransformer instance
        marker_positions_m: Dict of marker positions (meters)
        matching: Dict mapping marker names to point indices
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 6))

    # Get constellation points in OptiTrack frame (after transformation)
    constellation_in_optitrack = transformer.get_constellation_points_in_optitrack_frame()

    # Subplot 1: 3D view
    ax1 = fig.add_subplot(131, projection='3d')

    for i, point_idx in enumerate([18, 19, 20, 21]):
        marker_name = [k for k, v in matching.items() if v == point_idx][0]

        transformed = constellation_in_optitrack[i]
        actual = marker_positions_m[marker_name]

        # Plot transformed point (blue)
        ax1.scatter(transformed[0], transformed[1], transformed[2],
                    c='blue', s=100, marker='o', label='Transformed' if i == 0 else '')

        # Plot actual marker (red)
        ax1.scatter(actual[0], actual[1], actual[2],
                    c='red', s=100, marker='^', label='Actual' if i == 0 else '')

        # Draw error line
        ax1.plot([transformed[0], actual[0]],
                 [transformed[1], actual[1]],
                 [transformed[2], actual[2]],
                 'g--', linewidth=1, alpha=0.5)

        # Label
        ax1.text(actual[0], actual[1], actual[2],
                 f'  P{point_idx}', fontsize=10)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D View: Transformed vs Actual')
    ax1.legend()

    # Subplot 2: Error bars
    ax2 = fig.add_subplot(132)

    errors = []
    labels = []
    for i, point_idx in enumerate([18, 19, 20, 21]):
        marker_name = [k for k, v in matching.items() if v == point_idx][0]
        transformed = constellation_in_optitrack[i]
        actual = marker_positions_m[marker_name]
        error = np.linalg.norm(transformed - actual)
        errors.append(error)
        labels.append(f'P{point_idx}\n{marker_name}')

    ax2.bar(range(4), errors, color=['blue', 'green', 'orange', 'red'])
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Point-wise Errors')
    ax2.axhline(np.mean(errors), color='black', linestyle='--',
                label=f'Mean: {np.mean(errors):.6f} m')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Error statistics
    ax3 = fig.add_subplot(133)
    ax3.axis('off')

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    max_error = np.max(errors)
    min_error = np.min(errors)

    stats_text = "SVD FIT STATISTICS\n"
    stats_text += "=" * 30 + "\n\n"
    stats_text += f"RMSE:       {rmse:.6f} m\n"
    stats_text += f"Mean error: {np.mean(errors):.6f} m\n"
    stats_text += f"Max error:  {max_error:.6f} m\n"
    stats_text += f"Min error:  {min_error:.6f} m\n\n"
    stats_text += "Individual errors:\n"
    for i, (label, error) in enumerate(zip(labels, errors)):
        stats_text += f"  {label.replace(chr(10), ' ')}: {error:.6f} m\n"

    ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ SVD fit quality plot saved to: {save_path}")

    plt.show()


def plot_marker_matching(
    transformer: PyramidTransformer,
    marker_positions_m: Dict[str, np.ndarray],
    matching: Dict[str, int],
    save_path: Optional[Path] = None
):
    """
    Plot marker matching results to visualize the correspondence.

    Args:
        transformer: PyramidTransformer instance
        marker_positions_m: Dict of marker positions in constellation local frame (meters)
        matching: Dict mapping marker names to point indices
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Get constellation points (in world frame)
    constellation_points = transformer.points_m[transformer.constellation_indices]
    barycenter = transformer.constellation_barycenter_m

    # Transform marker positions to world frame
    marker_positions_world = {}
    for marker_name, pos_local in marker_positions_m.items():
        # Transform: world = R_constellation @ local + barycenter
        pos_world = transformer.R_constellation @ pos_local + barycenter
        marker_positions_world[marker_name] = pos_world

    # Plot constellation points (JSON)
    for i, idx in enumerate(transformer.constellation_indices):
        point = constellation_points[i]
        ax.scatter(point[0], point[1], point[2],
                   c='blue', s=200, marker='o',
                   edgecolors='darkblue', linewidths=2,
                   label='JSON Points' if i == 0 else '')
        ax.text(point[0], point[1], point[2],
                f'  Point {idx}', fontsize=14, fontweight='bold', color='blue')

    # Plot OptiTrack markers
    for i, (marker_name, pos_world) in enumerate(marker_positions_world.items()):
        ax.scatter(pos_world[0], pos_world[1], pos_world[2],
                   c='red', s=200, marker='^',
                   edgecolors='darkred', linewidths=2,
                   label='OptiTrack Markers' if i == 0 else '')
        ax.text(pos_world[0], pos_world[1], pos_world[2],
                f'  {marker_name}', fontsize=14, fontweight='bold', color='red')

    # Draw lines connecting matched pairs
    for marker_name, point_idx in matching.items():
        # Get point position
        point_array_idx = transformer.constellation_indices.index(point_idx)
        point_pos = constellation_points[point_array_idx]

        # Get marker position
        marker_pos = marker_positions_world[marker_name]

        # Draw line
        ax.plot([point_pos[0], marker_pos[0]],
                [point_pos[1], marker_pos[1]],
                [point_pos[2], marker_pos[2]],
                'g--', linewidth=2, alpha=0.7)

        # Calculate midpoint for label
        mid_x = (point_pos[0] + marker_pos[0]) / 2
        mid_y = (point_pos[1] + marker_pos[1]) / 2
        mid_z = (point_pos[2] + marker_pos[2]) / 2

        # Add label at midpoint
        ax.text(mid_x, mid_y, mid_z,
                f'{marker_name}\n→\nPoint {point_idx}',
                fontsize=10, color='green', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Plot barycenter
    ax.scatter(barycenter[0], barycenter[1], barycenter[2],
               c='purple', s=300, marker='*',
               edgecolors='black', linewidths=2,
               label='Barycenter')

    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('Marker Matching Results\n(Green lines connect matched pairs)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')

    # Equal aspect ratio
    all_points = np.vstack([constellation_points] +
                           [list(marker_positions_world.values())])
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add text box with matching summary
    matching_text = "MATCHING RESULTS:\n" + "=" * 30 + "\n"
    for marker_name in ['Marker 001', 'Marker 002', 'Marker 003', 'Marker 004']:
        if marker_name in matching:
            matching_text += f"{marker_name} → Point {matching[marker_name]}\n"

    plt.figtext(0.02, 0.02, matching_text, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Marker matching plot saved to: {save_path}")

    plt.show()
    print(f"\n✓ Marker matching plot displayed")


def print_matching_table(
    initial_guess: Dict[str, int],
    final_matching: Dict[str, int]
):
    """
    Print a clear table showing the matching results.

    Args:
        initial_guess: Initial matching guess
        final_matching: Final matching from brute force
    """
    print("\n" + "=" * 70)
    print("MARKER MATCHING RESULTS")
    print("=" * 70)

    print(f"\n{'OptiTrack Marker':<20} {'Initial Guess':<15} {'Final Match':<15} {'Status':<10}")
    print("-" * 70)

    for marker in ['Marker 001', 'Marker 002', 'Marker 003', 'Marker 004']:
        initial = initial_guess.get(marker, '?')
        final = final_matching.get(marker, '?')

        if initial == final:
            status = "✓ Same"
        else:
            status = "⚠️ Changed"

        print(f"{marker:<20} Point {initial:<12} Point {final:<12} {status}")

    print("=" * 70)

    if initial_guess == final_matching:
        print("✓ Initial guess was CORRECT! No changes needed.")
    else:
        print("⚠️ Initial guess was MODIFIED by brute force algorithm.")
        print("\nChanges:")
        for marker in initial_guess:
            if initial_guess[marker] != final_matching[marker]:
                print(f"  • {marker}: Point {initial_guess[marker]} → Point {final_matching[marker]}")

    print("=" * 70 + "\n")


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("PYRAMID TRANSFORMER - Example Usage")
    print("="*70)

    # Initialize
    transformer = PyramidTransformer(Path("ModelMire3DSLAM.json"))

    # Print info
    transformer.print_info()

    # Plot constellation frame
    transformer.plot_constellation_frame()

    # Plot distance rankings (if referential point is available)
    if transformer.referential_point_m is not None:
        transformer.plot_distance_ranking_with_3d()
        transformer.plot_distance_ranking()

    print("\n" + "="*70)
    print("To complete setup:")
    print("="*70)
    print("""
1. Extract marker positions from rb_data:
   marker_pos, rb_pos, rb_quat = extract_marker_positions_from_rb_data(rb_data)

2. Match constellation markers:
   initial_guess = {
       'Marker 002': 20,
       'Marker 003': 19,
       'Marker 001': 21,
       'Marker 004': 18
   }
   matching = transformer.match_constellation_markers(marker_pos, initial_guess)

3. Compute OptiTrack rotation from markers (using SVD):
   R = transformer.compute_optitrack_rotation_from_markers(marker_pos, matching)
   
   OR manually set rotation matrix:
   R_constellation_to_optitrack = your_3x3_matrix
   transformer.set_optitrack_rotation(R_constellation_to_optitrack)

4. Transform points:
   points_optitrack = transformer.transform_pyramid_to_optitrack(points_pyramid)

5. Visualize results:
   plot_marker_matching(transformer, marker_pos, matching)
   plot_svd_fit_quality(transformer, marker_pos, matching)
    """)