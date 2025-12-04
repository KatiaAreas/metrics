"""
Pyramid Transformer: Transform between pyramid frame and OptiTrack frame.

Frame definitions:
- Pyramid frame: Origin at point 0, Y axis (0→1), X axis (0→5), Z = X × Y
- Constellation frame: Barycenter of points 18-21, Y toward point 20, Z perpendicular to plane
- OptiTrack frame: From rigid body orientation (given as input)

Transformation chain:
  R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ConstellationMatch:
    """Match between 3D model points and OptiTrack markers."""
    json_point: int  # Point index in JSON (18-21)
    marker_name: str  # OptiTrack marker name (Marker 001-004)
    distance_error: float  # Geometric matching error


class PyramidTransformer:
    """
    Transform points between pyramid frame and OptiTrack frame.

    Transformation: R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation
    """

    def __init__(self, json_path: Path):
        """
        Initialize transformer.

        Args:
            json_path: Path to 3D model JSON file
        """
        self.json_path = json_path
        self.points_mm = None  # All points 0-21 in mm
        self.pyramid_origin_mm = None  # Point 0 in mm
        self.R_pyramid = None  # Rotation matrix of pyramid frame (world frame)
        self.constellation_indices = [18, 19, 20, 21]  # Points forming constellation
        self.constellation_barycenter_mm = None
        self.R_constellation = None  # Rotation matrix of constellation frame (world frame)
        self.marker_match = None  # Dict mapping marker names to point indices

        # Load and process JSON
        self._load_json()
        self._compute_pyramid_frame()
        self._compute_constellation_frame()

        # Compute relative rotation
        self.compute_relative_rotation()

    def _load_json(self) -> None:
        """Load 3D model JSON and extract points."""
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

        # Convert to mm
        self.points_mm = np.array(markers) * 1000.0

        print(f"✓ Loaded {len(self.points_mm)} points from JSON")

    def _compute_pyramid_frame(self) -> None:
        """
        Compute pyramid frame definition.

        Origin: Point 0
        Y axis: Direction from point 0 to point 1
        X axis: Direction from point 0 to point 5
        Z axis: X × Y (right-handed, opposite to point 16)
        """
        # Origin
        self.pyramid_origin_mm = self.points_mm[0].copy()

        # Y axis: 0 → 1
        y_vec = self.points_mm[1] - self.points_mm[0]
        y_axis = y_vec / np.linalg.norm(y_vec)

        # X axis: 0 → 5
        x_vec = self.points_mm[5] - self.points_mm[0]
        x_axis = x_vec / np.linalg.norm(x_vec)

        # Z axis: X × Y (right-handed)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Verify it points away from point 16 (should have negative dot product with 0→16)
        vec_to_16 = self.points_mm[16] - self.points_mm[0]
        dot_with_16 = np.dot(z_axis, vec_to_16)
        if dot_with_16 > 0:
            z_axis = -z_axis  # Flip if pointing toward point 16

        # Recompute X to ensure orthonormal
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Build rotation matrix [X | Y | Z]
        self.R_pyramid = np.column_stack([x_axis, y_axis, z_axis])

        print(f"\n✓ Pyramid frame computed:")
        print(f"  Origin (point 0): {self.pyramid_origin_mm} mm")
        print(f"  X axis (0→5): {x_axis}")
        print(f"  Y axis (0→1): {y_axis}")
        print(f"  Z axis (X×Y, away from 16): {z_axis}")
        print(f"  Dot(Z, vec to point 16): {dot_with_16:.4f} (should be negative)")

    def _compute_constellation_frame(self) -> None:
        """
        Compute constellation frame definition.

        Origin: Barycenter of points 18, 19, 20, 21
        Y axis: Direction toward point 20 (on plane of 4 points)
        Z axis: Normal to plane of 4 points
        X axis: Y × Z (right-handed, on plane)
        """
        # Get constellation points
        constellation_points = self.points_mm[self.constellation_indices]

        # Barycenter (origin)
        self.constellation_barycenter_mm = np.mean(constellation_points, axis=0)

        # Compute plane normal using cross product of two vectors in plane
        # Use points 18, 19, 20 to define plane
        v1 = constellation_points[1] - constellation_points[0]  # 19 - 18
        v2 = constellation_points[2] - constellation_points[0]  # 20 - 18

        z_axis = np.cross(v1, v2)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Y axis: Direction from barycenter to point 20
        y_vec = self.points_mm[20] - self.constellation_barycenter_mm
        # Project onto plane (remove Z component)
        y_vec = y_vec - np.dot(y_vec, z_axis) * z_axis
        y_axis = y_vec / np.linalg.norm(y_vec)

        # X axis: Y × Z (on plane, right-handed)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Build rotation matrix [X | Y | Z]
        self.R_constellation = np.column_stack([x_axis, y_axis, z_axis])

        print(f"\n✓ Constellation frame computed:")
        print(f"  Barycenter: {self.constellation_barycenter_mm} mm")
        print(f"  X axis (on plane): {x_axis}")
        print(f"  Y axis (toward point 20): {y_axis}")
        print(f"  Z axis (normal to plane): {z_axis}")

    def compute_relative_rotation(self) -> np.ndarray:
        """
        Compute relative rotation from pyramid frame to constellation frame.

        R_pyramid_to_constellation transforms vectors from pyramid frame to constellation frame:
          v_constellation = R_pyramid_to_constellation @ v_pyramid

        This is computed as:
          R_pyramid_to_constellation = R_constellation @ R_pyramid^T

        Where:
        - R_pyramid: columns are [X, Y, Z] axes of pyramid frame expressed in world coordinates
        - R_constellation: columns are [X, Y, Z] axes of constellation frame expressed in world coordinates

        Returns:
            3×3 rotation matrix from pyramid to constellation frame
        """
        # Relative rotation from pyramid to constellation
        self.R_pyramid_to_constellation = self.R_constellation @ self.R_pyramid.T

        # Verify it's a valid rotation matrix
        det = np.linalg.det(self.R_pyramid_to_constellation)
        orthogonality = np.linalg.norm(
            self.R_pyramid_to_constellation @ self.R_pyramid_to_constellation.T - np.eye(3)
        )

        print(f"\n✓ Relative rotation R_pyramid_to_constellation computed:")
        print(f"  Determinant: {det:.6f} (should be 1.0)")
        print(f"  Orthogonality error: {orthogonality:.2e} (should be ~0)")
        print(f"  Matrix:")
        print(f"{self.R_pyramid_to_constellation}")

        return self.R_pyramid_to_constellation

    def match_constellation_markers(
        self,
        marker_positions_mm: Dict[str, np.ndarray],
        initial_guess: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Match OptiTrack markers to 3D model constellation points using geometry.

        Uses brute force to find the matching that minimizes geometric distortion.

        Args:
            marker_positions_mm: Dict of marker names to positions in mm
                                 (in constellation local frame)
            initial_guess: Optional initial matching to try first
                          {'Marker 001': 21, 'Marker 002': 20, ...}

        Returns:
            Best matching dict: marker_name → point_index
        """
        marker_names = ['Marker 001', 'Marker 002', 'Marker 003', 'Marker 004']
        point_indices = [18, 19, 20, 21]

        # Get constellation points in constellation frame
        constellation_points_mm = self.points_mm[point_indices]

        # Transform to constellation frame (centered at barycenter)
        constellation_points_centered = constellation_points_mm - self.constellation_barycenter_mm

        # Get marker positions as array (already in constellation frame)
        marker_positions = np.array([marker_positions_mm[name] for name in marker_names])

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
                    constellation_points_centered,
                    [point_indices.index(idx) for idx in guess_indices]
                )
                print(f"  Error: {error:.4f} mm²")

                # If error is reasonable, use it
                if error < 2500.0:  # 50mm RMS ~ 2500mm² total
                    print(f"  ✓ Initial guess accepted (error < 2500mm²)")
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
                constellation_points_centered,
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

        print(f"\n✓ Best matching found (error: {best_error:.4f} mm²):")
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
        """
        # Translation: pyramid origin → constellation barycenter → OptiTrack
        # First: transform pyramid origin to constellation frame
        t_in_constellation = (
            self.constellation_barycenter_mm -
            self.R_pyramid_to_constellation @ self.pyramid_origin_mm
        )

        # Then: transform to OptiTrack frame
        t_in_optitrack = self.R_constellation_to_optitrack @ t_in_constellation

        # Build 4×4 transformation matrix
        self.T_pyramid_to_optitrack = np.eye(4)
        self.T_pyramid_to_optitrack[0:3, 0:3] = self.R_pyramid_to_optitrack
        self.T_pyramid_to_optitrack[0:3, 3] = t_in_optitrack

        print(f"\n✓ Full 4×4 transformation T_pyramid_to_optitrack:")
        print(f"{self.T_pyramid_to_optitrack}")

    def transform_pyramid_to_optitrack(self, points_pyramid_mm: np.ndarray) -> np.ndarray:
        """
        Transform points from pyramid frame to OptiTrack frame.

        Args:
            points_pyramid_mm: N×3 array of points in pyramid frame (mm)

        Returns:
            N×3 array of points in OptiTrack frame (mm)
        """
        if not hasattr(self, 'T_pyramid_to_optitrack'):
            raise ValueError("Must call set_optitrack_rotation() first!")

        n_points = points_pyramid_mm.shape[0]
        points_hom = np.hstack([points_pyramid_mm, np.ones((n_points, 1))])
        points_optitrack = (self.T_pyramid_to_optitrack @ points_hom.T).T[:, 0:3]

        return points_optitrack

    def get_constellation_points_in_optitrack_frame(self) -> np.ndarray:
        """
        Get constellation points (18-21) in OptiTrack frame.

        Returns:
            4×3 array of constellation points in OptiTrack frame (mm)
        """
        constellation_points = self.points_mm[self.constellation_indices]
        return self.transform_pyramid_to_optitrack(constellation_points)

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
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("⚠️  matplotlib not available for plotting")
            return

        # Get constellation points
        points = self.points_mm[self.constellation_indices]
        barycenter = self.constellation_barycenter_mm

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
                  c='red', s=200, marker='*', label='Barycenter (origin)')

        # Plot axes
        axis_length = 20.0  # mm

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
        # Create a grid in the plane
        xx, yy = np.meshgrid(np.linspace(-30, 30, 10), np.linspace(-30, 30, 10))
        zz = np.zeros_like(xx)

        # Transform to world coordinates
        plane_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        plane_world = barycenter + (self.R_constellation @ plane_points.T).T
        xx_world = plane_world[:, 0].reshape(xx.shape)
        yy_world = plane_world[:, 1].reshape(yy.shape)
        zz_world = plane_world[:, 2].reshape(zz.shape)

        ax.plot_surface(xx_world, yy_world, zz_world, alpha=0.2, color='cyan')

        # Set labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
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

    def print_info(self) -> None:
        """Print transformer information."""
        print(f"\n{'='*70}")
        print("PYRAMID TRANSFORMER INFO")
        print(f"{'='*70}")
        print(f"JSON file: {self.json_path}")
        print(f"Total points: {len(self.points_mm)}")
        print(f"Pyramid origin (point 0): {self.pyramid_origin_mm} mm")
        print(f"Constellation barycenter: {self.constellation_barycenter_mm} mm")
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
        marker_positions_mm: Dict of marker positions in constellation local frame (mm)
        rb_position_m: Rigid body position in world frame (meters)
        rb_quaternion: Rigid body orientation [x, y, z, w]
    """
    pyramid_rb = rb_data[rigid_body_name][frame_id]

    # Extract marker positions (in meters, world frame)
    marker_positions_m = {}
    for marker_name in ['Marker 001', 'Marker 002', 'Marker 003', 'Marker 004']:
        marker = pyramid_rb.data.marker_positions[marker_name]
        if hasattr(marker, '_x'):
            pos_m = np.array([marker._x, marker._y, marker._z])
        else:
            pos_m = np.array(marker)
        marker_positions_m[marker_name] = pos_m

    # Extract rigid body pose
    rb_pos = pyramid_rb.data.position
    rb_position_m = np.array([rb_pos._x, rb_pos._y, rb_pos._z])

    rb_ori = pyramid_rb.data.orientation
    rb_quaternion = np.array([rb_ori._x, rb_ori._y, rb_ori._z, rb_ori._w])

    # Convert marker positions to constellation local frame (mm)
    marker_positions_local_mm = transform_world_to_local_frame(
        marker_positions_m,
        rb_position_m,
        rb_quaternion
    )

    return marker_positions_local_mm, rb_position_m, rb_quaternion


def transform_world_to_local_frame(
    positions_m: Dict[str, np.ndarray],
    rb_position_m: np.ndarray,
    rb_quaternion: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Transform positions from world frame to rigid body local frame.

    Args:
        positions_m: Dict of positions in world frame (meters)
        rb_position_m: Rigid body position (meters)
        rb_quaternion: Rigid body orientation [x, y, z, w]

    Returns:
        Dict of positions in local frame (mm)
    """
    from scipy.spatial.transform import Rotation

    # Get rotation matrix
    rot = Rotation.from_quat(rb_quaternion)
    R = rot.as_matrix()

    # Transform: p_local = R^T × (p_world - t_world)
    result = {}
    for name, pos_world in positions_m.items():
        pos_local_m = R.T @ (pos_world - rb_position_m)
        result[name] = pos_local_m * 1000.0  # Convert to mm

    return result


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        quaternion: [x, y, z, w]

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

3. Set OptiTrack rotation (given as 3×3 matrix input):
   R_constellation_to_optitrack = your_3x3_matrix  # or from quaternion
   transformer.set_optitrack_rotation(R_constellation_to_optitrack)

4. Transform points:
   points_optitrack = transformer.transform_pyramid_to_optitrack(points_pyramid)
    """)