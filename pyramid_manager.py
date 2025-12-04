# pyramid_manager

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class PyramidManager:
    """
    Manages pyramid geometry and coordinate transformations.

    Now supports BOTH translation AND rotation between pyramid frame
    and OptiTrack rigid body frame.
    """

    def __init__(self, pyramid_json_path: Path, rotation_matrix: Optional[np.ndarray] = None):
        """
        Initialize the PyramidManager with pyramid geometry data.

        Args:
            pyramid_json_path: Path to the JSON file containing pyramid 3D scan data
            rotation_matrix: Optional 3x3 rotation matrix from pyramid frame to OptiTrack frame.
                           If None, assumes identity (no rotation).

        Common rotations:
            - 90° CCW around Z:  [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            - 90° CW around Z:   [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
            - 180° around Z:     [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        """
        self.pyramid_json_path = pyramid_json_path
        self.markers_pyramid_frame = None
        self.local_referential = None
        self.T_optitrack_pyramid = None
        self.rotation_matrix = rotation_matrix if rotation_matrix is not None else np.eye(3)

        self._load_pyramid_data()
        self._compute_transform()

    def _load_pyramid_data(self) -> None:
        """Load pyramid geometry from JSON file."""
        with open(self.pyramid_json_path, 'r') as f:
            data = json.load(f)

        # Extract marker positions in pyramid local frame
        markers = []
        for marker in data['Markers']:
            pos = marker['LocalPosition']
            x = pos.get('x', 0.0)
            y = pos.get('y', 0.0)
            z = pos.get('z', 0.0)
            markers.append([x, y, z])

        self.markers_pyramid_frame = np.array(markers)

        # Extract local referential (OptiTrack rigid body origin in pyramid frame)
        local_ref_pos = data['LocalReferential']['position']
        self.local_referential = np.array([
            local_ref_pos['x'],
            local_ref_pos['y'],
            local_ref_pos['z']
        ])

    def _compute_transform(self) -> None:
        """
        Compute the transformation matrix from pyramid frame to OptiTrack rigid body frame.

        The transformation consists of:
        - Rotation: specified rotation matrix (or identity if none provided)
        - Translation: from pyramid origin (0,0,0) to rigid body origin (local referential)
        """
        self.T_optitrack_pyramid = np.eye(4)

        # Apply rotation
        self.T_optitrack_pyramid[0:3, 0:3] = self.rotation_matrix

        # Apply translation
        # Note: Translation is applied AFTER rotation in the transform matrix
        self.T_optitrack_pyramid[0:3, 3] = self.local_referential

    def set_rotation(self, rotation_matrix: np.ndarray) -> None:
        """
        Update the rotation matrix and recompute the transformation.

        Args:
            rotation_matrix: 3x3 rotation matrix
        """
        self.rotation_matrix = rotation_matrix
        self._compute_transform()

    def set_rotation_from_angle(self, angle_degrees: float, axis: str = 'z') -> None:
        """
        Set rotation from angle and axis.

        Args:
            angle_degrees: Rotation angle in degrees
            axis: 'x', 'y', or 'z'
        """
        angle_rad = np.deg2rad(angle_degrees)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        if axis.lower() == 'z':
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
        elif axis.lower() == 'y':
            R = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
        elif axis.lower() == 'x':
            R = np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
        else:
            R = np.eye(3)

        self.set_rotation(R)

    def get_pyramid_summits_in_pyramid_frame(self) -> np.ndarray:
        """
        Extract the summit/corner points from the pyramid geometry.

        Returns:
            np.ndarray: Nx3 array of summit positions in pyramid local frame
        """
        return self.markers_pyramid_frame.copy()

    def get_specific_summits(self, indices: List[int]) -> np.ndarray:
        """
        Get specific summit points by their indices.

        Args:
            indices: List of marker indices to retrieve

        Returns:
            np.ndarray: Nx3 array of selected summit positions
        """
        return self.markers_pyramid_frame[indices].copy()

    def transform_pyramid_to_optitrack(self, points_pyramid: np.ndarray) -> np.ndarray:
        """
        Transform points from pyramid local frame to OptiTrack rigid body frame.

        Applies both rotation and translation.

        Args:
            points_pyramid: Nx3 array of points in pyramid local frame

        Returns:
            np.ndarray: Nx3 array of points in OptiTrack rigid body frame
        """
        n_points = points_pyramid.shape[0]
        points_hom = np.hstack([points_pyramid, np.ones((n_points, 1))])
        points_optitrack = (self.T_optitrack_pyramid @ points_hom.T).T[:, 0:3]
        return points_optitrack

    def get_summits_in_optitrack_frame(self) -> np.ndarray:
        """
        Get all pyramid summit positions in the OptiTrack rigid body frame.

        This is the main function to call from your code.

        Returns:
            np.ndarray: Nx3 array of summit positions in OptiTrack rigid body frame
        """
        summits_pyramid = self.get_pyramid_summits_in_pyramid_frame()
        summits_optitrack = self.transform_pyramid_to_optitrack(summits_pyramid)
        return summits_optitrack

    def get_pyramid_corners(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the corner points of the pyramid base and apex.

        Returns:
            corners_pyramid: Corner positions in pyramid frame
            corners_optitrack: Corner positions in OptiTrack frame
        """
        corners_pyramid = self.markers_pyramid_frame.copy()
        corners_optitrack = self.transform_pyramid_to_optitrack(corners_pyramid)

        return corners_pyramid, corners_optitrack

    def get_transform_matrix(self) -> np.ndarray:
        """
        Get the 4x4 transformation matrix from pyramid to OptiTrack frame.

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix
        """
        return self.T_optitrack_pyramid.copy()

    def __repr__(self) -> str:
        has_rotation = not np.allclose(self.rotation_matrix, np.eye(3))
        return (f"PyramidManager(markers={len(self.markers_pyramid_frame)}, "
                f"local_ref={self.local_referential}, "
                f"has_rotation={has_rotation})")


# Convenience functions
def load_pyramid_summits(pyramid_json_path: Path,
                         rotation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convenience function to load pyramid summits in OptiTrack frame.

    Args:
        pyramid_json_path: Path to pyramid JSON file
        rotation_matrix: Optional 3x3 rotation matrix

    Returns:
        np.ndarray: Nx3 array of summit positions in OptiTrack frame
    """
    manager = PyramidManager(pyramid_json_path, rotation_matrix)
    return manager.get_summits_in_optitrack_frame()


def create_rotation_matrix(angle_degrees: float, axis: str = 'z') -> np.ndarray:
    """
    Create a rotation matrix from angle and axis.

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
            [s, c, 0],
            [0, 0, 1]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    else:
        return np.eye(3)


if __name__ == "__main__":
    # Example usage
    json_path = Path("ModelMire3DSLAM.json")

    if json_path.exists():
        print("=" * 70)
        print("TESTING PYRAMID MANAGER WITH ROTATION")
        print("=" * 70)

        # Test 1: No rotation (original behavior)
        print("\nTest 1: No rotation (identity)")
        manager = PyramidManager(json_path)
        print(f"  {manager}")
        summits = manager.get_summits_in_optitrack_frame()
        print(f"  First summit: {summits[0]}")

        # Test 2: 90 degree rotation around Z
        print("\nTest 2: 90° rotation around Z")
        R_90 = create_rotation_matrix(90, 'z')
        manager.set_rotation(R_90)
        print(f"  {manager}")
        summits_rotated = manager.get_summits_in_optitrack_frame()
        print(f"  First summit: {summits_rotated[0]}")
        print(f"  Difference: {np.linalg.norm(summits_rotated[0] - summits[0]):.6f} m")

        # Test 3: 180 degree rotation around Z
        print("\nTest 3: 180° rotation around Z")
        R_180 = create_rotation_matrix(180, 'z')
        manager.set_rotation(R_180)
        summits_rotated = manager.get_summits_in_optitrack_frame()
        print(f"  First summit: {summits_rotated[0]}")

        # Test 4: Using convenience function
        print("\nTest 4: Using set_rotation_from_angle()")
        manager.set_rotation_from_angle(-90, 'z')
        summits_rotated = manager.get_summits_in_optitrack_frame()
        print(f"  First summit with -90° rotation: {summits_rotated[0]}")

        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)