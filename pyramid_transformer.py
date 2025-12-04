"""
Pyramid to OptiTrack transformation utilities.

This module provides functions to transform pyramid corner positions
from the pyramid local frame to the OptiTrack world frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class PyramidTransformer:
    """
    Handle transformations between pyramid local frame and OptiTrack world frame.
    """
    
    def __init__(self, transform_npz_path: str):
        """
        Initialize with pre-computed transformation matrix.
        
        Args:
            transform_npz_path: Path to .npz file containing T_optitrack_pyramid_json
        """
        data = np.load(transform_npz_path)
        self.T_optitrack_pyramid = data['T_optitrack_pyramid_json']
        self.T_pyramid_optitrack = data['T_pyramid_optitrack_json']
    
    def pyramid_to_rigid_body(self, points_pyramid: np.ndarray) -> np.ndarray:
        """
        Transform points from pyramid frame to rigid body frame.
        
        Args:
            points_pyramid: Nx3 array of points in pyramid local frame
            
        Returns:
            points_rb: Nx3 array of points in rigid body frame
        """
        n_points = points_pyramid.shape[0]
        points_hom = np.hstack([points_pyramid, np.ones((n_points, 1))])
        points_rb = (self.T_optitrack_pyramid @ points_hom.T).T[:, 0:3]
        return points_rb
    
    def rigid_body_to_pyramid(self, points_rb: np.ndarray) -> np.ndarray:
        """
        Transform points from rigid body frame to pyramid frame.
        
        Args:
            points_rb: Nx3 array of points in rigid body frame
            
        Returns:
            points_pyramid: Nx3 array of points in pyramid local frame
        """
        n_points = points_rb.shape[0]
        points_hom = np.hstack([points_rb, np.ones((n_points, 1))])
        points_pyramid = (self.T_pyramid_optitrack @ points_hom.T).T[:, 0:3]
        return points_pyramid
    
    def pyramid_to_world(self, 
                        points_pyramid: np.ndarray,
                        rb_position: np.ndarray,
                        rb_quaternion: np.ndarray) -> np.ndarray:
        """
        Transform points from pyramid frame to world frame.
        
        Args:
            points_pyramid: Nx3 array of points in pyramid local frame
            rb_position: [x, y, z] rigid body position in world frame
            rb_quaternion: [x, y, z, w] rigid body orientation
            
        Returns:
            points_world: Nx3 array of points in world frame
        """
        # Pyramid -> Rigid body
        points_rb = self.pyramid_to_rigid_body(points_pyramid)
        
        # Rigid body -> World
        T_world_rb = self._create_transform(rb_position, rb_quaternion)
        n_points = points_rb.shape[0]
        points_hom = np.hstack([points_rb, np.ones((n_points, 1))])
        points_world = (T_world_rb @ points_hom.T).T[:, 0:3]
        
        return points_world
    
    def world_to_pyramid(self,
                        points_world: np.ndarray,
                        rb_position: np.ndarray,
                        rb_quaternion: np.ndarray) -> np.ndarray:
        """
        Transform points from world frame to pyramid frame.
        
        Args:
            points_world: Nx3 array of points in world frame
            rb_position: [x, y, z] rigid body position in world frame
            rb_quaternion: [x, y, z, w] rigid body orientation
            
        Returns:
            points_pyramid: Nx3 array of points in pyramid local frame
        """
        # World -> Rigid body
        T_world_rb = self._create_transform(rb_position, rb_quaternion)
        T_rb_world = self._invert_transform(T_world_rb)
        
        n_points = points_world.shape[0]
        points_hom = np.hstack([points_world, np.ones((n_points, 1))])
        points_rb = (T_rb_world @ points_hom.T).T[:, 0:3]
        
        # Rigid body -> Pyramid
        points_pyramid = self.rigid_body_to_pyramid(points_rb)
        
        return points_pyramid
    
    @staticmethod
    def _create_transform(position: np.ndarray, 
                         quaternion: np.ndarray = None) -> np.ndarray:
        """Create 4x4 transformation matrix from position and quaternion."""
        T = np.eye(4)
        if quaternion is not None:
            rot = Rotation.from_quat(quaternion)
            T[0:3, 0:3] = rot.as_matrix()
        T[0:3, 3] = position
        return T
    
    @staticmethod
    def _invert_transform(T: np.ndarray) -> np.ndarray:
        """Invert a 4x4 homogeneous transformation matrix."""
        T_inv = np.eye(4)
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        T_inv[0:3, 0:3] = R.T
        T_inv[0:3, 3] = -R.T @ t
        return T_inv


# Example usage:
if __name__ == "__main__":
    # Initialize transformer
    transformer = PyramidTransformer('transform_pyramid_optitrack.npz')
    
    # Define pyramid corners
    pyramid_corners = np.array([
        [0.0, 0.0, 0.0],
        [0.050, 0.0, 0.0],
        [0.050, -0.040, 0.0],
        [0.0, -0.040, 0.0]
    ])
    
    # Get rigid body pose from OptiTrack (example)
    rb_position = np.array([1.5, 2.0, 0.8])
    rb_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    
    # Transform to world frame
    corners_world = transformer.pyramid_to_world(
        pyramid_corners,
        rb_position,
        rb_quaternion
    )
    
    print("Corners in world frame:")
    print(corners_world)
