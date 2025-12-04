import numpy as np
from scipy.spatial.transform import Rotation

class OTSAngleEstimator:
    def __init__(self, ref_rb_pose, cam_rb_data, lens_rb_data):
        """
        Initializes the estimator with a reference pose and the relevant data.

        Args:
            ref_rb_pose (np.ndarray): A 4x4 transformation matrix for the reference pose.
            cam_rb_data (list): A list of camera rigid body transforms.
            lens_rb_data (list): A list of lens rigid body transforms.
        """
        self._ref_rb_pose = np.asarray(ref_rb_pose)
        self._ref_rotation_inv = Rotation.from_matrix(self._ref_rb_pose[:3, :3]).inv()
        self._cam_rb_data = cam_rb_data
        self._lens_rb_data = lens_rb_data

        # The axis is unknown, so we'll learn it on the first call.
        self._estimated_axis = None
        self._axis_estimation_threshold_rad = 0.01 # Corresponds to ~0.5 degrees

        # Init angle estimation with rotation direction detection
        for i in range(len(cam_rb_data)):
            if self.is_axis_estimated():
                break
            if i % 10 == 0:
                self.estimate_angle(np.linalg.inv(cam_rb_data[i].get_transform()) @ lens_rb_data[i].get_transform())


    def is_axis_estimated(self):
        """Checks if the rotation axis has been estimated."""
        return self._estimated_axis is not None

    def get_estimated_axis(self):
        """Returns the learned rotation axis."""
        return self._estimated_axis

    def estimate_angle(self, current_rb_pose):
        """
        Computes the signed rotation angle in degrees around a fixed but initially unknown axis.
        The axis is estimated from the first significant rotation.
        
        Args:
            current_rb_pose (np.ndarray): A 4x4 transformation matrix for the current pose.
            
        Returns:
            float: The signed rotation angle in degrees [-180, 180].
        """
        # 1. Calculate the relative rotation from the reference pose
        current_rotation_matrix = np.asarray(current_rb_pose)[:3, :3]
        curr_rotation = Rotation.from_matrix(current_rotation_matrix)
        relative_rotation = curr_rotation * self._ref_rotation_inv

        # 2. Get the rotation in vector form (axis * angle)
        rot_vec = relative_rotation.as_rotvec()
        angle_rad = np.linalg.norm(rot_vec)

        # 3. If the axis hasn't been estimated yet, try to estimate it.
        if self._estimated_axis is None:
            # Only estimate the axis if the rotation is large enough to be reliable
            if angle_rad > self._axis_estimation_threshold_rad:
                # The axis is the direction of the rotation vector. We save this
                # normalized vector as our reference for "positive" rotation.
                self._estimated_axis = rot_vec / angle_rad
                print("--- Calibrated rotation axis ---") # Optional: for feedback

        # 4. If the axis is known, calculate the signed angle
        if self._estimated_axis is not None:
            # Project the current rotation vector onto our saved axis.
            # The dot product gives us the component of rotation along the axis,
            # effectively giving us the angle with the correct sign.
            signed_angle_rad = np.dot(rot_vec, self._estimated_axis)
            return np.rad2deg(signed_angle_rad)
        else:
            # If we're here, the rotation is too small to estimate an axis, so the angle is 0.
            return 0.0

    def get_estimated_axis(self):
        """Returns the learned rotation axis."""
        return self._estimated_axis

    def reset(self):
        """Resets the learned axis, forcing recalibration on the next call."""
        self._estimated_axis = None