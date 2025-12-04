import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from camera_model_loader import UpdatableCameraModel

class CalibData:
    """
    A container for loading and storing camera calibration data.
    
    This class handles loading the intrinsics and extrinsics matrices from .npz files,
    and the camera model from a .json file.
    """
    def __init__(self, intrinsics_path: Path, extrinsics_path: Path, camera_model_path: Path, ots_ref_pose_path: Optional[Path]=None):
        """
        Initializes and loads all calibration data from the specified paths.

        Args:
            intrinsics_path: Path to the intrinsics.npz file.
            extrinsics_path: Path to the extrinsics.npz file.
            camera_model_path: Path to the camera_model.json file.
        """
        print("Loading calibration data...")
        self._intrinsics: Dict[str, np.ndarray] = self._load_npz_file(intrinsics_path)
        self._extrinsics: Dict[str, np.ndarray] = self._load_npz_file(extrinsics_path)
        self._camera_model: Dict[str, Any] = self._load_npz_file(camera_model_path)

        self.K = self._intrinsics.get('camera_matrix')
        self.dist_coeffs = self._intrinsics.get('dist_coeffs')
        self.RT = self._extrinsics.get('T_cam_rb_to_optical')

        self.RT[0:3, 3] = self.RT[0:3, 3] / 1000.0

        self.camera_model = UpdatableCameraModel(camera_model_path)

        if ots_ref_pose_path is not None:
            try:
                self.ots_ref_pose = np.load(ots_ref_pose_path)
            except FileNotFoundError as e:
                print(f"Error loading OTS reference pose from '{ots_ref_pose_path}': {e}")
                self.ots_ref_pose = None
                
        print("Calibration data loaded successfully.")

    def _load_npz_file(self, path: Path) -> Dict[str, np.ndarray]:
        """Safely loads data from a .npz file."""
        try:
            # np.load returns a lazy loader object that acts like a dictionary
            return np.load(path)
        except FileNotFoundError:
            print(f"Error: Calibration file not found at '{path}'")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to load or parse .npz file at '{path}': {e}")
            sys.exit(1)

    def __repr__(self) -> str:
        """Provides a user-friendly summary of the loaded calibration data."""
        intr_keys = list(self.intrinsics.keys())
        extr_keys = list(self.extrinsics.keys())
        model_keys = list(self.camera_model.keys())
        
        return (
            f"CalibData(\n"
            f"  Intrinsics loaded: {intr_keys}\n"
            f"  Extrinsics loaded: {extr_keys}\n"
            f"  Camera Model Keys: {model_keys}\n"
            f")"
        )
