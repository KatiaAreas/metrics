# paths.py

import sys
from pathlib import Path
from config import ExperimentConfig

class ExperimentPaths:
    """
    Manages and constructs the file and directory paths for an experiment.

    This class centralizes all path-related logic, making it easy to update
    if the directory structure changes. It takes a configuration object
    and builds all paths from it.
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config

        # --- Base Directories ---
        self.experiment_dir: Path = config.base_data_dir / config.experiment_name
        self.tracking_dir: Path = self.experiment_dir / "tracking"
        self.video_dir: Path = self.experiment_dir / "video"
        self.calib_dir: Path = self.experiment_dir / "calib"
        self.camera_models_dir: Path = config.base_data_dir / "camera_models"

        # --- File Paths ---
        # Tracking data files
        self.headers_path: Path = self.tracking_dir / f"{config.data_prefix}_headers.json"
        self.data_path: Path = self.tracking_dir / f"{config.data_prefix}_data.json"

        # Video and timestamp files
        self.video_path: Path = self.video_dir / f"{config.data_prefix}.mp4"
        self.timestamp_path: Path = self.video_dir / f"{config.data_prefix}.txt"

        # Calibration files
        self.intrinsics_path: Path = self.calib_dir / "intrinsics.npz"
        self.extrinsics_path: Path = self.calib_dir / "extrinsics.npz"
        self.camera_model_path: Path = self.camera_models_dir / f"camera_{config.camera_model_type}_model.npz"
        self.reference_ots_angle_pose = self.experiment_dir / "R0.npy"

    def validate_paths(self) -> None:
        """
        Checks if all essential data files exist before processing.

        This proactive check prevents runtime errors from missing files
        and provides clear, immediate feedback to the user.
        """
        print("Validating required file paths...")
        required_paths = [
            self.headers_path,
            self.data_path,
            self.video_path,
            self.timestamp_path,
        ]
        
        
        for path in required_paths:
            if not path.exists():
                print(f"Error: Required file not found at '{path}'")
                sys.exit(1) # Exit the script with an error code
        
        print("All required paths are valid.")
