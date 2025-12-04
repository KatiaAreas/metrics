# config.py

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Define a type for the camera model to ensure only valid values are used.
CameraModel = Literal["circle", "ellipse"]
AngleDetectorType = Literal["ots", "notch"]
DisplayType = Literal["calib", "pen"]

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Holds all configuration settings for a specific experiment.

    Using a frozen dataclass provides immutability, ensuring that
    configuration values are not accidentally changed during runtime.
    """
    base_data_dir: Path = Path("data")
    experiment_name: str = "arthrex_19_10_2024"
    camera_model_type: CameraModel = "circle"
    data_prefix: str = "1_1"  # Common prefix for related files
    angle_detector_type: AngleDetectorType = "ots"
    display_type: DisplayType = "calib" 
