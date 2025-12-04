# main.py

from pathlib import Path
from config import ExperimentConfig
from paths import ExperimentPaths
from utils import display_calib
from calib_data import CalibData

from areas_common.data_loading.rigid_body import read_data

def main():
    """
    Main workflow for loading experiment data and playing the associated video.
    """
    # 1. Configure the experiment
    # To run a different experiment, you only need to change the values here.
    config = ExperimentConfig(
        base_data_dir=Path("/mnt/areas_nas/data/AREAS data/Knee/SLAM/bdd_SLAM/areas_2025-11-20/knee_pyramid/water"),
        experiment_name="pyramid",
        camera_model_type="ellipse",
        data_prefix="pyramid_0",
        angle_detector_type="ots",
        display_type="calib"
    )

    # 2. Set up and validate all required file paths
    paths = ExperimentPaths(config)
    paths.validate_paths() # Ensures files exist before we proceed

    # 3. Load and process the rigid body tracking data
    print("Loading rigid body data...")
    rb_data = read_data(
        header_path=paths.headers_path,
        data_path=paths.data_path,
        video_timestamps_path=paths.timestamp_path,
        unit_scale=1.0  # Defines the scaling factor (1.0 = no change)
    )
    print(f"Rigid bodies Data loaded successfully. Found {len(rb_data)} data points.")

    # 4. Load camera calibration data
    calib_data = CalibData(
        intrinsics_path=paths.intrinsics_path,
        extrinsics_path=paths.extrinsics_path,
        camera_model_path=paths.camera_model_path,
        ots_ref_pose_path=paths.reference_ots_angle_pose
    )
    print(f"Camera Calibration Data loaded successfully.")


    # 5. Play the corresponding video
    display_calib(paths.video_path,
                rb_data, calib_data,
                use_notch=(config.angle_detector_type == "notch"),
                pen_mode=(config.display_type == "pen")
            )


if __name__ == "__main__":
    main()
