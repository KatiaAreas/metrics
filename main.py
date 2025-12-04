# main.py

from pathlib import Path
from config import ExperimentConfig
from paths import ExperimentPaths
from utils import display_calib, display_pyramid
from calib_data import CalibData

from areas_common.data_loading.rigid_body import read_data


def main():
    """
    Main workflow for loading experiment data and displaying video with overlays.
    """
    # 1. Configure the experiment
    config = ExperimentConfig(
        base_data_dir=Path("/mnt/areas_nas/data/AREAS data/Knee/SLAM/bdd_SLAM/areas_2025-11-20/knee_pyramid/water"),
        experiment_name="pyramid",
        camera_model_type="circle",
        data_prefix="pyramid_0",
        angle_detector_type="notch",
        display_type="pyramid"  # Options: "calib", "pen", "pyramid"
    )

    # 2. Set up and validate all required file paths
    paths = ExperimentPaths(config)
    paths.validate_paths()

    # 3. Load and process the rigid body tracking data
    print("Loading rigid body data...")
    rb_data = read_data(
        header_path=paths.headers_path,
        data_path=paths.data_path,
        video_timestamps_path=paths.timestamp_path,
        unit_scale=1.0
    )
    print(f"Rigid bodies data loaded successfully. Found {len(rb_data)} data points.")

    # 4. Load camera calibration data
    calib_data = CalibData(
        intrinsics_path=paths.intrinsics_path,
        extrinsics_path=paths.extrinsics_path,
        camera_model_path=paths.camera_model_path,
        ots_ref_pose_path=paths.reference_ots_angle_pose
    )
    print(f"Camera calibration data loaded successfully.")

    # 5. Display based on mode
    if config.display_type == "pyramid":
        # Display pyramid summits overlay
        print("Starting pyramid display mode...")

        # Define path to pyramid JSON file
        pyramid_json_path = Path(config.base_data_dir) / "ModelMire3DSLAM.json"

        if not pyramid_json_path.exists():
            print(f"Error: Pyramid JSON file not found at {pyramid_json_path}")
            print("Please update the pyramid_json_path in main.py")
            return

        display_pyramid(
            video_path=paths.video_path,
            rb_data=rb_data,
            calib_data=calib_data,
            pyramid_json_path=pyramid_json_path,
            use_notch=(config.angle_detector_type == "notch")
        )
    else:
        # Display calibration or pen markers
        display_calib(
            video_path=paths.video_path,
            rb_data=rb_data,
            calib_data=calib_data,
            use_notch=(config.angle_detector_type == "notch"),
            pen_mode=(config.display_type == "pen"),
            pyramid_mode=(config.display_type == "pyramid")
        )


if __name__ == "__main__":
    main()