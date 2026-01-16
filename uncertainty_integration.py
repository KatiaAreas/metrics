"""
Integration script to add uncertainty analysis to the existing pipeline.

This script demonstrates how to use the UncertaintyAnalyzer with your existing
display_pyramid workflow in utils.py.
"""

from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional
import cv2

from utils import (
    display_pyramid,
    parse_vectors_log,
    rotation_correction_cw
)
from calib_data import CalibData
from pyramid_transformer import PyramidTransformer, extract_marker_positions_from_rb_data
from uncertainty_analysis import UncertaintyAnalyzer


def display_pyramid_with_uncertainty_analysis(
        video_path: Path,
        rb_data: Dict[str, Any],
        calib_data: CalibData,
        pyramid_json_path: Path,
        keypoints_json_path: Optional[Path] = None,
        use_notch: bool = False,
        vectors_log_path: Optional[Path] = None,
        uncertainty_output_dir: Optional[Path] = None,
        **kwargs
) -> None:
    """
    Enhanced version of display_pyramid that includes uncertainty analysis.
    
    This function:
    1. Performs full uncertainty analysis before video playback
    2. Saves uncertainty report and visualizations
    3. Then proceeds with normal video display
    
    Args:
        video_path: Path to video file
        rb_data: Rigid body tracking data
        calib_data: Camera calibration data
        pyramid_json_path: Path to pyramid geometry JSON
        keypoints_json_path: Path to 2D keypoints JSON (optional)
        use_notch: Whether to use notch detection
        vectors_log_path: Path to vectors.log file
        uncertainty_output_dir: Directory to save uncertainty analysis results
        **kwargs: Additional arguments passed to display_pyramid
    """
    
    print("\n" + "="*80)
    print("ENHANCED PIPELINE WITH UNCERTAINTY ANALYSIS")
    print("="*80)
    
    # =========================================================================
    # STEP 1: Initialize transformer (same as original)
    # =========================================================================
    print("\n📦 Initializing PyramidTransformer...")
    transformer = PyramidTransformer(pyramid_json_path)
    
    # Extract OptiTrack marker positions
    marker_positions_m, rb_position_m, rb_quaternion = extract_marker_positions_from_rb_data(
        rb_data,
        frame_id=0
    )
    
    # Define marker matching
    matching = {
        'Marker 002': 20,
        'Marker 001': 21,
        'Marker 003': 18,
        'Marker 004': 19
    }
    
    # Compute transformation
    R_constellation_to_optitrack = transformer.compute_optitrack_rotation_from_markers(
        marker_positions_m,
        matching
    )
    
    print("✓ PyramidTransformer initialized")
    
    # =========================================================================
    # STEP 2: Perform uncertainty analysis
    # =========================================================================
    print("\n🔍 Performing uncertainty analysis...")
    
    analyzer = UncertaintyAnalyzer(
        transformer=transformer,
        calib_data=calib_data,
        verbose=True
    )
    
    # Load 2D keypoints if provided for reprojection error analysis
    keypoints_2d = None
    if keypoints_json_path is not None:
        from utils import load_keypoints_from_json
        keypoints_dict = load_keypoints_from_json(keypoints_json_path)
        
        # Get keypoints for frame 0 (for uncertainty analysis)
        if 0 in keypoints_dict:
            keypoints_list = keypoints_dict[0]
            keypoints_2d = np.array([[kp['x'], kp['y']] for kp in keypoints_list])
            print(f"✓ Loaded {len(keypoints_2d)} keypoints for uncertainty analysis")
    
    # Compute rotation correction for frame 0
    R_cor = None
    if use_notch and vectors_log_path is not None:
        initial_theta = parse_vectors_log(vectors_log_path)
        if initial_theta is not None:
            camera_center = calib_data.camera_model.get_center()
            R_cor = rotation_correction_cw(initial_theta, camera_center[0], camera_center[1])
            print(f"✓ Using rotation correction with theta={initial_theta:.2f}°")
    
    # Run full pipeline uncertainty analysis
    report = analyzer.analyze_full_pipeline(
        rb_data=rb_data,
        marker_positions_m=marker_positions_m,
        matching=matching,
        frame_id=0,
        keypoints_2d=keypoints_2d,
        R_cor=R_cor
    )
    
    # =========================================================================
    # STEP 3: Save and visualize uncertainty results
    # =========================================================================
    if uncertainty_output_dir is not None:
        uncertainty_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = uncertainty_output_dir / "uncertainty_report.json"
        report.save_json(json_path)
        
        # Save visualization
        viz_path = uncertainty_output_dir / "uncertainty_visualization.png"
        analyzer.visualize_uncertainties(save_path=viz_path, show=False)
        
        print(f"\n✓ Uncertainty analysis results saved to: {uncertainty_output_dir}")
    
    # Print summary
    analyzer.print_summary()
    
    # =========================================================================
    # STEP 4: Proceed with normal video display
    # =========================================================================
    print("\n" + "="*80)
    print("PROCEEDING TO VIDEO DISPLAY")
    print("="*80)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # Call the original display_pyramid function
    display_pyramid(
        video_path=video_path,
        rb_data=rb_data,
        calib_data=calib_data,
        pyramid_json_path=pyramid_json_path,
        keypoints_json_path=keypoints_json_path,
        use_notch=use_notch,
        vectors_log_path=vectors_log_path,
        **kwargs
    )


def analyze_per_frame_uncertainties(
        video_path: Path,
        rb_data: Dict[str, Any],
        calib_data: CalibData,
        transformer: PyramidTransformer,
        keypoints_json_path: Path,
        output_path: Path,
        num_frames: int = 100,
        frame_stride: int = 10
) -> None:
    """
    Analyze uncertainty across multiple video frames to see temporal variations.
    
    This is useful for understanding:
    - How tracking quality varies over time
    - Whether there are systematic drift patterns
    - Frame-dependent error characteristics
    
    Args:
        video_path: Path to video file
        rb_data: Rigid body tracking data
        calib_data: Camera calibration data
        transformer: PyramidTransformer with computed transformations
        keypoints_json_path: Path to 2D keypoints JSON
        output_path: Path to save temporal uncertainty analysis
        num_frames: Number of frames to analyze
        frame_stride: Analyze every Nth frame
    """
    from utils import load_keypoints_from_json
    import matplotlib.pyplot as plt
    
    print("\n" + "="*80)
    print("TEMPORAL UNCERTAINTY ANALYSIS")
    print("="*80)
    
    # Load keypoints
    keypoints_dict = load_keypoints_from_json(keypoints_json_path)
    
    # Initialize storage for temporal metrics
    frame_ids = []
    mean_errors = []
    max_errors = []
    std_errors = []
    
    analyzer = UncertaintyAnalyzer(
        transformer=transformer,
        calib_data=calib_data,
        verbose=False
    )
    
    # Extract marker info for Stage 1 analysis (only needed once)
    marker_positions_m, _, _ = extract_marker_positions_from_rb_data(rb_data, frame_id=0)
    matching = {
        'Marker 002': 20,
        'Marker 001': 21,
        'Marker 003': 18,
        'Marker 004': 19
    }
    
    print(f"\nAnalyzing {num_frames} frames with stride {frame_stride}...")
    
    for i in range(0, num_frames, frame_stride):
        if i not in keypoints_dict or i >= len(rb_data["Pyramid_RB"]):
            continue
        
        try:
            # Get keypoints for this frame
            keypoints_list = keypoints_dict[i]
            keypoints_2d = np.array([[kp['x'], kp['y']] for kp in keypoints_list])
            
            # Analyze this frame
            report = analyzer.analyze_full_pipeline(
                rb_data=rb_data,
                marker_positions_m=marker_positions_m,
                matching=matching,
                frame_id=i,
                keypoints_2d=keypoints_2d,
                R_cor=None
            )
            
            # Store metrics
            if report.camera_projection is not None:
                if report.camera_projection.mean_reprojection_error_px is not None:
                    frame_ids.append(i)
                    mean_errors.append(report.camera_projection.mean_reprojection_error_px)
                    max_errors.append(report.camera_projection.max_reprojection_error_px)
                    std_errors.append(report.camera_projection.std_reprojection_error_px)
            
            if (len(frame_ids) % 10) == 0:
                print(f"  Processed {len(frame_ids)} frames...")
        
        except Exception as e:
            print(f"  Warning: Error analyzing frame {i}: {e}")
            continue
    
    # =========================================================================
    # Visualize temporal uncertainty
    # =========================================================================
    if len(frame_ids) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Mean reprojection error over time
        ax1 = axes[0]
        ax1.plot(frame_ids, mean_errors, 'b-', linewidth=2, label='Mean error')
        ax1.fill_between(frame_ids, 
                         np.array(mean_errors) - np.array(std_errors),
                         np.array(mean_errors) + np.array(std_errors),
                         alpha=0.3, color='blue', label='±1 std')
        ax1.set_xlabel("Frame ID", fontsize=12)
        ax1.set_ylabel("Mean Reprojection Error (px)", fontsize=12)
        ax1.set_title("Temporal Variation in Reprojection Error", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Max error over time
        ax2 = axes[1]
        ax2.plot(frame_ids, max_errors, 'r-', linewidth=2)
        ax2.set_xlabel("Frame ID", fontsize=12)
        ax2.set_ylabel("Max Reprojection Error (px)", fontsize=12)
        ax2.set_title("Maximum Error Per Frame", fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Temporal uncertainty analysis saved to: {output_path}")
        
        # Print summary statistics
        print("\n📊 Temporal Uncertainty Summary:")
        print(f"  Frames analyzed: {len(frame_ids)}")
        print(f"  Mean error:      {np.mean(mean_errors):.2f} ± {np.std(mean_errors):.2f} px")
        print(f"  Max error:       {np.max(max_errors):.2f} px")
        print(f"  Min error:       {np.min(mean_errors):.2f} px")
        
        plt.show()
    else:
        print("⚠ No valid frames analyzed")


# Example usage function
def example_usage():
    """
    Example of how to use the uncertainty analysis pipeline.
    """
    # Setup paths (adjust these to your actual paths)
    video_path = Path("/path/to/your/video.mp4")
    pyramid_json_path = Path("/path/to/pyramid_geometry.json")
    keypoints_json_path = Path("/path/to/keypoints_2d.json")
    vectors_log_path = Path("/path/to/vectors.log")
    output_dir = Path("/path/to/output/uncertainty_analysis")
    
    # Load your rb_data and calib_data (you already have these in your code)
    # rb_data = load_rigid_body_data(...)
    # calib_data = CalibData(...)
    
    # Option 1: Single-frame uncertainty analysis with video display
    display_pyramid_with_uncertainty_analysis(
        video_path=video_path,
        rb_data=None,  # Replace with actual rb_data
        calib_data=None,  # Replace with actual calib_data
        pyramid_json_path=pyramid_json_path,
        keypoints_json_path=keypoints_json_path,
        use_notch=True,
        vectors_log_path=vectors_log_path,
        uncertainty_output_dir=output_dir,
        compute_metrics=True,
        verify_transformation=True
    )
    
    # Option 2: Temporal uncertainty analysis (multiple frames)
    # transformer = PyramidTransformer(pyramid_json_path)
    # # ... compute transformations ...
    # 
    # analyze_per_frame_uncertainties(
    #     video_path=video_path,
    #     rb_data=rb_data,
    #     calib_data=calib_data,
    #     transformer=transformer,
    #     keypoints_json_path=keypoints_json_path,
    #     output_path=output_dir / "temporal_uncertainty.png",
    #     num_frames=100,
    #     frame_stride=5
    # )


if __name__ == "__main__":
    example_usage()
