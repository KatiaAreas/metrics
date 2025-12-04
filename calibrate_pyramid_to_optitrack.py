"""
COMPUTE:
  R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation

Where:
- R_pyramid_to_constellation: Relative rotation between pyramid and constellation frames (computed)
- R_constellation_to_optitrack: Given as input (3×3 matrix)
"""

import numpy as np
from pathlib import Path
from pyramid_transformer import (
    PyramidTransformer,
    extract_marker_positions_from_rb_data,
    quaternion_to_rotation_matrix
)


def complete_workflow_with_visualization(
        rb_data,
        pyramid_json_path: Path,
        frame_id: int = 0,
        R_constellation_to_optitrack: np.ndarray = None,
        save_plot: bool = True,
):
    """
    Complete workflow with visualization.

    Args:
        rb_data: OptiTrack rigid body data
        frame_id: Frame to use
        R_constellation_to_optitrack: 3×3 rotation matrix from constellation to OptiTrack
                                      If None, will use identity matrix
        save_plot: Whether to save the constellation plot
    """

    # ========================================================================
    # STEP 1: Initialize transformer
    # ========================================================================
    print("\n[STEP 1] Loading 3D model and computing frames...")

    transformer = PyramidTransformer(pyramid_json_path)

    # This automatically:
    # - Loads points 0-21 from JSON
    # - Computes pyramid frame (origin at 0, Y→1, X→5, Z=X×Y away from 16)
    # - Computes constellation frame (barycenter 18-21, Y→20, Z⊥plane, X=Y×Z)
    # - Computes R_pyramid_to_constellation (relative rotation)

    # ========================================================================
    # STEP 2: Visualize constellation frame
    # ========================================================================
    print("\n[STEP 2] Visualizing constellation frame...")

    plot_path = Path("constellation_frame.png") if save_plot else None
    transformer.plot_constellation_frame(save_path=plot_path)

    # ========================================================================
    # STEP 3: Extract OptiTrack data
    # ========================================================================
    print("\n[STEP 3] Extracting OptiTrack data...")

    marker_positions_local_mm, rb_position_m, rb_quaternion = \
        extract_marker_positions_from_rb_data(rb_data, frame_id)

    print(f"\nRigid body pose (world frame):")
    print(f"  Position: {rb_position_m} m")
    print(f"  Quaternion [x,y,z,w]: {rb_quaternion}")

    print(f"\nMarker positions (constellation local frame, mm):")
    for name, pos in marker_positions_local_mm.items():
        print(f"  {name}: {pos}")

    # ========================================================================
    # STEP 4: Match constellation markers
    # ========================================================================
    print("\n[STEP 4] Matching constellation markers (brute force)...")

    # Initial guess from your specification
    initial_guess = {
        'Marker 002': 20,
        'Marker 003': 19,
        'Marker 001': 21,
        'Marker 004': 18
    }

    matching = transformer.match_constellation_markers(
        marker_positions_local_mm,
        initial_guess
    )

    # ========================================================================
    # STEP 5: Set OptiTrack rotation (given as input)
    # ========================================================================
    print("\n[STEP 5] Setting OptiTrack rotation...")

    if R_constellation_to_optitrack is None:
        # Use identity if not provided
        # (means constellation frame = OptiTrack frame)
        R_constellation_to_optitrack = np.eye(3)
        print("  Using identity matrix (constellation = OptiTrack frame)")
    else:
        print("  Using provided R_constellation_to_optitrack")

    transformer.set_optitrack_rotation(R_constellation_to_optitrack)

    # This computes:
    # R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation

    # ========================================================================
    # STEP 6: Transform and verify
    # ========================================================================
    print("\n[STEP 6] Transforming constellation points...")

    # Get constellation points in OptiTrack frame
    constellation_optitrack = transformer.get_constellation_points_in_optitrack_frame()

    print(f"\nConstellation points in OptiTrack frame (mm):")
    for i, idx in enumerate([18, 19, 20, 21]):
        print(f"  Point {idx}: {constellation_optitrack[i]}")

    # ========================================================================
    # STEP 7: Verification
    # ========================================================================
    print("\n[STEP 7] Verification - comparing with measured positions...")

    errors = []
    print(f"\nPoint-by-point comparison:")
    for i, idx in enumerate([18, 19, 20, 21]):
        # Find corresponding marker
        marker_name = None
        for name, point in matching.items():
            if point == idx:
                marker_name = name
                break

        if marker_name:
            computed = constellation_optitrack[i]
            measured = marker_positions_local_mm[marker_name]
            error = np.linalg.norm(computed - measured)
            errors.append(error)

            print(f"\n  Point {idx} ↔ {marker_name}:")
            print(f"    Computed:  {computed}")
            print(f"    Measured:  {measured}")
            print(f"    Error:     {error:.4f} mm")

    # RMSE
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))

    print(f"\n{'=' * 70}")
    print("VERIFICATION RESULTS")
    print(f"{'=' * 70}")
    print(f"RMSE:       {rmse:.4f} mm")
    print(f"Max error:  {np.max(errors):.4f} mm")
    print(f"Min error:  {np.min(errors):.4f} mm")

    if rmse < 1.0:
        print(f"\n✨ EXCELLENT! (RMSE < 1 mm)")
    elif rmse < 5.0:
        print(f"\n✓ GOOD (RMSE < 5 mm)")
    elif rmse < 10.0:
        print(f"\n⚠️  ACCEPTABLE (RMSE < 10 mm)")
    else:
        print(f"\n❌ POOR (RMSE > 10 mm) - Check your setup!")

    # ========================================================================
    # STEP 8: Example transformations
    # ========================================================================
    print("\n[STEP 8] Example point transformations...")

    # Test points in pyramid frame
    test_points_pyramid = np.array([
        [0.0, 0.0, 0.0],  # Pyramid origin (point 0)
        [10.0, 0.0, 0.0],  # 10mm along X
        [0.0, 10.0, 0.0],  # 10mm along Y
        [0.0, 0.0, 10.0]  # 10mm along Z
    ])

    test_points_optitrack = transformer.transform_pyramid_to_optitrack(test_points_pyramid)

    print(f"\nTest transformations:")
    for i, (p_pyr, p_opt) in enumerate(zip(test_points_pyramid, test_points_optitrack)):
        print(f"  Pyramid {p_pyr} → OptiTrack {p_opt}")

    print(f"\n{'=' * 70}")
    print("TRANSFORMATION COMPLETE!")
    print(f"{'=' * 70}")

    return transformer, rmse


def verify_rotation_chain(transformer):
    """
    Verify the rotation chain mathematically.

    Should have:
      R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation
    """
    print("\n" + "=" * 70)
    print("ROTATION CHAIN VERIFICATION")
    print("=" * 70)

    R_pyr_const = transformer.R_pyramid_to_constellation
    R_const_opt = transformer.R_constellation_to_optitrack
    R_pyr_opt = transformer.R_pyramid_to_optitrack

    # Compute expected
    R_expected = R_const_opt @ R_pyr_const

    # Compare
    diff = np.linalg.norm(R_pyr_opt - R_expected)

    print(f"\nR_pyramid_to_constellation:")
    print(R_pyr_const)

    print(f"\nR_constellation_to_optitrack:")
    print(R_const_opt)

    print(f"\nR_pyramid_to_optitrack (computed):")
    print(R_pyr_opt)

    print(f"\nR_pyramid_to_optitrack (expected = R_const_opt @ R_pyr_const):")
    print(R_expected)

    print(f"\nDifference: {diff:.2e}")

    if diff < 1e-10:
        print("✓ VERIFICATION PASSED!")
    else:
        print("❌ VERIFICATION FAILED!")

    return diff < 1e-10


def example_with_mock_data():
    """
    Example with mock data.
    """
    print("=" * 70)
    print("EXAMPLE WITH MOCK DATA")
    print("=" * 70)

    # Mock classes
    class Vector3:
        def __init__(self, x, y, z):
            self._x, self._y, self._z = x, y, z

    class RBData:
        def __init__(self):
            self.position = Vector3(0.450, 0.160, 0.310)
            self.orientation = Vector3(0.0, 0.0, 0.0)
            self.orientation._w = 1.0

            # Marker positions in world frame (mock)
            self.marker_positions = {
                'Marker 001': Vector3(0.454, 0.158, 0.308),
                'Marker 002': Vector3(0.446, 0.162, 0.310),
                'Marker 003': Vector3(0.448, 0.156, 0.305),
                'Marker 004': Vector3(0.453, 0.160, 0.313)
            }

    class Frame:
        def __init__(self):
            self.data = RBData()

    rb_data = {"Pyramid_RB": [Frame()]}

    # Run workflow
    try:
        transformer, rmse = complete_workflow_with_visualization(
            rb_data,
            frame_id=0,
            R_constellation_to_optitrack=np.eye(3),  # Identity for this example
            save_plot=True
        )

        # Verify rotation chain
        verify_rotation_chain(transformer)

    except FileNotFoundError:
        print("\n⚠️  ModelMire3DSLAM.json not found")


def print_usage_guide():
    """Print usage guide."""
    print("\n" + "=" * 70)
    print("USAGE GUIDE - CORRECTED WORKFLOW")
    print("=" * 70)
    print("""
KEY ROTATION CONCEPT:
  R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation

STEPS:

1. Initialize (automatic frame computation):
   transformer = PyramidTransformer("ModelMire3DSLAM.json")
   # This computes R_pyramid_to_constellation automatically

2. Visualize (optional):
   transformer.plot_constellation_frame(save_path="constellation.png")

3. Extract OptiTrack data:
   marker_pos, rb_pos, rb_quat = extract_marker_positions_from_rb_data(rb_data)

4. Match markers:
   matching = transformer.match_constellation_markers(marker_pos, initial_guess)

5. Set OptiTrack rotation (given as input 3×3 matrix):
   R_const_to_opt = your_3x3_matrix  # or np.eye(3) if same frame
   transformer.set_optitrack_rotation(R_const_to_opt)
   # This computes R_pyramid_to_optitrack = R_const_to_opt @ R_pyr_to_const

6. Transform points:
   points_opt = transformer.transform_pyramid_to_optitrack(points_pyr)

COMPLETE ONE-LINER:
   from example_usage_v2 import complete_workflow_with_visualization
   transformer, rmse = complete_workflow_with_visualization(your_rb_data)
    """)


if __name__ == "__main__":
    print_usage_guide()

    print("\n" + "=" * 70)
    print("To run with your data:")
    print("=" * 70)
    print("""
from example_usage_v2 import complete_workflow_with_visualization

transformer, rmse = complete_workflow_with_visualization(
    your_rb_data,
    frame_id=0,
    R_constellation_to_optitrack=your_3x3_matrix,  # or None for identity
    save_plot=True
)
    """)