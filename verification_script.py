import numpy as np
import cv2
from pathlib import Path
from typing import Any, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyramid_transformer import PyramidTransformer, plot_svd_fit_quality, plot_marker_matching, extract_marker_positions_from_rb_data, quaternion_to_rotation_matrix



def verify_pyramid_transformation(
    rb_data,
    calib_data,
    transformer,
    frame_id: int = 0
):
    """
    Comprehensive verification of pyramid transformation pipeline.
    
    Checks:
    1. Rotation matrix validity (determinant, orthogonality)
    2. Transform of pyramid origin
    3. Constellation points vs measured markers (RMSE)
    4. Frame visibility
    
    Args:
        rb_data: OptiTrack rigid body data
        calib_data: Camera calibration data
        transformer: PyramidTransformer instance
        frame_id: Frame to verify
    """
    print("\n" + "=" * 70)
    print("PYRAMID TRANSFORMATION VERIFICATION")
    print("=" * 70)
    
    # =========================================================================
    # 1. Check rotation matrix validity
    # =========================================================================
    print("\n[1] Rotation Matrix Validation")
    print("-" * 70)
    
    R_pyr_to_opt = transformer.R_pyramid_to_optitrack
    
    det = np.linalg.det(R_pyr_to_opt)
    orthogonality = np.linalg.norm(R_pyr_to_opt @ R_pyr_to_opt.T - np.eye(3))
    
    print(f"R_pyramid_to_optitrack:")
    print(f"{R_pyr_to_opt}")
    print(f"\nDeterminant: {det:.6f} (should be 1.0)")
    print(f"Orthogonality error: {orthogonality:.2e} (should be ~0)")
    
    if abs(det - 1.0) < 1e-6 and orthogonality < 1e-10:
        print("✓ PASSED: Rotation matrix is valid")
    else:
        print("❌ FAILED: Rotation matrix is invalid!")
        # return False
    
    # =========================================================================
    # 2. Check pyramid origin transformation
    # =========================================================================
    print("\n[2] Pyramid Origin Transformation")
    print("-" * 70)
    
    # Transform pyramid origin [0,0,0] to OptiTrack frame
    origin_transformed = transformer.transform_pyramid_to_optitrack(
        np.array([[0.0, 0.0, 0.0]])
    )
    print(f"Pyramid origin [0,0,0] transforms to:")
    print(f"  {origin_transformed[0]} m in OptiTrack frame")

    # Get constellation barycenter in OptiTrack frame
    # The barycenter should be at the rigid body position
    frame_data = rb_data["Pyramid_RB"][0].data

    # Get barycenter position
    barycenter_optitrack = np.array(frame_data.position) * 1000.0  # Convert from meters to m
    
    print(f"\nConstellation barycenter (from OptiTrack Pyramid_RB position):")
    print(f"  {barycenter_optitrack} m")
    
    # These should be close (within a few m)
    distance = np.linalg.norm(origin_transformed[0] - barycenter_optitrack)
    print(f"\nDistance between transformed origin and barycenter: {distance:.4f} m")
    
    if distance < 50.0:  # Reasonable threshold
        print(f"✓ PASSED: Distance < 50m")
    else:
        print(f"⚠️  WARNING: Distance > 50m - check transformation")
    
    # =========================================================================
    # 3. Check constellation points vs measured markers
    # =========================================================================
    print("\n[3] Constellation Points vs Measured Markers")
    print("-" * 70)
    
    # Get constellation points in OptiTrack frame (from transformation)
    constellation_optitrack = transformer.get_constellation_points_in_optitrack_frame()
    
    print(f"\nConstellation points in OptiTrack frame (from transformation):")
    for i, idx in enumerate([18, 19, 20, 21]):
        print(f"  Point {idx}: {constellation_optitrack[i]} m")
    
    # Get measured marker positions
    from pyramid_transformer import extract_marker_positions_from_rb_data
    
    marker_positions_local_m, rb_position_m, rb_quaternion = \
        extract_marker_positions_from_rb_data(rb_data, frame_id)
    
    print(f"\nMeasured marker positions (in constellation local frame):")
    for name, pos in marker_positions_local_m.items():
        print(f"  {name}: {pos} m")
    
    # Compare using the matching
    if hasattr(transformer, 'marker_match') and transformer.marker_match:
        print(f"\nPoint-by-point comparison (using matching):")
        errors = []
        
        for i, idx in enumerate([18, 19, 20, 21]):
            # Find corresponding marker
            marker_name = None
            for name, point in transformer.marker_match.items():
                if point == idx:
                    marker_name = name
                    break
            
            if marker_name:
                computed = constellation_optitrack[i]
                measured = marker_positions_local_m[marker_name]
                error = np.linalg.norm(computed - measured)
                errors.append(error)
                
                print(f"\n  Point {idx} ↔ {marker_name}:")
                print(f"    Computed:  {computed}")
                print(f"    Measured:  {measured}")
                print(f"    Error:     {error:.4f} m")
        
        if errors:
            rmse = np.sqrt(np.mean(np.array(errors) ** 2))
            max_error = np.max(errors)
            
            print(f"\nRMSE:       {rmse:.4f} m")
            print(f"Max error:  {max_error:.4f} m")
            
            if rmse < 5.0:
                print("✓ PASSED: RMSE < 5m")
            elif rmse < 10.0:
                print("⚠️  WARNING: RMSE < 10m but > 5m")
            else:
                print("❌ FAILED: RMSE > 10m")
                # return False
    else:
        print("\n⚠️  No marker matching available - skip comparison")
    
    # =========================================================================
    # 4. Check frame visibility
    # =========================================================================
    print("\n[4] Frame Visibility Check")
    print("-" * 70)
    
    is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
    is_cam_visible = rb_data["Cam_RB"][frame_id].data.is_visible
    is_pyramid_visible = rb_data["Pyramid_RB"][frame_id].data.is_visible
    
    print(f"Lens_RB visible:    {is_lens_visible}")
    print(f"Cam_RB visible:     {is_cam_visible}")
    print(f"Pyramid_RB visible: {is_pyramid_visible}")
    
    if is_lens_visible and is_cam_visible and is_pyramid_visible:
        print("✓ PASSED: All rigid bodies visible")
    else:
        print("❌ FAILED: Some rigid bodies not visible")
        # return False
    
    # =========================================================================
    # 5. Test projection pipeline with pyramid origin
    # =========================================================================
    print("\n[5] Projection Pipeline Test")
    print("-" * 70)
    
    # Get transformation matrices (same as in draw_pyramid_points)
    T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
    T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
    RT = np.linalg.inv(T_World_Lens @ calib_data.RT)
    
    # Test pyramid origin
    pyramid_origin_in_pyramid_frame = np.array([[0.0, 0.0, 0.0]])
    
    # Transform to OptiTrack rigid body frame (m)
    origin_optitrack_m = transformer.transform_pyramid_to_optitrack(pyramid_origin_in_pyramid_frame)
    
    print(f"Pyramid origin in OptiTrack frame: {origin_optitrack_m[0]} m")
    
    # Transform to world frame
    origin_hom = np.hstack([origin_optitrack_m, np.ones((1, 1))])
    origin_world_hom = (T_World_Pyramid @ origin_hom.T).T
    origin_world = origin_world_hom[:, 0:3]
    
    print(f"Pyramid origin in world frame: {origin_world[0]} m")
    
    # Project to image
    proj_2d = cv2.projectPoints(
        origin_world,
        cv2.Rodrigues(RT[:3, :3])[0],
        RT[:3, 3],
        calib_data.K,
        calib_data.dist_coeffs
    )[0]
    
    print(f"Pyramid origin projected to image: {proj_2d[0, 0]} pixels")
    

    #################### Distance barycentre to markers + plot 18 to 21 + distance ref point au billet 18 a 21 ######################################

    # Extract data
    frame_data = rb_data["Pyramid_RB"][frame_id].data

    # Get barycenter position
    barycenter = np.array(frame_data.position)

    # Get orientation quaternion
    quaternion = np.array(frame_data.orientation)
    marker_names = ['Marker 001', 'Marker 002', 'Marker 003', 'Marker 004']
    # Get marker positions
    markers = {}
    for marker_name in marker_names:
        if marker_name in frame_data.marker_positions:
            markers[marker_name] = np.array(frame_data.marker_positions[marker_name])

    # Calculate distances from each marker to barycenter
    marker_distances = {}
    for marker_name, marker_pos in markers.items():
        distance = np.linalg.norm(marker_pos - barycenter)
        marker_distances[marker_name] = distance

    # Sort markers by distance (smallest to biggest)
    sorted_markers = sorted(marker_distances.items(), key=lambda x: x[1])

    for rank, (marker_name, distance) in enumerate(sorted_markers, start=1):
        print(f"{rank:<6} {marker_name:<20} {distance:>10.4f} m")


    ################ distance in JSON
    # Get constellation points in OptiTrack frame (from transformation)
    sorted_points, distances, point_indices, sorted_idx, sorted_distances = transformer.plot_distance_ranking_with_3d()

    for rank in range(4):
        opti_name = sorted_markers[rank][0] if rank < len(sorted_markers) else "N/A"
        opti_dist = sorted_markers[rank][1] if rank < len(sorted_markers) else 0

        pyr_name = f"P{sorted_points[rank]}" if rank < len(sorted_points) else "N/A"
        pyr_dist = sorted_distances[rank] if rank < len(sorted_distances) else 0

        print(f"{rank + 1:<6} {opti_name:<25} {opti_dist:>8.6f}   m   | {pyr_name:<15} {pyr_dist:>8.6f} m")

    print("=" * 80)


def add_verification_overlay(
    frame: np.ndarray,
    transformer,
    frame_id: int,
    rb_data,
    calib_data
) -> None:
    """
    Add verification information overlay to video frame.
    
    Shows:
    - Rotation matrix determinant
    - RMSE if available
    - Origin projection
    
    Args:
        frame: Video frame
        transformer: PyramidTransformer instance
        frame_id: Current frame
        rb_data: OptiTrack data
        calib_data: Camera calibration
    """
    y_offset = 100
    
    # Show determinant
    det = np.linalg.det(transformer.R_pyramid_to_optitrack)
    color = (0, 255, 0) if abs(det - 1.0) < 1e-6 else (0, 0, 255)
    cv2.putText(frame, f"Det(R): {det:.6f}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_offset += 30
    
    # Show orthogonality
    orthogonality = np.linalg.norm(
        transformer.R_pyramid_to_optitrack @ transformer.R_pyramid_to_optitrack.T - np.eye(3)
    )
    color = (0, 255, 0) if orthogonality < 1e-10 else (0, 0, 255)
    cv2.putText(frame, f"Orthogonality: {orthogonality:.2e}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_offset += 30

def visualize_pyramid_frame_and_points(transformer, interactive=True, save_path=None):
    """
    Create interactive 3D visualization of pyramid frame and points.

    Args:
        transformer: PyramidTransformer instance with computed frame
        interactive: Whether to use interactive mode (rotatable)
        save_path: Optional path to save the figure
    """
    # Get pyramid points (0-17) in millimeters
    point_indices = list(range(18))
    points_m = transformer.points_m[point_indices]

    # Get pyramid frame origin and rotation matrix
    origin_m = transformer.pyramid_origin_m
    R_pyramid = transformer.R_pyramid

    # Extract axes from rotation matrix
    x_axis = R_pyramid[:, 0]
    y_axis = R_pyramid[:, 1]
    z_axis = R_pyramid[:, 2]

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # =========================================================================
    # 1. Plot pyramid origin
    # =========================================================================
    ax.scatter(origin_m[0], origin_m[1], origin_m[2],
               c='red', s=300, marker='*',
               edgecolors='black', linewidths=2,
               label='Origin (Point 0)', zorder=10)

    ax.text(origin_m[0], origin_m[1], origin_m[2],
            '  ORIGIN\n  (Point 0)', fontsize=12, fontweight='bold',
            color='red', ha='left')

    # =========================================================================
    # 2. Plot pyramid frame axes
    # =========================================================================
    axis_length = 0.05  # m

    # X axis (red arrow)
    ax.quiver(origin_m[0], origin_m[1], origin_m[2],
              x_axis[0] * axis_length, x_axis[1] * axis_length, x_axis[2] * axis_length,
              color='red', arrow_length_ratio=0.2, linewidth=3,
              label='X axis (⊥Z, in plane 0-1-4)', zorder=5)

    # Y axis (green arrow)
    ax.quiver(origin_m[0], origin_m[1], origin_m[2],
              y_axis[0] * axis_length, y_axis[1] * axis_length, y_axis[2] * axis_length,
              color='green', arrow_length_ratio=0.2, linewidth=3,
              label='Y axis (Z×X)', zorder=5)

    # Z axis (blue arrow)
    ax.quiver(origin_m[0], origin_m[1], origin_m[2],
              z_axis[0] * axis_length, z_axis[1] * axis_length, z_axis[2] * axis_length,
              color='blue', arrow_length_ratio=0.2, linewidth=3,
              label='Z axis (0→1)', zorder=5)

    # Add axis labels at arrow tips
    x_tip = origin_m + x_axis * axis_length
    y_tip = origin_m + y_axis * axis_length
    z_tip = origin_m + z_axis * axis_length

    ax.text(x_tip[0], x_tip[1], x_tip[2], '  X', fontsize=14,
            fontweight='bold', color='red')
    ax.text(y_tip[0], y_tip[1], y_tip[2], '  Y', fontsize=14,
            fontweight='bold', color='green')
    ax.text(z_tip[0], z_tip[1], z_tip[2], '  Z', fontsize=14,
            fontweight='bold', color='blue')

    # =========================================================================
    # 3. Plot all 18 pyramid points
    # =========================================================================
    # Color gradient for points
    colors = plt.cm.viridis(np.linspace(0, 1, len(points_m)))

    for i, point in enumerate(points_m):
        ax.scatter(point[0], point[1], point[2],
                   c=[colors[i]], s=100, marker='o',
                   edgecolors='black', linewidths=1,
                   label=f'Point {i}' if i == 0 else '', zorder=3)

        # Add point label
        ax.text(point[0], point[1], point[2],
                f'  {i}', fontsize=10, color='black',
                fontweight='bold' if i in [0, 1, 4] else 'normal')

    # =========================================================================
    # 4. Highlight special points
    # =========================================================================
    # Point 1 (defines Z axis)
    ax.scatter(points_m[1, 0], points_m[1, 1], points_m[1, 2],
               c='blue', s=200, marker='s',
               edgecolors='darkblue', linewidths=2, zorder=4)

    # Point 4 (used to define X axis)
    ax.scatter(points_m[4, 0], points_m[4, 1], points_m[4, 2],
               c='red', s=200, marker='s',
               edgecolors='darkred', linewidths=2, zorder=4)

    # Draw line from 0 to 1 (Z axis direction)
    ax.plot([origin_m[0], points_m[1, 0]],
            [origin_m[1], points_m[1, 1]],
            [origin_m[2], points_m[1, 2]],
            'b--', linewidth=2, alpha=0.5, label='0→1 (Z direction)')

    # Draw line from 0 to 4 (used for X axis computation)
    ax.plot([origin_m[0], points_m[4, 0]],
            [origin_m[1], points_m[4, 1]],
            [origin_m[2], points_m[4, 2]],
            'r--', linewidth=2, alpha=0.5, label='0→4 (for X definition)')

    # =========================================================================
    # 5. Draw plane formed by points 0, 1, 4
    # =========================================================================
    # Create a mesh grid in the plane
    vec_01 = points_m[1] - origin_m
    vec_04 = points_m[4] - origin_m

    # Parametric representation: P = origin + s*vec_01 + t*vec_04
    s_range = np.linspace(-0.2, 1.2, 10)
    t_range = np.linspace(-0.2, 1.2, 10)
    S, T = np.meshgrid(s_range, t_range)

    X_plane = origin_m[0] + S * vec_01[0] + T * vec_04[0]
    Y_plane = origin_m[1] + S * vec_01[1] + T * vec_04[1]
    Z_plane = origin_m[2] + S * vec_01[2] + T * vec_04[2]

    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.1, color='yellow',
                    label='Plane 0-1-4')

    # =========================================================================
    # 6. Configure plot
    # =========================================================================
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('Pyramid Frame and Points (0-17)\n' +
                 'Z: 0→1 | X: ⊥Z in plane 0-1-4 | Y: Z×X (right-handed)',
                 fontsize=14, fontweight='bold', pad=20)

    # Equal aspect ratio
    all_points = np.vstack([points_m, origin_m.reshape(1, 3)])
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Legend (without duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper left', fontsize=10, framealpha=0.9)

    # =========================================================================
    # 7. Add information text box
    # =========================================================================
    info_text = "FRAME DEFINITION:\n"
    info_text += "=" * 40 + "\n"
    info_text += f"Origin: Point 0\n"
    info_text += f"Z axis: 0 → 1\n"
    info_text += f"X axis: ⊥Z, in plane 0-1-4\n"
    info_text += f"Y axis: Z × X (right-handed)\n\n"
    info_text += "ROTATION MATRIX:\n"
    info_text += f"Det(R): {np.linalg.det(R_pyramid):.6f}\n"
    info_text += f"Orthogonality: {np.linalg.norm(R_pyramid @ R_pyramid.T - np.eye(3)):.2e}\n\n"
    info_text += "SPECIAL POINTS:\n"
    info_text += f"Point 0 (origin): red star\n"
    info_text += f"Point 1 (Z dir): blue square\n"
    info_text += f"Point 4 (X def): red square\n"

    plt.figtext(0.02, 0.02, info_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # =========================================================================
    # 8. Print coordinates to console
    # =========================================================================
    print("\n" + "=" * 70)
    print("PYRAMID FRAME AND POINTS")
    print("=" * 70)

    print(f"\nOrigin (Point 0): {origin_m}")
    print(f"\nFrame axes:")
    print(f"  X axis: {x_axis} (⊥Z, in plane 0-1-4)")
    print(f"  Y axis: {y_axis} (Z×X, right-handed)")
    print(f"  Z axis: {z_axis} (0→1)")

    print(f"\nRotation matrix verification:")
    print(f"  Determinant: {np.linalg.det(R_pyramid):.6f}")
    print(f"  Orthogonality: {np.linalg.norm(R_pyramid @ R_pyramid.T - np.eye(3)):.2e}")

    print(f"\nPyramid points (0-17) in m:")
    for i, point in enumerate(points_m):
        print(f"  Point {i:2d}: [{point[0]:8.3f}, {point[1]:8.3f}, {point[2]:8.3f}]")

    print("=" * 70)

    # =========================================================================
    # 9. Show or save
    # =========================================================================
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")

    if interactive:
        print("\n✓ Interactive 3D plot opened")
        print("  - Click and drag to rotate")
        print("  - Scroll to zoom")
        print("  - Close window to continue")
        plt.show()
    else:
        plt.close()

    return fig, ax


def verify_matching(transformer, rb_data):
    # 2. Extract OptiTrack data
    marker_positions_m, rb_position_m, rb_quaternion = extract_marker_positions_from_rb_data(
        rb_data,
        frame_id=0
    )

    # 3. Define known matching
    matching = {
        'Marker 002': 20,
        'Marker 001': 21,
        'Marker 003': 18,
        'Marker 004': 19
    }

    # 4. Verify matching (optional - will confirm your matching is correct)
    verified_matching = transformer.match_constellation_markers(
        marker_positions_m,
        initial_guess=matching
    )

    # 5. Get OptiTrack rotation from quaternion
    R_constellation_to_optitrack = quaternion_to_rotation_matrix(rb_quaternion)

    # 6. Set OptiTrack rotation in transformer
    transformer.set_optitrack_rotation(R_constellation_to_optitrack)

    # 7. Now you can transform any point from pyramid frame to OptiTrack frame
    # Example: transform point 0 (pyramid origin)
    point_0_pyramid = np.array([[0, 0, 0]])  # Origin in pyramid frame
    point_0_optitrack = transformer.transform_pyramid_to_optitrack(point_0_pyramid)
    print(f"Point 0 in OptiTrack frame: {point_0_optitrack}")

    # 8. Visualize the matching
    plot_marker_matching(
        transformer,
        marker_positions_m,
        verified_matching,
        save_path=Path("marker_matching.png")
    )


def verif_svd(rb_data, pyramid_json_path, matching: Optional[dict[str, int]] = None):
    """
    Verify SVD-based transformation using PyramidTransformer.

    Args:
        rb_data: Rigid body data structure
        pyramid_json_path: Path to pyramid JSON model file
        matching: Optional dict mapping marker names to point indices
                 If None, uses default: {'Marker 002': 20, 'Marker 001': 21,
                                        'Marker 003': 18, 'Marker 004': 19}

    Returns:
        transformer: The initialized PyramidTransformer object
    """

    # 1. Initialize transformer
    transformer = PyramidTransformer(pyramid_json_path)

    # 2. Extract OptiTrack marker positions
    marker_positions_m, rb_position_m, rb_quaternion = extract_marker_positions_from_rb_data(
        rb_data,
        frame_id=0
    )

    # 3. Define matching (use default if not provided)
    if matching is None:
        matching = {
            'Marker 002': 20,
            'Marker 001': 21,
            'Marker 003': 18,
            'Marker 004': 19
        }

    # 4. Compute rotation using SVD
    R_constellation_to_optitrack = transformer.compute_optitrack_rotation_from_markers(
        marker_positions_m,
        matching
    )
    R_pyramid_to_optitrack = transformer.R_pyramid_to_optitrack

    # 5. Now transformer is ready - transform points from pyramid to OptiTrack
    points_pyramid = np.array([[0, 0, 0]])  # Example: pyramid origin
    points_optitrack = transformer.transform_pyramid_to_optitrack(points_pyramid)

    print(f"\nPyramid origin in OptiTrack frame: {points_optitrack}")

    # 6. Verify by checking constellation points
    print("\n" + "=" * 70)
    print("VERIFICATION: Constellation points in OptiTrack frame")
    print("=" * 70)

    constellation_in_optitrack = transformer.get_constellation_points_in_optitrack_frame()

    for i, point_idx in enumerate([18, 19, 20, 21]):
        marker_name = [k for k, v in matching.items() if v == point_idx][0]

        # Get the transformed point
        transformed = constellation_in_optitrack[i]

        # Get the actual marker position (need to add barycenter offset)
        marker_pos_local = marker_positions_m[marker_name]

        # Calculate error
        error = np.linalg.norm(transformed - marker_pos_local)

        print(f"\nPoint {point_idx} → {marker_name}:")
        print(f"  Transformed: {transformed}")
        print(f"  Actual:      {marker_pos_local}")
        print(f"  Error:       {error * 1000:.6f} m")

    plot_svd_fit_quality(
        transformer,
        marker_positions_m,
        matching,
        save_path=Path("marker_svd_matching.png")
    )

    return transformer

# Example usage in your main script:
"""
from utils import verify_pyramid_transformation

# After setting up transformer:
verify_pyramid_transformation(
    rb_data=rb_data,
    calib_data=calib_data,
    transformer=transformer,
    frame_id=0
)
"""
