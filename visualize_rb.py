import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from areas_common.geo3d.quaternion import Quaternion


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix
    q = [w, x, y, z] or [x, y, z, w] depending on convention
    """
    # Assuming q is [x, y, z, w] format (common in robotics)
    # Adjust if your quaternion format is different
    if len(q) == 4:
        x, y, z, w = q
    else:
        raise ValueError("Quaternion must have 4 components")

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
    ])

    return R


def visualize_rigid_body(rb_data, frame_id, marker_names=['Marker 001', 'Marker 002', 'Marker 003', 'Marker 004'],
                         save_path=None):
    """
    Visualize rigid body markers, barycenter, and orientation

    Parameters:
    -----------
    rb_data : dict
        Dictionary containing rigid body data
    frame_id : int
        Frame ID to visualize
    marker_names : list
        List of marker names to display
    save_path : str, optional
        Path to save the figure. If None, returns the figure without saving.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes3D
        The 3D axes object
    marker_distances : dict
        Dictionary with marker names as keys and distances to barycenter as values
    rotation : numpy.ndarray (3x3)
        Rotation matrix from marker frame to barycenter frame (R_relative)
    rotation_data : dict
        Dictionary containing detailed rotation information
    """

    # Extract data
    frame_data = rb_data["Pyramid_RB"][frame_id].data

    # Get barycenter position
    barycenter = np.array(frame_data.position)

    # Get orientation quaternion
    quaternion = np.array(frame_data.orientation)

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

    # Sort markers by distance (closest to farthest)
    sorted_markers = sorted(marker_distances.items(), key=lambda x: x[1])

    # Print distance ranking
    print(f"\n{'=' * 70}")
    print(f"Distance Ranking for Frame {frame_id} (closest to farthest):")
    print(f"{'=' * 70}")
    for rank, (marker_name, distance) in enumerate(sorted_markers, 1):
        print(f"Rank {rank}: {marker_name:15s} - Distance: {distance:.6f} units")
    print(f"{'=' * 70}\n")

    # ========== CREATE MARKER-BASED REFERENCE FRAME ==========
    # Step 1: Calculate Z-axis (normal to the plane of 4 markers)
    marker_positions_array = np.array([markers[name].flatten() for name in marker_names if name in markers])
    barycenter_flat = barycenter.flatten()

    # Center the markers around barycenter
    centered_markers = marker_positions_array - barycenter_flat

    # Calculate the normal to the plane using SVD
    # The last singular vector corresponds to the normal of the best-fit plane
    U, S, Vt = np.linalg.svd(centered_markers)
    Z_marker = Vt[-1, :]  # Normal to the plane
    Z_marker = Z_marker / np.linalg.norm(Z_marker)  # Normalize

    # Ensure Z points in a consistent direction (positive Z if possible)
    if Z_marker[2] < 0:
        Z_marker = -Z_marker

    # Step 2: Calculate Y-axis (direction to Marker 002, projected onto the plane)
    marker_002_pos = markers['Marker 002'].flatten()
    Y_direction = marker_002_pos - barycenter_flat

    # Project Y onto the plane by removing the component along Z
    Y_marker = Y_direction - np.dot(Y_direction, Z_marker) * Z_marker
    Y_marker = Y_marker / np.linalg.norm(Y_marker)  # Normalize

    # Step 3: Calculate X-axis (in plane, orthogonal to Y)
    # X = Z × Y to ensure right-handed coordinate system
    X_marker = np.cross(Z_marker, Y_marker)
    X_marker = X_marker / np.linalg.norm(X_marker)  # Normalize

    # Create rotation matrix from marker-based frame
    R_marker = np.column_stack([X_marker, Y_marker, Z_marker])

    print(f"\n{'=' * 70}")
    print(f"Marker-Based Reference Frame:")
    print(f"{'=' * 70}")
    print(f"X-axis (in plane, ⊥ to Y):    [{X_marker[0]:8.5f}, {X_marker[1]:8.5f}, {X_marker[2]:8.5f}]")
    print(f"Y-axis (in plane, to M002):   [{Y_marker[0]:8.5f}, {Y_marker[1]:8.5f}, {Y_marker[2]:8.5f}]")
    print(f"Z-axis (normal to plane):     [{Z_marker[0]:8.5f}, {Z_marker[1]:8.5f}, {Z_marker[2]:8.5f}]")
    print(f"{'=' * 70}\n")

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ===== DRAW THE PLANE OF THE 4 MARKERS =====
    # Create a mesh grid for the plane
    # Find the bounding box of the markers in the plane coordinate system
    marker_coords_in_plane = []
    for marker_pos in markers.values():
        pos = marker_pos.flatten()
        # Express position relative to barycenter
        rel_pos = pos - barycenter_flat
        # Project onto plane axes (X and Y of marker frame)
        x_coord = np.dot(rel_pos, X_marker)
        y_coord = np.dot(rel_pos, Y_marker)
        marker_coords_in_plane.append([x_coord, y_coord])

    marker_coords_in_plane = np.array(marker_coords_in_plane)

    # Extend the plane beyond the markers
    x_min, x_max = marker_coords_in_plane[:, 0].min(), marker_coords_in_plane[:, 0].max()
    y_min, y_max = marker_coords_in_plane[:, 1].min(), marker_coords_in_plane[:, 1].max()

    # Add margin
    margin = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin

    # Create mesh grid in plane coordinates
    u = np.linspace(x_min, x_max, 10)
    v = np.linspace(y_min, y_max, 10)
    U, V = np.meshgrid(u, v)

    # Convert plane coordinates to 3D coordinates
    X_plane = barycenter_flat[0] + U * X_marker[0] + V * Y_marker[0]
    Y_plane = barycenter_flat[1] + U * X_marker[1] + V * Y_marker[1]
    Z_plane = barycenter_flat[2] + U * X_marker[2] + V * Y_marker[2]

    # Plot the plane as a semi-transparent surface
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.15, color='cyan',
                    edgecolor='none', label='Marker Plane')

    # Plot barycenter
    ax.scatter(barycenter[0][0], barycenter[0][1], barycenter[0][2],
               c='red', s=200, marker='o', label='Barycenter',
               edgecolors='black', linewidths=2)

    # Plot markers and lines to barycenter with distance information
    colors = ['blue', 'green', 'orange', 'purple']
    legend_entries = []

    for idx, (marker_name, marker_pos) in enumerate(markers.items()):
        color = colors[idx % len(colors)]
        distance = marker_distances[marker_name]

        # Get rank for this marker
        rank = next(i for i, (name, _) in enumerate(sorted_markers, 1) if name == marker_name)

        # Plot marker
        ax.scatter(marker_pos[0][0], marker_pos[0][1], marker_pos[0][2],
                   c=color, s=100, marker='^',
                   edgecolors='black', linewidths=1.5)

        # Add marker label with distance and rank
        label_text = f'{marker_name}\nd={distance:.4f}\n(rank {rank})'
        ax.text(marker_pos[0][0], marker_pos[0][1], marker_pos[0][2], label_text,
                fontsize=8, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))

        # Draw line from marker to barycenter
        line = ax.plot([marker_pos[0][0], barycenter[0][0]],
                       [marker_pos[0][1], barycenter[0][1]],
                       [marker_pos[0][2], barycenter[0][2]],
                       c=color, linestyle='--', linewidth=2, alpha=0.6)[0]

        legend_entries.append((line, f'{marker_name}: {distance:.4f} (rank {rank})'))

    # Convert quaternion to rotation matrix and plot orientation axes
    try:
        quaternion = np.asarray(quaternion).item()
        R_barycenter = quaternion_to_rotation_matrix([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

        # Calculate rotation between marker frame and barycenter frame
        # R_relative transforms from marker frame to barycenter frame
        R_relative = R_barycenter @ R_marker.T

        # Convert rotation matrix to axis-angle representation
        trace = np.trace(R_relative)
        angle_rad = np.arccos((trace - 1) / 2)
        angle_deg = np.degrees(angle_rad)

        # Calculate rotation axis (if angle is not zero)
        if angle_rad > 1e-6:
            rotation_axis = np.array([
                R_relative[2, 1] - R_relative[1, 2],
                R_relative[0, 2] - R_relative[2, 0],
                R_relative[1, 0] - R_relative[0, 1]
            ]) / (2 * np.sin(angle_rad))
        else:
            rotation_axis = np.array([0, 0, 1])  # Arbitrary axis for zero rotation

        # Convert to Euler angles (ZYX convention)
        def rotation_matrix_to_euler_angles(R):
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0
            return np.degrees([x, y, z])

        euler_angles = rotation_matrix_to_euler_angles(R_relative)

        print(f"{'=' * 70}")
        print(f"Rotation from Marker Frame to Barycenter Frame:")
        print(f"{'=' * 70}")
        print(f"Rotation angle: {angle_deg:.4f} degrees")
        print(f"Rotation axis:  [{rotation_axis[0]:8.5f}, {rotation_axis[1]:8.5f}, {rotation_axis[2]:8.5f}]")
        print(
            f"Euler angles (ZYX): Roll={euler_angles[0]:.4f}°, Pitch={euler_angles[1]:.4f}°, Yaw={euler_angles[2]:.4f}°")
        print(f"{'=' * 70}\n")

        # Scale for visualization - SMALLER ARROWS
        axis_length = 0.05  # Reduced from 0.1
        arrow_width = 2  # Thinner arrows

        # ===== Plot Barycenter Frame in RED =====
        # X-axis
        x_axis_bary = R_barycenter[:, 0] * axis_length
        arrow_x_bary = Arrow3D(barycenter[0][0], barycenter[0][1], barycenter[0][2],
                               x_axis_bary[0], x_axis_bary[1], x_axis_bary[2],
                               mutation_scale=15, lw=arrow_width, arrowstyle="-|>", color="red")
        ax.add_artist(arrow_x_bary)

        # Y-axis
        y_axis_bary = R_barycenter[:, 1] * axis_length
        arrow_y_bary = Arrow3D(barycenter[0][0], barycenter[0][1], barycenter[0][2],
                               y_axis_bary[0], y_axis_bary[1], y_axis_bary[2],
                               mutation_scale=15, lw=arrow_width, arrowstyle="-|>", color="red")
        ax.add_artist(arrow_y_bary)

        # Z-axis
        z_axis_bary = R_barycenter[:, 2] * axis_length
        arrow_z_bary = Arrow3D(barycenter[0][0], barycenter[0][1], barycenter[0][2],
                               z_axis_bary[0], z_axis_bary[1], z_axis_bary[2],
                               mutation_scale=15, lw=arrow_width, arrowstyle="-|>", color="red")
        ax.add_artist(arrow_z_bary)

        # Add labels for barycenter frame
        ax.text(barycenter[0][0] + x_axis_bary[0] * 1.5,
                barycenter[0][1] + x_axis_bary[1] * 1.5,
                barycenter[0][2] + x_axis_bary[2] * 1.5,
                'Xb', fontsize=9, color='red', fontweight='bold')
        ax.text(barycenter[0][0] + y_axis_bary[0] * 1.5,
                barycenter[0][1] + y_axis_bary[1] * 1.5,
                barycenter[0][2] + y_axis_bary[2] * 1.5,
                'Yb', fontsize=9, color='red', fontweight='bold')
        ax.text(barycenter[0][0] + z_axis_bary[0] * 1.5,
                barycenter[0][1] + z_axis_bary[1] * 1.5,
                barycenter[0][2] + z_axis_bary[2] * 1.5,
                'Zb', fontsize=9, color='red', fontweight='bold')

        # ===== Plot Marker Frame in CYAN =====
        # X-axis marker
        x_axis_marker = X_marker * axis_length
        arrow_x_marker = Arrow3D(barycenter[0][0], barycenter[0][1], barycenter[0][2],
                                 x_axis_marker[0], x_axis_marker[1], x_axis_marker[2],
                                 mutation_scale=15, lw=arrow_width, arrowstyle="-|>", color="cyan", linestyle='--')
        ax.add_artist(arrow_x_marker)

        # Y-axis marker
        y_axis_marker = Y_marker * axis_length
        arrow_y_marker = Arrow3D(barycenter[0][0], barycenter[0][1], barycenter[0][2],
                                 y_axis_marker[0], y_axis_marker[1], y_axis_marker[2],
                                 mutation_scale=15, lw=arrow_width, arrowstyle="-|>", color="cyan", linestyle='--')
        ax.add_artist(arrow_y_marker)

        # Z-axis marker
        z_axis_marker = Z_marker * axis_length
        arrow_z_marker = Arrow3D(barycenter[0][0], barycenter[0][1], barycenter[0][2],
                                 z_axis_marker[0], z_axis_marker[1], z_axis_marker[2],
                                 mutation_scale=15, lw=arrow_width, arrowstyle="-|>", color="cyan", linestyle='--')
        ax.add_artist(arrow_z_marker)

        # Add labels for marker frame
        ax.text(barycenter[0][0] + x_axis_marker[0] * 1.5,
                barycenter[0][1] + x_axis_marker[1] * 1.5,
                barycenter[0][2] + x_axis_marker[2] * 1.5,
                'Xm', fontsize=9, color='cyan', fontweight='bold')
        ax.text(barycenter[0][0] + y_axis_marker[0] * 1.5,
                barycenter[0][1] + y_axis_marker[1] * 1.5,
                barycenter[0][2] + y_axis_marker[2] * 1.5,
                'Ym', fontsize=9, color='cyan', fontweight='bold')
        ax.text(barycenter[0][0] + z_axis_marker[0] * 1.5,
                barycenter[0][1] + z_axis_marker[1] * 1.5,
                barycenter[0][2] + z_axis_marker[2] * 1.5,
                'Zm', fontsize=9, color='cyan', fontweight='bold')

    except Exception as e:
        print(f"Could not visualize orientation: {e}")
        R_relative = None
        angle_deg = None
        euler_angles = None

    # Labels and title
    ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Position', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Rigid Body Visualization - Frame {frame_id}\nPyramid_RB\nBarycenter Frame (RED solid) vs Marker Frame (CYAN dashed)',
        fontsize=13, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    all_points = np.vstack([barycenter] + list(markers.values()))
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                          all_points[:, 1].max() - all_points[:, 1].min(),
                          all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Create distance ranking text box
    ranking_text = "Distance Ranking:\n"
    for rank, (marker_name, distance) in enumerate(sorted_markers, 1):
        ranking_text += f"{rank}. {marker_name}: {distance:.4f}\n"

    # Add info text with rotation information
    quaternion = np.asarray(quaternion).item()
    info_text = f"Barycenter: ({barycenter[0][0]:.3f}, {barycenter[0][1]:.3f}, {barycenter[0][2]:.3f})\n"
    info_text += f"Orientation (quat): ({quaternion.x:.3f}, {quaternion.y:.3f}, {quaternion.z:.3f}, {quaternion.w:.3f})\n\n"
    info_text += ranking_text

    if angle_deg is not None:
        info_text += f"\nRotation Marker→Barycenter:\n"
        info_text += f"  Angle: {angle_deg:.2f}°\n"
        if euler_angles is not None:
            info_text += f"  Euler: R={euler_angles[0]:.1f}° P={euler_angles[1]:.1f}° Y={euler_angles[2]:.1f}°\n"

    plt.figtext(0.02, 0.02, info_text, fontsize=7.5, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add custom legend with distance information
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    # Save or return figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close(fig)

    # Prepare rotation data dictionary (detailed information)
    rotation_data = {
        'R_marker': R_marker,
        'R_barycenter': R_barycenter if 'R_barycenter' in locals() else None,
        'R_relative': R_relative if 'R_relative' in locals() else None,
        'angle_deg': angle_deg if 'angle_deg' in locals() else None,
        'euler_angles': euler_angles if 'euler_angles' in locals() else None,
        'marker_frame_axes': {
            'X': X_marker,
            'Y': Y_marker,
            'Z': Z_marker
        }
    }

    # Return the 3x3 rotation matrix as main 'rotation' parameter
    rotation_matrix = R_relative if 'R_relative' in locals() else np.eye(3)

    return fig, ax, marker_distances, rotation_matrix, rotation_data


# Example usage:
if __name__ == "__main__":
    # This is example code - replace with your actual rb_data structure
    print("To use this script, call visualize_rigid_body() with your rb_data:")
    print(
        "fig, ax, distances, rotation, rotation_data = visualize_rigid_body(rb_data, frame_id, save_path='output.png')")
    print("\nThe function now returns:")
    print("  - fig: matplotlib figure")
    print("  - ax: 3D axes")
    print("  - distances: dictionary of marker distances to barycenter")
    print("  - rotation: 3x3 numpy array - rotation matrix from marker frame to barycenter frame")
    print("  - rotation_data: dictionary containing detailed rotation information")