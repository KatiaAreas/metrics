"""
Point Cloud and Depth Map Comparison Metrics
Compares PLY point clouds and depth maps against STL reference models
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class PointCloudMetrics:
    """Metrics for comparing accumulated point clouds (PLY) against STL models"""
    
    def __init__(self, ply_points: np.ndarray, stl_path: str):
        """
        Initialize with point cloud and STL model
        
        Args:
            ply_points: Nx3 array of point coordinates
            stl_path: Path to STL file
        """
        self.points = np.array(ply_points)
        self.mesh = trimesh.load(stl_path)
        self.tree = cKDTree(self.mesh.vertices)
        
    def cluster_points(self, eps: float = 0.01, min_samples: int = 3) -> Dict:
        """
        Cluster points that should represent the same physical point
        
        Args:
            eps: Maximum distance between points in a cluster (meters)
            min_samples: Minimum number of points to form a cluster
            
        Returns:
            Dictionary with cluster labels and centroids
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(self.points)
        labels = clustering.labels_
        
        # Remove noise points (label = -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        clusters = {}
        for i in range(n_clusters):
            mask = labels == i
            cluster_points = self.points[mask]
            centroid = np.mean(cluster_points, axis=0)
            
            clusters[i] = {
                'points': cluster_points,
                'centroid': centroid,
                'size': len(cluster_points)
            }
        
        print(f"Found {n_clusters} clusters from {len(self.points)} points")
        return clusters
    
    def compute_cluster_dispersion(self, clusters: Dict) -> Dict:
        """
        Compute dispersion metrics for each cluster (noise quantification)
        
        Returns:
            Dictionary with dispersion metrics per cluster
        """
        dispersion_metrics = {}
        
        for cluster_id, cluster_data in clusters.items():
            points = cluster_data['points']
            centroid = cluster_data['centroid']
            
            # Standard deviation
            std_dev = np.std(points, axis=0)
            mean_std = np.mean(std_dev)
            
            # Distance from centroid
            distances = np.linalg.norm(points - centroid, axis=2)
            max_distance = np.max(distances)
            mean_distance = np.mean(distances)
            
            # PCA for ellipse fitting
            if len(points) >= 2:
                pca = PCA(n_components=min(3, len(points)))
                pca.fit(points)
                
                # Eigenvalues represent variance along principal axes
                ellipse_axes = np.sqrt(pca.explained_variance_)
            else:
                ellipse_axes = np.array([0, 0, 0])
            
            dispersion_metrics[cluster_id] = {
                'std_dev': std_dev,
                'mean_std': mean_std,
                'max_distance': max_distance,
                'mean_distance': mean_distance,
                'ellipse_axes': ellipse_axes,  # Principal axes lengths
                'ellipse_ratio': ellipse_axes[0] / (ellipse_axes[1] + 1e-10) if len(ellipse_axes) > 1 else 1.0
            }
        
        return dispersion_metrics
    
    def compute_distance_to_mesh(self, points: np.ndarray) -> np.ndarray:
        """
        Compute shortest distance from points to mesh surface
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Array of distances
        """
        # Find closest point on mesh for each point
        closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
            self.mesh, points
        )
        
        return distances
    
    def compute_point_to_surface_metrics(self, clusters: Dict) -> Dict:
        """
        Compute distance metrics from cluster centroids to STL surface
        
        Returns:
            Dictionary with distance metrics
        """
        centroids = np.array([c['centroid'] for c in clusters.values()])
        distances = self.compute_distance_to_mesh(centroids)
        
        metrics = {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances),
            'min_distance': np.min(distances),
            'rmse': np.sqrt(np.mean(distances**2)),
            'mae': np.mean(np.abs(distances)),
            'distances_per_cluster': distances
        }
        
        return metrics
    
    def compute_combined_metric(self, clusters: Dict, dispersion_metrics: Dict, 
                                distance_metrics: Dict, alpha: float = 0.5) -> float:
        """
        Compute combined metric: distance to STL + weighted dispersion
        
        Args:
            alpha: Weight factor for dispersion (0 to 1)
            
        Returns:
            Combined error metric
        """
        mean_distance_to_stl = distance_metrics['mean_distance']
        mean_dispersion = np.mean([d['mean_distance'] for d in dispersion_metrics.values()])
        
        combined = alpha * mean_distance_to_stl + (1 - alpha) * mean_dispersion
        
        return combined
    
    def visualize_clusters(self, clusters: Dict, save_path: str = None):
        """Visualize point clusters and their dispersion"""
        fig = plt.figure(figsize=(15, 5))
        
        # 3D scatter plot
        ax1 = fig.add_subplot(131, projection='3d')
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
        
        for idx, (cluster_id, cluster_data) in enumerate(clusters.items()):
            points = cluster_data['points']
            centroid = cluster_data['centroid']
            
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c=[colors[idx]], alpha=0.6, s=20)
            ax1.scatter(*centroid, c='red', marker='x', s=100)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Point Clusters (red X = centroids)')
        
        # Dispersion per cluster
        ax2 = fig.add_subplot(132)
        dispersion = self.compute_cluster_dispersion(clusters)
        cluster_ids = list(dispersion.keys())
        mean_distances = [d['mean_distance'] for d in dispersion.values()]
        
        ax2.bar(cluster_ids, mean_distances)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Mean Distance from Centroid (m)')
        ax2.set_title('Cluster Dispersion')
        
        # Distance to STL
        ax3 = fig.add_subplot(133)
        distance_metrics = self.compute_point_to_surface_metrics(clusters)
        distances = distance_metrics['distances_per_cluster']
        
        ax3.bar(cluster_ids, distances)
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Distance to STL Surface (m)')
        ax3.set_title('Centroid Distance to Reference Model')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, clusters: Dict) -> str:
        """Generate a comprehensive text report"""
        dispersion = self.compute_cluster_dispersion(clusters)
        distance_metrics = self.compute_point_to_surface_metrics(clusters)
        combined = self.compute_combined_metric(clusters, dispersion, distance_metrics)
        
        report = "=" * 70 + "\n"
        report += "POINT CLOUD COMPARISON REPORT (PLY vs STL)\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Total points: {len(self.points)}\n"
        report += f"Number of clusters: {len(clusters)}\n"
        report += f"Expected clusters: 18\n\n"
        
        report += "DISPERSION METRICS (Noise Quantification):\n"
        report += "-" * 70 + "\n"
        mean_dispersion = np.mean([d['mean_distance'] for d in dispersion.values()])
        report += f"Average cluster dispersion: {mean_dispersion*1000:.3f} mm\n"
        
        for cluster_id, disp in dispersion.items():
            report += f"\n  Cluster {cluster_id}:\n"
            report += f"    Points: {clusters[cluster_id]['size']}\n"
            report += f"    Mean distance from centroid: {disp['mean_distance']*1000:.3f} mm\n"
            report += f"    Max distance from centroid: {disp['max_distance']*1000:.3f} mm\n"
            report += f"    Ellipse axes (mm): [{disp['ellipse_axes'][0]*1000:.2f}, "
            report += f"{disp['ellipse_axes'][1]*1000:.2f}, {disp['ellipse_axes'][2]*1000:.2f}]\n"
            report += f"    Ellipse ratio: {disp['ellipse_ratio']:.2f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        report += "DISTANCE TO STL METRICS:\n"
        report += "-" * 70 + "\n"
        report += f"Mean distance: {distance_metrics['mean_distance']*1000:.3f} mm\n"
        report += f"Median distance: {distance_metrics['median_distance']*1000:.3f} mm\n"
        report += f"Std deviation: {distance_metrics['std_distance']*1000:.3f} mm\n"
        report += f"RMSE: {distance_metrics['rmse']*1000:.3f} mm\n"
        report += f"MAE: {distance_metrics['mae']*1000:.3f} mm\n"
        report += f"Min distance: {distance_metrics['min_distance']*1000:.3f} mm\n"
        report += f"Max distance: {distance_metrics['max_distance']*1000:.3f} mm\n"
        
        report += "\n" + "=" * 70 + "\n"
        report += f"COMBINED METRIC: {combined*1000:.3f} mm\n"
        report += "=" * 70 + "\n"
        
        return report


class DepthMapMetrics:
    """Metrics for comparing depth maps against STL models"""
    
    def __init__(self, depth_map: np.ndarray, K: np.ndarray, RT: np.ndarray, 
                 stl_path: str, depth_scale: float = 1000.0):
        """
        Initialize with depth map and camera parameters
        
        Args:
            depth_map: nxm depth map in mm (float values)
            K: 3x3 camera intrinsic matrix
            RT: 4x4 transformation matrix [R|t] or 3x4 [R|t]
            stl_path: Path to STL file (in meters)
            depth_scale: Scale factor to convert depth to meters (default: 1000 for mm→m)
        """
        self.depth_map = depth_map
        self.K = K
        self.depth_scale = depth_scale
        
        # Handle both 3x4 and 4x4 RT matrices
        if RT.shape == (3, 4):
            self.R = RT[:, :3]
            self.t = RT[:, 3]
        elif RT.shape == (4, 4):
            self.R = RT[:3, :3]
            self.t = RT[:3, 3]
        else:
            raise ValueError("RT must be either 3x4 or 4x4 matrix")
        
        self.mesh = trimesh.load(stl_path)
        self.point_cloud = None
        
    def deproject_depth_to_3d(self) -> np.ndarray:
        """
        Convert depth map to 3D point cloud using camera intrinsics
        
        Returns:
            Nx3 array of 3D points in world coordinates
        """
        h, w = self.depth_map.shape
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Get valid depth pixels (non-zero, non-nan)
        valid_mask = (self.depth_map > 0) & (~np.isnan(self.depth_map))
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = self.depth_map[valid_mask] / self.depth_scale  # Convert to meters
        
        # Camera intrinsics
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        # Deproject to camera coordinates
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        z_cam = z_valid
        
        # Stack to Nx3
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Transform to world coordinates
        points_world = (self.R @ points_cam.T).T + self.t
        
        self.point_cloud = points_world
        self.valid_mask = valid_mask
        self.pixel_coords = np.stack([v_valid, u_valid], axis=1)  # Store for back-projection
        
        print(f"Deprojected {len(points_world)} valid points from {h}x{w} depth map")
        
        return points_world
    
    def compute_point_to_mesh_distances(self) -> np.ndarray:
        """
        Compute distance from each point in cloud to mesh surface
        
        Returns:
            Array of distances (in meters)
        """
        if self.point_cloud is None:
            self.deproject_depth_to_3d()
        
        closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
            self.mesh, self.point_cloud
        )
        
        return distances
    
    def compute_accuracy_metrics(self) -> Dict:
        """
        Compute accuracy metrics (RMSE, MAE, etc.)
        
        Returns:
            Dictionary of accuracy metrics
        """
        distances = self.compute_point_to_mesh_distances()
        
        metrics = {
            'rmse': np.sqrt(np.mean(distances**2)),
            'mae': np.mean(np.abs(distances)),
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'percentile_95': np.percentile(distances, 95),
            'percentile_99': np.percentile(distances, 99),
            'num_valid_points': len(distances),
            'total_pixels': self.depth_map.size,
            'valid_ratio': len(distances) / self.depth_map.size
        }
        
        return metrics
    
    def compute_completeness(self, threshold: float = 0.01) -> Dict:
        """
        Compute completeness: percentage of STL surface covered by points
        
        Args:
            threshold: Distance threshold in meters to consider a surface point as "covered"
            
        Returns:
            Dictionary with completeness metrics
        """
        if self.point_cloud is None:
            self.deproject_depth_to_3d()
        
        # Sample points uniformly on mesh surface
        mesh_samples, _ = trimesh.sample.sample_surface(self.mesh, count=10000)
        
        # Build KD-tree of point cloud
        tree = cKDTree(self.point_cloud)
        
        # Find distance from each mesh sample to nearest point cloud point
        distances, _ = tree.query(mesh_samples)
        
        # Count how many mesh points are within threshold
        covered = np.sum(distances < threshold)
        total = len(mesh_samples)
        
        completeness = {
            'completeness_ratio': covered / total,
            'completeness_percentage': 100 * covered / total,
            'covered_points': covered,
            'total_surface_points': total,
            'threshold_m': threshold,
            'mean_distance_to_cloud': np.mean(distances),
            'median_distance_to_cloud': np.median(distances)
        }
        
        return completeness
    
    def create_error_heatmap(self, save_path: str = None):
        """
        Create 2D heatmap of errors projected back onto depth image
        """
        if self.point_cloud is None:
            self.deproject_depth_to_3d()
        
        distances = self.compute_point_to_mesh_distances()
        
        # Create error image
        h, w = self.depth_map.shape
        error_map = np.full((h, w), np.nan)
        
        # Fill in errors at valid pixel locations
        rows = self.pixel_coords[:, 0]
        cols = self.pixel_coords[:, 1]
        error_map[rows, cols] = distances * 1000  # Convert to mm
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original depth map
        im1 = axes[0].imshow(self.depth_map, cmap='viridis')
        axes[0].set_title('Original Depth Map (mm)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Error heatmap
        im2 = axes[1].imshow(error_map, cmap='jet', vmin=0, vmax=np.nanpercentile(error_map, 95))
        axes[1].set_title('Error Heatmap (mm)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Distance to STL (mm)')
        
        # Error histogram
        axes[2].hist(distances * 1000, bins=50, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Distance to STL (mm)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Error Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return error_map
    
    def compute_regional_metrics(self, regions: List[Tuple[slice, slice]]) -> Dict:
        """
        Compute metrics for specific regions of interest in the depth map
        
        Args:
            regions: List of (row_slice, col_slice) tuples defining regions
            
        Returns:
            Dictionary with metrics per region
        """
        if self.point_cloud is None:
            self.deproject_depth_to_3d()
        
        distances = self.compute_point_to_mesh_distances()
        h, w = self.depth_map.shape
        
        regional_metrics = {}
        
        for idx, (row_slice, col_slice) in enumerate(regions):
            # Create mask for this region
            region_mask = np.zeros((h, w), dtype=bool)
            region_mask[row_slice, col_slice] = True
            
            # Find points that belong to this region
            rows = self.pixel_coords[:, 0]
            cols = self.pixel_coords[:, 1]
            
            in_region = region_mask[rows, cols]
            region_distances = distances[in_region]
            
            if len(region_distances) > 0:
                regional_metrics[f'region_{idx}'] = {
                    'rmse': np.sqrt(np.mean(region_distances**2)),
                    'mae': np.mean(np.abs(region_distances)),
                    'mean': np.mean(region_distances),
                    'std': np.std(region_distances),
                    'max': np.max(region_distances),
                    'num_points': len(region_distances)
                }
            else:
                regional_metrics[f'region_{idx}'] = {
                    'rmse': np.nan,
                    'mae': np.nan,
                    'mean': np.nan,
                    'std': np.nan,
                    'max': np.nan,
                    'num_points': 0
                }
        
        return regional_metrics
    
    def generate_report(self) -> str:
        """Generate comprehensive text report"""
        if self.point_cloud is None:
            self.deproject_depth_to_3d()
        
        accuracy = self.compute_accuracy_metrics()
        completeness = self.compute_completeness(threshold=0.01)
        
        report = "=" * 70 + "\n"
        report += "DEPTH MAP COMPARISON REPORT (Depth → 3D vs STL)\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Depth map size: {self.depth_map.shape[0]} x {self.depth_map.shape[1]}\n"
        report += f"Total pixels: {accuracy['total_pixels']}\n"
        report += f"Valid depth pixels: {accuracy['num_valid_points']} ({accuracy['valid_ratio']*100:.1f}%)\n\n"
        
        report += "ACCURACY METRICS:\n"
        report += "-" * 70 + "\n"
        report += f"RMSE: {accuracy['rmse']*1000:.3f} mm\n"
        report += f"MAE: {accuracy['mae']*1000:.3f} mm\n"
        report += f"Mean distance: {accuracy['mean_distance']*1000:.3f} mm\n"
        report += f"Median distance: {accuracy['median_distance']*1000:.3f} mm\n"
        report += f"Std deviation: {accuracy['std_distance']*1000:.3f} mm\n"
        report += f"Min distance: {accuracy['min_distance']*1000:.3f} mm\n"
        report += f"Max distance: {accuracy['max_distance']*1000:.3f} mm\n"
        report += f"95th percentile: {accuracy['percentile_95']*1000:.3f} mm\n"
        report += f"99th percentile: {accuracy['percentile_99']*1000:.3f} mm\n\n"
        
        report += "COMPLETENESS METRICS:\n"
        report += "-" * 70 + "\n"
        report += f"Surface coverage: {completeness['completeness_percentage']:.2f}%\n"
        report += f"Covered surface points: {completeness['covered_points']} / {completeness['total_surface_points']}\n"
        report += f"Coverage threshold: {completeness['threshold_m']*1000:.1f} mm\n"
        report += f"Mean STL-to-cloud distance: {completeness['mean_distance_to_cloud']*1000:.3f} mm\n"
        report += f"Median STL-to-cloud distance: {completeness['median_distance_to_cloud']*1000:.3f} mm\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report


# Example usage functions
def example_ply_comparison():
    """Example: Compare accumulated point cloud (PLY) to STL"""
    print("\n" + "="*70)
    print("EXAMPLE 1: PLY Point Cloud Comparison")
    print("="*70 + "\n")
    
    # Simulate accumulated points (18 points × N frames with noise)
    np.random.seed(42)
    n_frames = 50
    expected_points = 18
    
    # Generate 18 true positions on a simple surface
    true_positions = np.random.rand(expected_points, 3) * 0.1
    
    # Accumulate points with noise
    accumulated_points = []
    for _ in range(n_frames):
        for pos in true_positions:
            # Add Gaussian noise to simulate detection error
            noisy_point = pos + np.random.normal(0, 0.002, 3)
            accumulated_points.append(noisy_point)
    
    accumulated_points = np.array(accumulated_points)
    
    print(f"Generated {len(accumulated_points)} accumulated points")
    print(f"Expected clusters: {expected_points}")
    
    # Note: You would replace this with your actual STL file path
    # stl_path = "your_model.stl"
    # metrics = PointCloudMetrics(accumulated_points, stl_path)
    
    print("\nTo use with real data:")
    print("  metrics = PointCloudMetrics(ply_points, 'model.stl')")
    print("  clusters = metrics.cluster_points(eps=0.01)")
    print("  report = metrics.generate_report(clusters)")
    print("  metrics.visualize_clusters(clusters, 'output.png')")


def example_depth_comparison():
    """Example: Compare depth map to STL"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Depth Map Comparison")
    print("="*70 + "\n")
    
    # Simulate depth map
    h, w = 480, 640
    depth_map = np.random.rand(h, w) * 1000 + 500  # Random depths 500-1500mm
    depth_map[depth_map < 600] = 0  # Some invalid pixels
    
    # Camera intrinsic matrix (example)
    K = np.array([
        [525.0, 0, 319.5],
        [0, 525.0, 239.5],
        [0, 0, 1]
    ])
    
    # Extrinsic matrix [R|t]
    RT = np.eye(4)
    RT[:3, 3] = [0, 0, 1]  # 1 meter offset in Z
    
    print(f"Depth map size: {h} x {w}")
    print(f"Valid pixels: {np.sum(depth_map > 0)}")
    
    print("\nTo use with real data:")
    print("  metrics = DepthMapMetrics(depth_map, K, RT, 'model.stl')")
    print("  metrics.deproject_depth_to_3d()")
    print("  report = metrics.generate_report()")
    print("  metrics.create_error_heatmap('heatmap.png')")
    print("  completeness = metrics.compute_completeness(threshold=0.01)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("POINT CLOUD AND DEPTH MAP METRICS TOOLKIT")
    print("="*70)
    
    example_ply_comparison()
    example_depth_comparison()
    
    print("\n" + "="*70)
    print("Required libraries: numpy, trimesh, scipy, sklearn, matplotlib, seaborn")
    print("Install with: pip install numpy trimesh scipy scikit-learn matplotlib seaborn")
    print("="*70 + "\n")
