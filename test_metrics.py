"""
Unit tests for point cloud and depth map metrics
"""

import numpy as np
import pytest
import trimesh
from point_cloud_metrics import PointCloudMetrics, DepthMapMetrics


class TestPointCloudMetrics:
    """Tests for PointCloudMetrics class"""
    
    @pytest.fixture
    def simple_mesh(self, tmp_path):
        """Create a simple STL mesh for testing"""
        # Create a simple cube
        mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        stl_path = tmp_path / "test_cube.stl"
        mesh.export(stl_path)
        return str(stl_path)
    
    @pytest.fixture
    def accumulated_points(self):
        """Generate simulated accumulated points"""
        np.random.seed(42)
        n_points = 18
        n_frames = 20
        
        # True positions
        true_positions = np.random.rand(n_points, 3) * 0.05
        
        # Accumulate with noise
        points = []
        for _ in range(n_frames):
            for pos in true_positions:
                noisy = pos + np.random.normal(0, 0.001, 3)
                points.append(noisy)
        
        return np.array(points)
    
    def test_initialization(self, accumulated_points, simple_mesh):
        """Test metrics initialization"""
        metrics = PointCloudMetrics(accumulated_points, simple_mesh)
        
        assert metrics.points.shape[0] == len(accumulated_points)
        assert metrics.mesh is not None
        assert metrics.tree is not None
    
    def test_clustering(self, accumulated_points, simple_mesh):
        """Test point clustering"""
        metrics = PointCloudMetrics(accumulated_points, simple_mesh)
        clusters = metrics.cluster_points(eps=0.005, min_samples=3)
        
        # Should find approximately 18 clusters
        assert 15 <= len(clusters) <= 21  # Allow some tolerance
        
        # Each cluster should have centroid
        for cluster in clusters.values():
            assert 'centroid' in cluster
            assert 'points' in cluster
            assert 'size' in cluster
            assert cluster['centroid'].shape == (3,)
    
    def test_dispersion_metrics(self, accumulated_points, simple_mesh):
        """Test dispersion calculation"""
        metrics = PointCloudMetrics(accumulated_points, simple_mesh)
        clusters = metrics.cluster_points(eps=0.005, min_samples=3)
        dispersion = metrics.compute_cluster_dispersion(clusters)
        
        assert len(dispersion) == len(clusters)
        
        for disp in dispersion.values():
            assert 'mean_std' in disp
            assert 'mean_distance' in disp
            assert 'ellipse_axes' in disp
            assert disp['mean_std'] >= 0
            assert disp['mean_distance'] >= 0
    
    def test_distance_to_mesh(self, accumulated_points, simple_mesh):
        """Test distance calculation to mesh"""
        metrics = PointCloudMetrics(accumulated_points, simple_mesh)
        
        # Test with a few points
        test_points = accumulated_points[:10]
        distances = metrics.compute_distance_to_mesh(test_points)
        
        assert len(distances) == len(test_points)
        assert np.all(distances >= 0)
    
    def test_point_to_surface_metrics(self, accumulated_points, simple_mesh):
        """Test point-to-surface metrics"""
        metrics = PointCloudMetrics(accumulated_points, simple_mesh)
        clusters = metrics.cluster_points(eps=0.005, min_samples=3)
        distance_metrics = metrics.compute_point_to_surface_metrics(clusters)
        
        assert 'rmse' in distance_metrics
        assert 'mae' in distance_metrics
        assert 'mean_distance' in distance_metrics
        assert distance_metrics['rmse'] >= 0
        assert distance_metrics['mae'] >= 0
    
    def test_combined_metric(self, accumulated_points, simple_mesh):
        """Test combined metric calculation"""
        metrics = PointCloudMetrics(accumulated_points, simple_mesh)
        clusters = metrics.cluster_points(eps=0.005, min_samples=3)
        dispersion = metrics.compute_cluster_dispersion(clusters)
        distance_metrics = metrics.compute_point_to_surface_metrics(clusters)
        
        # Test different alpha values
        for alpha in [0.0, 0.5, 1.0]:
            combined = metrics.compute_combined_metric(
                clusters, dispersion, distance_metrics, alpha
            )
            assert combined >= 0
    
    def test_report_generation(self, accumulated_points, simple_mesh):
        """Test report generation"""
        metrics = PointCloudMetrics(accumulated_points, simple_mesh)
        clusters = metrics.cluster_points(eps=0.005, min_samples=3)
        report = metrics.generate_report(clusters)
        
        assert isinstance(report, str)
        assert len(report) > 100
        assert 'POINT CLOUD COMPARISON REPORT' in report
        assert 'RMSE' in report


class TestDepthMapMetrics:
    """Tests for DepthMapMetrics class"""
    
    @pytest.fixture
    def simple_mesh(self, tmp_path):
        """Create a simple STL mesh for testing"""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        stl_path = tmp_path / "test_cube.stl"
        mesh.export(stl_path)
        return str(stl_path)
    
    @pytest.fixture
    def depth_map(self):
        """Generate a simple depth map"""
        h, w = 100, 100
        depth = np.ones((h, w), dtype=np.float32) * 1000  # 1m in mm
        
        # Add some variation
        depth += np.random.randn(h, w) * 10
        
        # Add some invalid pixels
        depth[0:10, 0:10] = 0
        
        return depth
    
    @pytest.fixture
    def camera_matrix(self):
        """Generate camera intrinsic matrix"""
        return np.array([
            [500.0, 0, 50.0],
            [0, 500.0, 50.0],
            [0, 0, 1]
        ])
    
    @pytest.fixture
    def transform_matrix(self):
        """Generate transformation matrix"""
        RT = np.eye(4)
        RT[:3, 3] = [0, 0, 1]  # 1m translation in Z
        return RT
    
    def test_initialization(self, depth_map, camera_matrix, transform_matrix, simple_mesh):
        """Test depth metrics initialization"""
        metrics = DepthMapMetrics(
            depth_map, camera_matrix, transform_matrix, simple_mesh
        )
        
        assert metrics.depth_map.shape == depth_map.shape
        assert metrics.K.shape == (3, 3)
        assert metrics.R.shape == (3, 3)
        assert metrics.t.shape == (3,)
    
    def test_initialization_3x4_matrix(self, depth_map, camera_matrix, simple_mesh):
        """Test initialization with 3x4 RT matrix"""
        RT_3x4 = np.eye(3, 4)
        RT_3x4[:3, 3] = [0, 0, 1]
        
        metrics = DepthMapMetrics(
            depth_map, camera_matrix, RT_3x4, simple_mesh
        )
        
        assert metrics.R.shape == (3, 3)
        assert metrics.t.shape == (3,)
    
    def test_deprojection(self, depth_map, camera_matrix, transform_matrix, simple_mesh):
        """Test depth map deprojection to 3D"""
        metrics = DepthMapMetrics(
            depth_map, camera_matrix, transform_matrix, simple_mesh
        )
        
        point_cloud = metrics.deproject_depth_to_3d()
        
        # Should have valid points
        assert len(point_cloud) > 0
        assert point_cloud.shape[1] == 3
        
        # Should have fewer points than total pixels (due to invalid pixels)
        assert len(point_cloud) < depth_map.size
    
    def test_valid_mask(self, depth_map, camera_matrix, transform_matrix, simple_mesh):
        """Test valid pixel mask generation"""
        metrics = DepthMapMetrics(
            depth_map, camera_matrix, transform_matrix, simple_mesh
        )
        
        metrics.deproject_depth_to_3d()
        
        assert metrics.valid_mask is not None
        assert metrics.valid_mask.shape == depth_map.shape
        assert np.sum(metrics.valid_mask) == len(metrics.point_cloud)
    
    def test_accuracy_metrics(self, depth_map, camera_matrix, transform_matrix, simple_mesh):
        """Test accuracy metrics computation"""
        metrics = DepthMapMetrics(
            depth_map, camera_matrix, transform_matrix, simple_mesh
        )
        
        accuracy = metrics.compute_accuracy_metrics()
        
        assert 'rmse' in accuracy
        assert 'mae' in accuracy
        assert 'mean_distance' in accuracy
        assert 'percentile_95' in accuracy
        assert 'num_valid_points' in accuracy
        
        assert accuracy['rmse'] >= 0
        assert accuracy['mae'] >= 0
        assert accuracy['num_valid_points'] > 0
    
    def test_completeness(self, depth_map, camera_matrix, transform_matrix, simple_mesh):
        """Test completeness computation"""
        metrics = DepthMapMetrics(
            depth_map, camera_matrix, transform_matrix, simple_mesh
        )
        
        completeness = metrics.compute_completeness(threshold=0.1)
        
        assert 'completeness_ratio' in completeness
        assert 'completeness_percentage' in completeness
        assert 0 <= completeness['completeness_ratio'] <= 1
        assert 0 <= completeness['completeness_percentage'] <= 100
    
    def test_error_heatmap(self, depth_map, camera_matrix, transform_matrix, simple_mesh):
        """Test error heatmap creation"""
        metrics = DepthMapMetrics(
            depth_map, camera_matrix, transform_matrix, simple_mesh
        )
        
        error_map = metrics.create_error_heatmap()
        
        assert error_map.shape == depth_map.shape
        # Should have some valid error values
        assert np.sum(~np.isnan(error_map)) > 0
    
    def test_regional_metrics(self, depth_map, camera_matrix, transform_matrix, simple_mesh):
        """Test regional metrics computation"""
        metrics = DepthMapMetrics(
            depth_map, camera_matrix, transform_matrix, simple_mesh
        )
        
        h, w = depth_map.shape
        regions = [
            (slice(0, h//2), slice(0, w//2)),
            (slice(h//2, h), slice(w//2, w))
        ]
        
        regional = metrics.compute_regional_metrics(regions)
        
        assert 'region_0' in regional
        assert 'region_1' in regional
        
        for region_metrics in regional.values():
            assert 'rmse' in region_metrics
            assert 'mae' in region_metrics
            assert 'num_points' in region_metrics
    
    def test_report_generation(self, depth_map, camera_matrix, transform_matrix, simple_mesh):
        """Test report generation"""
        metrics = DepthMapMetrics(
            depth_map, camera_matrix, transform_matrix, simple_mesh
        )
        
        report = metrics.generate_report()
        
        assert isinstance(report, str)
        assert len(report) > 100
        assert 'DEPTH MAP COMPARISON REPORT' in report
        assert 'RMSE' in report


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_point_cloud(self, tmp_path):
        """Test with empty point cloud"""
        mesh = trimesh.creation.box()
        stl_path = tmp_path / "test.stl"
        mesh.export(stl_path)
        
        empty_points = np.array([]).reshape(0, 3)
        metrics = PointCloudMetrics(empty_points, str(stl_path))
        
        clusters = metrics.cluster_points(eps=0.005)
        assert len(clusters) == 0
    
    def test_single_cluster(self, tmp_path):
        """Test with all points in one cluster"""
        mesh = trimesh.creation.box()
        stl_path = tmp_path / "test.stl"
        mesh.export(stl_path)
        
        # All points very close together
        points = np.random.randn(50, 3) * 0.001
        metrics = PointCloudMetrics(points, str(stl_path))
        
        clusters = metrics.cluster_points(eps=0.01, min_samples=3)
        assert len(clusters) >= 1
    
    def test_all_invalid_depth(self, tmp_path):
        """Test depth map with all invalid pixels"""
        mesh = trimesh.creation.box()
        stl_path = tmp_path / "test.stl"
        mesh.export(stl_path)
        
        depth_map = np.zeros((100, 100))  # All invalid
        K = np.eye(3)
        RT = np.eye(4)
        
        metrics = DepthMapMetrics(depth_map, K, RT, str(stl_path))
        point_cloud = metrics.deproject_depth_to_3d()
        
        assert len(point_cloud) == 0
    
    def test_invalid_rt_matrix(self, tmp_path):
        """Test with invalid RT matrix shape"""
        mesh = trimesh.creation.box()
        stl_path = tmp_path / "test.stl"
        mesh.export(stl_path)
        
        depth_map = np.ones((10, 10))
        K = np.eye(3)
        RT_invalid = np.eye(2)  # Wrong shape
        
        with pytest.raises(ValueError):
            DepthMapMetrics(depth_map, K, RT_invalid, str(stl_path))


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_tests()
