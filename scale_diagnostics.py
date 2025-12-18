"""
Scale Diagnostic Tool for Pyramid Metrics

This script helps identify unit conversion and scale issues in the metrics calculation.
Expected errors should be in the range of 0.1-0.2 cm (1-2 mm) for well-calibrated systems.

Run this to diagnose if you're seeing errors like 100mm (10cm) which would be too large.
"""

import numpy as np
from pathlib import Path


def diagnose_scale_issues(
    calib_data,
    transformer,
    rb_data,
    keypoints_dict,
    frame_id=0
):
    """
    Diagnose potential scale and unit issues.
    
    Args:
        calib_data: Camera calibration data
        transformer: PyramidTransformer object
        rb_data: Rigid body tracking data
        keypoints_dict: Dictionary of keypoints by frame
        frame_id: Frame to analyze (default: 0)
    """
    
    print("=" * 80)
    print("SCALE DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # 1. Check pyramid geometry scale
    print("\n1. PYRAMID GEOMETRY SCALE CHECK")
    print("-" * 80)
    
    points_world = transformer.points_m
    
    # Get first 4 points (should form squares of ~8mm)
    if len(points_world) >= 4:
        p0, p1, p2, p3 = points_world[0], points_world[1], points_world[2], points_world[3]
        
        dist_01 = np.linalg.norm(p1 - p0)
        dist_12 = np.linalg.norm(p2 - p1)
        dist_23 = np.linalg.norm(p3 - p2)
        dist_30 = np.linalg.norm(p0 - p3)
        
        print(f"Point 0: {p0}")
        print(f"Point 1: {p1}")
        print(f"Point 2: {p2}")
        print(f"Point 3: {p3}")
        print(f"\nDistances between consecutive points:")
        print(f"  0→1: {dist_01:.6f} m = {dist_01*1000:.3f} mm")
        print(f"  1→2: {dist_12:.6f} m = {dist_12*1000:.3f} mm")
        print(f"  2→3: {dist_23:.6f} m = {dist_23*1000:.3f} mm")
        print(f"  3→0: {dist_30:.6f} m = {dist_30*1000:.3f} mm")
        
        avg_dist = np.mean([dist_01, dist_12, dist_23, dist_30])
        print(f"\nAverage distance: {avg_dist:.6f} m = {avg_dist*1000:.3f} mm")
        print(f"Expected: ~8 mm (0.008 m)")
        
        if abs(avg_dist * 1000 - 8) > 2:
            print("⚠️  WARNING: Pyramid geometry scale may be incorrect!")
            print(f"   Expected ~8mm, got {avg_dist*1000:.1f}mm")
            if avg_dist * 1000 > 100:
                print("   🔴 CRITICAL: Scale is in wrong units (cm instead of m?)")
        else:
            print("✓  Pyramid geometry scale looks correct")
    
    # 2. Check OptiTrack scale
    print("\n2. OPTITRACK COORDINATE SCALE CHECK")
    print("-" * 80)
    
    T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()
    T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
    T_World_Cam = rb_data["Cam_RB"][frame_id].get_transform()
    
    pyramid_pos = T_World_Pyramid[:3, 3]
    lens_pos = T_World_Lens[:3, 3]
    cam_pos = T_World_Cam[:3, 3]
    
    print(f"Pyramid position (world): {pyramid_pos} m")
    print(f"Lens position (world):    {lens_pos} m")
    print(f"Camera position (world):  {cam_pos} m")
    
    dist_pyramid_lens = np.linalg.norm(pyramid_pos - lens_pos)
    dist_pyramid_cam = np.linalg.norm(pyramid_pos - cam_pos)
    
    print(f"\nDistance pyramid→lens: {dist_pyramid_lens:.3f} m")
    print(f"Distance pyramid→cam:  {dist_pyramid_cam:.3f} m")
    
    if dist_pyramid_lens > 10 or dist_pyramid_cam > 10:
        print("⚠️  WARNING: OptiTrack distances are very large (>10m)")
        print("   This could indicate cm units being interpreted as meters")
    elif dist_pyramid_lens < 0.1:
        print("⚠️  WARNING: OptiTrack distances are very small (<10cm)")
        print("   Objects might be too close or scale issue")
    else:
        print("✓  OptiTrack scales look reasonable")
    
    # 3. Check camera calibration scale
    print("\n3. CAMERA CALIBRATION CHECK")
    print("-" * 80)
    
    fx = calib_data.K[0, 0]
    fy = calib_data.K[1, 1]
    cx = calib_data.K[0, 2]
    cy = calib_data.K[1, 2]
    
    print(f"Intrinsics K:")
    print(f"  fx = {fx:.2f} pixels")
    print(f"  fy = {fy:.2f} pixels")
    print(f"  cx = {cx:.2f} pixels")
    print(f"  cy = {cy:.2f} pixels")
    
    if fx < 100 or fy < 100:
        print("⚠️  WARNING: Focal lengths seem too small")
    elif fx > 10000 or fy > 10000:
        print("⚠️  WARNING: Focal lengths seem too large")
    else:
        print("✓  Focal lengths look reasonable")
    
    # 4. Check pixel-to-metric conversion
    print("\n4. PIXEL-TO-METRIC SCALE ESTIMATION")
    print("-" * 80)
    
    # Typical distance to pyramid
    typical_distance = dist_pyramid_lens
    
    # Pixel size at this distance (rough estimate)
    pixel_size_at_distance = typical_distance / fx  # meters per pixel
    
    print(f"At distance {typical_distance:.3f} m from camera:")
    print(f"  1 pixel ≈ {pixel_size_at_distance*1000:.3f} mm")
    print(f"  10 pixels ≈ {pixel_size_at_distance*10*1000:.3f} mm")
    
    expected_pixel_error = 1.5  # pixels (typical)
    expected_3d_error_mm = expected_pixel_error * pixel_size_at_distance * 1000
    
    print(f"\nExpected errors for {expected_pixel_error:.1f} pixel error:")
    print(f"  3D error ≈ {expected_3d_error_mm:.2f} mm")
    print(f"  (This should be in 0.1-2 mm range for good calibration)")
    
    # 5. Check transformation pipeline
    print("\n5. TRANSFORMATION PIPELINE CHECK")
    print("-" * 80)
    
    # Get points in different frames
    R_pyramid = transformer.R_pyramid
    pyramid_origin = transformer.pyramid_origin_m
    points_pyramid = (R_pyramid.T @ (points_world - pyramid_origin).T).T
    points_optitrack = transformer.transform_pyramid_to_optitrack(points_pyramid)
    
    print(f"Point 0 transformations:")
    print(f"  World frame:     {points_world[0]} m")
    print(f"  Pyramid frame:   {points_pyramid[0]} m")
    print(f"  OptiTrack frame: {points_optitrack[0]} m")
    
    # Check if transformation preserves scale
    dist_world = np.linalg.norm(points_world[1] - points_world[0])
    dist_pyramid = np.linalg.norm(points_pyramid[1] - points_pyramid[0])
    dist_optitrack = np.linalg.norm(points_optitrack[1] - points_optitrack[0])
    
    print(f"\nDistance between points 0 and 1:")
    print(f"  World frame:     {dist_world*1000:.3f} mm")
    print(f"  Pyramid frame:   {dist_pyramid*1000:.3f} mm")
    print(f"  OptiTrack frame: {dist_optitrack*1000:.3f} mm")
    
    if abs(dist_world - dist_pyramid) > 0.001 or abs(dist_world - dist_optitrack) > 0.001:
        print("⚠️  WARNING: Transformations are not preserving scale!")
    else:
        print("✓  Transformations preserve scale correctly")
    
    # 6. Check actual reprojection
    print("\n6. REPROJECTION TEST")
    print("-" * 80)
    
    if keypoints_dict and frame_id in keypoints_dict:
        keypoints = keypoints_dict[frame_id]
        
        # Get transformation for reprojection
        RT = np.linalg.inv(T_World_Lens @ calib_data.RT)
        
        # Transform first point to world frame
        points_hom = np.append(points_optitrack[0], 1.0)
        point_world_hom = T_World_Pyramid @ points_hom
        point_world = point_world_hom[:3]
        
        # Project to image
        import cv2
        proj_2d = cv2.projectPoints(
            point_world.reshape(1, 3),
            cv2.Rodrigues(RT[:3, :3])[0],
            RT[:3, 3],
            calib_data.K,
            calib_data.dist_coeffs
        )[0].reshape(2)
        
        # Get detected keypoint
        if len(keypoints) > 0:
            kp = keypoints[0]
            detected = np.array([kp['x'], kp['y']])
            
            pixel_error = np.linalg.norm(detected - proj_2d)
            
            print(f"Point 0:")
            print(f"  Detected:    ({detected[0]:.1f}, {detected[1]:.1f}) px")
            print(f"  Reprojected: ({proj_2d[0]:.1f}, {proj_2d[1]:.1f}) px")
            print(f"  Pixel error: {pixel_error:.2f} px")
            
            # Estimate 3D error from pixel error
            estimated_3d_error = pixel_error * pixel_size_at_distance * 1000
            print(f"  Estimated 3D error: {estimated_3d_error:.2f} mm")
            
            if estimated_3d_error > 10:
                print("⚠️  WARNING: Estimated 3D error is too large!")
                print("   Possible causes:")
                print("   - Camera calibration incorrect")
                print("   - Pyramid geometry wrong scale")
                print("   - Transformation errors")
            else:
                print("✓  Estimated 3D error is reasonable")
    
    # 7. Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    issues = []
    
    if len(points_world) >= 4:
        if abs(avg_dist * 1000 - 8) > 2:
            issues.append("Pyramid geometry scale incorrect")
    
    if dist_pyramid_lens > 10:
        issues.append("OptiTrack distances too large")
    elif dist_pyramid_lens < 0.1:
        issues.append("OptiTrack distances too small")
    
    if fx < 100 or fy < 100:
        issues.append("Focal lengths too small")
    elif fx > 10000 or fy > 10000:
        issues.append("Focal lengths too large")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nRECOMMENDATIONS:")
        print("  1. Check that pyramid JSON file uses METERS (not mm or cm)")
        print("  2. Verify OptiTrack is exporting in METERS")
        print("  3. Re-run camera calibration if focal lengths are wrong")
        print("  4. Check coordinate frame conventions")
    else:
        print("✓  No obvious scale issues detected")
        print("\nIf you're still seeing large errors (>10mm):")
        print("  1. Check detector accuracy (pixel errors should be <5px)")
        print("  2. Verify temporal synchronization (video vs tracking)")
        print("  3. Check for systematic biases in calibration")
    
    print("=" * 80)


def quick_scale_check(pyramid_json_path: Path):
    """
    Quick check of pyramid geometry file for scale issues.
    
    Args:
        pyramid_json_path: Path to pyramid geometry JSON
    """
    import json
    
    print("=" * 80)
    print("QUICK PYRAMID GEOMETRY SCALE CHECK")
    print("=" * 80)
    
    with open(pyramid_json_path, 'r') as f:
        data = json.load(f)
    
    # Get first few points
    points = []
    for i in range(min(4, len(data['points']))):
        p = data['points'][i]
        points.append([p['x'], p['y'], p['z']])
    
    points = np.array(points)
    
    print(f"\nFirst 4 points from JSON:")
    for i, p in enumerate(points):
        print(f"  Point {i}: x={p[0]:.6f}, y={p[1]:.6f}, z={p[2]:.6f}")
    
    # Check distances
    if len(points) >= 2:
        dist = np.linalg.norm(points[1] - points[0])
        print(f"\nDistance between points 0 and 1:")
        print(f"  Raw value: {dist:.6f}")
        print(f"  In mm: {dist*1000:.3f}")
        print(f"  In cm: {dist*100:.3f}")
        print(f"  In m: {dist:.6f}")
        
        print(f"\nExpected: ~8 mm (0.008 m) for pyramid squares")
        
        if dist < 0.001:
            print("⚠️  Values seem to be in METERS already (very small)")
        elif 0.007 < dist < 0.009:
            print("✓  Values appear to be in METERS (correct!)")
        elif 0.7 < dist < 0.9:
            print("🔴  Values appear to be in CENTIMETERS!")
            print("   → Need to divide by 100 or convert JSON to meters")
        elif 7 < dist < 9:
            print("🔴  Values appear to be in MILLIMETERS!")
            print("   → Need to divide by 1000 or convert JSON to meters")
        else:
            print("⚠️  Scale is unclear - manual inspection needed")
    
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    print("Run this diagnostic in your main script after loading data:")
    print()
    print("from scale_diagnostics import diagnose_scale_issues, quick_scale_check")
    print()
    print("# Quick check of JSON file")
    print("quick_scale_check(Path('ModelMire3DSLAM.json'))")
    print()
    print("# Full diagnostic")
    print("diagnose_scale_issues(")
    print("    calib_data=calib_data,")
    print("    transformer=transformer,")
    print("    rb_data=rb_data,")
    print("    keypoints_dict=keypoints_dict,")
    print("    frame_id=0")
    print(")")
