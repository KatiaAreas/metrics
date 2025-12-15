# Pyramid Tracking and Visualization System

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Main Components](#main-components)
4. [Coordinate Frames](#coordinate-frames)
5. [Transformation Pipeline](#transformation-pipeline)
6. [Usage Guide](#usage-guide)
7. [Verification and Testing](#verification-and-testing)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This system tracks and visualizes a 3D pyramid geometry in video footage using OptiTrack motion capture data. It performs real-time coordinate transformations from the pyramid's local frame through OptiTrack coordinates to camera image coordinates, with optional notch-based rotation correction.

### Key Features
- **Multi-frame coordinate transformation**: Pyramid → OptiTrack → World → Camera → Image
- **SVD-based alignment**: Automatic computation of transformation matrices using marker correspondences
- **Real-time visualization**: Video overlay with projected pyramid points
- **Notch detection**: Optional rotation correction using computer vision
- **Verification tools**: Built-in validation of transformations

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                              │
├─────────────────────────────────────────────────────────────────┤
│ • Video file (.mp4)                                             │
│ • OptiTrack tracking data (rigid bodies)                        │
│ • Camera calibration (intrinsics, extrinsics)                   │
│ • Pyramid geometry (JSON with 22 points in pyramid frame)       │
│ • Initial rotation angle (vectors.log, optional)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    COORDINATE TRANSFORMATIONS                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pyramid Frame → OptiTrack RB → World → Camera → Image         │
│       (JSON)         (SVD)      (Tracking)  (Calib)  (Project) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT VISUALIZATION                       │
├─────────────────────────────────────────────────────────────────┤
│ • Real-time video with projected pyramid points                 │
│ • Point labels and visibility indicators                         │
│ • Rotation angle display (if using notch detection)            │
│ • Verification plots and error metrics                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Main Components

### 1. `main.py` - Entry Point and Workflow Orchestration

**Purpose**: Configures experiment parameters, loads data, and initiates visualization.

**Key Steps**:
```python
# 1. Configure experiment
config = ExperimentConfig(
    base_data_dir=Path("..."),
    experiment_name="pyramid",
    display_type="pyramid"  # "calib", "pen", or "pyramid"
)

# 2. Set up file paths
paths = ExperimentPaths(config)
paths.validate_paths()

# 3. Load OptiTrack data
rb_data = read_data(
    header_path=paths.headers_path,
    data_path=paths.data_path,
    video_timestamps_path=paths.timestamp_path,
    unit_scale=1.0  # meters
)

# 4. Load camera calibration
calib_data = CalibData(
    intrinsics_path=paths.intrinsics_path,
    extrinsics_path=paths.extrinsics_path,
    camera_model_path=paths.camera_model_path,
    ots_ref_pose_path=paths.reference_ots_angle_pose
)

# 5. Display pyramid overlay
display_pyramid(
    video_path=paths.video_path,
    rb_data=rb_data,
    calib_data=calib_data,
    pyramid_json_path=pyramid_json_path,
    use_notch=True,  # Enable notch detection
    verify_transformation=False  # Set True for verification
)
```

**Configuration Options**:
- `display_type`: 
  - `"pyramid"` - Display pyramid points overlay
  - `"calib"` - Display calibration markers
  - `"pen"` - Display pen marker
- `angle_detector_type`: 
  - `"notch"` - Use notch detection for rotation
  - `None` - No rotation correction
- `unit_scale`: 
  - `1.0` for meters
  - `1000.0` for millimeters

---

### 2. `utils.py` - Display and Drawing Functions

**Purpose**: Handles video playback, coordinate projection, and overlay rendering.

#### 2.1 `display_pyramid()` - Main Display Loop

**Workflow**:
```
1. Load pyramid geometry (PyramidTransformer)
   ├─ Initialize transformer with JSON
   ├─ Extract marker positions from OptiTrack data
   ├─ Match markers to constellation points (18-21)
   └─ Compute transformation using SVD

2. Optional: Initialize notch detector
   ├─ Load deep learning models
   └─ Load initial theta from vectors.log

3. Video playback loop (for each frame):
   ├─ Read frame from video
   ├─ Detect notch (if enabled) → compute theta
   ├─ Compute rotation correction matrix R_cor
   ├─ Draw pyramid points on frame
   └─ Display frame with overlays
```

**Key Parameters**:
- `video_path`: Path to video file
- `rb_data`: OptiTrack rigid body tracking data
- `calib_data`: Camera calibration data
- `pyramid_json_path`: Path to pyramid geometry JSON
- `use_notch`: Enable notch-based rotation correction
- `workflow_type`: "visualization" or "ranking"
- `verify_transformation`: Enable verification plots

#### 2.2 `draw_pyramid_points()` - Point Projection

**Transformation Pipeline**:
```python
# Step 1: Get rigid body poses from OptiTrack
T_World_Lens = rb_data["Lens_RB"][frame_id].get_transform()
T_World_Pyramid = rb_data["Pyramid_RB"][frame_id].get_transform()

# Step 2: Compute camera transformation
RT = np.linalg.inv(T_World_Lens @ calib_data.RT)

# Step 3: Transform points from OptiTrack RB to world frame
points_hom = np.hstack([points_optitrack_m, np.ones((n_points, 1))])
points_world_hom = (T_World_Pyramid @ points_hom.T).T
obj_pts = points_world_hom[:, 0:3]

# Step 4: Project to image coordinates
proj_marker_2d = cv2.projectPoints(
    obj_pts,
    cv2.Rodrigues(RT[:3, :3])[0],
    RT[:3, 3],
    calib_data.K,
    calib_data.dist_coeffs
)[0]

# Step 5: Apply 2D rotation correction
homog_marker_2d = np.hstack([proj_marker_2d.reshape(-1, 2), 
                              np.ones((n_points, 1))]).T
homog_marker_2d_cor = R_cor @ homog_marker_2d

# Step 6: Draw on frame
for i in range(n_points):
    x, y = homog_marker_2d_cor[:2, i].flatten()
    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    cv2.putText(frame, str(i), (int(x)+8, int(y)-8), ...)
```

**Color Coding**:
- Green circles: Pyramid points
- Yellow text: Point indices
- Red text: Error messages

---

### 3. `pyramid_transformer.py` - Coordinate Transformation Engine

**Purpose**: Core transformation logic between pyramid frame and OptiTrack frame.

#### 3.1 PyramidTransformer Class

**Initialization**:
```python
transformer = PyramidTransformer(json_path)
# Automatically computes:
# - Pyramid frame definition (origin, axes)
# - Constellation frame (points 18-21)
# - Relative rotation R_pyramid_to_constellation
```

**Key Attributes**:
```python
# Input data (loaded from JSON)
transformer.points_m              # All 22 points in pyramid frame (Nx3)
transformer.referential_point_m   # Reference point (if in JSON)

# Pyramid frame definition
transformer.pyramid_origin_m      # Origin at point 0
transformer.R_pyramid             # Identity (JSON is in pyramid frame!)

# Constellation frame (points 18-21)
transformer.constellation_barycenter_m  # Center of 18-21
transformer.R_constellation             # Rotation matrix
transformer.constellation_indices       # [18, 19, 20, 21]

# Transformations (computed after marker matching)
transformer.R_pyramid_to_constellation  # 3x3 rotation
transformer.R_constellation_to_optitrack # 3x3 rotation (from SVD)
transformer.R_pyramid_to_optitrack      # 3x3 combined rotation
transformer.T_pyramid_to_optitrack      # 4x4 homogeneous transform
```

**Key Methods**:

##### 3.1.1 `compute_optitrack_rotation_from_markers()`
```python
R = transformer.compute_optitrack_rotation_from_markers(
    marker_positions_m,  # Dict: marker_name → position (meters)
    matching            # Dict: marker_name → point_index
)
```

**What it does**:
1. Extracts constellation points (18-21) in constellation frame
2. Matches with OptiTrack marker positions
3. Uses SVD (Kabsch algorithm) to compute optimal rotation
4. Computes full transformation chain:
   - `R_constellation_to_optitrack` (from SVD)
   - `R_pyramid_to_optitrack = R_constellation_to_optitrack @ R_pyramid_to_constellation`
   - `T_pyramid_to_optitrack` (4x4 with translation)

**SVD Algorithm** (Kabsch):
```python
# Center point clouds
centroid_source = mean(source_points)
centroid_target = mean(target_points)
centered_source = source_points - centroid_source
centered_target = target_points - centroid_target

# Compute covariance matrix
H = centered_source.T @ centered_target

# SVD decomposition
U, S, Vt = svd(H)

# Optimal rotation
R = Vt.T @ U.T

# Handle reflection (ensure det(R) = 1)
if det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T
```

##### 3.1.2 `transform_pyramid_to_optitrack()`
```python
points_optitrack = transformer.transform_pyramid_to_optitrack(points_pyramid)
```

**What it does**:
- Applies full 4x4 transformation: `p_optitrack = T @ [p_pyramid; 1]`
- Handles rotation and translation
- Returns points in OptiTrack rigid body frame

##### 3.1.3 Helper Functions

**`extract_marker_positions_from_rb_data()`**:
```python
marker_positions_m, rb_position, rb_quaternion = \
    extract_marker_positions_from_rb_data(rb_data, frame_id=0)
```
- Extracts OptiTrack marker positions from tracking data
- Transforms from world frame to rigid body local frame
- Returns positions in meters

**`quaternion_to_rotation_matrix()`**:
```python
R = quaternion_to_rotation_matrix([x, y, z, w])
```
- Converts quaternion to 3x3 rotation matrix
- Uses scipy if available, otherwise manual computation

---

### 4. Coordinate Frames Explained

#### 4.1 Pyramid Frame (JSON Coordinates)

**Definition**:
- **Origin**: Point 0
- **Z axis**: Point 0 → Point 1
- **X axis**: Perpendicular to Z, in plane formed by points 0, 1, 4
- **Y axis**: Z × X (right-handed)

**CRITICAL**: The JSON file contains points **already expressed in pyramid frame**:
- Point 0 is at `[0, 0, 0]`
- Point 1 is at `[0, 0, z]` (along Z axis)
- Points are measured in meters

**Usage**:
```python
# Points from JSON are ALREADY in pyramid frame
points_pyramid = transformer.points_m  # Shape: (22, 3)

# No conversion needed! These are pyramid coordinates.
```

#### 4.2 Constellation Frame (Points 18-21)

**Definition**:
- **Origin**: Barycenter of points 18, 19, 20, 21 (or referential point if available)
- **Y axis**: Direction toward point 20 (projected on plane)
- **Z axis**: Normal to plane of 4 points
- **X axis**: Y × Z (right-handed, on plane)

**Purpose**: Intermediate frame for marker matching with OptiTrack.

#### 4.3 OptiTrack Frame (Pyramid_RB)

**Definition**:
- **Origin**: OptiTrack rigid body origin
- **Axes**: Defined by OptiTrack marker placement
- Provided by OptiTrack tracking system

**Marker Correspondence**:
```python
matching = {
    'Marker 001': 21,  # OptiTrack marker → JSON point
    'Marker 002': 20,
    'Marker 003': 18,
    'Marker 004': 19
}
```

#### 4.4 World Frame (OptiTrack Global)

**Definition**:
- **Origin**: OptiTrack system origin
- **Axes**: OptiTrack global coordinate system
- All rigid bodies are tracked in this frame

#### 4.5 Camera Frame

**Definition**:
- **Origin**: Camera optical center
- **Z axis**: Optical axis (looking direction)
- **X, Y axes**: Image horizontal and vertical
- Defined by camera calibration

---

## Transformation Pipeline

### Complete Chain

```
Pyramid Frame (JSON)
    │
    │ transformer.points_m (already in pyramid frame)
    │
    ↓
[R_pyramid_to_optitrack, t]  ← Computed by SVD
    │
    ↓
OptiTrack RB Frame (Pyramid_RB local)
    │
    │ T_World_Pyramid (from tracking)
    │
    ↓
World Frame (OptiTrack global)
    │
    │ RT = inv(T_World_Lens @ calib_data.RT)
    │
    ↓
Camera Frame
    │
    │ cv2.projectPoints(K, dist_coeffs)
    │
    ↓
Image Coordinates (pixels)
    │
    │ R_cor (notch rotation correction)
    │
    ↓
Corrected Image Coordinates
```

### Mathematical Formulation

**Step 1: Pyramid → OptiTrack RB**
```
p_optitrack = R_pyramid_to_optitrack @ p_pyramid + t_pyramid_to_optitrack

Or in homogeneous coordinates:
[p_optitrack]   [R_pyramid_to_optitrack | t] [p_pyramid]
[    1      ] = [        0, 0, 0       | 1] [    1     ]
```

**Step 2: OptiTrack RB → World**
```
p_world = T_World_Pyramid @ [p_optitrack; 1]

Where T_World_Pyramid is from OptiTrack tracking at frame t
```

**Step 3: World → Camera**
```
p_camera = RT @ [p_world; 1]

Where RT = inv(T_World_Lens @ calib_data.RT)
```

**Step 4: Camera → Image**
```
[u]     K @ [R | t] @ p_world
[v] = ─────────────────────────
[1]            Z

Using cv2.projectPoints with distortion correction
```

**Step 5: 2D Rotation Correction** (if notch enabled)
```
[u']   [cos(θ)  -sin(θ)  tx] [u]
[v'] = [sin(θ)   cos(θ)  ty] [v]
[1 ]   [  0        0      1] [1]

Where θ = initial_theta - detected_theta
```

---

## Usage Guide

### Basic Usage

```python
from pathlib import Path
from main import main

# Run with default configuration
main()
```

### Custom Configuration

```python
from pathlib import Path
from utils import display_pyramid
from calib_data import CalibData
from areas_common.data_loading.rigid_body import read_data

# 1. Load data
rb_data = read_data(
    header_path="path/to/headers.csv",
    data_path="path/to/data.csv",
    video_timestamps_path="path/to/timestamps.csv",
    unit_scale=1.0  # meters
)

calib_data = CalibData(
    intrinsics_path="path/to/intrinsics.json",
    extrinsics_path="path/to/extrinsics.json",
    camera_model_path="path/to/camera_model.json",
    ots_ref_pose_path="path/to/reference_pose.json"
)

# 2. Display with pyramid overlay
display_pyramid(
    video_path=Path("video.mp4"),
    rb_data=rb_data,
    calib_data=calib_data,
    pyramid_json_path=Path("ModelMire3DSLAM3.json"),
    use_notch=True,
    workflow_type="visualization",
    vectors_log_path=Path("vectors.log"),  # Optional: initial rotation
    verify_transformation=True  # Enable verification plots
)
```

### Verification Mode

```python
# Enable comprehensive verification
display_pyramid(
    ...,
    verify_transformation=True
)

# This will show:
# 1. Constellation frame visualization
# 2. Distance ranking plots
# 3. SVD fit quality analysis
# 4. Interactive 3D visualization
# 5. Transformation accuracy metrics
```

---

## Verification and Testing

### Built-in Verification Tools

#### 1. Constellation Frame Visualization
```python
transformer.plot_constellation_frame()
```
**Shows**:
- Constellation points (18-21)
- Barycenter (origin)
- X, Y, Z axes
- Plane formed by 4 points

#### 2. Distance Ranking
```python
transformer.plot_distance_ranking_with_3d()
transformer.plot_distance_ranking()
```
**Shows**:
- Distance from each point to referential point
- 3D scatter plot with color-coded distances
- Bar chart of sorted distances

#### 3. SVD Fit Quality
```python
from pyramid_transformer import plot_svd_fit_quality

plot_svd_fit_quality(transformer, marker_positions_m, matching)
```
**Shows**:
- 3D view: transformed vs actual marker positions
- Point-wise error bars
- RMSE and error statistics

#### 4. Full Transformation Verification
```python
from verification_script import verify_pyramid_transformation

verify_pyramid_transformation(
    rb_data=rb_data,
    calib_data=calib_data,
    transformer=transformer,
    frame_id=0
)
```
**Checks**:
- Point visibility in camera frame
- Projection accuracy
- Coordinate frame consistency
- Transformation matrix properties

### Expected Verification Results

**Good SVD Fit**:
- RMSE < 0.01 m (10mm)
- Individual errors < 0.02 m (20mm)
- Uniform error distribution

**Good Projection**:
- Points within image bounds
- Consistent with video observations
- No systematic bias

**Good Transformation**:
- Determinant of R = 1.0 (orthonormal)
- R @ R.T = Identity
- No reflection (right-handed frame)

---

## Troubleshooting

### Common Issues

#### 1. "Cannot import name 'compute_optitrack_rotation_from_markers'"

**Cause**: Trying to import as standalone function instead of class method.

**Solution**:
```python
# WRONG
from pyramid_transformer import compute_optitrack_rotation_from_markers

# CORRECT
from pyramid_transformer import PyramidTransformer
transformer = PyramidTransformer(json_path)
R = transformer.compute_optitrack_rotation_from_markers(...)
```

#### 2. "Points not visible in frame"

**Cause**: Transformation chain is broken or rigid bodies not tracked.

**Check**:
```python
# Verify rigid body visibility
is_pyramid_visible = rb_data["Pyramid_RB"][frame_id].data.is_visible
is_lens_visible = rb_data["Lens_RB"][frame_id].data.is_visible
is_cam_visible = rb_data["Cam_RB"][frame_id].data.is_visible

print(f"Pyramid: {is_pyramid_visible}")
print(f"Lens: {is_lens_visible}")
print(f"Camera: {is_cam_visible}")
```

**Solution**:
- Ensure all rigid bodies are visible in OptiTrack
- Check camera calibration
- Verify frame synchronization

#### 3. "High SVD error (RMSE > 0.05m)"

**Cause**: Incorrect marker matching or poor OptiTrack data.

**Check**:
```python
# Verify marker correspondence
for marker_name, point_idx in matching.items():
    print(f"{marker_name} → Point {point_idx}")

# Check marker positions
for name, pos in marker_positions_m.items():
    print(f"{name}: {pos}")
```

**Solution**:
- Double-check marker-to-point matching
- Verify OptiTrack calibration
- Check for marker occlusion
- Try different frame_id for initialization

#### 4. "Rotation correction not working"

**Cause**: Notch not detected or vectors.log missing.

**Check**:
```python
# Verify vectors.log exists and is readable
vectors_log_path = Path("camera_models/vectors.log")
print(f"Exists: {vectors_log_path.exists()}")

# Check file contents
with open(vectors_log_path) as f:
    print(f.read())
```

**Solution**:
- Ensure vectors.log contains "Angle: XX.XX" line
- Wait for first notch detection if file missing
- Check notch model weights are loaded

#### 5. "Points at wrong location"

**Cause**: Unit scale mismatch (meters vs millimeters).

**Check**:
```python
# Check point magnitudes
print(f"Point 0: {transformer.points_m[0]}")
print(f"Point 1: {transformer.points_m[1]}")

# Expected: values in range [-1, 1] for meters
# If values are in range [-1000, 1000], data is in millimeters
```

**Solution**:
```python
# If data is in millimeters:
rb_data = read_data(..., unit_scale=1000.0)

# If data is in meters:
rb_data = read_data(..., unit_scale=1.0)
```

---

## File Structure

```
project/
├── main.py                           # Entry point
├── utils.py                          # Display and drawing functions
├── pyramid_transformer.py            # Core transformation logic
├── verification_script.py            # Verification tools
├── calib_data.py                     # Camera calibration
├── config.py                         # Configuration dataclass
├── paths.py                          # Path management
│
├── data/
│   ├── ModelMire3DSLAM3.json        # Pyramid geometry
│   ├── video.mp4                    # Video file
│   ├── optitrack/
│   │   ├── headers.csv              # OptiTrack headers
│   │   ├── data.csv                 # OptiTrack data
│   │   └── timestamps.csv           # Video timestamps
│   └── camera_models/
│       ├── intrinsics.json          # Camera intrinsics
│       ├── extrinsics.json          # Camera extrinsics
│       ├── camera_model.json        # Distortion model
│       ├── reference_pose.json      # Reference pose
│       └── vectors.log              # Initial rotation (optional)
│
└── output/
    ├── verification_plots/          # Verification outputs
    └── videos/                      # Processed videos
```

---

## API Reference

### PyramidTransformer

```python
class PyramidTransformer:
    def __init__(self, json_path: Path)
    
    def compute_optitrack_rotation_from_markers(
        self,
        marker_positions_m: Dict[str, np.ndarray],
        matching: Dict[str, int]
    ) -> np.ndarray
    
    def transform_pyramid_to_optitrack(
        self,
        points_pyramid_m: np.ndarray
    ) -> np.ndarray
    
    def get_constellation_points_in_optitrack_frame(
        self
    ) -> np.ndarray
    
    def match_constellation_markers(
        self,
        marker_positions_m: Dict[str, np.ndarray],
        initial_guess: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]
```

### Utils Functions

```python
def display_pyramid(
    video_path: Path,
    rb_data: dict,
    calib_data: CalibData,
    pyramid_json_path: Path,
    use_notch: bool = False,
    workflow_type: str = "visualization",
    R_const_to_opt: Optional[np.ndarray] = None,
    vectors_log_path: Optional[Path] = None,
    verify_transformation: bool = True
) -> None

def draw_pyramid_points(
    frame: np.ndarray,
    frame_id: int,
    rb_data: dict,
    calib_data: CalibData,
    points_optitrack_m: np.ndarray,
    R_cor: np.ndarray
) -> None
```

---





