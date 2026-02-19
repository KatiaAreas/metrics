#!/usr/bin/env python3
"""
Pyramid Corner Detection and Tracking System

This application detects and tracks pyramid corners where different colored faces meet.
It uses SIFT for robust initial detection combined with color filtering, and Lucas-Kanade
optical flow for temporal tracking across frames.

Features:
- Automatic corner detection using color-based filtering and SIFT
- Lucas-Kanade tracking for temporal consistency
- Interactive keypoint editing (add/remove/modify)
- Space: pause/resume, Q: quit, Click: select/modify keypoints
- JSON export compatible with pyramid_points_coordinates.json format
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class KalmanTracker:
    """Simple Kalman filter for 2D point tracking with constant velocity model"""

    def __init__(self):
        """Initialize Kalman filter for tracking a 2D point"""
        # State: [x, y, vx, vy] - position and velocity
        self.kf = cv2.KalmanFilter(4, 2)

        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]  # vy = vy
        ], dtype=np.float32)

        # Measurement matrix (we only measure x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise (how much we trust the model)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise (how much we trust measurements)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        # Error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False

    def initialize(self, x, y):
        """Initialize filter with first measurement"""
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True

    def predict(self):
        """Predict next state"""
        prediction = self.kf.predict()
        return prediction[0, 0], prediction[1, 0]

    def update(self, x, y):
        """Update with new measurement"""
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(measurement)
        return self.kf.statePost[0, 0], self.kf.statePost[1, 0]


class EditMode(Enum):
    """Editing modes for interactive keypoint management"""
    NONE = 0
    SELECT = 1
    MOVE = 2
    DELETE = 3


@dataclass
class Keypoint:
    """Represents a tracked keypoint with visibility status"""
    id: int
    x: float
    y: float
    visibility: int  # 0: not visible, 1: visible
    template: np.ndarray = None  # Template patch for matching
    kalman: KalmanTracker = None  # Kalman filter for prediction

    def __post_init__(self):
        """Initialize Kalman filter"""
        if self.kalman is None:
            self.kalman = KalmanTracker()

    def to_dict(self):
        return {
            'id': self.id,
            'x': float(self.x),
            'y': float(self.y),
            'visibility': self.visibility
        }


class PyramidCornerDetector:
    """Detects pyramid corners at color intersections"""

    def __init__(self, min_contrast=30, color_threshold=50):
        """
        Initialize the detector

        Args:
            min_contrast: Minimum contrast for corner detection
            color_threshold: Threshold for color segmentation
        """
        self.min_contrast = min_contrast
        self.color_threshold = color_threshold

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=500,
            contrastThreshold=0.03,
            edgeThreshold=10
        )

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def detect_color_regions(self, frame):
        """
        Segment the frame into distinct color regions

        Args:
            frame: Input BGR frame

        Returns:
            List of color masks
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Apply bilateral filter to reduce noise while preserving edges
        hsv_filtered = cv2.bilateralFilter(hsv, 9, 75, 75)

        # Use k-means to find dominant colors
        pixels = frame.reshape(-1, 3).astype(np.float32)
        k = 6  # Assume up to 6 distinct colors on pyramid

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                        cv2.KMEANS_PP_CENTERS)

        # Create masks for each color cluster
        labels = labels.reshape(frame.shape[:2])
        masks = []

        for i in range(k):
            mask = (labels == i).astype(np.uint8) * 255
            # Remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            if cv2.countNonZero(mask) > 100:  # Only keep significant regions
                masks.append(mask)

        return masks

    def find_color_intersections(self, masks):
        """
        Find points where three or more color regions intersect

        Args:
            masks: List of binary masks for each color region

        Returns:
            List of intersection points (x, y)
        """
        intersections = []
        h, w = masks[0].shape

        # Create intersection map
        intersection_count = np.zeros((h, w), dtype=np.uint8)

        # Dilate masks slightly to ensure overlap at boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_masks = [cv2.dilate(mask, kernel, iterations=1) for mask in masks]

        # Count how many regions each pixel belongs to
        for mask in dilated_masks:
            intersection_count += (mask > 0).astype(np.uint8)

        # Find pixels where 3+ regions meet (pyramid corners)
        corner_mask = (intersection_count >= 3).astype(np.uint8) * 255

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        corner_mask = cv2.morphologyEx(corner_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of intersection regions
        contours, _ = cv2.findContours(corner_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Get centroid of each intersection region
        for contour in contours:
            if cv2.contourArea(contour) > 5:  # Minimum area threshold
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    intersections.append((cx, cy))

        return intersections

    def detect_sift_corners(self, frame, mask=None):
        """
        Detect corners using SIFT

        Args:
            frame: Input grayscale or BGR frame
            mask: Optional mask to restrict detection

        Returns:
            List of keypoint coordinates
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect keypoints
        keypoints = self.sift.detect(gray, mask)

        # Convert to coordinates
        corners = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

        return corners

    def detect_corners(self, frame):
        """
        Main corner detection combining color analysis and SIFT

        Args:
            frame: Input BGR frame

        Returns:
            List of detected corner points (x, y)
        """
        # Method 1: Color-based intersection detection
        color_masks = self.detect_color_regions(frame)
        color_corners = self.find_color_intersections(color_masks)

        # Method 2: SIFT-based detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift_corners = self.detect_sift_corners(frame)

        # Combine results: prioritize color intersections, add nearby SIFT points
        all_corners = []

        # Add all color-based corners
        all_corners.extend(color_corners)

        # Add SIFT corners that are close to color intersections
        for sift_pt in sift_corners:
            for color_pt in color_corners:
                dist = np.sqrt((sift_pt[0] - color_pt[0]) ** 2 +
                               (sift_pt[1] - color_pt[1]) ** 2)
                if dist < 20:  # Within 20 pixels
                    break
            else:
                # This SIFT point is not near any color intersection
                # Add it if it's a strong corner
                all_corners.append(sift_pt)

        # Remove duplicates
        unique_corners = []
        for corner in all_corners:
            is_duplicate = False
            for unique in unique_corners:
                dist = np.sqrt((corner[0] - unique[0]) ** 2 +
                               (corner[1] - unique[1]) ** 2)
                if dist < 10:  # Merge nearby points
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_corners.append(corner)

        return unique_corners


class PyramidTracker:
    """Tracks pyramid corners across video frames using hybrid approach"""

    def __init__(self, detector: PyramidCornerDetector, num_keypoints=18):
        """
        Initialize the tracker

        Args:
            detector: PyramidCornerDetector instance
            num_keypoints: Fixed number of keypoints to track
        """
        self.detector = detector
        self.num_keypoints = num_keypoints
        self.keypoints = []
        self.prev_gray = None
        self.tracking_points = None
        self.template_size = 21  # Size of template patch (odd number)
        self.template_update_interval = 30  # Update template every N frames
        self.frame_count = 0

    def extract_template(self, frame, x, y):
        """
        Extract template patch around a keypoint

        Args:
            frame: Grayscale frame
            x, y: Keypoint coordinates

        Returns:
            Template patch or None if near edge
        """
        half_size = self.template_size // 2
        h, w = frame.shape

        x_int, y_int = int(x), int(y)

        # Check boundaries
        if (x_int - half_size < 0 or x_int + half_size >= w or
                y_int - half_size < 0 or y_int + half_size >= h):
            return None

        # Extract patch
        template = frame[y_int - half_size:y_int + half_size + 1,
                   x_int - half_size:x_int + half_size + 1].copy()

        return template

    def match_template(self, frame, template, search_x, search_y, search_size=25):
        """
        Match template in a search region

        Args:
            frame: Current grayscale frame
            template: Template to match
            search_x, search_y: Center of search region
            search_size: Half-size of search region

        Returns:
            Best match (x, y) and confidence score
        """
        h, w = frame.shape

        # Define search region
        x1 = max(0, int(search_x) - search_size)
        y1 = max(0, int(search_y) - search_size)
        x2 = min(w, int(search_x) + search_size)
        y2 = min(h, int(search_y) + search_size)

        search_region = frame[y1:y2, x1:x2]

        if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
            return None, 0.0

        # Template matching
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)

        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Convert to frame coordinates
        best_x = x1 + max_loc[0] + template.shape[1] // 2
        best_y = y1 + max_loc[1] + template.shape[0] // 2

        return (best_x, best_y), max_val

    def initialize_keypoints(self, frame):
        """
        Initialize keypoints on the first frame with templates and Kalman filters

        Args:
            frame: First frame of the video
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract templates for each keypoint
        for kp in self.keypoints:
            if kp.visibility == 1:
                # Extract template
                kp.template = self.extract_template(gray, kp.x, kp.y)

                # Initialize Kalman filter
                kp.kalman.initialize(kp.x, kp.y)

        # Store for tracking
        self.prev_gray = gray
        self.tracking_points = np.array(
            [[kp.x, kp.y] for kp in self.keypoints if kp.visibility == 1],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        self.frame_count = 0

    def track_frame(self, frame):
        """
        Track keypoints to the next frame using hybrid approach:
        1. Kalman prediction
        2. Lucas-Kanade optical flow
        3. Template matching (periodic)
        4. Fusion of all methods

        Args:
            frame: Current frame

        Returns:
            Updated keypoints
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        if self.tracking_points is None or len(self.tracking_points) == 0:
            self.prev_gray = gray
            return self.keypoints

        # Track using Lucas-Kanade optical flow
        lk_points, lk_status, lk_error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.tracking_points, None,
            **self.detector.lk_params
        )

        # Update tracked keypoints with hybrid approach
        visible_idx = [i for i, kp in enumerate(self.keypoints) if kp.visibility == 1]

        # Whether to use template matching this frame
        use_templates = (self.frame_count % 5 == 0)  # Every 5 frames

        for i, idx in enumerate(visible_idx):
            kp = self.keypoints[idx]

            # 1. Get Kalman prediction
            pred_x, pred_y = kp.kalman.predict()

            # 2. Get LK tracking result
            lk_valid = lk_status[i] == 1
            if lk_valid:
                lk_x, lk_y = lk_points[i].ravel()
            else:
                lk_x, lk_y = pred_x, pred_y  # Fall back to prediction

            # 3. Template matching (periodic for drift correction)
            template_x, template_y = None, None
            template_conf = 0.0

            if use_templates and kp.template is not None:
                # Search around LK result (or prediction if LK failed)
                search_center_x = lk_x if lk_valid else pred_x
                search_center_y = lk_y if lk_valid else pred_y

                match_result, template_conf = self.match_template(
                    gray, kp.template, search_center_x, search_center_y, search_size=25
                )

                if match_result is not None and template_conf > 0.5:
                    template_x, template_y = match_result

            # 4. Fuse measurements
            h, w = frame.shape[:2]

            # Decide which measurement to use based on confidence
            if template_x is not None and template_conf > 0.7:
                # High confidence template match - use it
                final_x, final_y = template_x, template_y
            elif lk_valid:
                # LK tracking succeeded
                # Blend with Kalman prediction for smoothness
                alpha = 0.7  # Weight for LK
                final_x = alpha * lk_x + (1 - alpha) * pred_x
                final_y = alpha * lk_y + (1 - alpha) * pred_y
            else:
                # LK failed - use Kalman prediction
                final_x, final_y = pred_x, pred_y

            # Check if point is within frame bounds
            if 0 <= final_x < w and 0 <= final_y < h:
                # Update Kalman filter with measurement
                kp.x, kp.y = kp.kalman.update(final_x, final_y)

                # Update template occasionally to adapt to appearance changes
                if self.frame_count % self.template_update_interval == 0:
                    new_template = self.extract_template(gray, kp.x, kp.y)
                    if new_template is not None:
                        # Blend old and new template for gradual adaptation
                        alpha = 0.7
                        kp.template = cv2.addWeighted(kp.template, alpha,
                                                      new_template, 1 - alpha, 0)
            else:
                # Point went out of bounds
                kp.visibility = 0

        # Update tracking points for next iteration
        self.tracking_points = np.array(
            [[kp.x, kp.y] for kp in self.keypoints if kp.visibility == 1],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        self.prev_gray = gray

        return self.keypoints

    def redetect_and_update(self, frame, confidence_threshold=0.7):
        """
        Re-detect corners and update/add/remove keypoints

        Args:
            frame: Current frame
            confidence_threshold: Minimum confidence for detection
        """
        # Detect new corners
        detected_corners = self.detector.detect_corners(frame)

        # Match detected corners with existing keypoints
        for corner in detected_corners:
            cx, cy = corner

            # Find closest invisible keypoint
            min_dist = float('inf')
            best_kp_idx = None

            for i, kp in enumerate(self.keypoints):
                if kp.visibility == 0:
                    dist = np.sqrt((kp.x - cx) ** 2 + (kp.y - cy) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_kp_idx = i

            # Check if any visible keypoint is very close (might be duplicate)
            is_near_existing = False
            for kp in self.keypoints:
                if kp.visibility == 1:
                    dist = np.sqrt((kp.x - cx) ** 2 + (kp.y - cy) ** 2)
                    if dist < 15:  # Very close to existing point
                        is_near_existing = True
                        break

            # Add as new keypoint if not near existing and we have space
            if not is_near_existing and best_kp_idx is not None:
                self.keypoints[best_kp_idx].x = float(cx)
                self.keypoints[best_kp_idx].y = float(cy)
                self.keypoints[best_kp_idx].visibility = 1

        # Update tracking points
        self.tracking_points = np.array(
            [[kp.x, kp.y] for kp in self.keypoints if kp.visibility == 1],
            dtype=np.float32
        ).reshape(-1, 1, 2)


class InteractiveVideoPlayer:
    """Interactive video player with user-driven keypoint selection and tracking"""

    def __init__(self, video_path: str, output_json: str, num_keypoints=18):
        """
        Initialize the video player

        Args:
            video_path: Path to input video
            output_json: Path to output JSON file
            num_keypoints: Number of keypoints to track
        """
        self.video_path = video_path
        self.output_json = output_json
        self.num_keypoints = num_keypoints

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize detector and tracker
        self.detector = PyramidCornerDetector()
        self.tracker = PyramidTracker(self.detector, num_keypoints)

        # State variables
        self.initialization_mode = True  # Start in initialization mode
        self.paused = True  # Start paused for initialization
        self.current_frame_idx = 0
        self.current_frame = None
        self.first_frame = None
        self.frames_data = []

        # Track when changes are made (to reprocess from that point)
        self.changes_made_at_frame = None

        # Editing state
        self.selected_kp_idx = None
        self.edit_mode = EditMode.NONE
        self.mouse_x = 0
        self.mouse_y = 0

        # Setup OpenCV window and mouse callback
        self.window_name = "Pyramid Corner Tracker"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def refine_keypoint_location(self, frame, x, y, window_size=50):
        """
        Robustly refine keypoint location using multiple detection methods

        Combines:
        1. Edge detection (Canny) + line intersection
        2. Harris corner detection
        3. Shi-Tomasi corner detection

        Args:
            frame: Input frame
            x, y: User-selected coordinates
            window_size: Half-size of search window in pixels (default: 50)

        Returns:
            Refined (x, y) coordinates with confidence score
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Define search window (larger for robustness)
            x1 = max(0, int(x) - window_size)
            y1 = max(0, int(y) - window_size)
            x2 = min(w, int(x) + window_size)
            y2 = min(h, int(y) + window_size)

            # Check if window is too small (near edge)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                print(f"  Click too close to edge - using original position ({x:.1f}, {y:.1f})")
                return x, y

            # Extract window
            window = gray[y1:y2, x1:x2]
            window_color = frame[y1:y2, x1:x2]

            if window.size == 0:
                return x, y

            # Store all candidates with scores
            candidates = []

            # METHOD 1: Edge-based corner detection (best for pyramid edges)
            # Apply Canny edge detection
            edges = cv2.Canny(window, 50, 150, apertureSize=3)

            # Dilate edges slightly to ensure connectivity
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Find lines using Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                    minLineLength=20, maxLineGap=10)

            if lines is not None and len(lines) >= 2:
                # Find intersections of lines
                intersections = []
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        x1_l1, y1_l1, x2_l1, y2_l1 = lines[i][0]
                        x1_l2, y1_l2, x2_l2, y2_l2 = lines[j][0]

                        # Calculate intersection
                        intersection = self._line_intersection(
                            (x1_l1, y1_l1), (x2_l1, y2_l1),
                            (x1_l2, y1_l2), (x2_l2, y2_l2)
                        )

                        if intersection is not None:
                            ix, iy = intersection
                            # Check if intersection is within window
                            if 0 <= ix < window.shape[1] and 0 <= iy < window.shape[0]:
                                # Calculate angle between lines (corners have ~90 degree angles)
                                angle = self._angle_between_lines(
                                    (x1_l1, y1_l1), (x2_l1, y2_l1),
                                    (x1_l2, y1_l2), (x2_l2, y2_l2)
                                )
                                # Prefer angles close to 90 degrees (pyramid corners)
                                angle_score = 1.0 - abs(90 - angle) / 90.0

                                # Distance from user click (prefer closer points)
                                dist_from_click = np.sqrt((ix - window_size) ** 2 + (iy - window_size) ** 2)
                                dist_score = max(0, 1.0 - dist_from_click / window_size)

                                # Combined score
                                score = angle_score * 0.6 + dist_score * 0.4

                                candidates.append({
                                    'x': x1 + ix,
                                    'y': y1 + iy,
                                    'score': score,
                                    'method': 'edge_intersection'
                                })

            # METHOD 2: Harris corner detection
            harris_corners = cv2.cornerHarris(window, blockSize=3, ksize=3, k=0.04)

            # Find local maxima in Harris response
            harris_threshold = 0.01 * harris_corners.max()
            corner_coords = np.where(harris_corners > harris_threshold)

            for hy, hx in zip(corner_coords[0], corner_coords[1]):
                dist_from_click = np.sqrt((hx - window_size) ** 2 + (hy - window_size) ** 2)
                if dist_from_click < window_size:
                    # Score based on Harris response and distance
                    harris_strength = harris_corners[hy, hx] / harris_corners.max()
                    dist_score = max(0, 1.0 - dist_from_click / window_size)
                    score = harris_strength * 0.5 + dist_score * 0.5

                    candidates.append({
                        'x': x1 + hx,
                        'y': y1 + hy,
                        'score': score * 0.8,  # Weight Harris slightly lower than edges
                        'method': 'harris'
                    })

            # METHOD 3: Shi-Tomasi (Good Features to Track)
            shi_tomasi_corners = cv2.goodFeaturesToTrack(
                window,
                maxCorners=10,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=3
            )

            if shi_tomasi_corners is not None:
                for corner in shi_tomasi_corners:
                    cx, cy = corner[0]
                    dist_from_click = np.sqrt((cx - window_size) ** 2 + (cy - window_size) ** 2)
                    if dist_from_click < window_size:
                        dist_score = max(0, 1.0 - dist_from_click / window_size)

                        candidates.append({
                            'x': x1 + cx,
                            'y': y1 + cy,
                            'score': dist_score * 0.7,  # Weight Shi-Tomasi moderately
                            'method': 'shi_tomasi'
                        })

            # If no candidates found, return original position
            if not candidates:
                print(f"  No refinement found - using original position ({x:.1f}, {y:.1f})")
                return x, y

            # Sort candidates by score
            candidates.sort(key=lambda c: c['score'], reverse=True)

            # Take best candidate
            best = candidates[0]

            # Apply sub-pixel refinement to best candidate
            best_x, best_y = best['x'], best['y']

            # Extract small region around best candidate for sub-pixel refinement
            refine_size = 5
            rx1 = max(0, int(best_y) - refine_size)
            ry1 = max(0, int(best_x) - refine_size)
            rx2 = min(h, int(best_y) + refine_size)
            ry2 = min(w, int(best_x) + refine_size)

            refine_window = gray[rx1:rx2, ry1:ry2]

            # Check if window is large enough for cornerSubPix (needs at least 11x11)
            min_window_size = 11
            if refine_window.shape[0] >= min_window_size and refine_window.shape[1] >= min_window_size:
                # Find corner in refined window
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.goodFeaturesToTrack(refine_window, maxCorners=1,
                                                  qualityLevel=0.01, minDistance=1)

                if corners is not None:
                    try:
                        corners = cv2.cornerSubPix(refine_window, corners, (3, 3), (-1, -1), criteria)
                        refined_x = ry1 + corners[0][0][0]
                        refined_y = rx1 + corners[0][0][1]
                        best_x, best_y = refined_x, refined_y
                    except cv2.error:
                        # If cornerSubPix fails, use the best candidate as-is
                        pass

            print(f"  Refined: ({x:.1f}, {y:.1f}) -> ({best_x:.1f}, {best_y:.1f}) "
                  f"[method: {best['method']}, score: {best['score']:.2f}]")

            return best_x, best_y

        except Exception as e:
            print(f"  Refinement failed: {e}")
            print(f"  Using original position ({x:.1f}, {y:.1f})")
            return x, y

    def _line_intersection(self, p1, p2, p3, p4):
        """
        Calculate intersection point of two lines

        Args:
            p1, p2: Points defining first line
            p3, p4: Points defining second line

        Returns:
            (x, y) intersection point or None if parallel
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-10:  # Lines are parallel
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def _angle_between_lines(self, p1, p2, p3, p4):
        """
        Calculate angle between two lines in degrees

        Args:
            p1, p2: Points defining first line
            p3, p4: Points defining second line

        Returns:
            Angle in degrees (0-180)
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        # Direction vectors
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x4 - x3, y4 - y3])

        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

        # Calculate angle
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product) * 180 / np.pi

        # Return acute angle
        if angle > 90:
            angle = 180 - angle

        return angle

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for keypoint editing"""
        self.mouse_x = x
        self.mouse_y = y

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.initialization_mode:
                # Add new keypoint during initialization
                num_visible = sum(1 for kp in self.tracker.keypoints if kp.visibility == 1)
                if num_visible < self.num_keypoints:
                    # Find first invisible keypoint slot
                    for i, kp in enumerate(self.tracker.keypoints):
                        if kp.visibility == 0:
                            print(f"Adding keypoint {i} at ({x}, {y})")
                            # Refine the location with 50px window
                            refined_x, refined_y = self.refine_keypoint_location(
                                self.first_frame, x, y, window_size=50
                            )
                            kp.x = float(refined_x)
                            kp.y = float(refined_y)
                            kp.visibility = 1
                            self.selected_kp_idx = i

                            # Extract template for this keypoint
                            gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
                            kp.template = self.tracker.extract_template(gray, kp.x, kp.y)
                            kp.kalman.initialize(kp.x, kp.y)
                            break
                else:
                    print(f"Already have {self.num_keypoints} keypoints. Delete one first.")
            else:
                # Editing mode during playback
                # Check if clicking near existing keypoint
                clicked_kp = None
                min_dist = 15  # Pixels

                for i, kp in enumerate(self.tracker.keypoints):
                    if kp.visibility == 1:
                        dist = np.sqrt((kp.x - x) ** 2 + (kp.y - y) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            clicked_kp = i

                if clicked_kp is not None:
                    # Select existing keypoint for moving
                    self.selected_kp_idx = clicked_kp
                    self.edit_mode = EditMode.MOVE
                else:
                    # Add new keypoint
                    for i, kp in enumerate(self.tracker.keypoints):
                        if kp.visibility == 0:
                            print(f"Adding keypoint {i} at frame {self.current_frame_idx}")
                            refined_x, refined_y = self.refine_keypoint_location(
                                self.current_frame, x, y, window_size=50
                            )
                            kp.x = float(refined_x)
                            kp.y = float(refined_y)
                            kp.visibility = 1
                            self.selected_kp_idx = i
                            self.changes_made_at_frame = self.current_frame_idx

                            # Extract template and initialize Kalman
                            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                            kp.template = self.tracker.extract_template(gray, kp.x, kp.y)
                            kp.kalman.initialize(kp.x, kp.y)
                            break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.edit_mode == EditMode.MOVE and self.selected_kp_idx is not None:
                # Move selected keypoint
                kp = self.tracker.keypoints[self.selected_kp_idx]
                kp.x = float(x)
                kp.y = float(y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.edit_mode == EditMode.MOVE and self.selected_kp_idx is not None:
                # Refine the final position
                kp = self.tracker.keypoints[self.selected_kp_idx]
                if not self.initialization_mode:
                    refined_x, refined_y = self.refine_keypoint_location(
                        self.current_frame, kp.x, kp.y, window_size=50
                    )
                else:
                    refined_x, refined_y = self.refine_keypoint_location(
                        self.first_frame, kp.x, kp.y, window_size=50
                    )
                kp.x = float(refined_x)
                kp.y = float(refined_y)
                print(f"Moved keypoint {self.selected_kp_idx} to ({refined_x:.1f}, {refined_y:.1f})")

                if not self.initialization_mode:
                    self.changes_made_at_frame = self.current_frame_idx
                    # Update template and Kalman filter
                    gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                    kp.template = self.tracker.extract_template(gray, kp.x, kp.y)
                    kp.kalman.initialize(kp.x, kp.y)
                else:
                    # Update template for initialization
                    gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
                    kp.template = self.tracker.extract_template(gray, kp.x, kp.y)
                    kp.kalman.initialize(kp.x, kp.y)

                self.edit_mode = EditMode.NONE

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete keypoint
            for i, kp in enumerate(self.tracker.keypoints):
                if kp.visibility == 1:
                    dist = np.sqrt((kp.x - x) ** 2 + (kp.y - y) ** 2)
                    if dist < 15:
                        kp.visibility = 0
                        print(f"Deleted keypoint {i}")
                        if not self.initialization_mode:
                            self.changes_made_at_frame = self.current_frame_idx
                        break

    def draw_keypoints(self, frame):
        """
        Draw keypoints on the frame

        Args:
            frame: Frame to draw on

        Returns:
            Frame with keypoints drawn
        """
        overlay = frame.copy()

        for kp in self.tracker.keypoints:
            if kp.visibility == 1:
                x, y = int(kp.x), int(kp.y)

                # Determine color based on selection
                if self.selected_kp_idx == kp.id:
                    color = (0, 255, 255)  # Yellow for selected
                    thickness = 3
                else:
                    color = (0, 255, 0)  # Green for visible
                    thickness = 2

                # Draw circle
                cv2.circle(overlay, (x, y), 8, color, thickness)

                # Draw crosshair for precision
                cv2.line(overlay, (x - 5, y), (x + 5, y), color, 1)
                cv2.line(overlay, (x, y - 5), (x, y + 5), color, 1)

                # Draw ID
                cv2.putText(overlay, str(kp.id), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Blend overlay
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    def draw_info(self, frame):
        """Draw information overlay on the frame"""
        if self.initialization_mode:
            num_visible = sum(1 for kp in self.tracker.keypoints if kp.visibility == 1)
            info_text = [
                "=== INITIALIZATION MODE ===",
                f"Select {self.num_keypoints} keypoints on the pyramid corners",
                f"Selected: {num_visible}/{self.num_keypoints}",
                "Left click: Add keypoint | Right click: Delete",
                "Press SPACE when done to start tracking"
            ]
        else:
            info_text = [
                f"Frame: {self.current_frame_idx}/{self.total_frames}",
                f"Visible KPs: {sum(1 for kp in self.tracker.keypoints if kp.visibility == 1)}/{self.num_keypoints}",
                "SPACE: pause/play | Q: quit",
            ]

            if self.paused:
                info_text.append("PAUSED - Left click: add/move | Right click: delete")

        y_offset = 30
        for i, text in enumerate(info_text):
            # Draw background for better visibility
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (5, y_offset + i * 25 - 20),
                          (15 + text_width, y_offset + i * 25 + 5),
                          (0, 0, 0), -1)

            # Draw text with outline
            cv2.putText(frame, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.initialization_mode else (255, 255, 255), 1)

        return frame

    def process_video(self):
        """Main processing loop"""
        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Resolution: {self.width}x{self.height}")
        print("\n" + "=" * 60)
        print("STEP 1: MANUAL KEYPOINT SELECTION")
        print("=" * 60)
        print("Click on pyramid corners to select keypoints")
        print("The system will automatically refine each click to the nearest corner")
        print(f"Select {self.num_keypoints} keypoints, then press SPACE to start tracking")
        print("=" * 60 + "\n")

        # Read first frame for initialization
        ret, self.first_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")

        self.current_frame = self.first_frame.copy()
        self.current_frame_idx = 0

        # Initialize keypoint slots (all invisible)
        self.tracker.keypoints = [
            Keypoint(id=i, x=0.0, y=0.0, visibility=0)
            for i in range(self.num_keypoints)
        ]

        # Initialization loop - wait for user to select points
        while self.initialization_mode:
            display_frame = self.draw_keypoints(self.first_frame.copy())
            display_frame = self.draw_info(display_frame)
            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord(' '):  # Space to finish initialization
                num_visible = sum(1 for kp in self.tracker.keypoints if kp.visibility == 1)
                if num_visible > 0:
                    print(f"\n{'=' * 60}")
                    print(f"Initialization complete with {num_visible} keypoints")
                    print("=" * 60)
                    print("\nSTEP 2: TRACKING")
                    print("=" * 60)
                    print("Controls:")
                    print("  SPACE: Pause/Resume")
                    print("  Q: Quit and save")
                    print("  Left Click (paused): Add or move keypoint")
                    print("  Right Click (paused): Delete keypoint")
                    print("=" * 60 + "\n")

                    self.initialization_mode = False
                    self.paused = False

                    # Initialize tracker with user-selected points
                    self.tracker.initialize_keypoints(self.first_frame)

                    # Store first frame data
                    frame_data = {
                        'frame_id': 0,
                        'keypoints': [kp.to_dict() for kp in self.tracker.keypoints]
                    }
                    self.frames_data.append(frame_data)
                else:
                    print("Please select at least one keypoint before starting")

            elif key == ord('q') or key == 27:  # Q or ESC
                print("\nCancelled by user")
                self.cap.release()
                cv2.destroyAllWindows()
                return

        # Main tracking loop
        frame_idx = 1  # Start from frame 1 (frame 0 already stored)

        while True:
            if not self.paused:
                ret, frame = self.cap.read()

                if not ret:
                    print("\nEnd of video reached.")
                    break

                self.current_frame = frame.copy()
                self.current_frame_idx = frame_idx

                # Track keypoints
                self.tracker.track_frame(frame)

                # Store frame data
                frame_data = {
                    'frame_id': frame_idx,
                    'keypoints': [kp.to_dict() for kp in self.tracker.keypoints]
                }
                self.frames_data.append(frame_data)

                frame_idx += 1

                # Display frame
                display_frame = self.draw_keypoints(frame.copy())
                display_frame = self.draw_info(display_frame)
            else:
                # Paused - show current frame
                display_frame = self.draw_keypoints(self.current_frame.copy())
                display_frame = self.draw_info(display_frame)

            cv2.imshow(self.window_name, display_frame)

            # Handle key presses
            key = cv2.waitKey(1 if not self.paused else 30) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                print("\nQuitting...")
                break
            elif key == ord(' '):  # Space
                was_paused = self.paused
                self.paused = not self.paused

                if self.paused:
                    print(f"\nPaused at frame {self.current_frame_idx}")
                else:
                    print("Resuming...")

                    # If changes were made, update tracking from this point
                    if self.changes_made_at_frame == self.current_frame_idx:
                        print(f"Applying changes made at frame {self.current_frame_idx}")

                        # Update the stored frame data with edited keypoints
                        if self.current_frame_idx < len(self.frames_data):
                            self.frames_data[self.current_frame_idx] = {
                                'frame_id': self.current_frame_idx,
                                'keypoints': [kp.to_dict() for kp in self.tracker.keypoints]
                            }

                        # Reinitialize tracking with current keypoints
                        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                        self.tracker.prev_gray = gray

                        # Reinitialize templates and Kalman for visible keypoints
                        for kp in self.tracker.keypoints:
                            if kp.visibility == 1:
                                if kp.template is None:
                                    kp.template = self.tracker.extract_template(gray, kp.x, kp.y)
                                if not kp.kalman.initialized:
                                    kp.kalman.initialize(kp.x, kp.y)

                        self.tracker.tracking_points = np.array(
                            [[kp.x, kp.y] for kp in self.tracker.keypoints if kp.visibility == 1],
                            dtype=np.float32
                        ).reshape(-1, 1, 2)

                        self.changes_made_at_frame = None

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        # Save results
        self.save_results()

    def save_results(self):
        """Save tracking results to JSON file"""
        print(f"\nSaving results to {self.output_json}...")

        output_data = {
            'video': str(Path(self.video_path).name),
            'num_keypoints': self.num_keypoints,
            'frames': self.frames_data
        }

        with open(self.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Successfully saved {len(self.frames_data)} frames to {self.output_json}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Pyramid Corner Detection and Tracking System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pyramid_corner_tracker.py input_video.avi
  python pyramid_corner_tracker.py input_video.mp4 -o output.json -n 12
  python pyramid_corner_tracker.py pyramid_recording.avi --num-keypoints 18

Controls:
  SPACE: Pause/Resume video
  Q: Quit and save results
  Left Click (when paused): Add new keypoint or move existing one
  Right Click (when paused): Delete keypoint
        """
    )

    parser.add_argument('video', type=str, help='Path to input video file (.avi, .mp4, etc.)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path to output JSON file (default: input_name_tracked.json)')
    parser.add_argument('-n', '--num-keypoints', type=int, default=18,
                        help='Number of keypoints to track (default: 18)')

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        video_path = Path(args.video)
        output_json = video_path.parent / f"{video_path.stem}_tracked.json"
    else:
        output_json = args.output

    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return

    # Run tracker
    try:
        player = InteractiveVideoPlayer(args.video, str(output_json), args.num_keypoints)
        player.process_video()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()