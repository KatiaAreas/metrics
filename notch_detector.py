# notch_detector.py
import time
import cv2
import numpy as np

class NotchDetector:
    """
    A class to detect the angle of a notch in a circular object from a video frame.

    This class maintains state between frames to provide temporally smoothed
    and robust detection of a circle and the notch angle within it.

    The core strategy is a two-step process for efficiency:
    1.  DETECT LOW: A circle is quickly detected on a small, downscaled version
        of the input frame.
    2.  REFINE HIGH: A high-resolution Region of Interest (ROI) is extracted
        from the original frame based on the detected circle's position. The
        precise notch angle is then calculated only within this small ROI.
    """

    # --- Constants for the detection algorithm ---
    # These can be tuned for different lighting or video conditions.
    _ROI_MARGIN_PX = 25  # Margin in pixels to add around the detected circle for the ROI.
    _NOTCH_SLOPE_DEGREES = 26.57  # Expected slope of the notch's V-shape edges.
    _MIN_SNR_THRESHOLD = 0.7  # Minimum signal-to-noise ratio to consider an angle detection valid.

    def __init__(self, circle_smoothing=0.1, notch_smoothing=1.0):
        """
        Initializes the NotchDetector.

        Args:
            circle_smoothing (float): A factor between 0.0 and 1.0 that controls
                how quickly the detected circle position adapts to new frames.
                A lower value results in smoother, but slower, tracking.
            notch_smoothing (float): A factor between 0.0 and 1.0 that controls
                how quickly the detected notch angle adapts. A value of 1.0
                means no smoothing is applied to the angle.
        """
        if not (0.0 <= circle_smoothing <= 1.0 and 0.0 <= notch_smoothing <= 1.0):
            raise ValueError("Smoothing factors must be between 0.0 and 1.0")

        self.circle_smoothing = circle_smoothing
        self.notch_smoothing = notch_smoothing

        # State variables to hold smoothed values between frames
        self.cur_circle = None  # Smoothed circle parameters (x, y, radius)
        self.cur_vec = None     # Smoothed notch angle unit vector [cos(a), sin(a)]

    def process_frame(self, frame):
        """
        Processes a single video frame to find the notch angle.

        This is the main method to be called for each frame of a video.

        Args:
            frame (np.ndarray): The input video frame in BGR format from OpenCV.

        Returns:
            dict: A dictionary containing the detection results:
                  {
                      'angle': float or None,  # The detected angle in degrees, or None if not found.
                      'center': tuple or None, # (x, y) coordinates of the circle center.
                      'radius': float or None, # Radius of the detected circle.
                  }
        """
        if frame is None:
            return {'angle': None, 'center': None, 'radius': None}

        original_h, original_w = frame.shape[:2]

        # 1. DETECT LOW: Find circle on a small, preprocessed image for speed.
        img_small, scale = self._preprocess_image(frame)
        best_circle = self._detect_best_circle(img_small)

        # Update and smooth the circle's position over time
        if best_circle is not None:
            if self.cur_circle is None:
                self.cur_circle = best_circle
            else:
                self.cur_circle = (1.0 - self.circle_smoothing) * self.cur_circle + \
                                  self.circle_smoothing * best_circle
        
        # If no circle has been found yet, we cannot proceed.
        if self.cur_circle is None:
            return {'angle': None, 'center': None, 'radius': None}

        # 2. REFINE HIGH: Scale coordinates and create a high-resolution ROI.
        c_x_small, c_y_small, c_r_small = self.cur_circle
        
        inv_scale = 1.0 / scale
        c_x_orig = c_x_small * inv_scale
        c_y_orig = c_y_small * inv_scale
        c_r_orig = c_r_small * inv_scale

        # Define the ROI on the original, full-resolution frame
        x1 = max(0, int(c_x_orig - c_r_orig - self._ROI_MARGIN_PX))
        y1 = max(0, int(c_y_orig - c_r_orig - self._ROI_MARGIN_PX))
        x2 = min(original_w, int(c_x_orig + c_r_orig + self._ROI_MARGIN_PX))
        y2 = min(original_h, int(c_y_orig + c_r_orig + self._ROI_MARGIN_PX))
        
        # Extract ROI and convert to grayscale for gradient analysis
        roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        
        # Calculate circle center relative to the ROI's top-left corner
        c_x_roi = c_x_orig - x1
        c_y_roi = c_y_orig - y1

        # 3. Find the precise angle ONLY on the small, high-res ROI.
        angle, sig_to_noise = self._find_best_angle(c_x_roi, c_y_roi, c_r_orig, roi_gray)

        # Discard detections with low signal-to-noise ratio
        if sig_to_noise < self._MIN_SNR_THRESHOLD:
            angle = None

        # Smooth the angle vector over time for stability
        if angle is not None:
            norm_vec = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            if self.cur_vec is None:
                self.cur_vec = norm_vec
            else:
                self.cur_vec = (1.0 - self.notch_smoothing) * self.cur_vec + \
                               self.notch_smoothing * norm_vec
                # Re-normalize the vector to keep it a unit vector
                self.cur_vec /= np.linalg.norm(self.cur_vec)

            # Convert the final smoothed vector back to an angle
            final_angle = np.arctan2(self.cur_vec[1], self.cur_vec[0]) * 180 / np.pi
        else:
            final_angle = None
            
        return {
            'angle': final_angle,
            'center': (c_x_orig, c_y_orig),
            'radius': c_r_orig
        }

    def _preprocess_image(self, img):
        """
        Converts an image to grayscale and resizes it for fast circle detection.
        """
        # Ensure the input is 3-channel BGR for consistent conversion
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        max_dim = max(img_gray.shape)
        scale = 380 / max_dim  # Target dimension for faster Hough transform
        img_resized = cv2.resize(img_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return img_resized, scale

    def _detect_best_circle(self, img):
        """
        Applies adaptive thresholding and Hough Circle Transform to find the main circle.
        """
        # Adaptive thresholding helps in variable lighting conditions
        img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 1)
        circles = cv2.HoughCircles(
            img_thresh, cv2.HOUGH_GRADIENT, 0.5, 40, param1=50, param2=30,
            minRadius=int(min(img.shape) / 2 * 0.8),  # 4/5
            maxRadius=int(min(img.shape) / 2 * 1.2)   # 6/5
        )
        if circles is not None:
            return circles[0, 0, :]  # Return the best circle found (x, y, radius)
        return None
    
    @staticmethod
    def _rotate(vectors, angle):
        """Rotates a batch of 2D vectors by a given angle in degrees."""
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        # Efficiently apply rotation matrix R to all vectors
        return np.tensordot(R, vectors, axes=[1, 0])

    def _find_best_angle(self, c_x, c_y, c_r, img):
        """
        Finds the notch angle using gradient analysis within a specific annulus of the circle.
        This function is called on a high-resolution ROI.
        """
        # Create a grid of coordinates corresponding to the image pixels
        coord_grid = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float64)
        
        # Calculate image gradients using Scharr operator for higher accuracy
        grad = np.array([cv2.Scharr(img, cv2.CV_64F, 0, 1), cv2.Scharr(img, cv2.CV_64F, 1, 0)])
        
        # Normalize gradients to prevent large values from dominating
        grad_mag = np.sqrt(grad[1] ** 2 + grad[0] ** 2)
        max_grad_mag = np.max(grad_mag)
        if max_grad_mag > 0:
            grad /= max_grad_mag

        # Calculate radial vectors from the circle center to each pixel
        radials = coord_grid - np.array([c_y, c_x])[:, None, None]
        radial_dist = np.sqrt(radials[0] ** 2 + radials[1] ** 2)
        
        # Normalize radial vectors (handle division by zero at the center)
        radials[:, radial_dist > 0] /= radial_dist[radial_dist > 0]
        
        # Define an annulus (a ring) just outside the circle's radius to search for the notch
        min_r = 1.04 * c_r
        max_r = 1.075 * c_r
        center_dist_sq = (coord_grid[0] - c_y) ** 2 + (coord_grid[1] - c_x) ** 2
        
        # Create a mask to zero out vectors outside the annulus
        mask = (center_dist_sq < min_r ** 2) | (center_dist_sq > max_r ** 2)
        radials[:, mask] = 0

        # Create two rotated vector fields to match the V-shape of the notch
        grid1 = self._rotate(radials, +self._NOTCH_SLOPE_DEGREES + 90)
        grid2 = self._rotate(radials, -self._NOTCH_SLOPE_DEGREES + 90)

        # Project the image gradients onto these rotated vector fields
        grid1 = np.sum(grad * grid1, axis=0)
        grid2 = np.sum(grad * grid2, axis=0)
        
        # The core of the algorithm: find where gradients match the notch pattern.
        # This creates four "heatmaps" corresponding to the four edges of the V-notch.
        grida = cv2.GaussianBlur(np.where(grid1 < 0, -grid1, 0), (17, 17), 0)
        gridb = cv2.GaussianBlur(np.where(grid1 > 0, grid1, 0), (17, 17), 0)
        gridc = cv2.GaussianBlur(np.where(grid2 < 0, -grid2, 0), (17, 17), 0)
        gridd = cv2.GaussianBlur(np.where(grid2 > 0, grid2, 0), (17, 17), 0)

        # Multiply the heatmaps to find the point of maximum correlation
        grid = grida * gridb * gridc * gridd
        grid = cv2.GaussianBlur(grid, (13, 13), 0, 0)
        
        # Find the coordinates of the brightest spot in the final grid
        max_y, max_x = np.unravel_index(np.argmax(grid), grid.shape)
        best_angle = np.arctan2(max_y - c_y, max_x - c_x) * 180 / np.pi

        # Calculate a signal-to-noise ratio for confidence scoring
        noise_floor = 1e-12
        grid_max1 = max(np.max(grid), noise_floor)
        # Find the max value again, but excluding a radius around the primary peak
        mask_dist_sq = (coord_grid[0] - max_y) ** 2 + (coord_grid[1] - max_x) ** 2
        grid_max2 = max(np.max(grid[mask_dist_sq > 30 ** 2]), noise_floor)
        sig_to_noise = np.log10(grid_max1 / grid_max2)

        return best_angle, sig_to_noise

# --- Example Usage ---

def render_results(img, results):
    """
    Helper function to draw the detection results on an image for visualization.
    """
    c_x, c_y = results['center']
    c_r = results['radius']
    angle = results['angle']

    # Draw the main circle
    cv2.circle(img, (int(c_x), int(c_y)), int(c_r), (0, 255, 0), 2)
    
    # Draw the angle line if it was detected
    if angle is not None:
        cv2.line(
            img, (int(c_x), int(c_y)),
            (int(c_x + c_r * np.cos(np.deg2rad(angle))),
             int(c_y + c_r * np.sin(np.deg2rad(angle)))),
            (255, 0, 0), 3
        )
        # Put angle text on the screen
        cv2.putText(img, f"Angle: {angle:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def process_video_file(video_file, detector):
    """
    Processes a video file using the NotchDetector and displays the results.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from {video_file}")
        return

    original_w, original_h = frame.shape[:2]
    print(f"\nProcessing {video_file} ({original_w}x{original_h})...")
    
    num_frame = 0
    start_time = time.time()

    while ret:
        # The main call to the detector class
        results = detector.process_frame(frame)

        # Render the results for display
        if results['center'] is not None:
            render_results(frame, results)

        cv2.imshow("Notch Angle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()
        num_frame += 1
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time > 0:
        print(f"Mean FPS = {num_frame / elapsed_time:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # You can list your video files here
    video_files = [
        "./videos/test_1.avi",
        "./videos/test_2.avi",
        # Add more video file paths here if needed
    ]

    # --- How to use the NotchDetector class ---
    # 1. Create an instance of the detector.
    #    You only need one instance per video stream.
    notch_finder = NotchDetector(circle_smoothing=0.1, notch_smoothing=0.5)

    # 2. Process each video file.
    for video in video_files:
        process_video_file(video, notch_finder)