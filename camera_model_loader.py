import numpy as np

class UpdatableCameraModel:
    def __init__(self, model_path: str):
        """
        Loads and initializes the updatable camera model from a .npz file.
        """
        try:
            model_data = np.load(model_path)
            self.avg_fx = model_data['avg_fx']
            self.avg_fy = model_data['avg_fy']
            self.avg_k = model_data['avg_k']
            self.avg_p = model_data['avg_p']
            
            # Check if this is a circle or ellipse model
            self.is_circle_model = model_data.get('is_circle_model', False)
            
            if self.is_circle_model:
                self.pp_circle_params = model_data['pp_circle_params']
                print(f"Updatable camera model (CIRCLE) loaded successfully from '{model_path}'")
            else:
                # Legacy support - check for ellipse params
                self.pp_ellipse_params = model_data['pp_ellipse_params']
                print(f"Updatable camera model (ELLIPSE) loaded successfully from '{model_path}'")
        except FileNotFoundError:
            print(f"Error: Model file not found at '{model_path}'")
            raise
    
    def get_center(self):
        if self.is_circle_model:
            return self.pp_circle_params[:2]
        else:
            return self.pp_ellipse_params[:2]

    def get_camera_parameters(self, theta_deg: float):
        """
        Calculates the camera matrix and distortion coefficients for a given angle.
        
        Args:
            theta_deg: The telescope rotation angle in degrees.

        Returns:
            A tuple of (camera_matrix, distortion_coefficients).
        """
        if self.is_circle_model:
            # Predict principal point using circle model parameters
            cx, cy, r = self.pp_circle_params
            r = abs(r)  # Ensure radius is positive
            
            theta_rad = np.deg2rad(theta_deg)
            predicted_cx = cx + r * np.cos(theta_rad)
            predicted_cy = cy + r * np.sin(theta_rad)
        else:
            # Predict principal point using ellipse model parameters
            cx, cy, a, b, phi = self.pp_ellipse_params
            
            theta_rad = np.deg2rad(theta_deg)
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)

            predicted_cx = cx + a * np.cos(theta_rad) * cos_phi - b * np.sin(theta_rad) * sin_phi
            predicted_cy = cy + a * np.cos(theta_rad) * sin_phi + b * np.sin(theta_rad) * cos_phi
        
        # Assemble the final camera matrix
        camera_matrix = np.array([
            [self.avg_fx, 0, predicted_cx],
            [0, self.avg_fy, predicted_cy],
            [0, 0, 1]
        ])
        
        # Assemble distortion coefficients
        dist_coeffs = np.array([self.avg_k[0], self.avg_k[1], self.avg_p[0], self.avg_p[1], self.avg_k[2]])
        
        return camera_matrix, dist_coeffs
