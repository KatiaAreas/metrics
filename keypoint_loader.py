import json
from typing import Dict, List, Tuple, Optional


class KeypointLoader:
    """
    A class to load and access keypoint data from a JSON file.
    
    The JSON structure should contain:
    - video: video filename
    - num_keypoints: number of keypoints per frame
    - frames: list of frame data, each containing frame_id and keypoints
    """
    
    def __init__(self, json_path: str):
        """
        Initialize the KeypointLoader with a JSON file.
        
        Args:
            json_path: Path to the JSON file containing keypoint data
        """
        self.json_path = json_path
        self.data = None
        self.frames_dict = {}
        self.load_data()
    
    def load_data(self):
        """Load the JSON file and organize keypoints by frame."""
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        # Organize frames into a dictionary for easy access
        for frame in self.data['frames']:
            frame_id = frame['frame_id']
            self.frames_dict[frame_id] = frame['keypoints']
    
    def get_frame_keypoints(self, frame_id: int) -> Dict[int, Dict[str, float]]:
        """
        Get all keypoints for a specific frame.
        
        Args:
            frame_id: The frame ID to retrieve
            
        Returns:
            Dictionary mapping keypoint ID to keypoint data (x, y, visibility)
        """
        if frame_id not in self.frames_dict:
            raise ValueError(f"Frame {frame_id} not found in data")
        
        keypoints = {}
        for kp in self.frames_dict[frame_id]:
            keypoints[kp['id']] = {
                'x': kp['x'],
                'y': kp['y'],
                'visibility': kp.get('visibility', 2)  # Default to 2 if not present
            }
        return keypoints
    
    def get_all_frames_keypoints(self) -> Dict[int, Dict[int, Dict[str, float]]]:
        """
        Get all keypoints for all frames in the video.
        
        Returns:
            Dictionary structure:
            {
                frame_id: {
                    keypoint_id: {'x': float, 'y': float, 'visibility': int}
                }
            }
        """
        all_keypoints = {}
        for frame_id in sorted(self.frames_dict.keys()):
            all_keypoints[frame_id] = self.get_frame_keypoints(frame_id)
        return all_keypoints
    
    def get_keypoint_trajectory(self, keypoint_id: int) -> List[Tuple[int, float, float]]:
        """
        Get the trajectory of a specific keypoint across all frames.
        
        Args:
            keypoint_id: The ID of the keypoint to track
            
        Returns:
            List of tuples (frame_id, x, y) for the specified keypoint
        """
        trajectory = []
        for frame_id in sorted(self.frames_dict.keys()):
            keypoints = self.frames_dict[frame_id]
            for kp in keypoints:
                if kp['id'] == keypoint_id:
                    trajectory.append((frame_id, kp['x'], kp['y']))
                    break
        return trajectory
    
    def get_frame_ids(self) -> List[int]:
        """Get all available frame IDs."""
        return sorted(self.frames_dict.keys())
    
    @property
    def num_keypoints(self) -> int:
        """Get the number of keypoints per frame."""
        return self.data.get('num_keypoints', 0)
    
    @property
    def video_name(self) -> str:
        """Get the video filename."""
        return self.data.get('video', '')
    
    @property
    def num_frames(self) -> int:
        """Get the total number of frames."""
        return len(self.frames_dict)


# Convenience function
def load_keypoints(json_path: str) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Convenience function to load keypoints from JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary mapping frame_id -> keypoint_id -> {x, y, visibility}
    """
    loader = KeypointLoader(json_path)
    return loader.get_all_frames_keypoints()


# Example usage
if __name__ == "__main__":
    # Example: Load keypoints
    json_path = "path/to/your/keypoints.json"
    
    # Method 1: Using the convenience function
    keypoints_dict = load_keypoints(json_path)
    
    # Access frame 1, keypoint 0
    frame_1_kp_0 = keypoints_dict[1][0]
    print(f"Frame 1, Keypoint 0: x={frame_1_kp_0['x']}, y={frame_1_kp_0['y']}")
    
    # Method 2: Using the class for more features
    loader = KeypointLoader(json_path)
    
    # Get all keypoints for frame 1
    frame_1_keypoints = loader.get_frame_keypoints(1)
    
    # Get trajectory of keypoint 0 across all frames
    trajectory = loader.get_keypoint_trajectory(0)
    
    # Get metadata
    print(f"Video: {loader.video_name}")
    print(f"Number of keypoints: {loader.num_keypoints}")
    print(f"Number of frames: {loader.num_frames}")
    
    # Iterate through all frames and keypoints
    all_keypoints = loader.get_all_frames_keypoints()
    for frame_id, keypoints in all_keypoints.items():
        print(f"Frame {frame_id}:")
        for kp_id, kp_data in keypoints.items():
            print(f"  Keypoint {kp_id}: ({kp_data['x']:.2f}, {kp_data['y']:.2f})")
