import cv2
import os
import platform
from pathlib import Path

__all__ = ["VideoUtils"]


def _to_wsl_path(path: str) -> str:
    """
    Convert Windows path to WSL mount path if running in WSL environment.
    
    Args:
        path: Windows or POSIX path string
        
    Returns:
        Converted path string for WSL or original path
    """
    # Only convert if we're in WSL environment
    if not _is_wsl():
        return path
    
    p = Path(path)
    
    # Check if it's a Windows path (has drive letter)
    if len(str(p).split(':')) == 2 and str(p)[1] == ':':
        drive = str(p)[0].lower()
        # Remove drive and convert backslashes to forward slashes
        rest = str(p)[2:].replace('\\', '/')
        # Remove leading slash if present
        rest = rest.lstrip('/')
        return f"/mnt/{drive}/{rest}"
    
    return str(p)


def _is_wsl() -> bool:
    """Check if running in WSL environment"""
    try:
        # Check for WSL-specific indicators
        if platform.system() == "Linux":
            # Check for WSL in /proc/version
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                if 'microsoft' in version_info or 'wsl' in version_info:
                    return True
            
            # Check for WSL environment variable
            if 'WSL_DISTRO_NAME' in os.environ:
                return True
                
        return False
    except:
        return False


class VideoUtils:
    """Utility class for video file operations"""
    
    @classmethod
    def _open_video_capture(cls, path: str) -> cv2.VideoCapture:
        """
        Open video capture with proper path handling for different environments.
        
        Args:
            path: Path to video file
            
        Returns:
            OpenCV VideoCapture object
            
        Raises:
            ValueError: If video cannot be opened
        """
        # Convert path for WSL if needed
        resolved_path = _to_wsl_path(path)
        
        # Try to open with resolved path
        cap = cv2.VideoCapture(resolved_path)
        
        if not cap.isOpened():
            # Try original path as fallback
            cap = cv2.VideoCapture(path)
            
            if not cap.isOpened():
                raise ValueError(
                    f"Cannot open video file: {path}\n"
                    f"Tried paths: {path}, {resolved_path}\n"
                    f"Make sure the file exists and is a valid video format."
                )
        
        return cap
    
    @classmethod
    def get_video_info(cls, path: str) -> dict:
        """
        Get video file information.
        
        Args:
            path: Path to video file
            
        Returns:
            Dictionary containing video properties:
            - frame_count: Number of frames
            - fps: Frames per second
            - width: Video width in pixels
            - height: Video height in pixels
            - duration: Video duration in seconds
            - codec: Video codec (if available)
            
        Raises:
            ValueError: If video cannot be opened or has invalid properties
        """
        cap = cls._open_video_capture(path)
        
        try:
            # Get basic properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Handle invalid FPS
            if fps <= 0 or fps is None:
                fps = 30.0  # Default fallback
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Get codec info (if available)
            codec = None
            try:
                fourcc = cap.get(cv2.CAP_PROP_FOURCC)
                if fourcc:
                    codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            except:
                pass
            
            # Validate properties
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid video dimensions: {width}x{height} for {path}"
                )
            
            if frame_count <= 0:
                raise ValueError(
                    f"Invalid frame count: {frame_count} for {path}"
                )
            
            info = {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration,
                "codec": codec,
                "path": path
            }
            
            return info
            
        finally:
            cap.release()
    
    @classmethod
    def validate_video_file(cls, path: str) -> bool:
        """
        Validate if a file is a readable video file.
        
        Args:
            path: Path to video file
            
        Returns:
            True if valid video file, False otherwise
        """
        try:
            cls.get_video_info(path)
            return True
        except:
            return False
    
    @classmethod
    def get_video_frame(cls, path: str, frame_number: int):
        """
        Get a specific frame from video.
        
        Args:
            path: Path to video file
            frame_number: Frame number to extract (0-based)
            
        Returns:
            numpy array of the frame, or None if frame cannot be read
        """
        cap = cls._open_video_capture(path)
        
        try:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            
            if ret:
                return frame
            else:
                return None
                
        finally:
            cap.release()
    
    @classmethod
    def get_video_frames_count(cls, path: str) -> int:
        """
        Get total number of frames in video.
        
        Args:
            path: Path to video file
            
        Returns:
            Number of frames
        """
        try:
            info = cls.get_video_info(path)
            return info["frame_count"]
        except:
            return 0
    
    @classmethod
    def create_video_writer(cls, output_path: str, fps: float, width: int, height: int, 
                           codec: str = "mp4v") -> cv2.VideoWriter:
        """
        Create a video writer for output.
        
        Args:
            output_path: Path for output video
            fps: Frames per second
            width: Video width
            height: Video height
            codec: Video codec (default: mp4v)
            
        Returns:
            OpenCV VideoWriter object
        """
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create fourcc
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Create writer
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise ValueError(f"Cannot create video writer for {output_path}")
        
        return writer