import time
from typing import List
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.console import Console

# Create console instance
console = Console()

class FastProgressTracker:
    """Fast progress tracker for video processing."""
    
    def __init__(self, total_videos: int):
        self.total_videos = total_videos
        self.start_time = time.time()
        self.processed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        self.current_video = ""
        self.current_status = "Initializing..."
        self.processing_times = []
        self.video_sizes = []
        self.detection_counts = []
        self.video_start_time = None

    def start_video(self, video_name: str, size: int = 0):
        """Start processing a video."""
        self.current_video = video_name
        self.current_status = "Processing..."
        self.video_start_time = time.time()
        self.video_sizes.append(size)

    def finish_video(self, success: bool = True, detections: int = 0):
        """Finish processing a video."""
        if self.video_start_time:
            processing_time = time.time() - self.video_start_time
            self.processing_times.append(processing_time)
            
        self.processed_count += 1
        self.detection_counts.append(detections)
        
        if success:
            self.successful_count += 1
            self.current_status = "✓ Success"
        else:
            self.failed_count += 1
            self.current_status = "✗ Failed"

    def get_display(self):
        """Get display table for progress."""
        table = Table(title="Video Processing Progress", header_style="bold blue")
        table.add_column("Video", style="cyan", width=20)
        table.add_column("Progress", style="green", width=15)
        table.add_column("Status", style="yellow", width=15)
        table.add_column("Time", style="magenta", width=10)
        
        elapsed = time.time() - self.start_time
        
        # Truncate video name if too long
        display_name = self.current_video[:17] + "..." if len(self.current_video) > 20 else self.current_video
        
        table.add_row(
            display_name,
            f"{self.processed_count}/{self.total_videos}",
            self.current_status,
            f"{elapsed:.1f}s"
        )
        
        # Statistics panel
        stats = Text()
        eta = 0
        if self.processed_count > 0:
            avg_time = elapsed / self.processed_count
            eta = avg_time * (self.total_videos - self.processed_count)
            
        stats.append(f"Processed: {self.processed_count}/{self.total_videos}\n", style="bold green")
        stats.append(f"Success: {self.successful_count}\n", style="green")
        stats.append(f"Failed: {self.failed_count}\n", style="red")
        stats.append(f"ETA: {eta:.1f}s\n", style="blue")
        stats.append(f"Total Detections: {sum(self.detection_counts)}", style="yellow")
        
        stats_panel = Panel(stats, title="Statistics", border_style="blue")
        
        return Columns([table, stats_panel], equal=True)


class LiveProgressDisplay:
    """Live progress display manager."""
    
    def __init__(self, tracker: FastProgressTracker):
        self.tracker = tracker
        self.live = Live(
            self.tracker.get_display(),
            console=console,
            refresh_per_second=4,
            transient=False
        )
        
    def __enter__(self):
        self.live.start()
        return self
        
    def update(self):
        """Update the live display."""
        self.live.update(self.tracker.get_display())
        
    def __exit__(self, *args):
        self.live.stop()