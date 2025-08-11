import time
import sys
from typing import Optional, Union, Callable
import threading


class ProgressTracker:
    """Progress tracking utility with ETA and FPS"""
    
    def __init__(self, total: int, description: str = "Processing", width: int = 50, 
                 update_interval: float = 0.1, show_rate: bool = True, 
                 show_eta: bool = True, show_percentage: bool = True):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.update_interval = update_interval
        self.show_rate = show_rate
        self.show_eta = show_eta
        self.show_percentage = show_percentage
        self.start_time = time.time()
        self.last_update = 0
        self.completed = False
        self.paused = False
        self.pause_start = 0
        self.total_pause_time = 0
        self._lock = threading.Lock()
        
    def update(self, increment: int = 1, eta: Optional[float] = None, 
               current_fps: Optional[float] = None) -> None:
        """Update progress with optional ETA and FPS override"""
        with self._lock:
            if self.paused or self.completed:
                return
                
            self.current += increment
            self.current = min(self.current, self.total)  # Prevent overflow
            current_time = time.time()
            
            if current_time - self.last_update >= self.update_interval:
                calculated_eta = eta if eta is not None else self.estimate_remaining_time()
                calculated_fps = current_fps if current_fps is not None else self.calculate_fps()
                self._display_progress(calculated_eta, calculated_fps)
                self.last_update = current_time
    
    def set_current(self, value: int, eta: Optional[float] = None, 
                   current_fps: Optional[float] = None) -> None:
        """Set current progress to a specific value"""
        with self._lock:
            if self.paused or self.completed:
                return
                
            self.current = min(max(0, value), self.total)
            current_time = time.time()
            
            if current_time - self.last_update >= self.update_interval:
                calculated_eta = eta if eta is not None else self.estimate_remaining_time()
                calculated_fps = current_fps if current_fps is not None else self.calculate_fps()
                self._display_progress(calculated_eta, calculated_fps)
                self.last_update = current_time
    
    def _display_progress(self, eta: float, current_fps: float) -> None:
        """Display progress bar with ETA and FPS"""
        if self.total == 0:
            return
        
        progress = self.current / self.total
        
        # Build status components
        components = []
        
        # Progress bar
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)
        components.append(f"|{bar}|")
        
        # Current/Total
        components.append(f"{self.current}/{self.total}")
        
        # Percentage
        if self.show_percentage:
            components.append(f"({progress*100:.1f}%)")
        
        # ETA
        if self.show_eta:
            if eta > 0:
                if eta < 60:
                    eta_str = f"ETA: {eta:.1f}s"
                elif eta < 3600:
                    eta_str = f"ETA: {eta/60:.1f}m"
                else:
                    eta_str = f"ETA: {eta/3600:.1f}h"
            else:
                eta_str = "ETA: --"
            components.append(eta_str)
        
        # Rate/FPS
        if self.show_rate:
            if current_fps > 0:
                if current_fps < 1:
                    rate_str = f"Rate: {current_fps:.2f}/s"
                else:
                    rate_str = f"Rate: {current_fps:.1f}/s"
            else:
                rate_str = "Rate: --"
            components.append(rate_str)
        
        # Pause indicator
        if self.paused:
            components.append("[PAUSED]")
        
        # Status line
        status = f"\r{self.description}: " + " ".join(components)
        
        sys.stdout.write(status)
        sys.stdout.flush()
    
    def pause(self) -> None:
        """Pause the progress tracker"""
        with self._lock:
            if not self.paused and not self.completed:
                self.paused = True
                self.pause_start = time.time()
                self._display_progress(0, 0)
    
    def resume(self) -> None:
        """Resume the progress tracker"""
        with self._lock:
            if self.paused:
                self.total_pause_time += time.time() - self.pause_start
                self.paused = False
                self.pause_start = 0
    
    def finish(self) -> None:
        """Finish progress tracking"""
        with self._lock:
            if not self.completed:
                self.current = self.total
                self._display_progress(0, 0)
                elapsed = self.get_elapsed_time()
                avg_rate = self.calculate_fps()
                print(f"\nCompleted in {elapsed:.2f}s (avg rate: {avg_rate:.1f}/s)")
                self.completed = True
    
    def reset(self) -> None:
        """Reset progress tracking"""
        with self._lock:
            self.current = 0
            self.start_time = time.time()
            self.last_update = 0
            self.completed = False
            self.paused = False
            self.pause_start = 0
            self.total_pause_time = 0
    
    def set_description(self, description: str) -> None:
        """Update the description"""
        with self._lock:
            self.description = description
    
    def get_progress(self) -> float:
        """Get current progress as a percentage (0-100)"""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds (excluding paused time)"""
        elapsed = time.time() - self.start_time - self.total_pause_time
        if self.paused:
            elapsed -= (time.time() - self.pause_start)
        return max(0, elapsed)
    
    def estimate_remaining_time(self) -> float:
        """Estimate remaining time based on current progress"""
        if self.current == 0 or self.paused:
            return 0.0
        
        elapsed = self.get_elapsed_time()
        progress = self.current / self.total
        
        if progress > 0:
            total_estimated = elapsed / progress
            return max(0, total_estimated - elapsed)
        return 0.0
    
    def calculate_fps(self) -> float:
        """Calculate current rate based on elapsed time"""
        elapsed = self.get_elapsed_time()
        if elapsed > 0:
            return self.current / elapsed
        return 0.0
    
    def set_total(self, new_total: int) -> None:
        """Update the total count"""
        with self._lock:
            self.total = max(0, new_total)
            self.current = min(self.current, self.total)
    
    def increment_total(self, increment: int) -> None:
        """Increment the total count"""
        with self._lock:
            self.total += increment
    
    def is_complete(self) -> bool:
        """Check if progress is complete"""
        return self.current >= self.total or self.completed
    
    def get_status_dict(self) -> dict:
        """Get current status as a dictionary"""
        return {
            'current': self.current,
            'total': self.total,
            'progress_percent': self.get_progress(),
            'elapsed_time': self.get_elapsed_time(),
            'eta': self.estimate_remaining_time(),
            'rate': self.calculate_fps(),
            'paused': self.paused,
            'completed': self.completed
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish()


class MultiProgressTracker:
    """Manage multiple progress trackers"""
    
    def __init__(self):
        self.trackers = {}
        self._lock = threading.Lock()
    
    def add_tracker(self, name: str, total: int, description: str = None) -> ProgressTracker:
        """Add a new progress tracker"""
        with self._lock:
            if description is None:
                description = name
            tracker = ProgressTracker(total, description, show_rate=False, show_eta=False)
            self.trackers[name] = tracker
            return tracker
    
    def remove_tracker(self, name: str) -> None:
        """Remove a progress tracker"""
        with self._lock:
            if name in self.trackers:
                del self.trackers[name]
    
    def update_all(self) -> None:
        """Update display for all trackers"""
        with self._lock:
            # Clear previous lines
            for _ in range(len(self.trackers)):
                sys.stdout.write('\033[A\033[K')
            
            # Display all trackers
            for name, tracker in self.trackers.items():
                tracker._display_progress(0, 0)
                print()
    
    def finish_all(self) -> None:
        """Finish all trackers"""
        with self._lock:
            for tracker in self.trackers.values():
                tracker.finish()


# Example usage and demonstrations
if __name__ == "__main__":
    print("=== Basic Progress Tracker Demo ===")
    
    # Basic usage with automatic ETA and FPS
    with ProgressTracker(100, "Processing files") as tracker:
        for i in range(100):
            time.sleep(0.02)  # Simulate work
            tracker.update(1)
    
    print("\n=== Pause/Resume Demo ===")
    
    # Pause/resume demo
    tracker = ProgressTracker(50, "Downloading", width=30)
    
    for i in range(20):
        time.sleep(0.05)
        tracker.update(1)
    
    print("\nPausing for 2 seconds...")
    tracker.pause()
    time.sleep(2)
    tracker.resume()
    
    for i in range(20, 50):
        time.sleep(0.05)
        tracker.update(1)
    
    tracker.finish()
    
    print("\n=== Custom Configuration Demo ===")
    
    # Custom configuration
    with ProgressTracker(30, "Custom Config", width=20, 
                        show_rate=True, show_eta=True, 
                        show_percentage=False) as tracker:
        for i in range(30):
            time.sleep(0.1)
            tracker.update(1)
    
    print("\n=== Status Dictionary Demo ===")
    
    # Status dictionary example
    tracker = ProgressTracker(10, "Status Demo")
    for i in range(5):
        tracker.update(1)
        time.sleep(0.1)
    
    status = tracker.get_status_dict()
    print(f"Status: {status}")
    tracker.finish()
    
    print("\n=== Multi-Progress Tracker Demo ===")
    
    # Multi-progress tracker demo
    multi = MultiProgressTracker()
    
    # Add multiple trackers
    file_tracker = multi.add_tracker("files", 10, "Processing files")
    download_tracker = multi.add_tracker("downloads", 5, "Downloading")
    
    # Simulate concurrent progress
    for i in range(10):
        if i < 5:
            download_tracker.update(1)
        file_tracker.update(1)
        multi.update_all()
        time.sleep(0.5)
    
    multi.finish_all()
    
    print("\nAll demos completed!")