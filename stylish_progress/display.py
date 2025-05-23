from torch.utils.tensorboard import SummaryWriter
import time
import os
import torch
import sys


class Colors:
    """ANSI color codes for terminal output"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'  # End color
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_CYAN = '\033[96m'

class ProgressBarStatus:
    """Status constants for progress bar"""
    TRAINING = "training"
    COMPLETED = "completed" 
    INTERRUPTED = "interrupted"

class Bar(object):
    def __init__(self, dataloader, desc="Training", color=Colors.CYAN):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloader.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloader.')
        
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 40  # Wider bar for better visualization
        self.desc = desc
        self.default_color = color
        self.last_loss = None
        self.compact = True  # Enable compact mode
        
        # Status tracking
        self.status = ProgressBarStatus.TRAINING
        self._completed_naturally = False
        self._start_time = time.time()
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())
        
        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)
        
        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            # Natural completion
            self._completed_naturally = True
            self.status = ProgressBarStatus.COMPLETED
            self._display_final()
            raise StopIteration()
        except KeyboardInterrupt:
            # Manual interruption
            self.status = ProgressBarStatus.INTERRUPTED
            self._display_final()
            raise KeyboardInterrupt()
        except Exception as e:
            # Other exceptions
            self.status = ProgressBarStatus.INTERRUPTED
            self._display_final()
            raise e
        
        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._completed_naturally = True
            self.status = ProgressBarStatus.COMPLETED
            self._reset()
        
        return batch
    
    def update_loss(self, loss_value):
        """Update current loss to display in the progress bar"""
        self.last_loss = loss_value
    
    def mark_completed(self):
        """Manually mark as completed (for successful finish)"""
        self.status = ProgressBarStatus.COMPLETED
        self._completed_naturally = True
        self._display_final()
    
    def mark_interrupted(self):
        """Manually mark as interrupted (for error/cancellation)"""
        self.status = ProgressBarStatus.INTERRUPTED
        self._display_final()
    
    def _get_status_colors(self):
        """Get colors based on current status"""
        if self.status == ProgressBarStatus.TRAINING:
            return {
                'bar_color': Colors.CYAN,
                'desc_color': Colors.BOLD + Colors.CYAN,
                'time_color': Colors.GREEN,
                'stats_color': Colors.BOLD
            }
        elif self.status == ProgressBarStatus.COMPLETED:
            return {
                'bar_color': Colors.BRIGHT_GREEN,
                'desc_color': Colors.BOLD + Colors.BRIGHT_GREEN,
                'time_color': Colors.BRIGHT_GREEN,
                'stats_color': Colors.BOLD + Colors.BRIGHT_GREEN
            }
        elif self.status == ProgressBarStatus.INTERRUPTED:
            return {
                'bar_color': Colors.BRIGHT_RED,
                'desc_color': Colors.BOLD + Colors.BRIGHT_RED,
                'time_color': Colors.BRIGHT_RED,
                'stats_color': Colors.BOLD + Colors.BRIGHT_RED
            }
        else:
            return {
                'bar_color': self.default_color,
                'desc_color': Colors.BOLD,
                'time_color': Colors.GREEN,
                'stats_color': Colors.BOLD
            }
    
    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0
        
        rate = self._idx / len(self.dataloader)
        percentage = int(rate * 100)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        
        # Get status-based colors
        colors = self._get_status_colors()
        
        # Use thinner characters for a more compact display
        bar_fill = '━' * len_bar  # Horizontal line instead of block
        bar_empty = '╌' * (self._DISPLAY_LENGTH - len_bar)  # Dotted line instead of block
        
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')
        
        # Format with status-based colors
        prefix = f"{colors['desc_color']}{self.desc}{Colors.ENDC}"
        progress = f"{colors['bar_color']}{bar_fill}{bar_empty}{Colors.ENDC}"
        stats = f"{colors['stats_color']}{percentage:3d}%{Colors.ENDC}"
        
        # Add loss display if available
        loss_display = ""
        if self.last_loss is not None:
            loss_display = f" {Colors.YELLOW}loss:{self.last_loss:.4f}{Colors.ENDC}"
        
        # Time display with green color
        time_display = f"{colors['time_color']}{eta:.1f}s{Colors.ENDC}"
        
        # Compact display in a single line
        if self.compact:
            tmpl = f"\r{prefix} {stats} {idx}/{len(self.dataset)} [{progress}] {time_display}{loss_display}"
        else:
            # Original multi-component display
            time_info = f"ETA: {time_display}"
            tmpl = f"\r{prefix}: |{progress}| {stats} {idx}/{len(self.dataset)} {time_info}{loss_display}"
        
        sys.stdout.write(tmpl)
        sys.stdout.flush()
        
        # Don't print newline here - let _display_final handle it
    
    def _display_final(self):
        """Display final status with appropriate colors"""
        total_time = time.time() - self._start_time
        colors = self._get_status_colors()
        
        # Calculate final percentage
        if self.status == ProgressBarStatus.COMPLETED:
            percentage = 100
            len_bar = self._DISPLAY_LENGTH
            status_text = "COMPLETED"
            status_icon = "✓"
        else:
            rate = self._idx / len(self.dataloader)
            percentage = int(rate * 100)
            len_bar = int(rate * self._DISPLAY_LENGTH)
            status_text = "INTERRUPTED"
            status_icon = "✗"
        
        # Create final progress bar
        bar_fill = '━' * len_bar
        bar_empty = '╌' * (self._DISPLAY_LENGTH - len_bar)
        
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')
        
        # Format final display
        prefix = f"{colors['desc_color']}{self.desc}{Colors.ENDC}"
        progress = f"{colors['bar_color']}{bar_fill}{bar_empty}{Colors.ENDC}"
        stats = f"{colors['stats_color']}{percentage:3d}%{Colors.ENDC}"
        
        # Add loss display if available
        loss_display = ""
        if self.last_loss is not None:
            loss_display = f" {Colors.YELLOW}loss:{self.last_loss:.4f}{Colors.ENDC}"
        
        # Final time display
        time_display = f"{colors['time_color']}{total_time:.1f}s{Colors.ENDC}"
        status_display = f"{colors['desc_color']}{status_icon} {status_text}{Colors.ENDC}"
        
        # Final display line
        if self.compact:
            tmpl = f"\r{prefix} {stats} {idx}/{len(self.dataset)} [{progress}] {time_display}{loss_display} {status_display}"
        else:
            tmpl = f"\r{prefix}: |{progress}| {stats} {idx}/{len(self.dataset)} Total: {time_display}{loss_display} {status_display}"
        
        sys.stdout.write(tmpl)
        sys.stdout.flush()
        print()  # Add newline after final display
    
    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []

# Enhanced Bar for training with context manager support
class TrainingBar(Bar):
    """Enhanced Bar with context manager support for training loops"""
    
    def __init__(self, dataloader, desc="Training", color=Colors.CYAN):
        super().__init__(dataloader, desc, color)
        self.epoch = 1
        self.total_epochs = None
    
    def set_epoch_info(self, current_epoch, total_epochs):
        """Set epoch information for display"""
        self.epoch = current_epoch
        self.total_epochs = total_epochs
        if total_epochs:
            self.desc = f"Epoch {current_epoch}/{total_epochs}"
        else:
            self.desc = f"Epoch {current_epoch}"
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - handle different exit scenarios"""
        if exc_type is None:
            # Normal completion
            self.mark_completed()
        elif exc_type == KeyboardInterrupt:
            # Manual interruption
            self.mark_interrupted()
            return False  # Re-raise the exception
        else:
            # Other exceptions
            self.mark_interrupted()
            return False  # Re-raise the exception
