from torch.utils.tensorboard import SummaryWriter
import time
import os
import torch
import sys

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Bar(object):
    def __init__(self, iterable, desc="Progress", total=None, color=Colors.CYAN):
        self.iterable = iterable
        self.iterator = iter(iterable)
        self.desc = desc
        self.color = color
        self.last_loss = None
        self.compact = True
        self._DISPLAY_LENGTH = 40
        
        # total이 None인 경우 iterable의 길이를 사용
        if total is None:
            if hasattr(iterable, '__len__'):
                self.total = len(iterable)
            else:
                raise ValueError("iterable에 __len__ 메서드가 없거나 total을 명시적으로 지정해야 합니다.")
        else:
            self.total = total
            
        self._idx = 0
        self._time = []
    
    def __len__(self):
        return self.total
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())
        
        try:
            item = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()
        
        self._idx += 1
        if self._idx >= self.total:
            self._reset()
        
        return item
    
    def update_loss(self, loss_value):
        """Update current loss to display in the progress bar"""
        self.last_loss = loss_value
    
    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (self.total - self._idx)
        else:
            eta = 0
        
        rate = self._idx / self.total
        percentage = int(rate * 100)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        
        bar_fill = '━' * len_bar
        bar_empty = '╌' * (self._DISPLAY_LENGTH - len_bar)
        
        idx = str(self._idx).rjust(len(str(self.total)), ' ')
        
        prefix = f"{Colors.BOLD}{self.desc}{Colors.ENDC}"
        progress = f"{self.color}{bar_fill}{bar_empty}{Colors.ENDC}"
        stats = f"{Colors.BOLD}{percentage:3d}%{Colors.ENDC}"
        
        loss_display = ""
        if self.last_loss is not None:
            loss_display = f" {Colors.GREEN}loss:{self.last_loss:.4f}{Colors.ENDC}"
            
        if self.compact:
            tmpl = f"\r{prefix} {stats} {idx}/{self.total} [{progress}] {Colors.YELLOW}{eta:.1f}s{Colors.ENDC}{loss_display}"
        else:
            time_info = f"ETA: {Colors.YELLOW}{eta:.1f}s{Colors.ENDC}"
            tmpl = f"\r{prefix}: |{progress}| {stats} {idx}/{self.total} {time_info}{loss_display}"
        
        sys.stdout.write(tmpl)
        sys.stdout.flush()
        
        if self._idx == self.total:
            print()
    
    def _reset(self):
        self._idx = 0
        self._time = []

# Enhanced writer class that extends SummaryWriter to log training/validation metrics
class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)
        self.epoch = 0
        self.best_metric = float('inf')
    
    def log_train_loss(self, loss_type, train_loss, step):
        """Log training loss with colorful output"""
        self.add_scalar(f'train_{loss_type}_loss', train_loss, step)
        print(f"{Colors.BOLD}Train {loss_type} Loss:{Colors.ENDC} {Colors.GREEN}{train_loss:.6f}{Colors.ENDC}")
    
    def log_valid_loss(self, loss_type, valid_loss, step):
        """Log validation loss with colorful output"""
        self.add_scalar(f'valid_{loss_type}_loss', valid_loss, step)
        print(f"{Colors.BOLD}Valid {loss_type} Loss:{Colors.ENDC} {Colors.BLUE}{valid_loss:.6f}{Colors.ENDC}")
    
    def log_score(self, metrics_name, metrics, step):
        """Log metrics with colorful output"""
        self.add_scalar(metrics_name, metrics, step)
        print(f"{Colors.BOLD}{metrics_name}:{Colors.ENDC} {Colors.CYAN}{metrics:.6f}{Colors.ENDC}")
        
        # Track best metrics for saving checkpoints
        if metrics_name == 'validation_loss' and metrics < self.best_metric:
            self.best_metric = metrics
            return True
        return False

def save_checkpoint(exp_log_dir, model, epoch, optimizer=None, is_best=False):
    """Enhanced checkpoint saving with colored output"""
    save_dict = {
        "model": model.state_dict(),
        "epoch": epoch
    }
    
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    
    save_path = os.path.join(exp_log_dir, "ckpt_latest.pt")
    torch.save(save_dict, save_path)
    
    if is_best:
        best_path = os.path.join(exp_log_dir, "ckpt_best.pt")
        torch.save(save_dict, best_path)
        print(f"{Colors.BOLD}{Colors.GREEN}✓ Saved new best model checkpoint!{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}→ Saved checkpoint at epoch {epoch}{Colors.ENDC}")

# Example usage:
# 
# # Training loop
# writer = EnhancedWriter('./logs/experiment1')
# train_bar = StylishBar(train_loader, desc="Train")  # Shorter description for compact display
# 
# for epoch in range(num_epochs):
#     print(f"{Colors.HEADER}{Colors.BOLD}Epoch {epoch+1}/{num_epochs}{Colors.ENDC}")
#     
#     # Training phase
#     model.train()
#     total_loss = 0
#     
#     for batch in train_bar:
#         inputs, targets = batch
#         
#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         
#         # Update progress bar with current loss
#         train_bar.update_loss(loss.item())
#         
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         
#         total_loss += loss.item()
#     
#     # Log training metrics
#     avg_loss = total_loss / len(train_loader)
#     writer.log_train_loss('total', avg_loss, epoch)
#     
#     # Validation phase with a different color
#     valid_bar = StylishBar(valid_loader, desc="Valid", color=Colors.BLUE)  # Shorter description
#     model.eval()
#     valid_loss = 0
#     
#     with torch.no_grad():
#         for batch in valid_bar:
#             inputs, targets = batch
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             valid_bar.update_loss(loss.item())
#             valid_loss += loss.item()
#     
#     avg_val_loss = valid_loss / len(valid_loader)
#     is_best = writer.log_valid_loss('total', avg_val_loss, epoch)
#     
#     # Save checkpoint
#     save_checkpoint('./checkpoints', model, epoch, optimizer, is_best)