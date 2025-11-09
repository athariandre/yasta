"""
Logging utilities for PPO training.

Provides CSV and optional TensorBoard logging helpers.
"""

import csv
import os
from pathlib import Path
from typing import Dict, Any, Optional


class CSVLogger:
    """CSV logger for training metrics."""
    
    def __init__(self, log_dir: str, run_name: str, headers: list):
        """
        Initialize CSV logger.
        
        Args:
            log_dir: Directory for logs
            run_name: Name of the run
            headers: List of column headers
        """
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.log_dir / "metrics.csv"
        self.headers = headers
        
        # Create CSV file with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        
        print(f"[CSVLogger] Logging to: {self.csv_path}")
    
    def log(self, metrics: Dict[str, Any]):
        """
        Log metrics to CSV.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(metrics)


class TensorBoardLogger:
    """Optional TensorBoard logger for training metrics."""
    
    def __init__(self, log_dir: str, run_name: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for logs
            run_name: Name of the run
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=Path(log_dir) / run_name)
            self.enabled = True
            print(f"[TensorBoardLogger] Logging to: {Path(log_dir) / run_name}")
        except ImportError:
            print("[TensorBoardLogger] TensorBoard not available, skipping")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log scalar value.
        
        Args:
            tag: Name of the scalar
            value: Value to log
            step: Global step
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, metrics: Dict[str, float], step: int):
        """
        Log multiple scalars.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Global step
        """
        if self.enabled and self.writer is not None:
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
