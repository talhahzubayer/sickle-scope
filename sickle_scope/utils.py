"""
SickleScope Utility Functions

Performance optimisation utilities and helper functions for the SickleScope package.
"""

import time
import functools
import gc
import sys
import os
from typing import Any, Callable, Dict, Optional
import warnings
import logging
import pandas as pd

# Optional psutil import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging for performance monitoring
logger = logging.getLogger(__name__)


def memory_usage() -> float:
    """Get current memory usage in MB.
    
    Returns:
        Memory usage in megabytes
    """
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            pass
    
    # Fallback if psutil not available or fails
    return sum(sys.getsizeof(obj) for obj in gc.get_objects()) / 1024 / 1024


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance and memory usage.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Start monitoring
        start_time = time.time()
        start_memory = memory_usage()
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} failed after {end_time - start_time:.2f}s: {e}")
            raise
        
        # End monitoring
        end_time = time.time()
        end_memory = memory_usage()
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Log performance metrics
        if execution_time > 1.0:  # Only log slow functions
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s, "
                       f"memory delta: {memory_delta:+.1f}MB")
        
        return result
    
    return wrapper


def optimise_dataframe(df) -> Any:
    """Optimise DataFrame memory usage by downcasting dtypes.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Optimized DataFrame with reduced memory footprint
    """
    if df is None or df.empty:
        return df
        
    original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize object columns to categories if they have few unique values
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    optimised_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    reduction = (original_memory - optimised_memory) / original_memory * 100
    
    if reduction > 10:  # Only log significant reductions
        logger.info(f"DataFrame optimised: {reduction:.1f}% memory reduction "
                   f"({original_memory:.1f}MB -> {optimised_memory:.1f}MB)")
    
    return df


def clear_cache(func: Optional[Callable] = None) -> None:
    """Clear LRU cache for a function or all caches if no function specified.
    
    Args:
        func: Optional function to clear cache for
    """
    if func is not None and hasattr(func, 'cache_clear'):
        func.cache_clear()
        logger.debug(f"Cleared cache for {func.__name__}")
    else:
        # Force garbage collection
        gc.collect()
        logger.debug("Performed garbage collection")


def batch_process(items: list, batch_size: int = 1000, progress_callback: Optional[Callable] = None) -> list:
    """Process items in batches to optimise memory usage.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of processed results
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if progress_callback:
            progress_callback(batch_num, total_batches)
        
        # Process batch (this would be customised per use case)
        results.extend(batch)
        
        # Clear memory periodically
        if batch_num % 10 == 0:
            gc.collect()
    
    return results


class ConfigManager:
    """Manage configuration settings with validation and optimisation."""
    
    DEFAULT_CONFIG = {
        'batch_size': 1000,
        'cache_size': 1000,
        'memory_limit_mb': 1024,
        'parallel_workers': 4,
        'optimisation_level': 1
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize configuration manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.config['batch_size'] < 1:
            warnings.warn("batch_size must be >= 1, setting to 1000")
            self.config['batch_size'] = 1000
        
        if self.config['cache_size'] < 100:
            warnings.warn("cache_size must be >= 100, setting to 1000")
            self.config['cache_size'] = 1000
        
        if self.config['memory_limit_mb'] < 512:
            warnings.warn("memory_limit_mb must be >= 512, setting to 1024")
            self.config['memory_limit_mb'] = 1024
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def update(self, updates: Dict) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config.update(updates)
        self._validate_config()


def validate_input_size(df, max_size_mb: float = 100.0) -> bool:
    """Validate input DataFrame size for memory safety.
    
    Args:
        df: Input DataFrame
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if size is acceptable, False otherwise
    """
    if df is None or df.empty:
        return True
    
    size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    if size_mb > max_size_mb:
        logger.warning(f"Input DataFrame is large ({size_mb:.1f}MB > {max_size_mb}MB). "
                      f"Consider processing in batches.")
        return False
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value for division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# Export key utilities
__all__ = [
    'memory_usage',
    'performance_monitor', 
    'optimise_dataframe',
    'clear_cache',
    'batch_process',
    'ConfigManager',
    'validate_input_size',
    'safe_divide',
    'format_file_size'
]