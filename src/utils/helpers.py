"""
Utility Functions for CPS-SOC Framework
Provides: logging, file I/O, random seed setting, path management
"""

import os
import sys
import pickle
import logging
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def setup_logger(name: str = "cps_soc", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to save logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy, random, and TensorFlow.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
        # Additional TF settings for reproducibility
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_pickle(filepath: str) -> Any:
    """
    Load pickle file with proper encoding handling.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Unpickled object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If loading fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            # Try default encoding first
            try:
                return pickle.load(f)
            except UnicodeDecodeError:
                # Fallback to latin1 (for Python 2 pickles)
                f.seek(0)
                return pickle.load(f, encoding='latin1')
    except Exception as e:
        raise Exception(f"Failed to load pickle from {filepath}: {str(e)}")


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object as pickle file.
    
    Args:
        obj: Object to pickle
        filepath: Output filepath
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def ensure_dir(directory: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
    }


def print_dict(d: Dict, indent: int = 0, max_items: int = 10) -> None:
    """
    Pretty print nested dictionary.
    
    Args:
        d: Dictionary to print
        indent: Indentation level
        max_items: Maximum items to show per level
    """
    prefix = "  " * indent
    
    if isinstance(d, dict):
        for i, (key, value) in enumerate(d.items()):
            if i >= max_items:
                print(f"{prefix}... ({len(d) - max_items} more items)")
                break
            
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}:")
                print_dict(value, indent + 1, max_items)
            elif isinstance(value, np.ndarray):
                print(f"{prefix}{key}: array{value.shape} dtype={value.dtype}")
            else:
                print(f"{prefix}{key}: {value}")
    
    elif isinstance(d, list):
        for i, item in enumerate(d[:max_items]):
            if isinstance(item, (dict, list)):
                print(f"{prefix}[{i}]:")
                print_dict(item, indent + 1, max_items)
            else:
                print(f"{prefix}[{i}]: {item}")
        
        if len(d) > max_items:
            print(f"{prefix}... ({len(d) - max_items} more items)")


def validate_wesad_subject(subject_id: str) -> bool:
    """
    Validate if subject ID is in expected WESAD range.
    
    Args:
        subject_id: Subject ID (e.g., 'S2', 'S17')
        
    Returns:
        True if valid, False otherwise
    """
    if not subject_id.startswith('S'):
        return False
    
    try:
        num = int(subject_id[1:])
        # WESAD has S2-S17 (S1 and S12 are excluded)
        return 2 <= num <= 17 and num != 12
    except ValueError:
        return False


def get_available_wesad_subjects(wesad_dir: str) -> List[str]:
    """
    Get list of available WESAD subject folders.
    
    Args:
        wesad_dir: Path to WESAD dataset directory
        
    Returns:
        List of subject IDs (e.g., ['S2', 'S3', ...])
    """
    subjects = []
    
    if not os.path.exists(wesad_dir):
        return subjects
    
    for item in os.listdir(wesad_dir):
        item_path = os.path.join(wesad_dir, item)
        if os.path.isdir(item_path) and validate_wesad_subject(item):
            subjects.append(item)
    
    return sorted(subjects)


# ============================================================================
# NUMERICAL UTILITIES
# ============================================================================

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Safe division that handles division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result of division with safe handling
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result[~np.isfinite(result)] = fill_value
    return result


def normalize_signal(signal: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize signal using specified method.
    
    Args:
        signal: Input signal
        method: 'zscore', 'minmax', or 'robust'
        
    Returns:
        Normalized signal
    """
    if method == 'zscore':
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal - mean
        return (signal - mean) / std
    
    elif method == 'minmax':
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val == min_val:
            return np.zeros_like(signal)
        return (signal - min_val) / (max_val - min_val)
    
    elif method == 'robust':
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        if mad == 0:
            return signal - median
        return (signal - median) / (1.4826 * mad)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def check_array_quality(arr: np.ndarray, name: str = "array") -> Dict[str, Any]:
    """
    Check array for NaN, Inf, and other quality issues.
    
    Args:
        arr: Array to check
        name: Name for reporting
        
    Returns:
        Dictionary with quality metrics
    """
    return {
        'name': name,
        'shape': arr.shape,
        'dtype': arr.dtype,
        'has_nan': np.isnan(arr).any(),
        'num_nan': np.isnan(arr).sum(),
        'has_inf': np.isinf(arr).any(),
        'num_inf': np.isinf(arr).sum(),
        'min': np.nanmin(arr) if not np.isnan(arr).all() else np.nan,
        'max': np.nanmax(arr) if not np.isnan(arr).all() else np.nan,
        'mean': np.nanmean(arr) if not np.isnan(arr).all() else np.nan,
        'std': np.nanstd(arr) if not np.isnan(arr).all() else np.nan,
    }


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or Python file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    elif config_path.endswith('.py'):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract all uppercase variables
        config_dict = {
            key: getattr(config_module, key)
            for key in dir(config_module)
            if key.isupper() and not key.startswith('_')
        }
        return config_dict
    
    else:
        raise ValueError(f"Unsupported config format: {config_path}")


if __name__ == "__main__":
    # Quick test
    logger = setup_logger()
    logger.info("Helper utilities loaded successfully!")
    
    set_random_seed(42)
    logger.info("Random seed set to 42")
    
    # Test signal normalization
    test_signal = np.random.randn(100)
    normalized = normalize_signal(test_signal, method='zscore')
    logger.info(f"Signal normalized: mean={normalized.mean():.6f}, std={normalized.std():.6f}")
    
    # Test quality check
    test_array = np.array([1, 2, np.nan, 4, np.inf])
    quality = check_array_quality(test_array, "test")
    logger.info(f"Array quality: {quality}")
    
    print("\n✅ All helper functions working!")
