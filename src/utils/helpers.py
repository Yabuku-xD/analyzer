"""
Helper functions for the Dynamic Workforce Skill Evolution Analyzer.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Name of the logger
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(f"logs/{name}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    return logger


def clean_filename(filename: str) -> str:
    """
    Clean a string to use as a filename.
    
    Args:
        filename: String to clean
        
    Returns:
        Cleaned filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_').lower()
    
    return filename


def extract_date_from_string(date_string: str) -> Optional[pd.Timestamp]:
    """
    Extract a date from a string using various formats.
    
    Args:
        date_string: String containing a date
        
    Returns:
        Extracted date as a pandas Timestamp or None if extraction fails
    """
    date_formats = [
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S'
    ]
    
    for date_format in date_formats:
        try:
            return pd.to_datetime(date_string, format=date_format)
        except:
            continue
    
    return None


def normalize_values(values: List[float], min_val: Optional[float] = None, max_val: Optional[float] = None) -> List[float]:
    """
    Normalize values to a 0-1 range.
    
    Args:
        values: List of values to normalize
        min_val: Minimum value for normalization (calculated from values if None)
        max_val: Maximum value for normalization (calculated from values if None)
        
    Returns:
        Normalized values
    """
    if not values:
        return []
    
    # Calculate min and max if not provided
    if min_val is None:
        min_val = min(values)
    if max_val is None:
        max_val = max(values)
    
    # Check for division by zero
    if max_val == min_val:
        return [0.5] * len(values)
    
    # Normalize values
    normalized = [(x - min_val) / (max_val - min_val) for x in values]
    
    return normalized


def split_data_by_date(df: pd.DataFrame, date_column: str, test_ratio: float = 0.2) -> tuple:
    """
    Split data into training and test sets based on date.
    
    Args:
        df: DataFrame to split
        date_column: Name of the date column
        test_ratio: Ratio of data to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by date
    df = df.sort_values(date_column)
    
    # Calculate split index
    split_idx = int(len(df) * (1 - test_ratio))
    
    # Split data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    return train_df, test_df


def calculate_trend(values: List[float]) -> Dict:
    """
    Calculate trend statistics for a series of values.
    
    Args:
        values: List of values
        
    Returns:
        Dictionary of trend statistics
    """
    # Convert to numpy array
    values = np.array(values)
    
    # Calculate statistics
    mean = np.mean(values)
    median = np.median(values)
    std_dev = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Calculate trend (slope)
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    
    # Determine trend direction
    if slope > 0.05:
        direction = "increasing"
    elif slope < -0.05:
        direction = "decreasing"
    else:
        direction = "stable"
    
    # Calculate volatility
    volatility = std_dev / mean if mean > 0 else 0
    
    # Create result dictionary
    result = {
        "mean": float(mean),
        "median": float(median),
        "std_dev": float(std_dev),
        "min": float(min_val),
        "max": float(max_val),
        "slope": float(slope),
        "direction": direction,
        "volatility": float(volatility)
    }
    
    return result


def convert_to_time_series(df: pd.DataFrame, value_column: str, date_column: str, freq: str = 'D') -> pd.DataFrame:
    """
    Convert a DataFrame to a time series with regular frequency.
    
    Args:
        df: DataFrame to convert
        value_column: Name of the value column
        date_column: Name of the date column
        freq: Frequency for resampling
        
    Returns:
        Time series DataFrame
    """
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index
    ts_df = df.set_index(date_column)
    
    # Resample to regular frequency
    ts_df = ts_df[[value_column]].resample(freq).mean()
    
    # Forward fill missing values
    ts_df = ts_df.fillna(method='ffill')
    
    return ts_df


def get_date_range_str(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    """
    Get a formatted date range string.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Formatted date range string
    """
    start_str = start_date.strftime('%B %d, %Y')
    end_str = end_date.strftime('%B %d, %Y')
    
    return f"{start_str} to {end_str}"


def calculate_growth_rate(current_value: float, previous_value: float) -> float:
    """
    Calculate growth rate between two values.
    
    Args:
        current_value: Current value
        previous_value: Previous value
        
    Returns:
        Growth rate as a percentage
    """
    if previous_value == 0:
        return float('inf') if current_value > 0 else float('-inf') if current_value < 0 else 0
    
    return (current_value - previous_value) / previous_value * 100


def format_number(number: float, precision: int = 2) -> str:
    """
    Format a number with specified precision.
    
    Args:
        number: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if number is None:
        return "N/A"
    
    if abs(number) >= 1000000:
        return f"{number / 1000000:.{precision}f}M"
    elif abs(number) >= 1000:
        return f"{number / 1000:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def format_growth(growth_rate: float) -> str:
    """
    Format a growth rate with an indicator.
    
    Args:
        growth_rate: Growth rate as a percentage
        
    Returns:
        Formatted growth rate string with indicator
    """
    if growth_rate > 0:
        return f"↑ {growth_rate:.1f}%"
    elif growth_rate < 0:
        return f"↓ {abs(growth_rate):.1f}%"
    else:
        return f"→ {growth_rate:.1f}%"


if __name__ == "__main__":
    # Example usage
    logger = setup_logger("example")
    logger.info("Example log message")
    
    # Test normalization
    values = [10, 20, 30, 40, 50]
    normalized = normalize_values(values)
    print(f"Normalized values: {normalized}")
    
    # Test trend calculation
    trend = calculate_trend(values)
    print(f"Trend: {trend}")