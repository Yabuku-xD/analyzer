"""
Storage utilities for saving and loading data.
"""

import os
import json
import pickle
from typing import Any, Dict, List, Union

import pandas as pd


def save_to_json(data: Union[List, Dict], filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save to
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_from_json(filepath: str) -> Union[List, Dict]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to load from
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_to_csv(data: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        data: DataFrame to save
        filepath: Path to save to
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath, index=False)


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Path to load from
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)


def save_to_pickle(data: Any, filepath: str) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save to
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Path to load from
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data