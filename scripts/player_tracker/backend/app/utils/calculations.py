"""
Statistical calculation utilities
"""
import numpy as np
from typing import List, Dict, Any
from config import config


def calculate_per90(stat_value: float, minutes: int) -> float:
    """
    Normalize stat to per 90 minutes
    """
    if minutes == 0:
        return 0.0
    return (stat_value / minutes) * 90


def calculate_percentile(value: float, all_values: List[float]) -> float:
    """
    Calculate percentile rank of a value within a distribution
    Returns value between 0-100
    """
    if not all_values or len(all_values) == 0:
        return 50.0
    
    return (np.sum(np.array(all_values) < value) / len(all_values)) * 100


def calculate_z_score(value: float, all_values: List[float]) -> float:
    """
    Calculate z-score (standard score) of a value
    """
    if not all_values or len(all_values) < 2:
        return 0.0
    
    mean = np.mean(all_values)
    std = np.std(all_values)
    
    if std == 0:
        return 0.0
    
    return (value - mean) / std


def calculate_ema(values: List[float], alpha: float = 0.3) -> float:
    """
    Calculate Exponential Moving Average
    
    Args:
        values: List of values in chronological order
        alpha: Smoothing factor (0-1), higher = more weight on recent values
    
    Returns:
        EMA value
    """
    if not values:
        return 0.0
    
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1 - alpha) * ema
    
    return ema


def calculate_form_trend(recent_scores: List[float], window: int = None) -> Dict[str, Any]:
    """
    Calculate form trend from recent match scores
    
    Returns:
        dict with direction, last_5_avg, last_10_avg
    """
    if not recent_scores:
        return {
            "direction": "unknown",
            "last_5_avg": 0.0,
            "last_10_avg": 0.0
        }
    
    window = window or config.FORM_TREND_WINDOW
    
    # Calculate averages
    last_5 = recent_scores[-5:] if len(recent_scores) >= 5 else recent_scores
    last_10 = recent_scores[-10:] if len(recent_scores) >= 10 else recent_scores
    
    last_5_avg = np.mean(last_5) if last_5 else 0.0
    last_10_avg = np.mean(last_10) if last_10 else 0.0
    
    # Determine direction
    if len(recent_scores) < 3:
        direction = "insufficient_data"
    elif last_5_avg > last_10_avg + 0.5:
        direction = "improving"
    elif last_5_avg < last_10_avg - 0.5:
        direction = "declining"
    else:
        direction = "stable"
    
    return {
        "direction": direction,
        "last_5_avg": float(last_5_avg),
        "last_10_avg": float(last_10_avg)
    }


def calculate_composite_score(
    breakdown: Dict[str, float],
    position: str,
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate weighted composite score based on position
    
    Args:
        breakdown: Dict with attacking, defensive, possession, efficiency, consistency scores
        position: Player position (GK, DEF, MID, ATT)
        weights: Optional custom weights, otherwise uses config
    
    Returns:
        Composite score (0-10 scale)
    """
    if weights is None:
        weights = config.STAT_WEIGHTS.get(position, config.STAT_WEIGHTS["MID"])
    
    total_weight = 0
    weighted_sum = 0
    
    for metric, score in breakdown.items():
        if metric in weights:
            weight = weights[metric]
            weighted_sum += score * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return weighted_sum / total_weight


def normalize_to_scale(value: float, min_val: float, max_val: float, target_scale: tuple = (0, 10)) -> float:
    """
    Normalize a value from one range to another
    
    Args:
        value: Value to normalize
        min_val: Minimum of original range
        max_val: Maximum of original range
        target_scale: Target range (min, max)
    
    Returns:
        Normalized value
    """
    if max_val == min_val:
        return target_scale[0]
    
    normalized = (value - min_val) / (max_val - min_val)
    return normalized * (target_scale[1] - target_scale[0]) + target_scale[0]


def calculate_consistency_score(values: List[float]) -> float:
    """
    Calculate consistency score based on standard deviation
    Lower std = higher consistency
    
    Returns:
        Consistency score (0-10 scale)
    """
    if not values or len(values) < 2:
        return 5.0
    
    std = np.std(values)
    mean = np.mean(values)
    
    if mean == 0:
        return 5.0
    
    # Coefficient of variation
    cv = std / mean
    
    # Convert to 0-10 scale (lower CV = higher score)
    # Assuming CV of 0.5 is poor, 0.1 is excellent
    consistency = max(0, min(10, 10 - (cv * 20)))
    
    return float(consistency)
