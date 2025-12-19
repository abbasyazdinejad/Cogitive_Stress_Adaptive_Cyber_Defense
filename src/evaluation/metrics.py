"""
Evaluation Metrics Module
Provides metrics for:
1. Classification performance (Acc, F1, AUC, ECE)
2. Cognitive agent performance (latency, error rate, forgetting)
3. SOC-specific metrics (false escalation, policy volatility)

Based on Section VI of the paper.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: True labels (0/1)
        y_pred: Predicted labels (0/1)
        y_prob: Predicted probabilities (optional, for AUC)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # AUC (requires probabilities)
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0
    
    return metrics


def compute_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Union[int, float]]:
    """
    Compute detailed confusion matrix metrics.
    
    Returns:
        Dict with TP, TN, FP, FN, sensitivity, specificity
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    total = len(y_true)
    
    return {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # Recall
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0.0,  # Precision
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
    }


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match actual frequencies.
    Lower is better (0 = perfect calibration).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: List[str] = ['Non-Stress', 'Stress']
) -> None:
    """
    Print comprehensive classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: Names for classes
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    # Basic metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    print(f"\nOverall Metrics:")
    for name, value in metrics.items():
        print(f"  {name.capitalize()}: {value:.4f}")
    
    # Confusion matrix details
    cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TP: {cm_metrics['TP']}, TN: {cm_metrics['TN']}")
    print(f"  FP: {cm_metrics['FP']}, FN: {cm_metrics['FN']}")
    print(f"  Sensitivity: {cm_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {cm_metrics['specificity']:.4f}")
    
    # Calibration
    if y_prob is not None:
        ece = compute_expected_calibration_error(y_true, y_prob)
        print(f"\nCalibration:")
        print(f"  ECE: {ece:.4f}")
    
    # Sklearn report
    print(f"\nPer-Class Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


# ============================================================================
# COGNITIVE AGENT METRICS
# ============================================================================

def compute_decision_latency(decision_times: np.ndarray) -> Dict[str, float]:
    """
    Compute decision latency statistics.
    
    Args:
        decision_times: Array of decision times (seconds)
        
    Returns:
        Dict with mean, std, median latency
    """
    return {
        'mean_latency': np.mean(decision_times),
        'std_latency': np.std(decision_times),
        'median_latency': np.median(decision_times),
        'p95_latency': np.percentile(decision_times, 95),
    }


def compute_retrieval_success_rate(
    successful_retrievals: int,
    total_retrievals: int
) -> float:
    """
    Compute memory retrieval success rate.
    
    Args:
        successful_retrievals: Number of successful memory retrievals
        total_retrievals: Total retrieval attempts
        
    Returns:
        Success rate (0-100%)
    """
    if total_retrievals == 0:
        return 0.0
    return 100.0 * successful_retrievals / total_retrievals


def compute_error_rate(
    errors: int,
    total_decisions: int
) -> float:
    """
    Compute decision error rate.
    
    Args:
        errors: Number of incorrect decisions
        total_decisions: Total decisions made
        
    Returns:
        Error rate (0-100%)
    """
    if total_decisions == 0:
        return 0.0
    return 100.0 * errors / total_decisions


def compute_forgetting_index(
    memory_activations: np.ndarray,
    time_window: int = 10
) -> float:
    """
    Compute forgetting index based on memory activation decay.
    
    Args:
        memory_activations: Array of memory activation values over time
        time_window: Window for computing decay rate
        
    Returns:
        Forgetting index (higher = more forgetting)
    """
    if len(memory_activations) < time_window:
        return 0.0
    
    # Compute decay rate over sliding windows
    decay_rates = []
    for i in range(len(memory_activations) - time_window):
        window = memory_activations[i:i+time_window]
        if np.max(window) > 0:
            decay = (np.max(window) - np.min(window)) / np.max(window)
            decay_rates.append(decay)
    
    return np.mean(decay_rates) if decay_rates else 0.0


def print_cognitive_metrics(
    stress_level: str,
    latency: float,
    retrieval_rate: float,
    error_rate: float,
    forgetting_index: float
) -> None:
    """
    Print cognitive agent metrics (for Table III).
    
    Args:
        stress_level: 'Low', 'Medium', or 'High'
        latency: Decision latency (seconds)
        retrieval_rate: Retrieval success rate (%)
        error_rate: Error rate (%)
        forgetting_index: Forgetting index (%)
    """
    print(f"\n{'='*60}")
    print(f"COGNITIVE METRICS - {stress_level} Stress")
    print(f"{'='*60}")
    print(f"Decision Latency:    {latency:.2f}s")
    print(f"Retrieval Rate:      {retrieval_rate:.1f}%")
    print(f"Error Rate:          {error_rate:.1f}%")
    print(f"Forgetting Index:    {forgetting_index:.2%}")


# ============================================================================
# SOC-SPECIFIC METRICS
# ============================================================================

def compute_false_escalation_rate(
    escalations: np.ndarray,
    true_high_priority: np.ndarray
) -> float:
    """
    Compute false escalation rate (escalated but not high priority).
    
    Args:
        escalations: Boolean array (True = escalated)
        true_high_priority: Boolean array (True = actually high priority)
        
    Returns:
        False escalation rate (0-100%)
    """
    false_escalations = np.sum(escalations & ~true_high_priority)
    total_escalations = np.sum(escalations)
    
    if total_escalations == 0:
        return 0.0
    
    return 100.0 * false_escalations / total_escalations


def compute_policy_volatility(actions: np.ndarray) -> float:
    """
    Compute policy volatility (how often policy changes).
    
    Args:
        actions: Array of action indices over time
        
    Returns:
        Volatility (% of time steps where action changed)
    """
    if len(actions) < 2:
        return 0.0
    
    changes = np.sum(actions[1:] != actions[:-1])
    return 100.0 * changes / (len(actions) - 1)


# ============================================================================
# LOSO CROSS-VALIDATION RESULTS
# ============================================================================

def aggregate_loso_results(
    results_per_subject: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Aggregate LOSO cross-validation results.
    
    Args:
        results_per_subject: Dict of subject_id -> metrics_dict
        
    Returns:
        DataFrame with mean ± std for each metric
    """
    df = pd.DataFrame(results_per_subject).T
    
    summary = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max(),
    })
    
    return summary


def print_loso_summary(summary_df: pd.DataFrame) -> None:
    """
    Print LOSO summary table (like Table II in paper).
    
    Args:
        summary_df: Output from aggregate_loso_results()
    """
    print("\n" + "="*60)
    print("LOSO CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for metric in summary_df.index:
        mean = summary_df.loc[metric, 'mean']
        std = summary_df.loc[metric, 'std']
        min_val = summary_df.loc[metric, 'min']
        max_val = summary_df.loc[metric, 'max']
        
        print(f"{metric:<15} {mean:>9.4f} {std:>9.4f} {min_val:>9.4f} {max_val:>9.4f}")


if __name__ == "__main__":
    """
    Test script
    """
    print("="*60)
    print("Metrics Module Test")
    print("="*60)
    
    # Generate fake data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Test classification metrics
    print("\n[Test 1] Classification Metrics...")
    print_classification_report(y_true, y_pred, y_prob)
    print("✅ Classification metrics computed!")
    
    # Test cognitive metrics
    print("\n[Test 2] Cognitive Metrics...")
    print_cognitive_metrics(
        stress_level="Medium",
        latency=0.71,
        retrieval_rate=87.2,
        error_rate=10.1,
        forgetting_index=0.12
    )
    print("✅ Cognitive metrics displayed!")
    
    # Test LOSO aggregation
    print("\n[Test 3] LOSO Aggregation...")
    fake_results = {
        f'S{i}': {
            'accuracy': np.random.uniform(0.85, 0.95),
            'f1_score': np.random.uniform(0.83, 0.93),
            'auc': np.random.uniform(0.90, 0.97),
        }
        for i in range(2, 8)
    }
    summary = aggregate_loso_results(fake_results)
    print_loso_summary(summary)
    print("✅ LOSO results aggregated!")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
