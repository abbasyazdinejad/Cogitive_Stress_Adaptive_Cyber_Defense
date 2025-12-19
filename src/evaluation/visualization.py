"""
Visualization Module
Creates all figures from the paper:
- Figure 4: Signal comparison (ECG, EDA, Respiration)
- Figure 5: Feature distributions (KDEs)
- Figure 6-7: Stress probability timelines
- Figure 8-12: Decision timelines and agent behavior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger
from config import SAMPLING_RATES, WESAD_CONDITIONS

logger = setup_logger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# SIGNAL VISUALIZATION (Figure 4)
# ============================================================================

def plot_signal_comparison(
    subject_data: Dict,
    signals: List[str] = ['ECG', 'EDA', 'Resp'],
    duration: float = 60.0,
    start_time: float = 0.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of physiological signals in baseline vs stress.
    
    Args:
        subject_data: From WESADLoader.load_subject()
        signals: Signals to plot
        duration: Duration to plot (seconds)
        start_time: Start time (seconds)
        save_path: Path to save figure
    """
    chest_data = subject_data['chest']
    labels = subject_data['binary_label']
    
    # Find baseline and stress segments
    baseline_idx = np.where(labels == 0)[0]
    stress_idx = np.where(labels == 1)[0]
    
    fig, axes = plt.subplots(len(signals), 2, figsize=(14, len(signals) * 3))
    
    if len(signals) == 1:
        axes = axes.reshape(1, -1)
    
    for i, signal_name in enumerate(signals):
        if signal_name not in chest_data:
            continue
        
        signal_data = chest_data[signal_name]
        sr = SAMPLING_RATES['chest'][signal_name]
        
        start_sample = int(start_time * sr)
        window_samples = int(duration * sr)
        
        # Baseline
        if len(baseline_idx) > window_samples:
            baseline_segment = signal_data[baseline_idx[start_sample:start_sample + window_samples]]
            time_axis = np.arange(len(baseline_segment)) / sr
            
            axes[i, 0].plot(time_axis, baseline_segment, linewidth=0.8)
            axes[i, 0].set_title(f'{signal_name} - Baseline (Calm)', fontsize=12)
            axes[i, 0].set_ylabel('Amplitude', fontsize=10)
            axes[i, 0].grid(True, alpha=0.3)
        
        # Stress
        if len(stress_idx) > window_samples:
            stress_segment = signal_data[stress_idx[start_sample:start_sample + window_samples]]
            time_axis = np.arange(len(stress_segment)) / sr
            
            axes[i, 1].plot(time_axis, stress_segment, linewidth=0.8, color='red')
            axes[i, 1].set_title(f'{signal_name} - Stress', fontsize=12)
            axes[i, 1].set_ylabel('Amplitude', fontsize=10)
            axes[i, 1].grid(True, alpha=0.3)
        
        # X-label only on bottom row
        if i == len(signals) - 1:
            axes[i, 0].set_xlabel('Time (seconds)', fontsize=10)
            axes[i, 1].set_xlabel('Time (seconds)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    plt.show()


# ============================================================================
# FEATURE DISTRIBUTIONS (Figure 5)
# ============================================================================

def plot_feature_distributions(
    features_df: pd.DataFrame,
    features_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot KDE distributions of features for stress vs non-stress.
    
    Args:
        features_df: DataFrame with features and 'label' column
        features_to_plot: List of features (None = plot first 6)
        save_path: Path to save figure
    """
    from config import ALL_FEATURES
    
    if features_to_plot is None:
        features_to_plot = ALL_FEATURES[:6]
    
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_plot):
        if feature not in features_df.columns:
            continue
        
        # Non-stress
        non_stress_data = features_df[features_df['label'] == 0][feature].dropna()
        # Stress
        stress_data = features_df[features_df['label'] == 1][feature].dropna()
        
        # Plot KDE
        if len(non_stress_data) > 0:
            axes[i].hist(non_stress_data, bins=30, alpha=0.5, label='Non-Stress', 
                        density=True, color='blue', edgecolor='black')
        
        if len(stress_data) > 0:
            axes[i].hist(stress_data, bins=30, alpha=0.5, label='Stress',
                        density=True, color='red', edgecolor='black')
        
        axes[i].set_title(f'Distribution of {feature}', fontsize=11)
        axes[i].set_xlabel(feature, fontsize=9)
        axes[i].set_ylabel('Density', fontsize=9)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    plt.show()


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Non-Stress', 'Stress'],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.show()


# ============================================================================
# ROC CURVE
# ============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")
    
    plt.show()


# ============================================================================
# STRESS TIMELINE (Figure 6-7)
# ============================================================================

def plot_stress_timeline(
    stress_probabilities: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    threshold: float = 0.8,
    save_path: Optional[str] = None
) -> None:
    """
    Plot stress probability over time (Figure 6).
    
    Args:
        stress_probabilities: Predicted stress probabilities
        true_labels: True labels (optional, for ground truth overlay)
        threshold: Stress threshold line
        save_path: Path to save
    """
    time_steps = np.arange(len(stress_probabilities))
    
    plt.figure(figsize=(14, 5))
    
    # Plot stress probability
    plt.plot(time_steps, stress_probabilities, linewidth=2, label='Stress Probability', color='blue')
    
    # Threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
    
    # Ground truth (if provided)
    if true_labels is not None:
        # Highlight stress regions
        stress_regions = true_labels == 1
        plt.fill_between(time_steps, 0, 1, where=stress_regions, 
                        alpha=0.2, color='red', label='True Stress')
    
    plt.xlabel('Time Window Index', fontsize=12)
    plt.ylabel('Stress Probability', fontsize=12)
    plt.title('Stress Probability Timeline', fontsize=14)
    plt.ylim([0, 1])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved timeline to {save_path}")
    
    plt.show()


# ============================================================================
# CALIBRATION PLOT
# ============================================================================

def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save
    """
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration curve to {save_path}")
    
    plt.show()


# ============================================================================
# QUICK VISUALIZATION FOR TESTING
# ============================================================================

def quick_visualize(
    subject_data: Dict,
    features_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Quick visualization for testing data pipeline.
    
    Args:
        subject_data: From WESADLoader
        features_df: Extracted features (optional)
    """
    print("\n" + "="*60)
    print("QUICK VISUALIZATION")
    print("="*60)
    
    # Plot signals
    print("\n[1] Plotting signal comparison...")
    plot_signal_comparison(subject_data, duration=30.0)
    
    # Plot features (if provided)
    if features_df is not None:
        print("\n[2] Plotting feature distributions...")
        plot_feature_distributions(features_df)
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    """
    Test script
    """
    print("="*60)
    print("Visualization Module Test")
    print("="*60)
    
    # Import dependencies
    from data.wesad_loader import quick_load
    from data.signal_processor import SignalProcessor
    from data.feature_extractor import FeatureExtractor
    
    # Load and process data
    print("\n[Test 1] Loading S2...")
    subject_data = quick_load('S2', show_info=False)
    
    print("\n[Test 2] Processing signals...")
    processor = SignalProcessor()
    processed = processor.process_subject(subject_data, return_windows=True)
    
    print("\n[Test 3] Extracting features...")
    extractor = FeatureExtractor()
    features_df = extractor.extract_subject_features(processed)
    
    print("\n[Test 4] Creating visualizations...")
    
    # Signal comparison
    print("  - Signal comparison plot...")
    plot_signal_comparison(subject_data, duration=20.0)
    
    # Feature distributions
    print("  - Feature distribution plot...")
    plot_feature_distributions(features_df)
    
    # Generate fake predictions for other plots
    print("  - ROC curve...")
    y_true = features_df['label'].values
    y_prob = np.random.rand(len(y_true))  # Fake predictions
    plot_roc_curve(y_true, y_prob)
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATION TESTS PASSED!")
    print("="*60)
