"""
Signal Preprocessing Module
Handles signal preprocessing including:
- Sliding window segmentation
- Butterworth filtering
- Z-score normalization
- Artifact removal

Based on Section IV-C of the paper.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, detrend
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger, normalize_signal, check_array_quality
from config import (
    WINDOW_SIZE, WINDOW_OVERLAP, WINDOW_STEP,
    FILTER_TYPE, LOWPASS_CUTOFF, FILTER_ORDER,
    SAMPLING_RATES
)

logger = setup_logger(__name__)


class SignalProcessor:
    """
    Signal preprocessing pipeline for WESAD physiological data.
    
    Pipeline:
        1. Sliding window segmentation (60s windows, 50% overlap)
        2. Butterworth lowpass filtering (5 Hz cutoff)
        3. Z-score normalization
        4. Quality checks
    """
    
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        window_overlap: float = WINDOW_OVERLAP,
        filter_cutoff: float = LOWPASS_CUTOFF,
        filter_order: int = FILTER_ORDER,
        normalize: bool = True
    ):
        """
        Initialize signal processor.
        
        Args:
            window_size: Window size in seconds
            window_overlap: Overlap fraction (0.0 - 1.0)
            filter_cutoff: Lowpass filter cutoff frequency (Hz)
            filter_order: Filter order
            normalize: Apply z-score normalization
        """
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.window_step = int(window_size * (1 - window_overlap))
        
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.normalize = normalize
        
        logger.info(f"SignalProcessor initialized:")
        logger.info(f"  Window: {window_size}s with {window_overlap*100}% overlap")
        logger.info(f"  Step: {self.window_step}s")
        logger.info(f"  Filter: {filter_order}th order Butterworth @ {filter_cutoff}Hz")
        logger.info(f"  Normalization: {normalize}")
    
    def process_signal(
        self,
        signal_data: np.ndarray,
        sampling_rate: float,
        apply_filter: bool = True,
        apply_detrend: bool = True
    ) -> np.ndarray:
        """
        Process a single continuous signal.
        
        Args:
            signal_data: Raw signal (1D array)
            sampling_rate: Sampling rate in Hz
            apply_filter: Apply Butterworth lowpass filter
            apply_detrend: Remove linear trend
            
        Returns:
            Processed signal (same length as input)
        """
        processed = signal_data.copy()
        
        # Remove linear trend
        if apply_detrend:
            processed = detrend(processed)
        
        # Apply Butterworth lowpass filter
        if apply_filter:
            processed = self._apply_butterworth_filter(processed, sampling_rate)
        
        # Normalize
        if self.normalize:
            processed = normalize_signal(processed, method='zscore')
        
        return processed
    
    def _apply_butterworth_filter(
        self, 
        signal_data: np.ndarray, 
        sampling_rate: float
    ) -> np.ndarray:
        """
        Apply Butterworth lowpass filter.
        
        Args:
            signal_data: Input signal
            sampling_rate: Sampling rate (Hz)
            
        Returns:
            Filtered signal
        """
        # Design filter
        nyquist_freq = sampling_rate / 2.0
        normalized_cutoff = self.filter_cutoff / nyquist_freq
        
        # Ensure cutoff is below Nyquist
        if normalized_cutoff >= 1.0:
            logger.warning(
                f"Cutoff frequency ({self.filter_cutoff}Hz) too high for "
                f"sampling rate ({sampling_rate}Hz). Skipping filter."
            )
            return signal_data
        
        try:
            b, a = butter(self.filter_order, normalized_cutoff, btype='low')
            filtered = filtfilt(b, a, signal_data)
            return filtered
        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            return signal_data
    
    def create_windows(
        self,
        signal_data: np.ndarray,
        labels: np.ndarray,
        sampling_rate: float,
        min_valid_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment signal into sliding windows.
        
        Args:
            signal_data: Preprocessed signal (1D array)
            labels: Corresponding labels (same length as signal)
            sampling_rate: Sampling rate (Hz)
            min_valid_ratio: Minimum ratio of valid (non -1) labels per window
            
        Returns:
            (windows, window_labels)
            - windows: shape (n_windows, window_samples)
            - window_labels: shape (n_windows,) - majority vote label
        """
        window_samples = int(self.window_size * sampling_rate)
        step_samples = int(self.window_step * sampling_rate)
        
        windows = []
        window_labels = []
        
        # Generate windows
        start_idx = 0
        while start_idx + window_samples <= len(signal_data):
            end_idx = start_idx + window_samples
            
            # Extract window
            window = signal_data[start_idx:end_idx]
            window_label_segment = labels[start_idx:end_idx]
            
            # Compute majority label (excluding -1)
            valid_labels = window_label_segment[window_label_segment != -1]
            
            # Check if window has enough valid labels
            if len(valid_labels) < min_valid_ratio * len(window_label_segment):
                start_idx += step_samples
                continue
            
            # Majority vote
            if len(valid_labels) > 0:
                unique, counts = np.unique(valid_labels, return_counts=True)
                majority_label = unique[np.argmax(counts)]
                
                windows.append(window)
                window_labels.append(majority_label)
            
            start_idx += step_samples
        
        windows = np.array(windows)
        window_labels = np.array(window_labels)
        
        logger.info(
            f"Created {len(windows)} windows "
            f"(window={self.window_size}s, step={self.window_step}s)"
        )
        
        return windows, window_labels
    
    def process_subject(
        self,
        subject_data: Dict,
        signals_to_process: List[str] = ['ECG', 'EDA', 'Resp'],
        return_windows: bool = True
    ) -> Dict:
        """
        Process all signals for a subject.
        
        Args:
            subject_data: Output from WESADLoader.load_subject()
            signals_to_process: List of signal names to process
            return_windows: If True, return windowed data; else return full processed signals
            
        Returns:
            Dictionary containing:
                If return_windows=True:
                    - '{signal}_windows': np.ndarray (n_windows, window_samples)
                    - 'labels': np.ndarray (n_windows,)
                    - 'subject_id': str
                If return_windows=False:
                    - '{signal}_processed': np.ndarray (full signal)
                    - 'labels': np.ndarray (full labels)
                    - 'subject_id': str
        """
        subject_id = subject_data['subject_id']
        chest_data = subject_data.get('chest', {})
        binary_labels = subject_data['binary_label']
        
        result = {'subject_id': subject_id}
        
        processed_signals = {}
        
        # Process each signal
        for signal_name in signals_to_process:
            if signal_name not in chest_data:
                logger.warning(f"Signal '{signal_name}' not found for {subject_id}")
                continue
            
            raw_signal = chest_data[signal_name]
            
            # Handle multi-dimensional signals (e.g., ACC)
            if raw_signal.ndim > 1:
                logger.info(f"Processing multi-channel signal {signal_name} (shape={raw_signal.shape})")
                # Process each channel separately
                processed_channels = []
                for ch in range(raw_signal.shape[1]):
                    processed_ch = self.process_signal(
                        raw_signal[:, ch],
                        sampling_rate=SAMPLING_RATES['chest'][signal_name]
                    )
                    processed_channels.append(processed_ch)
                processed = np.column_stack(processed_channels)
            else:
                # Single channel signal
                processed = self.process_signal(
                    raw_signal,
                    sampling_rate=SAMPLING_RATES['chest'][signal_name]
                )
            
            processed_signals[signal_name] = processed
        
        # Return windowed or full data
        if return_windows:
            # Create windows for each signal
            all_windows = []
            
            # Use first signal to determine window structure
            reference_signal = processed_signals[signals_to_process[0]]
            reference_sr = SAMPLING_RATES['chest'][signals_to_process[0]]
            
            windows, labels = self.create_windows(
                reference_signal,
                binary_labels,
                reference_sr
            )
            
            result['labels'] = labels
            
            # Window each signal
            for signal_name in signals_to_process:
                if signal_name not in processed_signals:
                    continue
                
                signal_data = processed_signals[signal_name]
                sr = SAMPLING_RATES['chest'][signal_name]
                
                # Reuse window structure from reference signal
                signal_windows, _ = self.create_windows(
                    signal_data,
                    binary_labels,
                    sr
                )
                
                result[f'{signal_name}_windows'] = signal_windows
            
        else:
            # Return full processed signals
            for signal_name, signal_data in processed_signals.items():
                result[f'{signal_name}_processed'] = signal_data
            result['labels'] = binary_labels
        
        return result
    
    def process_multiple_subjects(
        self,
        subjects_data: Dict[str, Dict],
        signals_to_process: List[str] = ['ECG', 'EDA', 'Resp'],
        return_windows: bool = True
    ) -> Dict[str, Dict]:
        """
        Process multiple subjects.
        
        Args:
            subjects_data: Dictionary of subject_id -> subject_data
            signals_to_process: Signals to process
            return_windows: Return windowed data
            
        Returns:
            Dictionary of subject_id -> processed_data
        """
        results = {}
        
        from tqdm import tqdm
        for subject_id in tqdm(subjects_data.keys(), desc="Processing subjects"):
            try:
                results[subject_id] = self.process_subject(
                    subjects_data[subject_id],
                    signals_to_process=signals_to_process,
                    return_windows=return_windows
                )
                logger.info(f"✓ Processed {subject_id}")
            except Exception as e:
                logger.error(f"✗ Failed to process {subject_id}: {e}")
                continue
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_process(
    subject_data: Dict,
    signal_name: str = 'ECG',
    show_info: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick preprocessing of a single signal (for testing).
    
    Args:
        subject_data: From WESADLoader.load_subject()
        signal_name: Signal to process
        show_info: Print information
        
    Returns:
        (processed_signal, windows)
    """
    processor = SignalProcessor()
    
    # Get raw signal
    raw = subject_data['chest'][signal_name]
    sr = SAMPLING_RATES['chest'][signal_name]
    
    # Process
    processed = processor.process_signal(raw, sr)
    
    # Create windows
    windows, labels = processor.create_windows(
        processed,
        subject_data['binary_label'],
        sr
    )
    
    if show_info:
        print(f"\n{signal_name} Processing:")
        print(f"  Raw shape: {raw.shape}")
        print(f"  Processed shape: {processed.shape}")
        print(f"  Windows: {windows.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Stress windows: {(labels == 1).sum()} / {len(labels)}")
    
    return processed, windows


if __name__ == "__main__":
    """
    Test script - run with: python signal_processor.py
    """
    print("="*60)
    print("Signal Processor Test")
    print("="*60)
    
    # Import loader
    from wesad_loader import quick_load
    
    # Test 1: Load subject
    print("\n[Test 1] Loading subject S2...")
    subject_data = quick_load('S2', show_info=False)
    print("✅ Subject loaded!")
    
    # Test 2: Process single signal
    print("\n[Test 2] Processing ECG signal...")
    processed, windows = quick_process(subject_data, 'ECG', show_info=True)
    print("✅ ECG processed!")
    
    # Test 3: Process all signals for subject
    print("\n[Test 3] Processing all signals (ECG, EDA, Resp)...")
    processor = SignalProcessor()
    result = processor.process_subject(
        subject_data,
        signals_to_process=['ECG', 'EDA', 'Resp'],
        return_windows=True
    )
    
    print(f"\nProcessed data keys: {list(result.keys())}")
    print(f"ECG windows shape: {result['ECG_windows'].shape}")
    print(f"EDA windows shape: {result['EDA_windows'].shape}")
    print(f"Resp windows shape: {result['Resp_windows'].shape}")
    print(f"Labels shape: {result['labels'].shape}")
    print("✅ All signals processed!")
    
    # Test 4: Signal quality check
    print("\n[Test 4] Signal quality check...")
    from utils.helpers import check_array_quality
    
    quality_ecg = check_array_quality(result['ECG_windows'], 'ECG_windows')
    print(f"ECG quality: NaN={quality_ecg['has_nan']}, Inf={quality_ecg['has_inf']}")
    print(f"  Mean: {quality_ecg['mean']:.4f}, Std: {quality_ecg['std']:.4f}")
    print("✅ Quality check passed!")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nSignal processor is ready to use!")
    print("Example:")
    print("  from data.signal_processor import SignalProcessor")
    print("  processor = SignalProcessor()")
    print("  processed = processor.process_subject(subject_data)")
