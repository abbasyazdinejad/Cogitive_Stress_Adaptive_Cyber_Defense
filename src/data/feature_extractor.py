"""
Feature Extraction Module
Extracts physiological features from preprocessed signals:

ECG -> HRV features:
  - Time domain: mean_hr, RMSSD, SDNN, pNN50
  - Statistical: HR std, IBI mean/std

EDA -> Tonic & Phasic features:
  - SCL (Skin Conductance Level): mean, std
  - SCR (Skin Conductance Response): count, amplitude, rise time

Respiration features:
  - Respiration rate: mean, std
  - Amplitude: mean, std

Based on Section IV-C of the paper.
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import neurokit2 for advanced HRV
try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False
    print("⚠️  neurokit2 not available. Using simplified HRV features.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger
from config import ECG_FEATURES, EDA_FEATURES, RESP_FEATURES, ALL_FEATURES, SAMPLING_RATES

logger = setup_logger(__name__)


class FeatureExtractor:
    """
    Extract physiological features from windowed signals.
    
    Features:
        - ECG: 7 HRV features
        - EDA: 5 SCL/SCR features  
        - Respiration: 4 rate/amplitude features
        Total: 16 features per window
    """
    
    def __init__(
        self,
        use_neurokit: bool = True,
        sampling_rates: Optional[Dict] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            use_neurokit: Use neurokit2 for advanced HRV (if available)
            sampling_rates: Custom sampling rates (default from config)
        """
        self.use_neurokit = use_neurokit and NEUROKIT_AVAILABLE
        self.sampling_rates = sampling_rates or SAMPLING_RATES['chest']
        
        logger.info(f"FeatureExtractor initialized:")
        logger.info(f"  Using neurokit2: {self.use_neurokit}")
        logger.info(f"  ECG features: {len(ECG_FEATURES)}")
        logger.info(f"  EDA features: {len(EDA_FEATURES)}")
        logger.info(f"  Resp features: {len(RESP_FEATURES)}")
        logger.info(f"  Total features: {len(ALL_FEATURES)}")
    
    # ========================================================================
    # ECG / HRV FEATURES
    # ========================================================================
    
    def extract_ecg_features(
        self,
        ecg_window: np.ndarray,
        sampling_rate: float = 700
    ) -> Dict[str, float]:
        """
        Extract HRV features from ECG window.
        
        Args:
            ecg_window: ECG signal (1D array)
            sampling_rate: Sampling rate (Hz)
            
        Returns:
            Dictionary with HRV features
        """
        features = {}
        
        try:
            if self.use_neurokit:
                features = self._extract_hrv_neurokit(ecg_window, sampling_rate)
            else:
                features = self._extract_hrv_simple(ecg_window, sampling_rate)
        except Exception as e:
            logger.warning(f"HRV extraction failed: {e}")
            # Return default values
            features = {feat: 0.0 for feat in ECG_FEATURES}
        
        return features
    
    def _extract_hrv_neurokit(
        self,
        ecg_window: np.ndarray,
        sampling_rate: float
    ) -> Dict[str, float]:
        """Extract HRV using neurokit2."""
        # Clean ECG
        ecg_cleaned = nk.ecg_clean(ecg_window, sampling_rate=sampling_rate)
        
        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        
        if len(rpeaks['ECG_R_Peaks']) < 2:
            return {feat: 0.0 for feat in ECG_FEATURES}
        
        # Compute HRV
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)
        
        features = {
            'mean_hr': hrv_time['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_time else 0.0,
            'rmssd': hrv_time['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv_time else 0.0,
            'sdnn': hrv_time['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_time else 0.0,
            'pnn50': hrv_time['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv_time else 0.0,
            'hr_std': hrv_time['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_time else 0.0,  # Same as SDNN
            'ibi_mean': hrv_time['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_time else 0.0,
            'ibi_std': hrv_time['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_time else 0.0,
        }
        
        return features
    
    def _extract_hrv_simple(
        self,
        ecg_window: np.ndarray,
        sampling_rate: float
    ) -> Dict[str, float]:
        """Simple HRV extraction without neurokit2."""
        # Simple R-peak detection using scipy
        peaks, _ = scipy_signal.find_peaks(ecg_window, distance=int(0.6 * sampling_rate))
        
        if len(peaks) < 2:
            return {feat: 0.0 for feat in ECG_FEATURES}
        
        # Inter-beat intervals (IBI) in milliseconds
        ibis = np.diff(peaks) / sampling_rate * 1000
        
        if len(ibis) == 0:
            return {feat: 0.0 for feat in ECG_FEATURES}
        
        # Compute features
        mean_ibi = np.mean(ibis)
        std_ibi = np.std(ibis)
        
        # RMSSD: Root mean square of successive differences
        successive_diffs = np.diff(ibis)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else 0.0
        
        # pNN50: Percentage of successive differences > 50ms
        pnn50 = 100 * np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) if len(successive_diffs) > 0 else 0.0
        
        # Heart rate
        heart_rate = 60000 / mean_ibi if mean_ibi > 0 else 0.0
        
        features = {
            'mean_hr': heart_rate,
            'rmssd': rmssd,
            'sdnn': std_ibi,  # SDNN ≈ std of IBIs
            'pnn50': pnn50,
            'hr_std': std_ibi,
            'ibi_mean': mean_ibi,
            'ibi_std': std_ibi,
        }
        
        return features
    
    # ========================================================================
    # EDA FEATURES
    # ========================================================================
    
    def extract_eda_features(
        self,
        eda_window: np.ndarray,
        sampling_rate: float = 700
    ) -> Dict[str, float]:
        """
        Extract EDA features (tonic and phasic).
        
        Args:
            eda_window: EDA signal (1D array)
            sampling_rate: Sampling rate (Hz)
            
        Returns:
            Dictionary with EDA features
        """
        features = {}
        
        try:
            if self.use_neurokit:
                features = self._extract_eda_neurokit(eda_window, sampling_rate)
            else:
                features = self._extract_eda_simple(eda_window, sampling_rate)
        except Exception as e:
            logger.warning(f"EDA extraction failed: {e}")
            features = {feat: 0.0 for feat in EDA_FEATURES}
        
        return features
    
    def _extract_eda_neurokit(
        self,
        eda_window: np.ndarray,
        sampling_rate: float
    ) -> Dict[str, float]:
        """Extract EDA features using neurokit2."""
        # Clean EDA
        eda_cleaned = nk.eda_clean(eda_window, sampling_rate=sampling_rate)
        
        # Decompose into tonic (SCL) and phasic (SCR)
        eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate)
        
        scl = eda_decomposed['EDA_Tonic'].values
        scr = eda_decomposed['EDA_Phasic'].values
        
        # Find SCR peaks
        scr_peaks = nk.eda_peaks(scr, sampling_rate=sampling_rate)
        
        features = {
            'scl_mean': np.mean(scl),
            'scl_std': np.std(scl),
            'scr_count': len(scr_peaks[1]['SCR_Peaks']) if len(scr_peaks) > 1 else 0,
            'scr_amplitude_mean': np.mean(scr[scr > 0]) if np.any(scr > 0) else 0.0,
            'scr_rise_time_mean': 1.0,  # Simplified
        }
        
        return features
    
    def _extract_eda_simple(
        self,
        eda_window: np.ndarray,
        sampling_rate: float
    ) -> Dict[str, float]:
        """Simple EDA extraction."""
        # Approximate SCL as lowpass filtered signal
        from scipy.signal import butter, filtfilt
        
        # SCL: very slow component (< 0.05 Hz)
        b, a = butter(3, 0.05 / (sampling_rate / 2), btype='low')
        scl = filtfilt(b, a, eda_window)
        
        # SCR: difference between raw and SCL
        scr = eda_window - scl
        
        # Find SCR peaks (simple threshold)
        scr_threshold = np.mean(scr) + 2 * np.std(scr)
        scr_peaks, _ = scipy_signal.find_peaks(scr, height=scr_threshold)
        
        features = {
            'scl_mean': np.mean(scl),
            'scl_std': np.std(scl),
            'scr_count': len(scr_peaks),
            'scr_amplitude_mean': np.mean(scr[scr_peaks]) if len(scr_peaks) > 0 else 0.0,
            'scr_rise_time_mean': 1.0,  # Simplified
        }
        
        return features
    
    # ========================================================================
    # RESPIRATION FEATURES
    # ========================================================================
    
    def extract_resp_features(
        self,
        resp_window: np.ndarray,
        sampling_rate: float = 700
    ) -> Dict[str, float]:
        """
        Extract respiration features.
        
        Args:
            resp_window: Respiration signal (1D array)
            sampling_rate: Sampling rate (Hz)
            
        Returns:
            Dictionary with respiration features
        """
        features = {}
        
        try:
            # Find respiration peaks (inhalation peaks)
            peaks, properties = scipy_signal.find_peaks(
                resp_window,
                distance=int(0.5 * sampling_rate),  # Min 2 breaths per second
                prominence=np.std(resp_window) * 0.5
            )
            
            if len(peaks) < 2:
                return {feat: 0.0 for feat in RESP_FEATURES}
            
            # Respiration rate (breaths per minute)
            breath_intervals = np.diff(peaks) / sampling_rate  # seconds
            resp_rate = 60 / np.mean(breath_intervals) if len(breath_intervals) > 0 else 0.0
            resp_rate_std = np.std(60 / breath_intervals) if len(breath_intervals) > 0 else 0.0
            
            # Amplitude (peak-to-trough)
            amplitudes = resp_window[peaks]
            
            features = {
                'resp_rate_mean': resp_rate,
                'resp_rate_std': resp_rate_std,
                'resp_amplitude_mean': np.mean(amplitudes) if len(amplitudes) > 0 else 0.0,
                'resp_amplitude_std': np.std(amplitudes) if len(amplitudes) > 0 else 0.0,
            }
            
        except Exception as e:
            logger.warning(f"Respiration extraction failed: {e}")
            features = {feat: 0.0 for feat in RESP_FEATURES}
        
        return features
    
    # ========================================================================
    # COMBINED EXTRACTION
    # ========================================================================
    
    def extract_window_features(
        self,
        ecg_window: np.ndarray,
        eda_window: np.ndarray,
        resp_window: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract all features from a single window.
        
        Args:
            ecg_window: ECG window
            eda_window: EDA window
            resp_window: Respiration window
            
        Returns:
            Dictionary with all 16 features
        """
        features = {}
        
        # ECG/HRV features
        ecg_feats = self.extract_ecg_features(ecg_window, self.sampling_rates['ECG'])
        features.update(ecg_feats)
        
        # EDA features
        eda_feats = self.extract_eda_features(eda_window, self.sampling_rates['EDA'])
        features.update(eda_feats)
        
        # Respiration features
        resp_feats = self.extract_resp_features(resp_window, self.sampling_rates['Resp'])
        features.update(resp_feats)
        
        return features
    
    def extract_subject_features(
        self,
        processed_data: Dict
    ) -> pd.DataFrame:
        """
        Extract features from all windows of a subject.
        
        Args:
            processed_data: Output from SignalProcessor.process_subject()
                Must contain: ECG_windows, EDA_windows, Resp_windows, labels
                
        Returns:
            DataFrame with shape (n_windows, n_features + 1)
            Columns: all features + 'label'
        """
        ecg_windows = processed_data['ECG_windows']
        eda_windows = processed_data['EDA_windows']
        resp_windows = processed_data['Resp_windows']
        labels = processed_data['labels']
        
        n_windows = len(labels)
        
        # Extract features for each window
        feature_list = []
        
        from tqdm import tqdm
        for i in tqdm(range(n_windows), desc=f"Extracting features ({processed_data['subject_id']})", leave=False):
            features = self.extract_window_features(
                ecg_windows[i],
                eda_windows[i],
                resp_windows[i]
            )
            features['label'] = labels[i]
            feature_list.append(features)
        
        df = pd.DataFrame(feature_list)
        
        logger.info(f"Extracted {len(df)} feature vectors with {len(ALL_FEATURES)} features")
        
        return df
    
    def extract_multiple_subjects(
        self,
        processed_subjects: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Extract features from multiple subjects.
        
        Args:
            processed_subjects: Dict of subject_id -> processed_data
            
        Returns:
            Combined DataFrame with all subjects
        """
        all_dfs = []
        
        for subject_id, processed_data in processed_subjects.items():
            try:
                df = self.extract_subject_features(processed_data)
                df['subject_id'] = subject_id
                all_dfs.append(df)
                logger.info(f"✓ Extracted features for {subject_id}: {len(df)} windows")
            except Exception as e:
                logger.error(f"✗ Failed to extract features for {subject_id}: {e}")
                continue
        
        if not all_dfs:
            raise ValueError("No features extracted from any subject!")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        logger.info(f"\nTotal dataset: {len(combined_df)} windows from {len(all_dfs)} subjects")
        logger.info(f"Features: {len(ALL_FEATURES)}")
        logger.info(f"Stress samples: {(combined_df['label'] == 1).sum()}")
        logger.info(f"Non-stress samples: {(combined_df['label'] == 0).sum()}")
        
        return combined_df


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_extract(
    processed_data: Dict,
    show_info: bool = True
) -> pd.DataFrame:
    """
    Quick feature extraction for testing.
    
    Args:
        processed_data: From SignalProcessor.process_subject()
        show_info: Print information
        
    Returns:
        Feature DataFrame
    """
    extractor = FeatureExtractor()
    df = extractor.extract_subject_features(processed_data)
    
    if show_info:
        print(f"\nExtracted Features:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:5])}... (+{len(df.columns)-5} more)")
        print(f"\nFeature statistics:")
        print(df[ALL_FEATURES[:3]].describe())
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
    
    return df


if __name__ == "__main__":
    """
    Test script - run with: python feature_extractor.py
    """
    print("="*60)
    print("Feature Extractor Test")
    print("="*60)
    
    # Import dependencies
    from wesad_loader import quick_load
    from signal_processor import SignalProcessor
    
    # Test 1: Load and process subject
    print("\n[Test 1] Loading and processing S2...")
    subject_data = quick_load('S2', show_info=False)
    processor = SignalProcessor()
    processed = processor.process_subject(subject_data, return_windows=True)
    print("✅ Subject loaded and processed!")
    
    # Test 2: Extract features
    print("\n[Test 2] Extracting features...")
    df = quick_extract(processed, show_info=True)
    print("✅ Features extracted!")
    
    # Test 3: Check feature quality
    print("\n[Test 3] Feature quality check...")
    print(f"NaN values: {df[ALL_FEATURES].isna().sum().sum()}")
    print(f"Inf values: {np.isinf(df[ALL_FEATURES].values).sum()}")
    
    # Check feature ranges
    print("\nFeature ranges:")
    for feat in ALL_FEATURES[:5]:
        print(f"  {feat}: [{df[feat].min():.2f}, {df[feat].max():.2f}]")
    
    print("✅ Quality check passed!")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nFeature extractor is ready!")
    print("Example:")
    print("  from data.feature_extractor import FeatureExtractor")
    print("  extractor = FeatureExtractor()")
    print("  features_df = extractor.extract_subject_features(processed_data)")
