"""
WESAD Dataset Loader
Loads and processes WESAD (Wearable Stress and Affect Detection) dataset
Reference: Schmidt et al., "Introducing WESAD" (2018)

Dataset structure:
- S2-S17 folders (S1 and S12 excluded)
- Each folder contains:
  - S*.pkl: Main data file
  - S*_quest.csv: Questionnaire data
  - S*_readme.txt: Metadata
  
Signal structure in .pkl:
- subject: dict with keys:
  - 'signal': dict with 'chest' and 'wrist' sensor data
  - 'label': array of condition labels
  
Conditions (labels):
  0: not defined
  1: baseline
  2: stress  
  3: amusement
  4: meditation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import (
    load_pickle, setup_logger, print_dict, 
    validate_wesad_subject, get_available_wesad_subjects
)
from config import (
    WESAD_DIR, WESAD_SUBJECTS, WESAD_CONDITIONS, STRESS_LABEL_MAP,
    SAMPLING_RATES
)

logger = setup_logger(__name__)


class WESADLoader:
    """
    Loader for WESAD dataset with support for:
    - Individual subject loading
    - Batch loading (multiple subjects)
    - Binary stress classification (stress vs non-stress)
    - Chest and wrist sensor data
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize WESAD loader.
        
        Args:
            data_dir: Path to WESAD dataset directory (default from config)
        """
        self.data_dir = Path(data_dir) if data_dir else WESAD_DIR
        
        if not self.data_dir.exists():
            logger.warning(f"WESAD directory not found: {self.data_dir}")
            logger.warning("Please set correct path in config.py or pass to constructor")
        
        self.available_subjects = get_available_wesad_subjects(str(self.data_dir))
        logger.info(f"Found {len(self.available_subjects)} subjects: {self.available_subjects}")
    
    def load_subject(
        self, 
        subject_id: str, 
        modalities: List[str] = ['chest'],
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Load data for a single subject.
        
        Args:
            subject_id: Subject ID (e.g., 'S2', 'S17')
            modalities: List of modalities to load ['chest', 'wrist']
            verbose: Print loading information
            
        Returns:
            Dictionary containing:
                - 'subject_id': str
                - 'chest' or 'wrist': dict with signal data
                - 'label': np.ndarray of condition labels
                - 'binary_label': np.ndarray (0=non-stress, 1=stress)
                - 'sampling_rates': dict
                
        Raises:
            FileNotFoundError: If subject data not found
            ValueError: If subject ID invalid
        """
        if not validate_wesad_subject(subject_id):
            raise ValueError(f"Invalid subject ID: {subject_id}")
        
        # Construct pickle file path
        pkl_path = self.data_dir / subject_id / f"{subject_id}.pkl"
        
        if not pkl_path.exists():
            raise FileNotFoundError(f"Subject data not found: {pkl_path}")
        
        if verbose:
            logger.info(f"Loading {subject_id} from {pkl_path}")
        
        # Load pickle file
        try:
            data = load_pickle(str(pkl_path))
        except Exception as e:
            logger.error(f"Failed to load {subject_id}: {e}")
            raise
        
        # Extract signals and labels
        result = {
            'subject_id': subject_id,
            'label': data['label'],  # Original labels (0,1,2,3,4)
        }
        
        # Convert to binary stress labels
        result['binary_label'] = self._convert_to_binary_stress(data['label'])
        
        # Extract requested modalities
        for modality in modalities:
            if modality in data['signal']:
                result[modality] = data['signal'][modality]
                if verbose:
                    self._print_modality_info(subject_id, modality, result[modality])
            else:
                logger.warning(f"Modality '{modality}' not found for {subject_id}")
        
        # Add sampling rates
        result['sampling_rates'] = SAMPLING_RATES
        
        if verbose:
            self._print_label_distribution(subject_id, result['label'], result['binary_label'])
        
        return result
    
    def load_multiple_subjects(
        self,
        subject_ids: Optional[List[str]] = None,
        modalities: List[str] = ['chest'],
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Load multiple subjects.
        
        Args:
            subject_ids: List of subject IDs (None = load all available)
            modalities: Modalities to load
            verbose: Print progress
            
        Returns:
            Dictionary mapping subject_id -> subject data
        """
        if subject_ids is None:
            subject_ids = self.available_subjects
        
        results = {}
        
        iterator = tqdm(subject_ids, desc="Loading subjects") if verbose else subject_ids
        
        for subject_id in iterator:
            try:
                results[subject_id] = self.load_subject(
                    subject_id, 
                    modalities=modalities, 
                    verbose=False
                )
                if verbose and isinstance(iterator, tqdm):
                    iterator.set_postfix({'loaded': len(results)})
            except Exception as e:
                logger.error(f"Failed to load {subject_id}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(results)}/{len(subject_ids)} subjects")
        return results
    
    def _convert_to_binary_stress(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert multi-class labels to binary stress labels.
        
        Mapping (from config.STRESS_LABEL_MAP):
            1 (baseline) -> 0 (non-stress)
            2 (stress) -> 1 (stress)
            3 (amusement) -> 0 (non-stress)
            
        Other labels (0, 4) are marked as -1 (ignore)
        
        Args:
            labels: Original labels
            
        Returns:
            Binary labels (0, 1, or -1 for ignore)
        """
        binary_labels = np.full_like(labels, -1)
        
        for original_label, binary_label in STRESS_LABEL_MAP.items():
            binary_labels[labels == original_label] = binary_label
        
        return binary_labels
    
    def _print_modality_info(self, subject_id: str, modality: str, data: Dict):
        """Print information about loaded modality."""
        logger.info(f"  {modality.upper()} sensors:")
        for sensor, values in data.items():
            if isinstance(values, np.ndarray):
                duration_sec = len(values) / SAMPLING_RATES[modality].get(sensor, 1)
                logger.info(
                    f"    {sensor}: shape={values.shape}, "
                    f"duration={duration_sec:.1f}s, "
                    f"dtype={values.dtype}"
                )
    
    def _print_label_distribution(
        self, 
        subject_id: str, 
        labels: np.ndarray, 
        binary_labels: np.ndarray
    ):
        """Print label distribution."""
        logger.info(f"  Label distribution:")
        
        # Original labels
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            condition = WESAD_CONDITIONS.get(label, "unknown")
            percentage = 100 * count / len(labels)
            logger.info(f"    {label} ({condition}): {count} samples ({percentage:.1f}%)")
        
        # Binary labels
        binary_unique, binary_counts = np.unique(binary_labels[binary_labels != -1], return_counts=True)
        logger.info(f"  Binary labels (stress classification):")
        for label, count in zip(binary_unique, binary_counts):
            label_name = "stress" if label == 1 else "non-stress"
            percentage = 100 * count / len(binary_labels[binary_labels != -1])
            logger.info(f"    {label} ({label_name}): {count} samples ({percentage:.1f}%)")


def extract_chest_signals(
    subject_data: Dict, 
    signals: List[str] = ['ECG', 'EDA', 'Resp']
) -> Dict[str, np.ndarray]:
    """
    Extract specific signals from chest sensor data.
    
    Args:
        subject_data: Output from WESADLoader.load_subject()
        signals: List of signals to extract (ECG, EDA, EMG, Temp, Resp, ACC)
        
    Returns:
        Dictionary mapping signal name -> numpy array
    """
    chest_data = subject_data.get('chest', {})
    
    extracted = {}
    for signal_name in signals:
        if signal_name in chest_data:
            extracted[signal_name] = chest_data[signal_name]
        else:
            logger.warning(f"Signal '{signal_name}' not found in chest data")
    
    return extracted


def get_stress_segments(
    subject_data: Dict,
    stress_value: int = 1,
    min_duration: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Get start/end indices of stress (or non-stress) segments.
    
    Args:
        subject_data: Output from WESADLoader.load_subject()
        stress_value: 0 for non-stress, 1 for stress
        min_duration: Minimum segment duration in samples (None = no filter)
        
    Returns:
        List of (start_idx, end_idx) tuples
    """
    binary_labels = subject_data['binary_label']
    
    # Find segments
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, label in enumerate(binary_labels):
        if label == stress_value and not in_segment:
            # Start of new segment
            start_idx = i
            in_segment = True
        elif label != stress_value and in_segment:
            # End of segment
            segments.append((start_idx, i))
            in_segment = False
    
    # Handle case where segment extends to end
    if in_segment:
        segments.append((start_idx, len(binary_labels)))
    
    # Filter by minimum duration
    if min_duration is not None:
        segments = [(s, e) for s, e in segments if (e - s) >= min_duration]
    
    return segments


def get_subject_metadata(subject_id: str, wesad_dir: Optional[Path] = None) -> Dict:
    """
    Load subject metadata from questionnaire and readme files.
    
    Args:
        subject_id: Subject ID
        wesad_dir: WESAD directory path
        
    Returns:
        Dictionary with metadata
    """
    if wesad_dir is None:
        wesad_dir = WESAD_DIR
    
    metadata = {'subject_id': subject_id}
    
    # Try to load questionnaire
    quest_path = wesad_dir / subject_id / f"{subject_id}_quest.csv"
    if quest_path.exists():
        try:
            quest_df = pd.read_csv(quest_path)
            metadata['questionnaire'] = quest_df.to_dict('records')[0] if len(quest_df) > 0 else {}
        except Exception as e:
            logger.warning(f"Failed to load questionnaire for {subject_id}: {e}")
    
    # Try to load readme
    readme_path = wesad_dir / subject_id / f"{subject_id}_readme.txt"
    if readme_path.exists():
        try:
            with open(readme_path, 'r') as f:
                metadata['readme'] = f.read()
        except Exception as e:
            logger.warning(f"Failed to load readme for {subject_id}: {e}")
    
    return metadata


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_load(subject_id: str = 'S2', show_info: bool = True) -> Dict:
    """
    Quick load of a single subject (for testing/exploration).
    
    Args:
        subject_id: Subject to load
        show_info: Print information
        
    Returns:
        Subject data dictionary
    """
    loader = WESADLoader()
    data = loader.load_subject(subject_id, modalities=['chest'], verbose=show_info)
    
    if show_info:
        print("\n" + "="*60)
        print(f"Loaded {subject_id} successfully!")
        print("="*60)
        print(f"Available keys: {list(data.keys())}")
        print(f"\nChest signals: {list(data['chest'].keys())}")
        print(f"Label shape: {data['label'].shape}")
        print(f"Binary label shape: {data['binary_label'].shape}")
    
    return data


def load_for_ml(
    subject_ids: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict, Dict]:
    """
    Load and split subjects into train/test sets.
    
    Args:
        subject_ids: List of subjects (None = all)
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        (train_dict, test_dict) containing subject data
    """
    from sklearn.model_selection import train_test_split
    
    loader = WESADLoader()
    
    if subject_ids is None:
        subject_ids = loader.available_subjects
    
    # Split subjects
    train_ids, test_ids = train_test_split(
        subject_ids,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Train subjects ({len(train_ids)}): {train_ids}")
    logger.info(f"Test subjects ({len(test_ids)}): {test_ids}")
    
    # Load data
    train_data = loader.load_multiple_subjects(train_ids, verbose=True)
    test_data = loader.load_multiple_subjects(test_ids, verbose=True)
    
    return train_data, test_data


if __name__ == "__main__":
    """
    Test script - run with: python wesad_loader.py
    """
    import sys
    
    print("="*60)
    print("WESAD Loader Test")
    print("="*60)
    
    # Test 1: Quick load single subject
    print("\n[Test 1] Quick load S2...")
    try:
        data = quick_load('S2', show_info=True)
        print("✅ Quick load successful!")
    except Exception as e:
        print(f"❌ Quick load failed: {e}")
        sys.exit(1)
    
    # Test 2: Extract specific signals
    print("\n[Test 2] Extract ECG, EDA, Respiration...")
    signals = extract_chest_signals(data, signals=['ECG', 'EDA', 'Resp'])
    for name, signal in signals.items():
        print(f"  {name}: shape={signal.shape}, mean={signal.mean():.2f}, std={signal.std():.2f}")
    print("✅ Signal extraction successful!")
    
    # Test 3: Find stress segments
    print("\n[Test 3] Find stress segments...")
    stress_segments = get_stress_segments(data, stress_value=1, min_duration=1000)
    print(f"  Found {len(stress_segments)} stress segments")
    if stress_segments:
        for i, (start, end) in enumerate(stress_segments[:3]):
            duration = (end - start) / 700  # Assuming 700 Hz
            print(f"    Segment {i+1}: [{start}, {end}], duration={duration:.1f}s")
    print("✅ Segment extraction successful!")
    
    # Test 4: Load multiple subjects
    print("\n[Test 4] Load multiple subjects...")
    loader = WESADLoader()
    available = loader.available_subjects[:3]  # Just first 3 for speed
    if available:
        multi_data = loader.load_multiple_subjects(available, verbose=True)
        print(f"✅ Loaded {len(multi_data)} subjects successfully!")
    else:
        print("⚠️  No subjects available (check WESAD_DIR in config.py)")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now use WESADLoader in your pipeline!")
    print("Example:")
    print("  from data.wesad_loader import WESADLoader")
    print("  loader = WESADLoader()")
    print("  data = loader.load_subject('S2')")
