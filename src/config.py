"""
Configuration file for Stress-Adaptive Defense Framework
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset directories
WESAD_DIR = DATA_DIR / "raw" / "wesad"
CICIDS_DIR = DATA_DIR / "raw" / "cicids"

# Output directories
OUTPUT_DIR = RESULTS_DIR
FIGURES_DIR = RESULTS_DIR / "figures"

# ============================================================================
# WESAD DATASET CONFIGURATION
# ============================================================================

# Available subjects (excluding S1 and S12 as per dataset notes)
WESAD_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                   'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

# WESAD experimental conditions/labels
WESAD_CONDITIONS = {
    1: 'baseline',
    2: 'stress',
    3: 'amusement',
    4: 'meditation',
    5: 'not_defined_1',
    6: 'not_defined_2',
    7: 'not_defined_3',
}

# Sampling rates (Hz)
SAMPLING_RATES = {
    'chest': {
        'ACC': 700,
        'ECG': 700,
        'EDA': 700,
        'EMG': 700,
        'Resp': 700,
        'Temp': 700,
    },
    'wrist': {
        'ACC': 32,
        'BVP': 64,
        'EDA': 4,
        'TEMP': 4,
    }
}

# Label mapping (from WESAD paper)
STRESS_LABEL_MAP = {
    1: 0,  # baseline -> non-stress
    2: 1,  # stress -> stress
    3: 0,  # amusement -> non-stress
    4: -1, # meditation -> ignore
    5: -1, # ignore
    6: -1, # ignore
    7: -1, # ignore
}

# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

# Window parameters
WINDOW_SIZE = 60  # seconds
WINDOW_OVERLAP = 0.5  # 50% overlap
WINDOW_STEP = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))  # 30 seconds

# Filter parameters
LOWPASS_CUTOFF = 5.0  # Hz
FILTER_ORDER = 4
FILTER_TYPE = 'butterworth'  # Type of filter to use

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

# ECG/HRV features
ECG_FEATURES = [
    'mean_hr',
    'RMSSD',
    'SDNN',
    'pNN50',
    'hr_std',
    'ibi_mean',
    'ibi_std'
]

# EDA features
EDA_FEATURES = [
    'scl_mean',
    'scl_std',
    'scr_count',
    'scr_amplitude_mean',
    'scr_rise_time_mean'
]

# Respiration features
RESP_FEATURES = [
    'resp_rate_mean',
    'resp_rate_std',
    'resp_amplitude_mean',
    'resp_amplitude_std'
]

# All features
ALL_FEATURES = ECG_FEATURES + EDA_FEATURES + RESP_FEATURES

# ============================================================================
# MACHINE LEARNING MODELS
# ============================================================================

# DNN Configuration
DNN_CONFIG = {
    'hidden_layers': [128, 64],
    'activation': 'relu',
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
}

# XGBoost Configuration
XGBOOST_CONFIG = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
}

# ============================================================================
# COGNITIVE MODELS
# ============================================================================

# Instance-Based Learning (IBL) Configuration
IBL_CONFIG = {
    'baseline_decay': 0.5,       # β₀
    'stress_modulation': 0.5,    # γ
    'noise': 0.1,                # σ
    'temperature': 0.1,          # τ
}

# UATR Configuration
UATR_CONFIG = {
    'ema_alpha': 0.3,      # EMA smoothing factor
    'hmm_n_states': 2,     # Number of HMM states
}

# ============================================================================
# CPS LAYER CONFIGURATION
# ============================================================================

# Actions
ACTIONS = ['MONITOR', 'ESCALATE', 'REDISTRIBUTE', 'ASSIST']

# Lambda values for each action (action-specific scaling factors)
LAMBDA_ACTIONS = {
    'MONITOR': 0.2,
    'ESCALATE': 0.1,
    'REDISTRIBUTE': 0.5,
    'ASSIST': 0.8,
}

# CPS modulation clip range
CPS_MODULATION_CLIP = (-5.0, 5.0)

# ============================================================================
# SOC CONFIGURATION
# ============================================================================

SOC_CONFIG = {
    'task_types': ['alert', 'incident', 'investigation'],
    'arrival_rate': 0.5,  # tasks per minute
    'duration': 120.0,    # minutes
}

# ============================================================================
# CICIDS CONFIGURATION
# ============================================================================

CICIDS_CONFIG = {
    'data_dir': CICIDS_DIR,
    'attack_types': [
        'BENIGN',
        'DoS',
        'DDoS',
        'Web Attack',
        'Infiltration',
        'Brute Force',
        'Bot',
        'PortScan'
    ],
    'sample_rate': 0.01,  # sampling rate for events
}

# ============================================================================
# RANDOM SEED
# ============================================================================

RANDOM_SEED = 42
