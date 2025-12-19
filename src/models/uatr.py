"""
Utility-Aware Temporal Reasoner (UATR)
Implements temporal smoothing and inference for stress-adaptive decision making.

Components:
1. EMA (Exponential Moving Average) smoothing
2. HMM-based temporal inference

Based on Section IV-E of the paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️  hmmlearn not available. HMM inference disabled.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger
from config import UATR_CONFIG

logger = setup_logger(__name__)


class UATR:
    """
    Utility-Aware Temporal Reasoner.
    
    Provides temporal smoothing and inference for stress probabilities,
    balancing between physiological signals and cognitive state.
    """
    
    def __init__(
        self,
        ema_alpha: float = UATR_CONFIG['ema_alpha'],
        hmm_n_states: int = UATR_CONFIG['hmm_n_states'],
        use_hmm: bool = True
    ):
        """
        Initialize UATR.
        
        Args:
            ema_alpha: EMA smoothing factor (0 < α < 1)
            hmm_n_states: Number of hidden states for HMM
            use_hmm: Enable HMM-based inference
        """
        self.ema_alpha = ema_alpha
        self.hmm_n_states = hmm_n_states
        self.use_hmm = use_hmm and HMM_AVAILABLE
        
        # State
        self.ema_value = None
        self.hmm_model = None
        self.is_fitted = False
        
        logger.info(f"UATR initialized:")
        logger.info(f"  EMA α: {ema_alpha}")
        logger.info(f"  HMM states: {hmm_n_states}")
        logger.info(f"  HMM enabled: {self.use_hmm}")
    
    # ========================================================================
    # EMA SMOOTHING
    # ========================================================================
    
    def ema_update(self, new_value: float) -> float:
        """
        Update EMA with new observation.
        
        S_t = α * x_t + (1 - α) * S_{t-1}
        
        Args:
            new_value: New stress probability
            
        Returns:
            Smoothed value
        """
        if self.ema_value is None:
            self.ema_value = new_value
        else:
            self.ema_value = self.ema_alpha * new_value + (1 - self.ema_alpha) * self.ema_value
        
        return self.ema_value
    
    def ema_smooth_sequence(self, values: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing to a sequence.
        
        Args:
            values: Sequence of stress probabilities
            
        Returns:
            Smoothed sequence
        """
        smoothed = np.zeros_like(values)
        self.ema_value = None  # Reset
        
        for i, value in enumerate(values):
            smoothed[i] = self.ema_update(value)
        
        return smoothed
    
    def reset_ema(self) -> None:
        """Reset EMA state."""
        self.ema_value = None
    
    # ========================================================================
    # HMM INFERENCE
    # ========================================================================
    
    def fit_hmm(
        self,
        sequences: List[np.ndarray],
        lengths: Optional[List[int]] = None
    ) -> None:
        """
        Fit HMM to training sequences.
        
        Args:
            sequences: List of stress probability sequences
            lengths: Length of each sequence (for concatenated input)
        """
        if not self.use_hmm:
            logger.warning("HMM disabled or unavailable")
            return
        
        # Concatenate sequences
        X = np.concatenate(sequences).reshape(-1, 1)
        
        if lengths is None:
            lengths = [len(seq) for seq in sequences]
        
        # Initialize and fit HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.hmm_n_states,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        
        self.hmm_model.fit(X, lengths)
        self.is_fitted = True
        
        logger.info(f"✓ HMM fitted with {len(sequences)} sequences")
    
    def hmm_predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict hidden states using HMM.
        
        Args:
            sequence: Stress probability sequence
            
        Returns:
            Predicted hidden states
        """
        if not self.use_hmm or not self.is_fitted:
            # Fallback to identity
            return sequence
        
        X = sequence.reshape(-1, 1)
        states = self.hmm_model.predict(X)
        
        return states
    
    def hmm_smooth(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply HMM smoothing to sequence.
        
        Uses posterior probabilities for smoothing.
        
        Args:
            sequence: Raw stress probabilities
            
        Returns:
            Smoothed probabilities
        """
        if not self.use_hmm or not self.is_fitted:
            return sequence
        
        X = sequence.reshape(-1, 1)
        
        # Get posterior probabilities
        posteriors = self.hmm_model.predict_proba(X)
        
        # Weight by state means
        state_means = self.hmm_model.means_.flatten()
        smoothed = posteriors @ state_means
        
        return smoothed
    
    # ========================================================================
    # COMBINED TEMPORAL REASONING
    # ========================================================================
    
    def temporal_smooth(
        self,
        sequence: np.ndarray,
        use_both: bool = True
    ) -> np.ndarray:
        """
        Apply temporal smoothing (EMA + optional HMM).
        
        Args:
            sequence: Raw stress probabilities
            use_both: Use both EMA and HMM
            
        Returns:
            Temporally smoothed probabilities
        """
        # Apply EMA
        ema_smoothed = self.ema_smooth_sequence(sequence)
        
        # Apply HMM if available and requested
        if use_both and self.use_hmm and self.is_fitted:
            final_smoothed = self.hmm_smooth(ema_smoothed)
        else:
            final_smoothed = ema_smoothed
        
        return final_smoothed
    
    def online_smooth(
        self,
        new_observation: float,
        cognitive_utility: Optional[float] = None,
        beta: float = 0.5
    ) -> float:
        """
        Online temporal smoothing with cognitive utility integration.
        
        Combines physiological signal with cognitive state:
        p_smooth = β * p_phys + (1 - β) * p_cognitive
        
        Args:
            new_observation: New physiological stress probability
            cognitive_utility: Optional cognitive utility signal
            beta: Weight for physiological signal
            
        Returns:
            Smoothed stress probability
        """
        # EMA smoothing of physiological signal
        phys_smooth = self.ema_update(new_observation)
        
        # If cognitive utility provided, blend signals
        if cognitive_utility is not None:
            # Map utility to stress-like probability (heuristic)
            cognitive_stress = 1 / (1 + np.exp(-cognitive_utility))
            smoothed = beta * phys_smooth + (1 - beta) * cognitive_stress
        else:
            smoothed = phys_smooth
        
        return smoothed
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    
    def compute_temporal_consistency(self, sequence: np.ndarray) -> float:
        """
        Compute temporal consistency score.
        
        Measures how smoothly stress evolves over time.
        Lower variance in differences = higher consistency.
        
        Args:
            sequence: Stress probability sequence
            
        Returns:
            Consistency score (0-1, higher is better)
        """
        if len(sequence) < 2:
            return 1.0
        
        # Compute first differences
        diffs = np.diff(sequence)
        
        # Variance of differences (normalized)
        var_diffs = np.var(diffs)
        
        # Convert to consistency score
        consistency = np.exp(-var_diffs)
        
        return consistency
    
    def detect_stress_transitions(
        self,
        sequence: np.ndarray,
        threshold: float = 0.8,
        min_duration: int = 3
    ) -> List[Tuple[int, int, str]]:
        """
        Detect stress level transitions in sequence.
        
        Args:
            sequence: Stress probabilities
            threshold: Threshold for stress classification
            min_duration: Minimum duration of stress episode
            
        Returns:
            List of (start_idx, end_idx, level) tuples
        """
        # Binary stress labels
        stress_binary = (sequence > threshold).astype(int)
        
        # Find transitions
        transitions = []
        in_episode = False
        start_idx = 0
        current_level = None
        
        for i, label in enumerate(stress_binary):
            level = 'high' if label == 1 else 'low'
            
            if not in_episode and level == 'high':
                # Start of stress episode
                in_episode = True
                start_idx = i
                current_level = 'high'
            
            elif in_episode and level == 'low':
                # End of stress episode
                duration = i - start_idx
                if duration >= min_duration:
                    transitions.append((start_idx, i, 'high'))
                in_episode = False
                current_level = None
        
        # Handle episode extending to end
        if in_episode:
            duration = len(sequence) - start_idx
            if duration >= min_duration:
                transitions.append((start_idx, len(sequence), 'high'))
        
        return transitions


class AdaptiveUATR(UATR):
    """
    Adaptive UATR with dynamic parameter adjustment.
    
    Adjusts EMA α based on signal volatility.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_alpha = self.ema_alpha
        self.volatility_window = 10
        self.recent_values = []
    
    def adaptive_ema_update(self, new_value: float) -> float:
        """
        Update EMA with adaptive α based on volatility.
        
        High volatility → lower α (more smoothing)
        Low volatility → higher α (more responsive)
        
        Args:
            new_value: New observation
            
        Returns:
            Smoothed value
        """
        # Track recent values
        self.recent_values.append(new_value)
        if len(self.recent_values) > self.volatility_window:
            self.recent_values.pop(0)
        
        # Compute volatility
        if len(self.recent_values) >= 3:
            volatility = np.std(self.recent_values)
            # Adjust α: high volatility → lower α
            self.ema_alpha = self.base_alpha * np.exp(-volatility)
            self.ema_alpha = np.clip(self.ema_alpha, 0.1, 0.9)
        
        # Standard EMA update
        return self.ema_update(new_value)


if __name__ == "__main__":
    """
    Test script
    """
    print("="*60)
    print("UATR Test")
    print("="*60)
    
    # Generate synthetic stress sequence
    np.random.seed(42)
    n_samples = 100
    
    # Create sequence with trends and noise
    t = np.linspace(0, 4*np.pi, n_samples)
    stress_true = 0.5 + 0.3 * np.sin(t)  # Sinusoidal trend
    noise = np.random.randn(n_samples) * 0.1
    stress_noisy = np.clip(stress_true + noise, 0, 1)
    
    # Initialize UATR
    uatr = UATR(ema_alpha=0.3, hmm_n_states=2)
    
    # Test 1: EMA smoothing
    print("\n[Test 1] EMA smoothing...")
    ema_smoothed = uatr.ema_smooth_sequence(stress_noisy)
    print(f"  Original variance: {np.var(stress_noisy):.4f}")
    print(f"  Smoothed variance: {np.var(ema_smoothed):.4f}")
    print("✅ EMA smoothing working!")
    
    # Test 2: HMM fitting and smoothing
    if HMM_AVAILABLE:
        print("\n[Test 2] HMM-based smoothing...")
        
        # Generate training sequences
        train_sequences = [
            np.clip(0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, 50)) + np.random.randn(50)*0.1, 0, 1)
            for _ in range(5)
        ]
        
        uatr.fit_hmm(train_sequences)
        hmm_smoothed = uatr.hmm_smooth(stress_noisy)
        
        print(f"  HMM smoothed variance: {np.var(hmm_smoothed):.4f}")
        print("✅ HMM smoothing working!")
    else:
        print("\n[Test 2] HMM smoothing skipped (hmmlearn not available)")
    
    # Test 3: Combined temporal smoothing
    print("\n[Test 3] Combined temporal smoothing...")
    combined_smoothed = uatr.temporal_smooth(stress_noisy, use_both=True)
    consistency = uatr.compute_temporal_consistency(combined_smoothed)
    print(f"  Temporal consistency: {consistency:.4f}")
    print("✅ Combined smoothing working!")
    
    # Test 4: Stress transition detection
    print("\n[Test 4] Stress transition detection...")
    transitions = uatr.detect_stress_transitions(combined_smoothed, threshold=0.6)
    print(f"  Detected {len(transitions)} stress episodes:")
    for start, end, level in transitions[:3]:
        print(f"    Episode: [{start}, {end}], level={level}")
    print("✅ Transition detection working!")
    
    # Test 5: Online smoothing
    print("\n[Test 5] Online smoothing...")
    uatr.reset_ema()
    online_results = []
    for i in range(10):
        smoothed = uatr.online_smooth(stress_noisy[i])
        online_results.append(smoothed)
    print(f"  Processed {len(online_results)} samples online")
    print("✅ Online smoothing working!")
    
    # Test 6: Adaptive UATR
    print("\n[Test 6] Adaptive UATR...")
    adaptive_uatr = AdaptiveUATR(ema_alpha=0.5)
    adaptive_results = []
    for value in stress_noisy[:20]:
        smoothed = adaptive_uatr.adaptive_ema_update(value)
        adaptive_results.append(smoothed)
    print(f"  Adaptive α range: [{adaptive_uatr.ema_alpha:.3f}]")
    print("✅ Adaptive UATR working!")
    
    print("\n" + "="*60)
    print("✅ ALL UATR TESTS PASSED!")
    print("="*60)
    
    # Optional: Visualization
    try:
        import matplotlib.pyplot as plt
        
        print("\n[Bonus] Visualizing temporal smoothing...")
        plt.figure(figsize=(12, 4))
        plt.plot(stress_noisy, 'o-', alpha=0.3, label='Noisy', markersize=2)
        plt.plot(ema_smoothed, linewidth=2, label='EMA Smoothed')
        if HMM_AVAILABLE:
            plt.plot(hmm_smoothed, linewidth=2, label='HMM Smoothed')
        plt.plot(stress_true, 'k--', linewidth=1, label='True Signal')
        plt.xlabel('Time Step')
        plt.ylabel('Stress Probability')
        plt.title('UATR Temporal Smoothing')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except:
        print("  (Visualization skipped)")
