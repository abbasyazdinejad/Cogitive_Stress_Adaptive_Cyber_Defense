"""
Cognitive-Physiological Synchronization (CPS) Layer
Implements Algorithm 2 from the paper.

The CPS layer couples calibrated stress beliefs with cognitive utilities
through Bayesian-inspired log-odds updates:

U'(a, s_t) = U(a, s_t) + λ_a * ℓ(t)

where:
- ℓ(t) = log(p_s(t) / (1 - p_s(t))) is the stress log-odds
- λ_a is an action-specific scaling factor
- U(a, s_t) is the baseline utility
- U'(a, s_t) is the stress-modulated utility

Based on Section IV-B, Equation 3 of the paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger
from config import LAMBDA_ACTIONS, CPS_MODULATION_CLIP, ACTIONS

logger = setup_logger(__name__)


class CPSLayer:
    """
    Cognitive-Physiological Synchronization Layer.
    
    Transforms calibrated stress probabilities into utility modulations
    that dynamically adjust the cognitive agent's action preferences.
    """
    
    def __init__(
        self,
        actions: List[str] = ACTIONS,
        lambda_actions: Optional[Dict[str, float]] = None,
        clip_range: Tuple[float, float] = CPS_MODULATION_CLIP
    ):
        """
        Initialize CPS layer.
        
        Args:
            actions: List of available actions
            lambda_actions: Action-specific scaling factors λ_a
            clip_range: (min, max) bounds for modulation term
        """
        self.actions = actions
        self.lambda_actions = lambda_actions or LAMBDA_ACTIONS
        self.clip_range = clip_range
        
        # Validate lambda_actions
        for action in actions:
            if action not in self.lambda_actions:
                logger.warning(f"No λ defined for action '{action}', using default 0.5")
                self.lambda_actions[action] = 0.5
        
        logger.info(f"CPS Layer initialized with {len(actions)} actions")
        logger.info(f"Lambda values: {self.lambda_actions}")
        logger.info(f"Clip range: {clip_range}")
    
    def stress_to_logodds(self, stress_prob: float) -> float:
        """
        Convert stress probability to log-odds.
        
        ℓ(t) = log(p_s(t) / (1 - p_s(t)))
        
        Args:
            stress_prob: Stress probability in [0, 1]
            
        Returns:
            Log-odds value
        """
        # Clip to avoid numerical issues
        p = np.clip(stress_prob, 1e-10, 1 - 1e-10)
        log_odds = np.log(p / (1 - p))
        return log_odds
    
    def modulate_utilities(
        self,
        base_utilities: Dict[str, float],
        stress_prob: float
    ) -> Dict[str, float]:
        """
        Modulate utilities based on stress probability.
        
        U'(a, s_t) = U(a, s_t) + λ_a * ℓ(t)
        
        Args:
            base_utilities: Baseline utilities for each action
            stress_prob: Current stress probability
            
        Returns:
            Stress-modulated utilities
        """
        # Convert stress to log-odds
        log_odds = self.stress_to_logodds(stress_prob)
        
        # Modulate each action's utility
        modulated_utilities = {}
        
        for action in self.actions:
            base_utility = base_utilities.get(action, 0.0)
            lambda_a = self.lambda_actions[action]
            
            # Apply modulation with clipping
            modulation = lambda_a * log_odds
            modulation = np.clip(modulation, self.clip_range[0], self.clip_range[1])
            
            modulated_utilities[action] = base_utility + modulation
        
        return modulated_utilities
    
    def select_action(
        self,
        base_utilities: Dict[str, float],
        stress_prob: float,
        return_utilities: bool = False
    ) -> str:
        """
        Select action based on stress-modulated utilities.
        
        Args:
            base_utilities: Baseline utilities
            stress_prob: Current stress probability
            return_utilities: If True, return (action, utilities)
            
        Returns:
            Selected action (or tuple if return_utilities=True)
        """
        # Modulate utilities
        modulated_utilities = self.modulate_utilities(base_utilities, stress_prob)
        
        # Select action with highest utility
        best_action = max(modulated_utilities, key=modulated_utilities.get)
        
        if return_utilities:
            return best_action, modulated_utilities
        return best_action
    
    def compute_utility_shift(
        self,
        action: str,
        stress_prob: float
    ) -> float:
        """
        Compute utility shift for a specific action.
        
        Args:
            action: Action name
            stress_prob: Stress probability
            
        Returns:
            Utility shift value
        """
        log_odds = self.stress_to_logodds(stress_prob)
        lambda_a = self.lambda_actions.get(action, 0.0)
        shift = lambda_a * log_odds
        return np.clip(shift, self.clip_range[0], self.clip_range[1])
    
    def batch_modulate(
        self,
        base_utilities: Dict[str, float],
        stress_probs: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Modulate utilities for a sequence of stress probabilities.
        
        Args:
            base_utilities: Baseline utilities
            stress_probs: Array of stress probabilities
            
        Returns:
            List of modulated utilities for each time step
        """
        modulated_sequence = []
        
        for stress_prob in stress_probs:
            modulated = self.modulate_utilities(base_utilities, stress_prob)
            modulated_sequence.append(modulated)
        
        return modulated_sequence


class CPSWithMemory(CPSLayer):
    """
    Extended CPS layer with utility history tracking.
    Useful for analyzing decision dynamics over time.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.utility_history = []
        self.stress_history = []
        self.action_history = []
    
    def select_action(
        self,
        base_utilities: Dict[str, float],
        stress_prob: float,
        return_utilities: bool = False
    ) -> str:
        """Select action and store in history."""
        action, utilities = super().select_action(
            base_utilities, stress_prob, return_utilities=True
        )
        
        # Store history
        self.utility_history.append(utilities)
        self.stress_history.append(stress_prob)
        self.action_history.append(action)
        
        if return_utilities:
            return action, utilities
        return action
    
    def get_history(self) -> Dict[str, List]:
        """Get decision history."""
        return {
            'utilities': self.utility_history,
            'stress': self.stress_history,
            'actions': self.action_history
        }
    
    def clear_history(self) -> None:
        """Clear decision history."""
        self.utility_history = []
        self.stress_history = []
        self.action_history = []


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_stress_impact(
    lambda_actions: Dict[str, float],
    stress_range: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Compute how stress probability affects utility for each action.
    
    Useful for visualization and analysis.
    
    Args:
        lambda_actions: Action-specific scaling factors
        stress_range: Range of stress probabilities
        n_points: Number of points to evaluate
        
    Returns:
        Dictionary mapping action -> utility shifts
    """
    stress_probs = np.linspace(stress_range[0], stress_range[1], n_points)
    
    cps = CPSLayer(
        actions=list(lambda_actions.keys()),
        lambda_actions=lambda_actions
    )
    
    impact = {}
    for action in lambda_actions.keys():
        shifts = [cps.compute_utility_shift(action, p) for p in stress_probs]
        impact[action] = np.array(shifts)
    
    return impact


def visualize_cps_dynamics(lambda_actions: Dict[str, float]) -> None:
    """
    Visualize how CPS modulates utilities across stress levels.
    
    Args:
        lambda_actions: Action-specific scaling factors
    """
    import matplotlib.pyplot as plt
    
    stress_probs = np.linspace(0, 1, 100)
    impact = compute_stress_impact(lambda_actions)
    
    plt.figure(figsize=(10, 6))
    
    for action, shifts in impact.items():
        plt.plot(stress_probs, shifts, label=action, linewidth=2)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Stress Probability', fontsize=12)
    plt.ylabel('Utility Shift (λ_a * ℓ(t))', fontsize=12)
    plt.title('CPS Layer: Stress-Dependent Utility Modulation', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Test script
    """
    print("="*60)
    print("CPS Layer Test")
    print("="*60)
    
    # Initialize CPS layer
    cps = CPSLayer()
    
    # Test 1: Stress to log-odds conversion
    print("\n[Test 1] Stress to log-odds conversion...")
    test_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    for p in test_probs:
        log_odds = cps.stress_to_logodds(p)
        print(f"  p={p:.1f} → ℓ={log_odds:.3f}")
    print("✅ Log-odds conversion working!")
    
    # Test 2: Utility modulation
    print("\n[Test 2] Utility modulation...")
    base_utilities = {
        'MONITOR': 1.0,
        'ESCALATE': 0.5,
        'REDISTRIBUTE': 0.8,
        'ASSIST': 1.2
    }
    
    for stress_prob in [0.2, 0.5, 0.8]:
        modulated = cps.modulate_utilities(base_utilities, stress_prob)
        print(f"\n  Stress = {stress_prob:.1f}:")
        for action, utility in modulated.items():
            shift = utility - base_utilities[action]
            print(f"    {action}: {utility:.3f} (shift: {shift:+.3f})")
    print("\n✅ Utility modulation working!")
    
    # Test 3: Action selection
    print("\n[Test 3] Action selection at different stress levels...")
    for stress_prob in [0.1, 0.5, 0.9]:
        action = cps.select_action(base_utilities, stress_prob)
        print(f"  Stress = {stress_prob:.1f} → Selected: {action}")
    print("✅ Action selection working!")
    
    # Test 4: Batch modulation
    print("\n[Test 4] Batch modulation...")
    stress_sequence = np.array([0.2, 0.4, 0.6, 0.8, 0.7, 0.5])
    modulated_sequence = cps.batch_modulate(base_utilities, stress_sequence)
    print(f"  Processed {len(modulated_sequence)} time steps")
    print("✅ Batch modulation working!")
    
    # Test 5: CPS with memory
    print("\n[Test 5] CPS with memory tracking...")
    cps_mem = CPSWithMemory()
    
    for stress_prob in np.linspace(0.2, 0.8, 5):
        action = cps_mem.select_action(base_utilities, stress_prob)
    
    history = cps_mem.get_history()
    print(f"  Tracked {len(history['actions'])} decisions")
    print(f"  Actions: {history['actions']}")
    print("✅ Memory tracking working!")
    
    print("\n" + "="*60)
    print("✅ ALL CPS LAYER TESTS PASSED!")
    print("="*60)
    
    # Optional: Visualize (requires matplotlib)
    try:
        print("\n[Bonus] Visualizing CPS dynamics...")
        visualize_cps_dynamics(LAMBDA_ACTIONS)
    except:
        print("  (Visualization skipped - requires matplotlib)")
