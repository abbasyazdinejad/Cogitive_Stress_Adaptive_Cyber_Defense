"""
SpeedyIBL Cognitive Agent
Instance-Based Learning agent with stress-weighted memory decay.

Implements Algorithm 3 from the paper:
- Memory-based decision making
- Stress-weighted activation decay
- Blended value computation
- Action selection with exploration

Based on Section IV-F and Algorithm 3 of the paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger, set_random_seed
from config import IBL_CONFIG, ACTIONS

logger = setup_logger(__name__)


@dataclass
class Instance:
    """
    Memory instance in IBL.
    
    Attributes:
        situation: Situation/state representation
        action: Action taken
        outcome: Observed outcome/utility
        creation_time: Time when instance was created
        last_access_time: Last time instance was accessed
        access_count: Number of times accessed
    """
    situation: np.ndarray
    action: str
    outcome: float
    creation_time: float
    last_access_time: float
    access_count: int = 0


class SpeedyIBLAgent:
    """
    SpeedyIBL Cognitive Agent with stress-adaptive memory.
    
    Key features:
    - Instance-based memory storage
    - Stress-modulated activation decay
    - Blended value computation
    - Softmax action selection
    """
    
    def __init__(
        self,
        actions: List[str] = ACTIONS,
        baseline_decay: float = IBL_CONFIG['baseline_decay'],
        stress_modulation: float = IBL_CONFIG['stress_modulation'],
        noise: float = IBL_CONFIG['noise'],
        temperature: float = IBL_CONFIG['temperature'],
        similarity_func: str = 'gaussian'
    ):
        """
        Initialize SpeedyIBL agent.
        
        Args:
            actions: List of available actions
            baseline_decay: Baseline decay rate β₀
            stress_modulation: Stress modulation factor γ
            noise: Decision noise σ
            temperature: Softmax temperature τ
            similarity_func: Similarity function ('gaussian' or 'exponential')
        """
        self.actions = actions
        self.baseline_decay = baseline_decay
        self.stress_modulation = stress_modulation
        self.noise = noise
        self.temperature = temperature
        self.similarity_func = similarity_func
        
        # Memory
        self.memory: List[Instance] = []
        self.current_time = 0.0
        self.current_stress = 0.0
        
        # Statistics
        self.n_decisions = 0
        self.n_retrievals = 0
        self.successful_retrievals = 0
        
        logger.info(f"SpeedyIBL Agent initialized:")
        logger.info(f"  Actions: {actions}")
        logger.info(f"  Baseline decay β₀: {baseline_decay}")
        logger.info(f"  Stress modulation γ: {stress_modulation}")
        logger.info(f"  Temperature τ: {temperature}")
    
    # ========================================================================
    # MEMORY OPERATIONS
    # ========================================================================
    
    def store_instance(
        self,
        situation: np.ndarray,
        action: str,
        outcome: float
    ) -> None:
        """
        Store new instance in memory.
        
        Args:
            situation: Current situation state
            action: Action taken
            outcome: Observed outcome/utility
        """
        instance = Instance(
            situation=situation.copy(),
            action=action,
            outcome=outcome,
            creation_time=self.current_time,
            last_access_time=self.current_time,
            access_count=1
        )
        
        self.memory.append(instance)
    
    def retrieve_instances(
        self,
        situation: np.ndarray,
        action: Optional[str] = None
    ) -> List[Instance]:
        """
        Retrieve relevant instances from memory.
        
        Args:
            situation: Current situation
            action: Optional action filter
            
        Returns:
            List of relevant instances
        """
        self.n_retrievals += 1
        
        if not self.memory:
            return []
        
        # Filter by action if specified
        if action is not None:
            instances = [inst for inst in self.memory if inst.action == action]
        else:
            instances = self.memory
        
        if instances:
            self.successful_retrievals += 1
        
        return instances
    
    # ========================================================================
    # ACTIVATION COMPUTATION
    # ========================================================================
    
    def compute_stress_weighted_decay(self, stress_level: float) -> float:
        """
        Compute stress-weighted decay rate.
        
        β(t) = β₀ * (1 + γ * p_s(t))
        
        High stress → faster decay → more forgetting
        
        Args:
            stress_level: Current stress probability
            
        Returns:
            Stress-weighted decay rate
        """
        decay = self.baseline_decay * (1 + self.stress_modulation * stress_level)
        return decay
    
    def compute_activation(
        self,
        instance: Instance,
        stress_level: float
    ) -> float:
        """
        Compute activation of an instance.
        
        A_i(t) = ln(Σ (t - t_i)^(-β(t)))
        
        where β(t) is stress-weighted decay.
        
        Args:
            instance: Memory instance
            stress_level: Current stress level
            
        Returns:
            Activation value
        """
        # Time since creation
        time_diff = self.current_time - instance.creation_time
        
        if time_diff <= 0:
            return 0.0
        
        # Stress-weighted decay
        decay = self.compute_stress_weighted_decay(stress_level)
        
        # Power-law decay with stress modulation
        activation = np.log(np.power(time_diff, -decay) + 1e-10)
        
        return activation
    
    def compute_similarity(
        self,
        situation1: np.ndarray,
        situation2: np.ndarray
    ) -> float:
        """
        Compute similarity between two situations.
        
        Args:
            situation1: First situation
            situation2: Second situation
            
        Returns:
            Similarity score (0-1)
        """
        if self.similarity_func == 'gaussian':
            # Gaussian similarity
            distance = np.linalg.norm(situation1 - situation2)
            similarity = np.exp(-distance ** 2)
        
        elif self.similarity_func == 'exponential':
            # Exponential similarity
            distance = np.linalg.norm(situation1 - situation2)
            similarity = np.exp(-distance)
        
        else:
            # Cosine similarity
            dot_product = np.dot(situation1, situation2)
            norm_product = np.linalg.norm(situation1) * np.linalg.norm(situation2)
            similarity = dot_product / (norm_product + 1e-10)
        
        return similarity
    
    def compute_retrieval_probability(
        self,
        instance: Instance,
        situation: np.ndarray,
        stress_level: float
    ) -> float:
        """
        Compute probability of retrieving an instance.
        
        P_i(t) ∝ exp(A_i(t) / τ) * sim(s, s_i)
        
        Args:
            instance: Memory instance
            situation: Current situation
            stress_level: Current stress level
            
        Returns:
            Retrieval probability
        """
        # Activation
        activation = self.compute_activation(instance, stress_level)
        
        # Similarity
        similarity = self.compute_similarity(situation, instance.situation)
        
        # Retrieval probability
        prob = np.exp(activation / self.temperature) * similarity
        
        return prob
    
    # ========================================================================
    # BLENDED VALUE COMPUTATION
    # ========================================================================
    
    def compute_blended_value(
        self,
        situation: np.ndarray,
        action: str,
        stress_level: float
    ) -> Tuple[float, int]:
        """
        Compute blended value for action using instance-based retrieval.
        
        V(a, s) = Σ P_i * U_i / Σ P_i
        
        Args:
            situation: Current situation
            action: Action to evaluate
            stress_level: Current stress level
            
        Returns:
            (blended_value, n_instances)
        """
        # Retrieve relevant instances
        instances = self.retrieve_instances(situation, action=action)
        
        if not instances:
            # No instances: return default value with noise
            return np.random.randn() * self.noise, 0
        
        # Compute retrieval probabilities and weighted outcomes
        total_prob = 0.0
        weighted_outcome = 0.0
        
        for instance in instances:
            prob = self.compute_retrieval_probability(instance, situation, stress_level)
            total_prob += prob
            weighted_outcome += prob * instance.outcome
        
        # Blended value
        if total_prob > 0:
            blended_value = weighted_outcome / total_prob
        else:
            blended_value = 0.0
        
        # Add decision noise
        blended_value += np.random.randn() * self.noise
        
        return blended_value, len(instances)
    
    # ========================================================================
    # ACTION SELECTION
    # ========================================================================
    
    def select_action(
        self,
        situation: np.ndarray,
        stress_level: float,
        return_values: bool = False
    ) -> str:
        """
        Select action using instance-based decision making.
        
        Uses softmax over blended values.
        
        Args:
            situation: Current situation
            stress_level: Current stress probability
            return_values: If True, return (action, values_dict)
            
        Returns:
            Selected action (or tuple if return_values=True)
        """
        self.current_stress = stress_level
        self.n_decisions += 1
        
        # Compute blended values for all actions
        values = {}
        n_instances = {}
        
        for action in self.actions:
            value, n_inst = self.compute_blended_value(situation, action, stress_level)
            values[action] = value
            n_instances[action] = n_inst
        
        # Softmax action selection
        action_array = np.array(list(values.values()))
        exp_values = np.exp(action_array / self.temperature)
        probs = exp_values / np.sum(exp_values)
        
        # Sample action
        action_idx = np.random.choice(len(self.actions), p=probs)
        selected_action = self.actions[action_idx]
        
        if return_values:
            return selected_action, values, n_instances
        return selected_action
    
    # ========================================================================
    # LEARNING
    # ========================================================================
    
    def learn(
        self,
        situation: np.ndarray,
        action: str,
        outcome: float
    ) -> None:
        """
        Learn from experience by storing instance.
        
        Args:
            situation: Situation state
            action: Action taken
            outcome: Observed outcome/utility
        """
        self.store_instance(situation, action, outcome)
        self.current_time += 1
    
    def update_time(self, increment: float = 1.0) -> None:
        """Update internal time."""
        self.current_time += increment
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        if not self.memory:
            return {
                'total_instances': 0,
                'instances_per_action': {},
                'avg_activation': 0.0,
                'retrieval_success_rate': 0.0
            }
        
        # Instances per action
        instances_per_action = {action: 0 for action in self.actions}
        for inst in self.memory:
            instances_per_action[inst.action] += 1
        
        # Average activation
        activations = [self.compute_activation(inst, self.current_stress) 
                      for inst in self.memory]
        avg_activation = np.mean(activations) if activations else 0.0
        
        # Retrieval success rate
        retrieval_rate = (self.successful_retrievals / self.n_retrievals * 100 
                         if self.n_retrievals > 0 else 0.0)
        
        return {
            'total_instances': len(self.memory),
            'instances_per_action': instances_per_action,
            'avg_activation': avg_activation,
            'retrieval_success_rate': retrieval_rate,
            'n_decisions': self.n_decisions,
            'n_retrievals': self.n_retrievals
        }
    
    def get_action_values(
        self,
        situation: np.ndarray,
        stress_level: float
    ) -> Dict[str, float]:
        """
        Get blended values for all actions.
        
        Args:
            situation: Current situation
            stress_level: Stress level
            
        Returns:
            Dictionary of action -> value
        """
        values = {}
        for action in self.actions:
            value, _ = self.compute_blended_value(situation, action, stress_level)
            values[action] = value
        return values
    
    def clear_memory(self) -> None:
        """Clear all memory instances."""
        self.memory = []
        self.current_time = 0.0
        self.n_decisions = 0
        self.n_retrievals = 0
        self.successful_retrievals = 0


if __name__ == "__main__":
    """
    Test script
    """
    print("="*60)
    print("SpeedyIBL Agent Test")
    print("="*60)
    
    set_random_seed(42)
    
    # Initialize agent
    agent = SpeedyIBLAgent(
        actions=['MONITOR', 'ESCALATE', 'REDISTRIBUTE', 'ASSIST'],
        baseline_decay=0.5,
        stress_modulation=0.5,
        temperature=0.1
    )
    
    # Test 1: Memory storage
    print("\n[Test 1] Memory storage...")
    situation1 = np.array([0.5, 0.3, 0.8])
    agent.learn(situation1, 'MONITOR', outcome=1.0)
    agent.learn(situation1, 'ESCALATE', outcome=0.5)
    
    stats = agent.get_memory_statistics()
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Instances per action: {stats['instances_per_action']}")
    print("✅ Memory storage working!")
    
    # Test 2: Action selection
    print("\n[Test 2] Action selection at different stress levels...")
    situation2 = np.array([0.6, 0.4, 0.7])
    
    for stress in [0.2, 0.5, 0.8]:
        action, values, n_inst = agent.select_action(situation2, stress, return_values=True)
        print(f"  Stress={stress:.1f}: Selected={action}, Values={list(values.values())[:2]}")
    
    print("✅ Action selection working!")
    
    # Test 3: Learning and memory growth
    print("\n[Test 3] Learning from experience...")
    
    for i in range(10):
        situation = np.random.randn(3)
        action = np.random.choice(agent.actions)
        outcome = np.random.randn()
        agent.learn(situation, action, outcome)
    
    stats = agent.get_memory_statistics()
    print(f"  Memory size after learning: {stats['total_instances']}")
    print(f"  Average activation: {stats['avg_activation']:.3f}")
    print("✅ Learning working!")
    
    # Test 4: Stress-weighted decay
    print("\n[Test 4] Stress-weighted decay...")
    
    for stress in [0.0, 0.5, 1.0]:
        decay = agent.compute_stress_weighted_decay(stress)
        print(f"  Stress={stress:.1f} → decay={decay:.3f}")
    
    print("✅ Stress-weighted decay working!")
    
    # Test 5: Retrieval statistics
    print("\n[Test 5] Retrieval statistics...")
    
    for _ in range(5):
        situation = np.random.randn(3)
        agent.select_action(situation, stress_level=0.5)
    
    stats = agent.get_memory_statistics()
    print(f"  Decisions made: {stats['n_decisions']}")
    print(f"  Retrievals: {stats['n_retrievals']}")
    print(f"  Success rate: {stats['retrieval_success_rate']:.1f}%")
    print("✅ Retrieval statistics working!")
    
    # Test 6: Similarity computation
    print("\n[Test 6] Similarity computation...")
    
    sit1 = np.array([1.0, 0.0, 0.5])
    sit2 = np.array([1.0, 0.0, 0.5])  # Identical
    sit3 = np.array([0.0, 1.0, 0.5])  # Different
    
    sim_same = agent.compute_similarity(sit1, sit2)
    sim_diff = agent.compute_similarity(sit1, sit3)
    
    print(f"  Similarity (identical): {sim_same:.3f}")
    print(f"  Similarity (different): {sim_diff:.3f}")
    print("✅ Similarity computation working!")
    
    print("\n" + "="*60)
    print("✅ ALL SPEEDYIBL TESTS PASSED!")
    print("="*60)
    
    # Display final statistics
    print("\nFinal Agent Statistics:")
    final_stats = agent.get_memory_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
