"""
Synthetic SOC Environment
Generates synthetic Security Operations Center scenarios for testing.

Creates:
- Task sequences with varying difficulty
- Stress-inducing event patterns
- Analyst performance simulation

Based on Section VI of the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger, set_random_seed
from config import ACTIONS, SOC_CONFIG

logger = setup_logger(__name__)


@dataclass
class SOCTask:
    """
    SOC task representation.
    
    Attributes:
        task_id: Unique identifier
        task_type: Type of task ('alert', 'incident', 'investigation')
        priority: Priority level (1-5)
        complexity: Complexity score (0-1)
        estimated_time: Estimated completion time (minutes)
        arrival_time: When task arrives
        stress_factor: How much stress this task induces (0-1)
    """
    task_id: int
    task_type: str
    priority: int
    complexity: float
    estimated_time: float
    arrival_time: float
    stress_factor: float


class SyntheticSOC:
    """
    Synthetic SOC environment generator.
    
    Generates realistic SOC task sequences with controllable
    stress patterns and workload characteristics.
    """
    
    def __init__(
        self,
        task_types: List[str] = SOC_CONFIG['task_types'],
        seed: int = 42
    ):
        """
        Initialize synthetic SOC.
        
        Args:
            task_types: Available task types
            seed: Random seed
        """
        self.task_types = task_types
        self.seed = seed
        set_random_seed(seed)
        
        self.task_counter = 0
        
        logger.info(f"Synthetic SOC initialized:")
        logger.info(f"  Task types: {task_types}")
    
    # ========================================================================
    # TASK GENERATION
    # ========================================================================
    
    def generate_task(
        self,
        arrival_time: float,
        stress_level: str = 'medium'
    ) -> SOCTask:
        """
        Generate a single SOC task.
        
        Args:
            arrival_time: Time when task arrives
            stress_level: 'low', 'medium', or 'high'
            
        Returns:
            Generated SOC task
        """
        self.task_counter += 1
        
        # Task type
        task_type = np.random.choice(self.task_types)
        
        # Priority (1-5, higher = more urgent)
        if stress_level == 'low':
            priority = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        elif stress_level == 'medium':
            priority = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        else:  # high
            priority = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
        
        # Complexity (0-1)
        if task_type == 'alert':
            complexity = np.random.uniform(0.1, 0.4)
        elif task_type == 'incident':
            complexity = np.random.uniform(0.3, 0.7)
        else:  # investigation
            complexity = np.random.uniform(0.6, 0.9)
        
        # Estimated time (minutes)
        base_time = {
            'alert': 5,
            'incident': 15,
            'investigation': 30
        }[task_type]
        
        estimated_time = base_time * (1 + complexity) * np.random.uniform(0.8, 1.2)
        
        # Stress factor
        stress_factor = 0.2 * priority / 5 + 0.3 * complexity
        stress_factor = np.clip(stress_factor, 0, 1)
        
        task = SOCTask(
            task_id=self.task_counter,
            task_type=task_type,
            priority=priority,
            complexity=complexity,
            estimated_time=estimated_time,
            arrival_time=arrival_time,
            stress_factor=stress_factor
        )
        
        return task
    
    def generate_task_sequence(
        self,
        duration: float,
        arrival_rate: float,
        stress_pattern: str = 'constant'
    ) -> List[SOCTask]:
        """
        Generate sequence of SOC tasks.
        
        Args:
            duration: Total duration (minutes)
            arrival_rate: Average tasks per minute
            stress_pattern: 'constant', 'increasing', 'cyclic', or 'burst'
            
        Returns:
            List of SOC tasks
        """
        tasks = []
        current_time = 0.0
        
        while current_time < duration:
            # Inter-arrival time (Poisson process)
            inter_arrival = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival
            
            if current_time >= duration:
                break
            
            # Determine stress level based on pattern
            if stress_pattern == 'constant':
                stress_level = 'medium'
            
            elif stress_pattern == 'increasing':
                progress = current_time / duration
                if progress < 0.33:
                    stress_level = 'low'
                elif progress < 0.67:
                    stress_level = 'medium'
                else:
                    stress_level = 'high'
            
            elif stress_pattern == 'cyclic':
                # Sine wave pattern
                phase = (current_time / duration) * 2 * np.pi
                intensity = (np.sin(phase) + 1) / 2  # 0-1
                if intensity < 0.33:
                    stress_level = 'low'
                elif intensity < 0.67:
                    stress_level = 'medium'
                else:
                    stress_level = 'high'
            
            elif stress_pattern == 'burst':
                # Random bursts of high stress
                if np.random.rand() < 0.2:
                    stress_level = 'high'
                else:
                    stress_level = np.random.choice(['low', 'medium'])
            
            else:
                stress_level = 'medium'
            
            # Generate task
            task = self.generate_task(current_time, stress_level)
            tasks.append(task)
        
        logger.info(f"Generated {len(tasks)} tasks over {duration:.1f} minutes")
        
        return tasks
    
    # ========================================================================
    # STRESS MAPPING
    # ========================================================================
    
    def map_tasks_to_stress(
        self,
        tasks: List[SOCTask],
        window_size: float = 5.0
    ) -> np.ndarray:
        """
        Map task sequence to stress probability timeline.
        
        Args:
            tasks: List of SOC tasks
            window_size: Time window for aggregation (minutes)
            
        Returns:
            Stress probability array
        """
        if not tasks:
            return np.array([])
        
        # Determine time range
        max_time = max(task.arrival_time for task in tasks)
        n_windows = int(np.ceil(max_time / window_size))
        
        stress_timeline = np.zeros(n_windows)
        
        # Aggregate stress factors in each window
        for task in tasks:
            window_idx = int(task.arrival_time / window_size)
            if window_idx < n_windows:
                # Weight by priority and complexity
                contribution = task.stress_factor * (task.priority / 5)
                stress_timeline[window_idx] += contribution
        
        # Normalize to probabilities
        if stress_timeline.max() > 0:
            stress_timeline = stress_timeline / stress_timeline.max()
        
        # Add some smoothing
        stress_timeline = self._smooth_signal(stress_timeline)
        
        return stress_timeline
    
    def _smooth_signal(self, signal: np.ndarray, window: int = 3) -> np.ndarray:
        """Apply simple moving average smoothing."""
        if len(signal) < window:
            return signal
        
        smoothed = np.convolve(signal, np.ones(window)/window, mode='same')
        return smoothed
    
    # ========================================================================
    # ANALYST SIMULATION
    # ========================================================================
    
    def simulate_analyst_performance(
        self,
        tasks: List[SOCTask],
        stress_levels: np.ndarray,
        baseline_performance: float = 0.9
    ) -> Dict[str, np.ndarray]:
        """
        Simulate analyst performance under stress.
        
        Performance degrades with higher stress levels.
        
        Args:
            tasks: SOC tasks
            stress_levels: Stress probability timeline
            baseline_performance: Baseline performance (0-1)
            
        Returns:
            Dictionary with performance metrics
        """
        n_tasks = len(tasks)
        
        # Initialize arrays
        completion_times = np.zeros(n_tasks)
        error_rates = np.zeros(n_tasks)
        decision_latencies = np.zeros(n_tasks)
        
        for i, task in enumerate(tasks):
            # Get stress at task arrival
            window_idx = int(task.arrival_time / 5.0)
            if window_idx < len(stress_levels):
                stress = stress_levels[window_idx]
            else:
                stress = 0.5
            
            # Performance degradation with stress
            performance = baseline_performance * (1 - 0.3 * stress)
            
            # Completion time (increases with stress)
            completion_times[i] = task.estimated_time * (1 + 0.5 * stress)
            
            # Error rate (increases with stress)
            base_error = 1 - performance
            error_rates[i] = base_error * (1 + stress)
            
            # Decision latency (increases with stress)
            decision_latencies[i] = 0.5 + 1.0 * stress  # seconds
        
        return {
            'completion_times': completion_times,
            'error_rates': error_rates,
            'decision_latencies': decision_latencies,
            'mean_completion_time': np.mean(completion_times),
            'mean_error_rate': np.mean(error_rates),
            'mean_latency': np.mean(decision_latencies)
        }
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    def export_scenario(
        self,
        tasks: List[SOCTask],
        filepath: str
    ) -> None:
        """
        Export scenario to JSON file.
        
        Args:
            tasks: List of tasks
            filepath: Output file path
        """
        scenario_data = {
            'n_tasks': len(tasks),
            'duration': max(task.arrival_time for task in tasks) if tasks else 0,
            'tasks': [
                {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'priority': task.priority,
                    'complexity': task.complexity,
                    'estimated_time': task.estimated_time,
                    'arrival_time': task.arrival_time,
                    'stress_factor': task.stress_factor
                }
                for task in tasks
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(scenario_data, f, indent=2)
        
        logger.info(f"Scenario exported to {filepath}")
    
    def tasks_to_dataframe(self, tasks: List[SOCTask]) -> pd.DataFrame:
        """Convert tasks to pandas DataFrame."""
        return pd.DataFrame([
            {
                'task_id': task.task_id,
                'task_type': task.task_type,
                'priority': task.priority,
                'complexity': task.complexity,
                'estimated_time': task.estimated_time,
                'arrival_time': task.arrival_time,
                'stress_factor': task.stress_factor
            }
            for task in tasks
        ])


if __name__ == "__main__":
    """
    Test script
    """
    print("="*60)
    print("Synthetic SOC Test")
    print("="*60)
    
    # Initialize SOC
    soc = SyntheticSOC()
    
    # Test 1: Generate single task
    print("\n[Test 1] Generate single task...")
    task = soc.generate_task(arrival_time=0.0, stress_level='medium')
    print(f"  Task ID: {task.task_id}")
    print(f"  Type: {task.task_type}")
    print(f"  Priority: {task.priority}")
    print(f"  Complexity: {task.complexity:.2f}")
    print(f"  Stress factor: {task.stress_factor:.2f}")
    print("✅ Task generation working!")
    
    # Test 2: Generate task sequence
    print("\n[Test 2] Generate task sequence...")
    tasks = soc.generate_task_sequence(
        duration=60.0,  # 1 hour
        arrival_rate=0.5,  # 0.5 tasks per minute
        stress_pattern='increasing'
    )
    print(f"  Generated {len(tasks)} tasks")
    print(f"  Duration: {max(t.arrival_time for t in tasks):.1f} minutes")
    print("✅ Task sequence generation working!")
    
    # Test 3: Map tasks to stress
    print("\n[Test 3] Map tasks to stress timeline...")
    stress_timeline = soc.map_tasks_to_stress(tasks, window_size=5.0)
    print(f"  Stress timeline length: {len(stress_timeline)} windows")
    print(f"  Stress range: [{stress_timeline.min():.2f}, {stress_timeline.max():.2f}]")
    print(f"  Mean stress: {stress_timeline.mean():.2f}")
    print("✅ Stress mapping working!")
    
    # Test 4: Simulate analyst performance
    print("\n[Test 4] Simulate analyst performance...")
    performance = soc.simulate_analyst_performance(tasks, stress_timeline)
    print(f"  Mean completion time: {performance['mean_completion_time']:.1f} min")
    print(f"  Mean error rate: {performance['mean_error_rate']:.3f}")
    print(f"  Mean latency: {performance['mean_latency']:.2f} sec")
    print("✅ Performance simulation working!")
    
    # Test 5: Different stress patterns
    print("\n[Test 5] Test different stress patterns...")
    patterns = ['constant', 'increasing', 'cyclic', 'burst']
    
    for pattern in patterns:
        tasks = soc.generate_task_sequence(
            duration=30.0,
            arrival_rate=0.5,
            stress_pattern=pattern
        )
        stress = soc.map_tasks_to_stress(tasks)
        print(f"  {pattern}: {len(tasks)} tasks, stress std={np.std(stress):.3f}")
    
    print("✅ Stress patterns working!")
    
    # Test 6: Export scenario
    print("\n[Test 6] Export scenario...")
    test_tasks = soc.generate_task_sequence(duration=20.0, arrival_rate=0.5)
    soc.export_scenario(test_tasks, '/tmp/test_scenario.json')
    print("✅ Export working!")
    
    # Test 7: DataFrame conversion
    print("\n[Test 7] Convert to DataFrame...")
    df = soc.tasks_to_dataframe(test_tasks)
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print("✅ DataFrame conversion working!")
    
    print("\n" + "="*60)
    print("✅ ALL SYNTHETIC SOC TESTS PASSED!")
    print("="*60)
    
    # Display summary statistics
    print("\nScenario Summary:")
    print(f"  Total tasks: {len(test_tasks)}")
    print(f"  Task types: {df['task_type'].value_counts().to_dict()}")
    print(f"  Priority distribution: {df['priority'].value_counts().to_dict()}")
    print(f"  Mean complexity: {df['complexity'].mean():.2f}")
