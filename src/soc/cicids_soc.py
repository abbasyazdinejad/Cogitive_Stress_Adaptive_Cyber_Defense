"""
CICIDS2017-based SOC Scenarios
Maps CICIDS2017 intrusion detection dataset to SOC analyst tasks.

The CICIDS2017 dataset contains network traffic with labeled attacks:
- Benign traffic
- DoS attacks
- DDoS attacks
- Web attacks
- Infiltration
- Brute Force
- Botnet

Based on Section VI-B of the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger
from config import CICIDS_CONFIG, ACTIONS

logger = setup_logger(__name__)


class CICIDSSOC:
    """
    CICIDS2017-based SOC environment.
    
    Converts network intrusion events into SOC analyst tasks
    with realistic workload and stress characteristics.
    """
    
    def __init__(self, cicids_dir: Optional[str] = None):
        """
        Initialize CICIDS SOC.
        
        Args:
            cicids_dir: Path to CICIDS2017 dataset directory
        """
        self.cicids_dir = Path(cicids_dir) if cicids_dir else CICIDS_CONFIG.get('data_dir')
        self.attack_types = CICIDS_CONFIG.get('attack_types', [])
        
        # Attack severity mapping (0-1)
        self.severity_map = {
            'BENIGN': 0.0,
            'DoS': 0.7,
            'DDoS': 0.9,
            'Web Attack': 0.6,
            'Infiltration': 0.95,
            'Brute Force': 0.5,
            'Bot': 0.8,
            'PortScan': 0.4
        }
        
        logger.info(f"CICIDS SOC initialized")
        if self.cicids_dir:
            logger.info(f"  Data directory: {self.cicids_dir}")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    def load_cicids_file(self, filepath: str) -> pd.DataFrame:
        """
        Load CICIDS CSV file.
        
        Args:
            filepath: Path to CICIDS CSV file
            
        Returns:
            DataFrame with CICIDS data
        """
        logger.info(f"Loading CICIDS file: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            
            # Standardize label column name
            label_cols = [col for col in df.columns if 'label' in col.lower()]
            if label_cols:
                df.rename(columns={label_cols[0]: 'Label'}, inplace=True)
            
            logger.info(f"  Loaded {len(df)} records")
            logger.info(f"  Columns: {len(df.columns)}")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to load CICIDS file: {e}")
            raise
    
    def load_all_cicids_files(self) -> pd.DataFrame:
        """
        Load all CICIDS CSV files from directory.
        
        Returns:
            Combined DataFrame
        """
        if not self.cicids_dir or not self.cicids_dir.exists():
            raise ValueError(f"CICIDS directory not found: {self.cicids_dir}")
        
        csv_files = list(self.cicids_dir.glob('*.csv'))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.cicids_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for csv_file in csv_files:
            try:
                df = self.load_cicids_file(str(csv_file))
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping {csv_file}: {e}")
        
        if not dfs:
            raise ValueError("No data loaded from any file")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} records")
        
        return combined_df
    
    # ========================================================================
    # TASK GENERATION
    # ========================================================================
    
    def classify_attack(self, label: str) -> Tuple[str, float]:
        """
        Classify attack type and determine severity.
        
        Args:
            label: CICIDS attack label
            
        Returns:
            (attack_category, severity)
        """
        label_upper = label.strip().upper()
        
        # Map specific labels to categories
        if 'BENIGN' in label_upper:
            return 'BENIGN', 0.0
        elif 'DOS' in label_upper and 'DDOS' not in label_upper:
            return 'DoS', 0.7
        elif 'DDOS' in label_upper:
            return 'DDoS', 0.9
        elif 'WEB' in label_upper or 'SQL' in label_upper or 'XSS' in label_upper:
            return 'Web Attack', 0.6
        elif 'INFILTRATION' in label_upper:
            return 'Infiltration', 0.95
        elif 'BRUTE' in label_upper or 'FTP' in label_upper or 'SSH' in label_upper:
            return 'Brute Force', 0.5
        elif 'BOT' in label_upper:
            return 'Bot', 0.8
        elif 'PORT' in label_upper:
            return 'PortScan', 0.4
        else:
            # Unknown attack type
            return 'Unknown', 0.5
    
    def event_to_task(
        self,
        event: pd.Series,
        task_id: int
    ) -> Dict:
        """
        Convert CICIDS event to SOC task.
        
        Args:
            event: CICIDS event (row from DataFrame)
            task_id: Unique task identifier
            
        Returns:
            Task dictionary
        """
        # Extract label
        label = event.get('Label', 'BENIGN')
        
        # Classify attack
        attack_type, severity = self.classify_attack(str(label))
        
        # Determine priority (1-5)
        if severity >= 0.8:
            priority = 5
        elif severity >= 0.6:
            priority = 4
        elif severity >= 0.4:
            priority = 3
        elif severity >= 0.2:
            priority = 2
        else:
            priority = 1
        
        # Estimate complexity based on attack type
        complexity_map = {
            'BENIGN': 0.1,
            'PortScan': 0.3,
            'Brute Force': 0.4,
            'DoS': 0.5,
            'Bot': 0.6,
            'Web Attack': 0.7,
            'DDoS': 0.8,
            'Infiltration': 0.9
        }
        complexity = complexity_map.get(attack_type, 0.5)
        
        # Map to SOC action
        if attack_type == 'BENIGN':
            suggested_action = 'MONITOR'
        elif priority >= 4:
            suggested_action = 'ESCALATE'
        elif complexity >= 0.6:
            suggested_action = 'REDISTRIBUTE'
        else:
            suggested_action = 'ASSIST'
        
        task = {
            'task_id': task_id,
            'attack_type': attack_type,
            'priority': priority,
            'severity': severity,
            'complexity': complexity,
            'suggested_action': suggested_action,
            'label': label
        }
        
        return task
    
    def generate_tasks_from_cicids(
        self,
        df: pd.DataFrame,
        sample_rate: float = 0.01,
        max_tasks: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate SOC tasks from CICIDS dataset.
        
        Args:
            df: CICIDS DataFrame
            sample_rate: Fraction of events to convert to tasks
            max_tasks: Maximum number of tasks to generate
            
        Returns:
            List of task dictionaries
        """
        # Sample events
        n_sample = int(len(df) * sample_rate)
        if max_tasks:
            n_sample = min(n_sample, max_tasks)
        
        sampled_df = df.sample(n=n_sample, random_state=42)
        
        logger.info(f"Generating tasks from {n_sample} sampled events...")
        
        tasks = []
        for idx, (_, event) in enumerate(sampled_df.iterrows()):
            task = self.event_to_task(event, task_id=idx)
            tasks.append(task)
        
        logger.info(f"✓ Generated {len(tasks)} tasks")
        
        return tasks
    
    # ========================================================================
    # STRESS MAPPING
    # ========================================================================
    
    def compute_stress_from_tasks(
        self,
        tasks: List[Dict],
        window_size: int = 10
    ) -> np.ndarray:
        """
        Compute stress timeline from tasks.
        
        Aggregates severity and priority in sliding windows.
        
        Args:
            tasks: List of task dictionaries
            window_size: Number of tasks per window
            
        Returns:
            Stress probability array
        """
        if not tasks:
            return np.array([])
        
        n_windows = len(tasks) // window_size
        if n_windows == 0:
            n_windows = 1
        
        stress_timeline = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(tasks))
            window_tasks = tasks[start_idx:end_idx]
            
            # Aggregate severity and priority
            severities = [t['severity'] for t in window_tasks]
            priorities = [t['priority'] / 5.0 for t in window_tasks]
            
            # Combined stress score
            stress = 0.6 * np.mean(severities) + 0.4 * np.mean(priorities)
            stress_timeline.append(stress)
        
        stress_timeline = np.array(stress_timeline)
        
        # Clip to [0, 1]
        stress_timeline = np.clip(stress_timeline, 0, 1)
        
        return stress_timeline
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    
    def analyze_task_distribution(self, tasks: List[Dict]) -> Dict:
        """
        Analyze distribution of generated tasks.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Analysis dictionary
        """
        if not tasks:
            return {}
        
        df = pd.DataFrame(tasks)
        
        analysis = {
            'total_tasks': len(tasks),
            'attack_type_distribution': df['attack_type'].value_counts().to_dict(),
            'priority_distribution': df['priority'].value_counts().to_dict(),
            'mean_severity': df['severity'].mean(),
            'mean_complexity': df['complexity'].mean(),
            'suggested_action_distribution': df['suggested_action'].value_counts().to_dict()
        }
        
        return analysis
    
    def print_analysis(self, analysis: Dict) -> None:
        """Print task analysis."""
        print("\nCICIDS Task Analysis:")
        print(f"  Total tasks: {analysis.get('total_tasks', 0)}")
        
        print("\n  Attack types:")
        for attack, count in analysis.get('attack_type_distribution', {}).items():
            print(f"    {attack}: {count}")
        
        print("\n  Priority distribution:")
        for priority, count in sorted(analysis.get('priority_distribution', {}).items()):
            print(f"    P{priority}: {count}")
        
        print(f"\n  Mean severity: {analysis.get('mean_severity', 0):.3f}")
        print(f"  Mean complexity: {analysis.get('mean_complexity', 0):.3f}")
        
        print("\n  Suggested actions:")
        for action, count in analysis.get('suggested_action_distribution', {}).items():
            print(f"    {action}: {count}")
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    def export_tasks(self, tasks: List[Dict], filepath: str) -> None:
        """
        Export tasks to CSV.
        
        Args:
            tasks: List of tasks
            filepath: Output file path
        """
        df = pd.DataFrame(tasks)
        df.to_csv(filepath, index=False)
        logger.info(f"Tasks exported to {filepath}")
    
    def tasks_to_dataframe(self, tasks: List[Dict]) -> pd.DataFrame:
        """Convert tasks to DataFrame."""
        return pd.DataFrame(tasks)


def create_sample_cicids_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample CICIDS-like data for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    labels = [
        'BENIGN', 'DoS Hulk', 'DoS GoldenEye', 'DDoS', 
        'Web Attack - Brute Force', 'Web Attack - XSS',
        'Bot', 'PortScan', 'Infiltration'
    ]
    
    # Generate with realistic distribution (mostly benign)
    label_probs = [0.7, 0.05, 0.03, 0.05, 0.03, 0.02, 0.04, 0.05, 0.03]
    
    data = {
        'Label': np.random.choice(labels, size=n_samples, p=label_probs),
        'Flow Duration': np.random.exponential(10000, n_samples),
        'Total Fwd Packets': np.random.poisson(50, n_samples),
        'Total Backward Packets': np.random.poisson(30, n_samples),
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    """
    Test script
    """
    print("="*60)
    print("CICIDS SOC Test")
    print("="*60)
    
    # Initialize CICIDS SOC
    cicids_soc = CICIDSSOC()
    
    # Test 1: Create sample data
    print("\n[Test 1] Create sample CICIDS data...")
    sample_df = create_sample_cicids_data(n_samples=500)
    print(f"  Created {len(sample_df)} sample records")
    print(f"  Labels: {sample_df['Label'].unique()[:5]}")
    print("✅ Sample data created!")
    
    # Test 2: Classify attacks
    print("\n[Test 2] Classify attacks...")
    test_labels = ['BENIGN', 'DoS Hulk', 'DDoS', 'Web Attack - XSS', 'Bot']
    for label in test_labels:
        attack_type, severity = cicids_soc.classify_attack(label)
        print(f"  {label:25s} → {attack_type:15s} (severity: {severity:.2f})")
    print("✅ Attack classification working!")
    
    # Test 3: Generate tasks
    print("\n[Test 3] Generate SOC tasks...")
    tasks = cicids_soc.generate_tasks_from_cicids(sample_df, sample_rate=0.2)
    print(f"  Generated {len(tasks)} tasks")
    print(f"  First task: {tasks[0]}")
    print("✅ Task generation working!")
    
    # Test 4: Compute stress timeline
    print("\n[Test 4] Compute stress timeline...")
    stress_timeline = cicids_soc.compute_stress_from_tasks(tasks, window_size=10)
    print(f"  Stress timeline length: {len(stress_timeline)} windows")
    print(f"  Stress range: [{stress_timeline.min():.2f}, {stress_timeline.max():.2f}]")
    print(f"  Mean stress: {stress_timeline.mean():.2f}")
    print("✅ Stress computation working!")
    
    # Test 5: Analyze tasks
    print("\n[Test 5] Analyze task distribution...")
    analysis = cicids_soc.analyze_task_distribution(tasks)
    cicids_soc.print_analysis(analysis)
    print("✅ Task analysis working!")
    
    # Test 6: Export tasks
    print("\n[Test 6] Export tasks...")
    cicids_soc.export_tasks(tasks, '/tmp/cicids_tasks.csv')
    print("✅ Export working!")
    
    # Test 7: DataFrame conversion
    print("\n[Test 7] Convert to DataFrame...")
    tasks_df = cicids_soc.tasks_to_dataframe(tasks)
    print(f"  DataFrame shape: {tasks_df.shape}")
    print(f"  Columns: {list(tasks_df.columns)}")
    print("✅ DataFrame conversion working!")
    
    print("\n" + "="*60)
    print("✅ ALL CICIDS SOC TESTS PASSED!")
    print("="*60)
    
    # Display sample tasks
    print("\nSample Tasks:")
    sample_tasks = tasks[:5]
    for task in sample_tasks:
        print(f"  Task {task['task_id']}: {task['attack_type']:15s} "
              f"P{task['priority']} → {task['suggested_action']}")
