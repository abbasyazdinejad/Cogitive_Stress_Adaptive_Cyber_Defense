"""
Leave-One-Subject-Out (LOSO) Cross-Validation

Performs LOSO CV for stress classification, as reported in Table II of the paper.
Each subject is used as test set once, with remaining subjects for training.

Usage:
    python evaluate_loso.py --model dnn_xgboost --save-results
    python evaluate_loso.py --model all --output results/loso/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Optional, Dict, List, Any

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.wesad_loader import WESADLoader
from data.signal_processor import SignalProcessor
from data.feature_extractor import FeatureExtractor
from models.stress_classifier import StressClassifier
from evaluation.metrics import (
    compute_classification_metrics,
    compute_expected_calibration_error,
    aggregate_loso_results,
    print_loso_summary
)
from utils.helpers import setup_logger, set_random_seed, ensure_dir, get_timestamp
from config import ALL_FEATURES, RANDOM_SEED

logger = setup_logger(__name__)


class LOSOEvaluator:
    """
    Leave-One-Subject-Out Cross-Validation Evaluator.
    """
    
    def __init__(
        self,
        model_type: str = 'dnn_xgboost',
        output_dir: str = 'results/loso'
    ):
        """
        Initialize LOSO evaluator.
        
        Args:
            model_type: Type of model to evaluate
            output_dir: Directory for results
        """
        self.model_type = model_type
        self.output_dir = ensure_dir(output_dir)
        
        set_random_seed(RANDOM_SEED)
        
        logger.info("="*60)
        logger.info("LOSO Cross-Validation Evaluator")
        logger.info("="*60)
        logger.info(f"Model: {model_type}")
        logger.info(f"Output: {output_dir}")
    
    def load_subject_features(self, subject_id: str) -> pd.DataFrame:
        """
        Load and extract features for a single subject.
        
        Args:
            subject_id: Subject ID
            
        Returns:
            Features DataFrame
        """
        # Load subject
        loader = WESADLoader()
        subject_data = loader.load_subject(subject_id, verbose=False)
        
        # Process signals
        processor = SignalProcessor()
        processed = processor.process_subject(subject_data, return_windows=True)
        
        # Extract features
        extractor = FeatureExtractor()
        features_df = extractor.extract_subject_features(processed)
        features_df['subject_id'] = subject_id
        
        return features_df
    
    def load_all_subjects(self, subject_ids: List[str]) -> pd.DataFrame:
        """
        Load features for all subjects with caching.
        
        Args:
            subject_ids: List of subject IDs
            
        Returns:
            Combined features DataFrame
        """
        cache_file = self.output_dir / 'features_cache.pkl'
        
        # Try to load from cache
        if cache_file.exists():
            logger.info(f"Loading features from cache: {cache_file}")
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load from scratch
        logger.info(f"Loading features for {len(subject_ids)} subjects...")
        
        all_features = []
        for subject_id in tqdm(subject_ids, desc="Loading subjects"):
            try:
                features_df = self.load_subject_features(subject_id)
                all_features.append(features_df)
            except Exception as e:
                logger.error(f"Failed to load {subject_id}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No subjects loaded successfully!")
        
        # Combine
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Cache for future use
        logger.info(f"Caching features to {cache_file}")
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(combined_df, f)
        
        return combined_df
    
    def run_loso_fold(
        self,
        features_df: pd.DataFrame,
        test_subject: str
    ) -> Dict[str, Any]:
        """
        Run one LOSO fold.
        
        Args:
            features_df: Complete features DataFrame
            test_subject: Subject to use as test set
            
        Returns:
            Fold results dictionary
        """
        logger.info(f"\nFold: Test subject = {test_subject}")
        
        # Split train/test
        test_df = features_df[features_df['subject_id'] == test_subject]
        train_df = features_df[features_df['subject_id'] != test_subject]
        
        logger.info(f"  Train: {len(train_df)} samples from {train_df['subject_id'].nunique()} subjects")
        logger.info(f"  Test:  {len(test_df)} samples")
        
        # Extract features and labels
        X_train = train_df[ALL_FEATURES].values
        y_train = train_df['label'].values
        X_test = test_df[ALL_FEATURES].values
        y_test = test_df['label'].values
        
        # Train model
        logger.info(f"  Training {self.model_type}...")
        classifier = StressClassifier(model_type=self.model_type, random_state=RANDOM_SEED)
        classifier.fit(X_train, y_train)
        
        # Predict
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)
        
        # Compute metrics
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        ece = compute_expected_calibration_error(y_test, y_prob)
        
        results = {
            'test_subject': test_subject,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc': metrics.get('auc', 0.0),
            'ece': ece
        }
        
        logger.info(f"  Results: Acc={results['accuracy']:.3f}, F1={results['f1_score']:.3f}, AUC={results['auc']:.3f}")
        
        return results
    
    def run_loso_cv(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Run complete LOSO cross-validation.
        
        Args:
            features_df: Features for all subjects
            
        Returns:
            Dictionary of subject -> results
        """
        subjects = features_df['subject_id'].unique()
        logger.info(f"\nRunning LOSO CV with {len(subjects)} subjects...")
        
        loso_results = {}
        
        for subject in subjects:
            try:
                results = self.run_loso_fold(features_df, subject)
                loso_results[subject] = results
            except Exception as e:
                logger.error(f"Fold failed for {subject}: {e}")
                continue
        
        return loso_results
    
    def save_results(self, loso_results: Dict[str, Dict[str, Any]], summary: pd.DataFrame):
        """
        Save LOSO results.
        
        Args:
            loso_results: Per-subject results
            summary: Aggregated summary
        """
        timestamp = get_timestamp()
        
        # Save per-subject results
        results_file = self.output_dir / f"loso_{self.model_type}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(loso_results, f, indent=2)
        
        logger.info(f"✓ Results saved: {results_file}")
        
        # Save summary
        summary_file = self.output_dir / f"loso_{self.model_type}_{timestamp}_summary.csv"
        summary.to_csv(summary_file)
        
        logger.info(f"✓ Summary saved: {summary_file}")
    
    def run(self):
        """
        Run complete LOSO evaluation.
        """
        start_time = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Load Features")
        logger.info("="*60)
        
        # Load all subjects
        loader = WESADLoader()
        subject_ids = loader.available_subjects
        features_df = self.load_all_subjects(subject_ids)
        
        logger.info(f"\nTotal dataset:")
        logger.info(f"  Subjects: {features_df['subject_id'].nunique()}")
        logger.info(f"  Samples: {len(features_df)}")
        logger.info(f"  Features: {len(ALL_FEATURES)}")
        
        logger.info("\n" + "="*60)
        logger.info("STEP 2: LOSO Cross-Validation")
        logger.info("="*60)
        
        # Run LOSO
        loso_results = self.run_loso_cv(features_df)
        
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Aggregate Results")
        logger.info("="*60)
        
        # Aggregate
        summary = aggregate_loso_results(loso_results)
        
        # Print summary
        print_loso_summary(summary)
        
        # Save
        self.save_results(loso_results, summary)
        
        total_time = time.time() - start_time
        logger.info(f"\n✓ Total LOSO time: {total_time/60:.1f} minutes")
        logger.info("="*60)
        
        return loso_results, summary


class MultiModelLOSO:
    """
    Run LOSO for multiple models and compare.
    """
    
    def __init__(self, output_dir: str = 'results/loso'):
        self.output_dir = ensure_dir(output_dir)
        self.models = ['svm', 'rf', 'xgboost', 'dnn', 'dnn_xgboost']
    
    def run_all_models(self) -> Dict[str, pd.DataFrame]:
        """Run LOSO for all baseline models."""
        all_summaries = {}
        
        for model_type in self.models:
            logger.info("\n" + "="*70)
            logger.info(f"LOSO EVALUATION: {model_type.upper()}")
            logger.info("="*70)
            
            try:
                evaluator = LOSOEvaluator(
                    model_type=model_type,
                    output_dir=self.output_dir
                )
                
                loso_results, summary = evaluator.run()
                all_summaries[model_type] = summary
                
            except Exception as e:
                logger.error(f"Failed for {model_type}: {e}")
                continue
        
        # Print comparison
        self.print_comparison(all_summaries)
        
        return all_summaries
    
    def print_comparison(self, all_summaries: Dict[str, pd.DataFrame]):
        """
        Print comparison of all models (Table II format).
        
        Args:
            all_summaries: Dict of model -> summary DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("TABLE II: LOSO CROSS-VALIDATION RESULTS")
        logger.info("="*70)
        
        # Header
        header = f"{'Model':<15} {'Accuracy':>12} {'F1 Score':>12} {'AUC':>12} {'ECE':>12}"
        logger.info(header)
        logger.info("-" * 70)
        
        # Rows
        for model_name, summary in all_summaries.items():
            acc_mean = summary.loc['accuracy', 'mean']
            acc_std = summary.loc['accuracy', 'std']
            
            f1_mean = summary.loc['f1_score', 'mean']
            f1_std = summary.loc['f1_score', 'std']
            
            auc_mean = summary.loc['auc', 'mean']
            auc_std = summary.loc['auc', 'std']
            
            ece_mean = summary.loc['ece', 'mean']
            ece_std = summary.loc['ece', 'std']
            
            row = (
                f"{model_name.upper():<15} "
                f"{acc_mean:.3f}±{acc_std:.3f}  "
                f"{f1_mean:.3f}±{f1_std:.3f}  "
                f"{auc_mean:.3f}±{auc_std:.3f}  "
                f"{ece_mean:.3f}±{ece_std:.3f}"
            )
            logger.info(row)
        
        logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='LOSO cross-validation evaluation')
    
    parser.add_argument(
        '--model',
        type=str,
        default='dnn_xgboost',
        choices=['svm', 'rf', 'xgboost', 'dnn', 'dnn_xgboost', 'all'],
        help='Model type to evaluate'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/loso',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed results'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        # Evaluate all models
        evaluator = MultiModelLOSO(output_dir=args.output)
        evaluator.run_all_models()
    else:
        # Evaluate single model
        evaluator = LOSOEvaluator(
            model_type=args.model,
            output_dir=args.output
        )
        evaluator.run()


if __name__ == "__main__":
    main()
