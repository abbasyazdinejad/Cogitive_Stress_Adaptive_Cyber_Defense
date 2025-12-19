"""
Train Stress Classification Model

Trains the DNN-XGBoost hybrid model (and other baseline models) on WESAD data.
Saves trained models and reports performance metrics.

Usage:
    python train_stress_model.py --model dnn_xgboost --output models/
    python train_stress_model.py --model all --calibrate --save
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.wesad_loader import WESADLoader
from data.signal_processor import SignalProcessor
from data.feature_extractor import FeatureExtractor
from models.stress_classifier import StressClassifier
from evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix_metrics,
    compute_expected_calibration_error,
    print_classification_report
)
from utils.helpers import setup_logger, set_random_seed, ensure_dir, get_timestamp
from config import ALL_FEATURES, WESAD_SUBJECTS, RANDOM_SEED

logger = setup_logger(__name__)


class StressModelTrainer:
    """
    Trainer for stress classification models.
    """
    
    def __init__(
        self,
        model_type: str = 'dnn_xgboost',
        output_dir: str = 'models',
        calibrate: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('svm', 'rf', 'xgboost', 'dnn', 'dnn_xgboost', 'all')
            output_dir: Directory to save models
            calibrate: Apply temperature scaling calibration
        """
        self.model_type = model_type
        self.output_dir = ensure_dir(output_dir)
        self.calibrate = calibrate
        
        set_random_seed(RANDOM_SEED)
        
        logger.info("="*60)
        logger.info("Stress Model Trainer Initialized")
        logger.info("="*60)
        logger.info(f"Model type: {model_type}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Calibration: {calibrate}")
    
    def load_and_preprocess_data(
        self,
        subject_ids: Optional[list] = None,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess WESAD data.
        
        Args:
            subject_ids: List of subjects to use (None = all)
            test_size: Fraction for test set
            
        Returns:
            (X_train, X_test, y_train, y_test, X_val, y_val)
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Load and Preprocess Data")
        logger.info("="*60)
        
        # Load subjects
        loader = WESADLoader()
        if subject_ids is None:
            subject_ids = loader.available_subjects
        
        logger.info(f"Loading {len(subject_ids)} subjects...")
        subjects_data = loader.load_multiple_subjects(subject_ids, verbose=True)
        
        # Process signals
        logger.info("\nProcessing signals...")
        processor = SignalProcessor()
        processed_data = processor.process_multiple_subjects(subjects_data)
        
        # Extract features
        logger.info("\nExtracting features...")
        extractor = FeatureExtractor()
        features_df = extractor.extract_multiple_subjects(processed_data)
        
        # Split data
        logger.info("\nSplitting data...")
        from sklearn.model_selection import train_test_split
        
        X = features_df[ALL_FEATURES].values
        y = features_df['label'].values
        
        # Train/temp split
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp, y_train_temp, 
            test_size=0.2, 
            random_state=RANDOM_SEED,
            stratify=y_train_temp
        )
        
        logger.info(f"\nDataset sizes:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Val:   {len(X_val)} samples")
        logger.info(f"  Test:  {len(X_test)} samples")
        logger.info(f"  Features: {X_train.shape[1]}")
        
        # Class distribution
        logger.info(f"\nClass distribution (train):")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            logger.info(f"  Class {label}: {count} ({100*count/len(y_train):.1f}%)")
        
        return X_train, X_test, y_train, y_test, X_val, y_val
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_type: Optional[str] = None
    ) -> StressClassifier:
        """
        Train a single model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_type: Model type (overrides self.model_type)
            
        Returns:
            Trained classifier
        """
        model_type = model_type or self.model_type
        
        logger.info(f"\nTraining {model_type.upper()} model...")
        start_time = time.time()
        
        # Initialize and train
        classifier = StressClassifier(model_type=model_type, random_state=RANDOM_SEED)
        classifier.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        # Calibrate if requested
        if self.calibrate and X_val is not None and y_val is not None:
            logger.info("Applying temperature scaling calibration...")
            classifier.calibrate(X_val, y_val, method='temperature')
        
        train_time = time.time() - start_time
        logger.info(f"✓ Training completed in {train_time:.1f}s")
        
        return classifier
    
    def evaluate_model(
        self,
        classifier: StressClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate model and return metrics.
        
        Args:
            classifier: Trained classifier
            X_test: Test features
            y_test: Test labels
            model_name: Name for reporting
            
        Returns:
            Dictionary with metrics
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)
        
        # Compute metrics
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        cm_metrics = compute_confusion_matrix_metrics(y_test, y_pred)
        ece = compute_expected_calibration_error(y_test, y_prob)
        
        # Combine metrics
        results = {
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc': metrics.get('auc', 0.0),
            'ece': ece,
            'tp': cm_metrics['TP'],
            'tn': cm_metrics['TN'],
            'fp': cm_metrics['FP'],
            'fn': cm_metrics['FN'],
            'sensitivity': cm_metrics['sensitivity'],
            'specificity': cm_metrics['specificity']
        }
        
        # Print report
        logger.info(f"\n{model_name} Results:")
        for metric, value in results.items():
            if metric != 'model' and not metric.startswith('t') and not metric.startswith('f'):
                logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def save_model(
        self,
        classifier: StressClassifier,
        model_name: str,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Save trained model and metrics.
        
        Args:
            classifier: Trained classifier
            model_name: Model name
            metrics: Performance metrics
            
        Returns:
            Path to saved model
        """
        timestamp = get_timestamp()
        
        # Save model
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.output_dir / model_filename
        classifier.save(str(model_path))
        
        # Save metrics
        metrics_filename = f"{model_name}_{timestamp}_metrics.json"
        metrics_path = self.output_dir / metrics_filename
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✓ Model saved: {model_path}")
        logger.info(f"✓ Metrics saved: {metrics_path}")
        
        return str(model_path)
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all baseline models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            X_val, y_val: Validation data
            
        Returns:
            Dictionary of model_name -> results
        """
        models_to_train = ['svm', 'rf', 'xgboost', 'dnn', 'dnn_xgboost']
        
        all_results = {}
        
        for model_type in models_to_train:
            try:
                logger.info("\n" + "="*60)
                logger.info(f"Training {model_type.upper()}")
                logger.info("="*60)
                
                # Train
                classifier = self.train_model(
                    X_train, y_train, X_val, y_val, model_type=model_type
                )
                
                # Evaluate
                results = self.evaluate_model(
                    classifier, X_test, y_test, model_type
                )
                
                # Save
                self.save_model(classifier, model_type, results)
                
                all_results[model_type] = results
                
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                continue
        
        return all_results
    
    def print_comparison_table(self, all_results: Dict[str, Dict[str, Any]]):
        """
        Print comparison table of all models.
        
        Args:
            all_results: Dictionary of model -> results
        """
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON (Table II)")
        logger.info("="*60)
        
        # Header
        header = f"{'Model':<15} {'Acc':>6} {'F1':>6} {'AUC':>6} {'ECE':>6}"
        logger.info(header)
        logger.info("-" * 60)
        
        # Rows
        for model_name, results in all_results.items():
            row = (
                f"{model_name.upper():<15} "
                f"{results['accuracy']:>6.3f} "
                f"{results['f1_score']:>6.3f} "
                f"{results['auc']:>6.3f} "
                f"{results['ece']:>6.3f}"
            )
            logger.info(row)
        
        logger.info("="*60)
    
    def run(self):
        """
        Run complete training pipeline.
        """
        start_time = time.time()
        
        # Load data
        X_train, X_test, y_train, y_test, X_val, y_val = self.load_and_preprocess_data()
        
        # Train models
        if self.model_type == 'all':
            logger.info("\n" + "="*60)
            logger.info("Training ALL baseline models")
            logger.info("="*60)
            
            all_results = self.train_all_models(
                X_train, y_train, X_test, y_test, X_val, y_val
            )
            
            # Print comparison
            self.print_comparison_table(all_results)
            
        else:
            logger.info("\n" + "="*60)
            logger.info(f"Training {self.model_type.upper()} model")
            logger.info("="*60)
            
            # Train single model
            classifier = self.train_model(X_train, y_train, X_val, y_val)
            
            # Evaluate
            results = self.evaluate_model(classifier, X_test, y_test, self.model_type)
            
            # Save
            self.save_model(classifier, self.model_type, results)
        
        total_time = time.time() - start_time
        logger.info(f"\n✓ Total training time: {total_time/60:.1f} minutes")
        logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train stress classification model')
    
    parser.add_argument(
        '--model',
        type=str,
        default='dnn_xgboost',
        choices=['svm', 'rf', 'xgboost', 'dnn', 'dnn_xgboost', 'all'],
        help='Model type to train'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Output directory for models'
    )
    
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Apply temperature scaling calibration'
    )
    
    parser.add_argument(
        '--no-calibrate',
        dest='calibrate',
        action='store_false',
        help='Skip calibration'
    )
    
    parser.set_defaults(calibrate=True)
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = StressModelTrainer(
        model_type=args.model,
        output_dir=args.output,
        calibrate=args.calibrate
    )
    
    # Run training
    trainer.run()


if __name__ == "__main__":
    main()
