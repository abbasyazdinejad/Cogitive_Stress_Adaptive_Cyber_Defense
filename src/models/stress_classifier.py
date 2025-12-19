"""
Stress Classification Models
Implements multiple classifiers for binary stress classification:
- SVM with RBF kernel
- Random Forest
- XGBoost
- Deep Neural Network (DNN)
- Hybrid DNN-XGBoost (main model from paper)

Includes temperature scaling for calibration.
Based on Section IV-D and Table II of the paper.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pickle

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. DNN and Hybrid models will not work.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logger, set_random_seed
from config import DNN_CONFIG, XGBOOST_CONFIG, RANDOM_SEED

logger = setup_logger(__name__)


class StressClassifier:
    """
    Base class for stress classifiers.
    """
    
    def __init__(self, model_type: str = 'dnn_xgboost', random_state: int = RANDOM_SEED):
        """
        Initialize classifier.
        
        Args:
            model_type: One of ['svm', 'rf', 'xgboost', 'dnn', 'dnn_xgboost']
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
        set_random_seed(random_state)
        logger.info(f"Initialized {model_type} classifier")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'StressClassifier':
        """
        Fit classifier to training data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            **kwargs: Additional arguments for specific models
            
        Returns:
            Self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit model
        if self.model_type == 'svm':
            self.model = self._train_svm(X_scaled, y, **kwargs)
        elif self.model_type == 'rf':
            self.model = self._train_rf(X_scaled, y, **kwargs)
        elif self.model_type == 'xgboost':
            self.model = self._train_xgboost(X_scaled, y, **kwargs)
        elif self.model_type == 'dnn':
            self.model = self._train_dnn(X_scaled, y, **kwargs)
        elif self.model_type == 'dnn_xgboost':
            self.model = self._train_dnn_xgboost(X_scaled, y, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.is_fitted = True
        logger.info(f"✓ Model fitted: {self.model_type}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'dnn':
            y_prob = self.model.predict(X_scaled, verbose=0)
            return (y_prob[:, 0] > 0.5).astype(int)
        elif self.model_type == 'dnn_xgboost':
            # Get DNN embeddings
            embeddings = self.dnn_encoder.predict(X_scaled, verbose=0)
            # XGBoost prediction
            return (self.model.predict(embeddings) > 0.5).astype(int)
        else:
            return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted probabilities for stress class (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'dnn':
            y_prob = self.model.predict(X_scaled, verbose=0)
            return y_prob[:, 0]  # Probability of stress
        elif self.model_type == 'dnn_xgboost':
            # Get DNN embeddings
            embeddings = self.dnn_encoder.predict(X_scaled, verbose=0)
            # XGBoost probability
            return self.model.predict(embeddings)
        else:
            proba = self.model.predict_proba(X_scaled)
            return proba[:, 1]  # Probability of class 1 (stress)
    
    # ========================================================================
    # MODEL-SPECIFIC TRAINING
    # ========================================================================
    
    def _train_svm(self, X: np.ndarray, y: np.ndarray, **kwargs) -> SVC:
        """Train SVM with RBF kernel."""
        model = SVC(
            kernel='rbf',
            C=kwargs.get('C', 1.0),
            gamma=kwargs.get('gamma', 'scale'),
            probability=True,
            random_state=self.random_state
        )
        model.fit(X, y)
        return model
    
    def _train_rf(self, X: np.ndarray, y: np.ndarray, **kwargs) -> RandomForestClassifier:
        """Train Random Forest."""
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=self.random_state
        )
        model.fit(X, y)
        return model
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, **kwargs) -> xgb.XGBClassifier:
        """Train XGBoost."""
        model = xgb.XGBClassifier(
            max_depth=XGBOOST_CONFIG['max_depth'],
            learning_rate=XGBOOST_CONFIG['learning_rate'],
            n_estimators=XGBOOST_CONFIG['n_estimators'],
            objective='binary:logistic',
            eval_metric='auc',
            random_state=self.random_state,
            use_label_encoder=False
        )
        
        # Use validation set if provided
        X_val = kwargs.get('X_val')
        y_val = kwargs.get('y_val')
        
        if X_val is not None and y_val is not None:
            model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model.fit(X, y, verbose=False)
        
        return model
    
    def _train_dnn(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Model:
        """Train Deep Neural Network."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for DNN")
        
        # Build model
        input_dim = X.shape[1]
        
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        
        # Hidden layers
        for units in DNN_CONFIG['hidden_layers']:
            x = layers.Dense(units, activation=DNN_CONFIG['activation'])(x)
            x = layers.Dropout(DNN_CONFIG['dropout_rate'])(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=DNN_CONFIG['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        # Train
        history = model.fit(
            X, y,
            epochs=DNN_CONFIG['epochs'],
            batch_size=DNN_CONFIG['batch_size'],
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0
        )
        
        return model
    
    def _train_dnn_xgboost(self, X: np.ndarray, y: np.ndarray, **kwargs) -> xgb.XGBClassifier:
        """
        Train Hybrid DNN-XGBoost model.
        DNN generates embeddings, which are fed into XGBoost.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for DNN-XGBoost hybrid")
        
        logger.info("Building DNN encoder...")
        
        # Build DNN encoder (without final classification layer)
        input_dim = X.shape[1]
        
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        
        # Hidden layers
        for units in DNN_CONFIG['hidden_layers']:
            x = layers.Dense(units, activation=DNN_CONFIG['activation'])(x)
            x = layers.Dropout(DNN_CONFIG['dropout_rate'])(x)
        
        # Encoder model (output is the last hidden layer)
        encoder = Model(inputs=inputs, outputs=x, name='dnn_encoder')
        
        # First, train DNN end-to-end for pretraining
        logger.info("Pretraining DNN...")
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        pretrain_model = Model(inputs=inputs, outputs=outputs)
        
        pretrain_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=DNN_CONFIG['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        pretrain_model.fit(
            X, y,
            epochs=DNN_CONFIG['epochs'],
            batch_size=DNN_CONFIG['batch_size'],
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # Extract embeddings
        logger.info("Extracting DNN embeddings...")
        embeddings = encoder.predict(X, verbose=0)
        
        # Train XGBoost on embeddings
        logger.info("Training XGBoost on embeddings...")
        xgb_model = xgb.XGBClassifier(
            max_depth=XGBOOST_CONFIG['max_depth'],
            learning_rate=XGBOOST_CONFIG['learning_rate'],
            n_estimators=XGBOOST_CONFIG['n_estimators'],
            objective='binary:logistic',
            eval_metric='auc',
            random_state=self.random_state,
            use_label_encoder=False
        )
        
        xgb_model.fit(embeddings, y, verbose=False)
        
        # Save encoder for inference
        self.dnn_encoder = encoder
        
        return xgb_model
    
    # ========================================================================
    # CALIBRATION
    # ========================================================================
    
    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray, method: str = 'temperature') -> None:
        """
        Calibrate model probabilities using temperature scaling.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            method: Calibration method ('temperature' or 'isotonic')
        """
        if not self.is_fitted:
            raise ValueError("Fit model before calibration")
        
        logger.info(f"Calibrating with {method} scaling...")
        
        if method == 'temperature':
            # Temperature scaling
            logits = self.predict_proba(X_val)
            
            # Find optimal temperature
            from scipy.optimize import minimize
            
            def temperature_loss(T):
                calibrated_probs = 1 / (1 + np.exp(-np.log(logits / (1 - logits)) / T))
                loss = -np.mean(y_val * np.log(calibrated_probs + 1e-10) + 
                               (1 - y_val) * np.log(1 - calibrated_probs + 1e-10))
                return loss
            
            result = minimize(temperature_loss, x0=1.0, bounds=[(0.1, 10.0)])
            self.temperature = result.x[0]
            
            logger.info(f"Optimal temperature: {self.temperature:.3f}")
        
        elif method == 'isotonic':
            # Isotonic regression
            self.model = CalibratedClassifierCV(
                self.model,
                method='isotonic',
                cv='prefit'
            )
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(X_val_scaled, y_val)
        
        self.is_calibrated = True
    
    # ========================================================================
    # SAVE/LOAD
    # ========================================================================
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        import joblib
        
        save_dict = {
            'model_type': self.model_type,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'random_state': self.random_state,
        }
        
        if self.model_type == 'dnn_xgboost':
            # Save DNN encoder separately
            encoder_path = filepath.replace('.pkl', '_encoder.h5')
            self.dnn_encoder.save(encoder_path)
            save_dict['model'] = self.model
            save_dict['encoder_path'] = encoder_path
        elif self.model_type == 'dnn':
            # Save Keras model separately
            model_path = filepath.replace('.pkl', '_model.h5')
            self.model.save(model_path)
            save_dict['model_path'] = model_path
        else:
            save_dict['model'] = self.model
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'StressClassifier':
        """Load model from file."""
        import joblib
        
        save_dict = joblib.load(filepath)
        
        classifier = StressClassifier(
            model_type=save_dict['model_type'],
            random_state=save_dict['random_state']
        )
        
        classifier.scaler = save_dict['scaler']
        classifier.is_fitted = save_dict['is_fitted']
        
        if save_dict['model_type'] == 'dnn_xgboost':
            classifier.model = save_dict['model']
            classifier.dnn_encoder = keras.models.load_model(save_dict['encoder_path'])
        elif save_dict['model_type'] == 'dnn':
            classifier.model = keras.models.load_model(save_dict['model_path'])
        else:
            classifier.model = save_dict['model']
        
        logger.info(f"Model loaded from {filepath}")
        return classifier


if __name__ == "__main__":
    """
    Test script
    """
    print("="*60)
    print("Stress Classifier Test")
    print("="*60)
    
    # Generate fake data
    np.random.seed(42)
    n_samples = 1000
    n_features = 16
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test all models
    models_to_test = ['svm', 'rf', 'xgboost']
    if TF_AVAILABLE:
        models_to_test.extend(['dnn', 'dnn_xgboost'])
    
    for model_type in models_to_test:
        print(f"\n[Test] {model_type.upper()}...")
        
        try:
            classifier = StressClassifier(model_type=model_type)
            classifier.fit(X_train, y_train)
            
            y_pred = classifier.predict(X_test)
            y_prob = classifier.predict_proba(X_test)
            
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"  Accuracy: {acc:.3f}")
            print(f"  F1 Score: {f1:.3f}")
            print(f"  ✅ {model_type} working!")
        
        except Exception as e:
            print(f"  ❌ {model_type} failed: {e}")
    
    print("\n" + "="*60)
    print("✅ STRESS CLASSIFIER TESTS COMPLETE!")
    print("="*60)
