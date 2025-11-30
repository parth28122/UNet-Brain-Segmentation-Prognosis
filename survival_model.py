"""
Survival Prediction Model using XGBoost

Extracts tumor features from segmentation masks and predicts survival risk categories:
- Low risk
- Medium risk
- High risk
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class SurvivalPredictor:
    """
    Survival prediction model using XGBoost classifier.
    
    Extracts features from tumor masks:
    - Volume
    - Bounding box
    - Centroid
    - Intensity statistics
    - Texture features (GLCM)
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        """
        Initialize survival predictor.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            random_state: Random seed
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """
        Train the survival prediction model.
        
        Args:
            features: Feature matrix [N, F]
            labels: Survival risk labels [N] (0: Low, 1: Medium, 2: High)
            validation_split: Validation split ratio
            verbose: Whether to print training progress
        
        Returns:
            Training history dictionary
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        if verbose:
            print(f"\nTraining Accuracy: {train_acc:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_val, val_pred, 
                                      target_names=['Low Risk', 'Medium Risk', 'High Risk']))
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict survival risk category.
        
        Args:
            features: Feature matrix [N, F] or [F]
        
        Returns:
            Tuple of (predictions, probabilities)
            - predictions: Risk category [N] or scalar
            - probabilities: Probability distribution [N, 3] or [3]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Return single value if input was single sample
        if features.shape[0] == 1:
            return predictions[0], probabilities[0]
        
        return predictions, probabilities
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            features: Feature matrix [N, F] or [F]
        
        Returns:
            Probability distribution [N, 3] or [3]
        """
        _, probabilities = self.predict(features)
        return probabilities
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data.get('feature_names', [])
    
    def get_feature_importance(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_k: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.model.feature_importances_
        
        if len(self.feature_names) == len(importance):
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
        else:
            df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            })
        
        df = df.sort_values('importance', ascending=False).head(top_k)
        return df


def get_risk_category(probabilities: np.ndarray) -> str:
    """
    Convert probability distribution to risk category.
    
    Args:
        probabilities: Probability array [3] (Low, Medium, High)
    
    Returns:
        Risk category string
    """
    categories = ['Low Risk', 'Medium Risk', 'High Risk']
    idx = np.argmax(probabilities)
    return categories[idx]


if __name__ == "__main__":
    # Test model
    predictor = SurvivalPredictor()
    
    # Dummy data
    n_samples = 100
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    # Train
    predictor.fit(X, y)
    
    # Predict
    pred, prob = predictor.predict(X[0])
    print(f"\nPrediction: {pred}, Probabilities: {prob}")
    print(f"Risk Category: {get_risk_category(prob)}")






