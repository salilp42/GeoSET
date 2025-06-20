"""
Feature Scalpel adaptive classification framework.

This module implements the Feature Scalpel strategy that adapts feature selection
based on classification difficulty, using comprehensive features for easy pairs
and focused biomarkers for challenging distinctions.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional imports
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

warnings.filterwarnings('ignore')


class FeatureScalpelClassifier:
    """
    Adaptive classification framework that selects feature complexity based on
    expected classification difficulty.
    
    The Feature Scalpel strategy uses:
    - Comprehensive features for well-separated groups (high signal-to-noise)
    - Focused biomarker features for challenging distinctions (low signal-to-noise)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Feature Scalpel classifier.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters including:
            - cv_folds: Number of cross-validation folds (default: 5)
            - random_state: Random seed (default: 42)
            - use_class_weights: Whether to balance classes (default: True)
            - difficulty_threshold: AUC threshold for easy vs hard pairs (default: 0.9)
        """
        self.config = config or {}
        self.cv_folds = self.config.get('cv_folds', 5)
        self.random_state = self.config.get('random_state', 42)
        self.use_class_weights = self.config.get('use_class_weights', True)
        self.difficulty_threshold = self.config.get('difficulty_threshold', 0.9)
        
        # Strategy mapping: which pairs get which features
        self.strategy_map = {}
        self.fitted_models = {}
        
    def fit_predict(self, 
                   comprehensive_features: np.ndarray,
                   biomarker_features: np.ndarray,
                   labels: np.ndarray,
                   label_names: Dict[int, str],
                   participant_ids: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit Feature Scalpel classifiers and predict all pairwise comparisons.
        
        Parameters
        ----------
        comprehensive_features : array-like of shape (n_participants, n_comprehensive_features)
            Full feature set with all available information
        biomarker_features : array-like of shape (n_participants, n_biomarker_features)
            Focused biomarker feature set
        labels : array-like of shape (n_participants,)
            Participant labels
        label_names : dict
            Mapping from label codes to descriptive names
        participant_ids : array-like, optional
            Participant identifiers for group-aware cross-validation
            
        Returns
        -------
        dict
            Complete classification results including metrics, ROC curves, and strategy decisions
        """
        print("Running Feature Scalpel classification...")
        
        # Get unique labels and create pairwise combinations
        unique_labels = np.unique(labels)
        pair_combinations = list(combinations(unique_labels, 2))
        
        print(f"Analyzing {len(pair_combinations)} pairwise comparisons")
        
        results = {
            'pairwise_results': {},
            'auc_matrix': np.zeros((len(unique_labels), len(unique_labels))),
            'strategy_decisions': {},
            'roc_curves': {},
            'metrics_summary': [],
            'mean_auc': 0.0
        }
        
        all_aucs = []
        
        for label1, label2 in pair_combinations:
            pair_name = f"{label_names[label1]} vs {label_names[label2]}"
            print(f"  Processing: {pair_name}")
            
            # Extract data for this pair
            pair_mask = np.isin(labels, [label1, label2])
            pair_labels = labels[pair_mask]
            pair_comprehensive = comprehensive_features[pair_mask]
            pair_biomarker = biomarker_features[pair_mask]
            
            # Extract participant IDs for this pair if provided
            pair_participant_ids = participant_ids[pair_mask] if participant_ids is not None else None
            
            # Convert to binary labels
            binary_labels = (pair_labels == label2).astype(int)
            
            # Determine strategy based on expected difficulty
            strategy = self._determine_strategy(label1, label2, label_names)
            
            # Select appropriate features
            if strategy == 'comprehensive':
                features = pair_comprehensive
                feature_type = 'comprehensive'
            else:
                features = pair_biomarker
                feature_type = 'biomarker'
                
            # Perform classification
            pair_result = self._classify_pair(features, binary_labels, pair_name, pair_participant_ids)
            pair_result['strategy'] = strategy
            pair_result['feature_type'] = feature_type
            pair_result['n_features'] = features.shape[1]
            
            # Store results
            results['pairwise_results'][pair_name] = pair_result
            results['strategy_decisions'][pair_name] = strategy
            results['roc_curves'][pair_name] = pair_result['roc_curve']
            
            # Update AUC matrix
            idx1 = np.where(unique_labels == label1)[0][0]
            idx2 = np.where(unique_labels == label2)[0][0]
            results['auc_matrix'][idx1, idx2] = pair_result['auc']
            results['auc_matrix'][idx2, idx1] = pair_result['auc']
            
            # Add to metrics summary
            results['metrics_summary'].append({
                'comparison': pair_name,
                'auc': pair_result['auc'],
                'accuracy': pair_result['accuracy'],
                'f1_score': pair_result['f1_score'],
                'mcc': pair_result['mcc'],
                'strategy': strategy,
                'n_features': features.shape[1]
            })
            
            all_aucs.append(pair_result['auc'])
        
        # Compute overall statistics
        results['mean_auc'] = np.mean(all_aucs)
        results['std_auc'] = np.std(all_aucs)
        results['label_names'] = label_names
        results['unique_labels'] = unique_labels
        
        print(f"Feature Scalpel complete. Mean AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
        
        return results
    
    def _determine_strategy(self, label1: int, label2: int, label_names: Dict[int, str]) -> str:
        """
        Determine whether to use comprehensive or biomarker features based on expected difficulty.
        
        Parameters
        ----------
        label1, label2 : int
            Label codes for the pair
        label_names : dict
            Label name mapping
            
        Returns
        -------
        str
            'comprehensive' for easy pairs, 'biomarker' for hard pairs
        """
        # Strategy rules based on clinical knowledge and pilot analysis
        name1, name2 = label_names[label1], label_names[label2]
        
        # PSP vs any group: well-separated due to severe brainstem pathology
        if 'PSP' in name1 or 'PSP' in name2:
            return 'comprehensive'
            
        # Medicated vs De Novo PD: systematic medication effects
        if ('Medicated PD' in name1 and 'De Novo PD' in name2) or \
           ('De Novo PD' in name1 and 'Medicated PD' in name2):
            return 'comprehensive'
            
        # PD vs Controls: subtle early-stage differences
        if (('PD' in name1 and 'HC' in name2) or 
            ('HC' in name1 and 'PD' in name2)):
            return 'biomarker'
            
        # Default to comprehensive for other pairs
        return 'comprehensive'
    
    def _classify_pair(self, features: np.ndarray, labels: np.ndarray, pair_name: str, participant_ids: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform binary classification for a single pair.
        
        Parameters
        ----------
        features : array-like
            Feature matrix for this pair
        labels : array-like
            Binary labels for this pair
        pair_name : str
            Descriptive name for this comparison
        participant_ids : array-like, optional
            Participant identifiers for this pair
            
        Returns
        -------
        dict
            Classification results including metrics and ROC curve
        """
        # Setup cross-validation with participant-aware splitting if IDs provided
        if participant_ids is not None:
            cv = StratifiedGroupKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_method = 'participant_aware'
        else:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_method = 'standard'
        
        # Initialize classifier based on feature size and availability
        if features.shape[1] > 500 and LGBM_AVAILABLE:
            # Use LightGBM for high-dimensional features
            base_classifier = LGBMClassifier(
                random_state=self.random_state,
                verbose=-1,
                class_weight='balanced' if self.use_class_weights else None
            )
        elif features.shape[1] > 100:
            # Use Histogram Gradient Boosting for medium-dimensional features
            base_classifier = HistGradientBoostingClassifier(
                random_state=self.random_state,
                class_weight='balanced' if self.use_class_weights else None
            )
        else:
            # Use Random Forest for lower-dimensional features
            base_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced' if self.use_class_weights else None
            )
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', base_classifier)
        ])
        
        # Cross-validation predictions for ROC curve
        if participant_ids is not None:
            y_scores = cross_val_predict(pipeline, features, labels, cv=cv, groups=participant_ids, method='predict_proba')[:, 1]
            y_pred = cross_val_predict(pipeline, features, labels, cv=cv, groups=participant_ids)
        else:
            y_scores = cross_val_predict(pipeline, features, labels, cv=cv, method='predict_proba')[:, 1]
            y_pred = cross_val_predict(pipeline, features, labels, cv=cv)
        
        # Compute metrics
        auc = roc_auc_score(labels, y_scores)
        accuracy = accuracy_score(labels, y_pred)
        f1 = f1_score(labels, y_pred)
        mcc = matthews_corrcoef(labels, y_pred)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, y_scores)
        
        # Fit final model on all data for future use
        pipeline.fit(features, labels)
        self.fitted_models[pair_name] = pipeline
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'f1_score': f1,
            'mcc': mcc,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'predictions': y_pred,
            'probabilities': y_scores,
            'cv_folds': self.cv_folds,
            'participant_ids': participant_ids
        }
    
    def predict_new_data(self, 
                        comprehensive_features: np.ndarray,
                        biomarker_features: np.ndarray,
                        pair_name: str) -> Dict[str, np.ndarray]:
        """
        Predict on new data using fitted models.
        
        Parameters
        ----------
        comprehensive_features : array-like
            New comprehensive features
        biomarker_features : array-like
            New biomarker features  
        pair_name : str
            Name of the comparison pair
            
        Returns
        -------
        dict
            Predictions and probabilities
        """
        if pair_name not in self.fitted_models:
            raise ValueError(f"No fitted model found for pair: {pair_name}")
            
        # Determine which features to use based on strategy
        strategy = self.strategy_map.get(pair_name, 'comprehensive')
        
        if strategy == 'comprehensive':
            features = comprehensive_features
        else:
            features = biomarker_features
            
        model = self.fitted_models[pair_name]
        
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'strategy_used': strategy
        }
    
    def get_feature_importance(self, pair_name: str, feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Get feature importance for a specific pair.
        
        Parameters
        ----------
        pair_name : str
            Name of the comparison pair
        feature_names : list, optional
            Names of features
            
        Returns
        -------
        dict
            Feature importance scores
        """
        if pair_name not in self.fitted_models:
            raise ValueError(f"No fitted model found for pair: {pair_name}")
            
        model = self.fitted_models[pair_name]
        classifier = model.named_steps['classifier']
        
        # Extract feature importance based on classifier type
        if hasattr(classifier, 'feature_importances_'):
            importance = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importance = np.abs(classifier.coef_[0])
        else:
            importance = np.ones(len(feature_names) if feature_names else 1)
            
        result = {'importance': importance}
        
        if feature_names:
            result['feature_names'] = feature_names
            # Sort by importance
            sorted_idx = np.argsort(importance)[::-1]
            result['sorted_importance'] = importance[sorted_idx]
            result['sorted_names'] = [feature_names[i] for i in sorted_idx]
            
        return result 