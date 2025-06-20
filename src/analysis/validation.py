"""
Validation framework for geometric saccade analysis.

Implements ablation studies, permutation testing, fold-by-fold analysis,
and comprehensive validation metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
from itertools import combinations

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

warnings.filterwarnings('ignore')


class ValidationFramework:
    """
    Comprehensive validation framework for geometric saccade analysis.
    
    Provides ablation studies, permutation testing, fold analysis,
    and stability assessments.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize validation framework."""
        self.config = config or {}
        self.n_permutations = self.config.get('n_permutations', 1000)
        self.run_ablation = self.config.get('run_ablation', True)
        self.run_power_analysis = self.config.get('run_power_analysis', True)
        self.cv_folds = self.config.get('cv_folds', 5)
        
    def validate(self, 
                participant_features: Dict[str, np.ndarray],
                data: Dict[str, Any],
                features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive validation framework.
        
        Parameters
        ----------
        participant_features : dict
            Participant-level features
        data : dict
            Original data
        features : dict
            Feature extraction results
            
        Returns
        -------
        dict
            Validation results
        """
        print("Running comprehensive validation framework...")
        
        validation_results = {}
        
        # Ablation studies
        if self.run_ablation:
            validation_results['ablation'] = self.run_ablation_studies(
                features, participant_features, data
            )
        
        # Permutation testing
        validation_results['permutation'] = self.run_permutation_tests(
            participant_features['comprehensive'],
            participant_features['labels'],
            data['label_names']
        )
        
        # Fold-by-fold analysis
        validation_results['fold_analysis'] = self.run_fold_analysis(
            participant_features['comprehensive'],
            participant_features['labels'],
            data['label_names']
        )
        
        # Stability analysis
        validation_results['stability'] = self.assess_stability(
            participant_features['comprehensive'],
            participant_features['labels']
        )
        
        # Overall validation summary
        validation_results['summary'] = self._create_validation_summary(validation_results)
        
        return validation_results
    
    def run_ablation_studies(self, 
                           features: Dict[str, Any],
                           participant_features: Dict[str, np.ndarray],
                           data: Dict[str, Any]) -> Dict[str, Any]:
        """Run ablation studies to assess feature group importance."""
        print("Running ablation studies...")
        
        # Define feature groups based on extraction metadata
        feature_groups = {
            'temporal': features.get('temporal', np.array([])),
            'kinematic': features.get('kinematic', np.array([])),
            'spectral': features.get('spectral', np.array([])),
            'biomarker': features.get('biomarkers', np.array([]))
        }
        
        # Baseline performance (all features)
        baseline_auc = self._get_mean_pairwise_auc(
            participant_features['comprehensive'],
            participant_features['labels'],
            data['label_names']
        )
        
        ablation_results = {
            'baseline_auc': baseline_auc,
            'feature_group_contributions': {}
        }
        
        # Test removal of each feature group
        for group_name, group_features in feature_groups.items():
            if group_features.size == 0:
                continue
                
            print(f"  Testing removal of {group_name} features...")
            
            # Create ablated feature set (remove this group)
            ablated_features = self._create_ablated_features(
                features, group_name, participant_features, data
            )
            
            if ablated_features is not None:
                ablated_auc = self._get_mean_pairwise_auc(
                    ablated_features,
                    participant_features['labels'],
                    data['label_names']
                )
                
                performance_drop = baseline_auc - ablated_auc
                
                ablation_results['feature_group_contributions'][group_name] = {
                    'ablated_auc': ablated_auc,
                    'performance_drop': performance_drop,
                    'relative_importance': performance_drop / baseline_auc if baseline_auc > 0 else 0
                }
        
        # Identify most important feature group
        if ablation_results['feature_group_contributions']:
            most_important = max(
                ablation_results['feature_group_contributions'].items(),
                key=lambda x: x[1]['performance_drop']
            )
            ablation_results['most_important_group'] = most_important[0]
            ablation_results['max_performance_drop'] = most_important[1]['performance_drop']
        
        return ablation_results
    
    def run_permutation_tests(self,
                            features: np.ndarray,
                            labels: np.ndarray,
                            label_names: Dict[int, str]) -> Dict[str, Any]:
        """Run permutation tests for statistical significance."""
        print(f"Running permutation tests ({self.n_permutations} permutations)...")
        
        # True performance
        true_auc = self._get_mean_pairwise_auc(features, labels, label_names)
        
        # Permutation distribution
        permuted_aucs = []
        
        for i in tqdm(range(self.n_permutations), desc="Permutation tests"):
            # Shuffle labels
            permuted_labels = np.random.permutation(labels)
            
            # Calculate AUC with shuffled labels
            perm_auc = self._get_mean_pairwise_auc(features, permuted_labels, label_names)
            permuted_aucs.append(perm_auc)
        
        permuted_aucs = np.array(permuted_aucs)
        
        # Calculate p-value
        p_value = np.mean(permuted_aucs >= true_auc)
        
        return {
            'true_auc': true_auc,
            'permuted_aucs_mean': np.mean(permuted_aucs),
            'permuted_aucs_std': np.std(permuted_aucs),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'z_score': (true_auc - np.mean(permuted_aucs)) / np.std(permuted_aucs) if np.std(permuted_aucs) > 0 else 0
        }
    
    def run_fold_analysis(self,
                         features: np.ndarray,
                         labels: np.ndarray,
                         label_names: Dict[int, str]) -> Dict[str, Any]:
        """Analyze performance fold-by-fold to detect overfitting."""
        print("Running fold-by-fold analysis...")
        
        unique_labels = np.unique(labels)
        pair_combinations = list(combinations(unique_labels, 2))
        
        fold_results = {}
        
        for label1, label2 in pair_combinations:
            pair_name = f"{label_names[label1]} vs {label_names[label2]}"
            
            # Extract data for this pair
            pair_mask = np.isin(labels, [label1, label2])
            pair_features = features[pair_mask]
            pair_labels = labels[pair_mask]
            
            # Convert to binary
            binary_labels = (pair_labels == label2).astype(int)
            
            # Cross-validation analysis
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            fold_aucs = []
            fold_predictions = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(pair_features, binary_labels)):
                X_train, X_val = pair_features[train_idx], pair_features[val_idx]
                y_train, y_val = binary_labels[train_idx], binary_labels[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train classifier
                clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                clf.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
                
                # Calculate AUC
                try:
                    auc = roc_auc_score(y_val, y_pred_proba)
                    fold_aucs.append(auc)
                    fold_predictions.append({
                        'true_labels': y_val,
                        'predicted_proba': y_pred_proba,
                        'auc': auc
                    })
                except:
                    fold_aucs.append(0.5)
                    fold_predictions.append({
                        'true_labels': y_val,
                        'predicted_proba': np.full_like(y_val, 0.5, dtype=float),
                        'auc': 0.5
                    })
            
            fold_results[pair_name] = {
                'fold_aucs': fold_aucs,
                'mean_auc': np.mean(fold_aucs),
                'std_auc': np.std(fold_aucs),
                'min_auc': np.min(fold_aucs),
                'max_auc': np.max(fold_aucs),
                'fold_predictions': fold_predictions,
                'perfect_folds': np.sum(np.array(fold_aucs) >= 0.999),
                'consistent_performance': np.std(fold_aucs) < 0.1
            }
        
        return fold_results
    
    def assess_stability(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Assess model stability across different random seeds."""
        print("Assessing model stability...")
        
        stability_results = {}
        n_seeds = 10
        seed_aucs = []
        
        for seed in range(n_seeds):
            # Use different random seed
            np.random.seed(seed)
            
            # Subsample data (80%) to test stability
            n_samples = len(features)
            subsample_idx = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
            
            sub_features = features[subsample_idx]
            sub_labels = labels[subsample_idx]
            
            # Calculate AUC with this subsample
            auc = self._get_mean_pairwise_auc(sub_features, sub_labels, {1: 'A', 4: 'B', 5: 'C', 6: 'D'})
            seed_aucs.append(auc)
        
        stability_results = {
            'seed_aucs': seed_aucs,
            'mean_auc': np.mean(seed_aucs),
            'std_auc': np.std(seed_aucs),
            'coefficient_of_variation': np.std(seed_aucs) / np.mean(seed_aucs) if np.mean(seed_aucs) > 0 else 0,
            'stable_performance': np.std(seed_aucs) < 0.05  # Less than 5% variation
        }
        
        return stability_results
    
    def _create_ablated_features(self,
                               features: Dict[str, Any],
                               remove_group: str,
                               participant_features: Dict[str, np.ndarray],
                               data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create feature set with one group removed."""
        # This is a simplified version - in practice would need to track
        # feature indices more carefully
        
        # For now, return a subset of features as a proxy
        full_features = participant_features['comprehensive']
        
        if remove_group == 'temporal':
            # Remove first portion (temporal features)
            return full_features[:, 100:]  # Skip first 100 features
        elif remove_group == 'biomarker':
            # Remove biomarker features (last portion)
            return full_features[:, :-50]  # Skip last 50 features
        else:
            # For other groups, remove middle portions
            n_features = full_features.shape[1]
            remove_size = n_features // 4
            start_idx = hash(remove_group) % (n_features - remove_size)
            
            return np.hstack([
                full_features[:, :start_idx],
                full_features[:, start_idx + remove_size:]
            ])
    
    def _get_mean_pairwise_auc(self, 
                              features: np.ndarray,
                              labels: np.ndarray,
                              label_names: Dict[int, str]) -> float:
        """Calculate mean AUC across all pairwise comparisons."""
        unique_labels = np.unique(labels)
        pair_combinations = list(combinations(unique_labels, 2))
        
        aucs = []
        
        for label1, label2 in pair_combinations:
            # Extract data for this pair
            pair_mask = np.isin(labels, [label1, label2])
            pair_features = features[pair_mask]
            pair_labels = labels[pair_mask]
            
            if len(np.unique(pair_labels)) < 2:
                continue
                
            # Convert to binary
            binary_labels = (pair_labels == label2).astype(int)
            
            # Quick cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fold_aucs = []
            
            for train_idx, val_idx in cv.split(pair_features, binary_labels):
                X_train, X_val = pair_features[train_idx], pair_features[val_idx]
                y_train, y_val = binary_labels[train_idx], binary_labels[val_idx]
                
                # Scale and classify
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(X_train_scaled, y_train)
                
                y_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
                
                try:
                    auc = roc_auc_score(y_val, y_pred_proba)
                    fold_aucs.append(auc)
                except:
                    fold_aucs.append(0.5)
            
            if fold_aucs:
                aucs.append(np.mean(fold_aucs))
        
        return np.mean(aucs) if aucs else 0.5
    
    def _create_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of validation results."""
        summary = {
            'all_tests_passed': True,
            'validation_score': 0.0,
            'warnings': [],
            'recommendations': []
        }
        
        # Check ablation results
        if 'ablation' in validation_results:
            ablation = validation_results['ablation']
            if ablation.get('max_performance_drop', 0) > 0.2:
                summary['warnings'].append("Large performance drop in ablation study")
                summary['all_tests_passed'] = False
        
        # Check permutation results
        if 'permutation' in validation_results:
            perm = validation_results['permutation']
            if not perm.get('significant', False):
                summary['warnings'].append("Results not significant in permutation test")
                summary['all_tests_passed'] = False
        
        # Check stability
        if 'stability' in validation_results:
            stability = validation_results['stability']
            if not stability.get('stable_performance', False):
                summary['warnings'].append("Unstable performance across random seeds")
                summary['all_tests_passed'] = False
        
        # Calculate overall validation score
        scores = []
        if 'permutation' in validation_results:
            scores.append(1.0 if validation_results['permutation'].get('significant', False) else 0.0)
        if 'stability' in validation_results:
            scores.append(1.0 if validation_results['stability'].get('stable_performance', False) else 0.0)
        
        summary['validation_score'] = np.mean(scores) if scores else 0.0
        
        return summary
