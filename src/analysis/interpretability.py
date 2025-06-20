"""
Interpretability analysis for geometric saccade analysis.

Provides saliency mapping, temporal attribution analysis, and
group-specific feature importance for biological insight.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class InterpretabilityAnalyzer:
    """
    Interpretability analysis for understanding geometric patterns in saccade data.
    
    Provides temporal saliency mapping, group-specific feature importance,
    and biological interpretation of learned representations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize interpretability analyzer."""
        self.config = config or {}
        self.random_state = self.config.get('random_state', 42)
        
    def analyze_interpretability(self,
                               latent_vectors: np.ndarray,
                               waveforms: np.ndarray,
                               labels: np.ndarray,
                               label_names: Dict[int, str]) -> Dict[str, Any]:
        """
        Perform comprehensive interpretability analysis.
        
        Parameters
        ----------
        latent_vectors : array-like
            Latent space representations
        waveforms : array-like
            Original saccade waveforms
        labels : array-like
            Group labels
        label_names : dict
            Label name mapping
            
        Returns
        -------
        dict
            Interpretability analysis results
        """
        print("Performing interpretability analysis...")
        
        results = {}
        
        # Temporal saliency mapping
        results['temporal_saliency'] = self.compute_temporal_saliency(
            waveforms, labels, label_names
        )
        
        # Group-specific feature importance
        results['group_importance'] = self.compute_group_specific_importance(
            latent_vectors, labels, label_names
        )
        
        # Latent dimension analysis
        results['latent_analysis'] = self.analyze_latent_dimensions(
            latent_vectors, labels, label_names
        )
        
        # Biomarker interpretation
        results['biomarker_interpretation'] = self.interpret_biomarkers(
            waveforms, labels, label_names
        )
        
        return results
    
    def compute_temporal_saliency(self,
                                waveforms: np.ndarray,
                                labels: np.ndarray,
                                label_names: Dict[int, str]) -> Dict[str, Any]:
        """Compute temporal saliency maps for each group."""
        print("Computing temporal saliency maps...")
        
        unique_labels = np.unique(labels)
        n_timepoints = waveforms.shape[1]
        
        saliency_maps = {}
        
        for target_label in unique_labels:
            target_name = label_names[target_label]
            
            # One-vs-rest classification
            binary_labels = (labels == target_label).astype(int)
            
            if np.sum(binary_labels) < 5:  # Skip if too few samples
                continue
                
            # Train classifier on raw waveforms
            X_train, X_test, y_train, y_test = train_test_split(
                waveforms, binary_labels, test_size=0.3, 
                random_state=self.random_state, stratify=binary_labels
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            clf = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                class_weight='balanced'
            )
            clf.fit(X_train_scaled, y_train)
            
            # Compute permutation importance for each timepoint
            perm_importance = permutation_importance(
                clf, X_test_scaled, y_test,
                n_repeats=10,
                random_state=self.random_state
            )
            
            # Extract saliency scores
            saliency_scores = perm_importance.importances_mean
            saliency_std = perm_importance.importances_std
            
            # Compute velocity saliency
            velocity_saliency = self._compute_velocity_saliency(
                waveforms[labels == target_label]
            )
            
            saliency_maps[target_name] = {
                'position_saliency': saliency_scores,
                'position_saliency_std': saliency_std,
                'velocity_saliency': velocity_saliency,
                'peak_saliency_timepoint': np.argmax(saliency_scores),
                'saliency_concentration': np.std(saliency_scores) / np.mean(saliency_scores)
            }
        
        return saliency_maps
    
    def compute_group_specific_importance(self,
                                        latent_vectors: np.ndarray,
                                        labels: np.ndarray,
                                        label_names: Dict[int, str]) -> Dict[str, Any]:
        """Compute group-specific importance in latent space."""
        print("Computing group-specific importance...")
        
        unique_labels = np.unique(labels)
        group_importance = {}
        
        for target_label in unique_labels:
            target_name = label_names[target_label]
            
            # One-vs-rest classification in latent space
            binary_labels = (labels == target_label).astype(int)
            
            if np.sum(binary_labels) < 5:
                continue
            
            # Train classifier
            X_train, X_test, y_train, y_test = train_test_split(
                latent_vectors, binary_labels, test_size=0.3,
                random_state=self.random_state, stratify=binary_labels
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            )
            clf.fit(X_train_scaled, y_train)
            
            # Feature importance in latent space
            latent_importance = clf.feature_importances_
            
            # Identify top dimensions
            top_dimensions = np.argsort(latent_importance)[::-1][:5]
            
            group_importance[target_name] = {
                'latent_importance': latent_importance,
                'top_dimensions': top_dimensions,
                'top_importance_scores': latent_importance[top_dimensions],
                'discriminative_power': np.max(latent_importance) / np.mean(latent_importance)
            }
        
        return group_importance
    
    def analyze_latent_dimensions(self,
                                latent_vectors: np.ndarray,
                                labels: np.ndarray,
                                label_names: Dict[int, str]) -> Dict[str, Any]:
        """Analyze what each latent dimension captures."""
        print("Analyzing latent dimensions...")
        
        n_dimensions = latent_vectors.shape[1]
        unique_labels = np.unique(labels)
        
        dimension_analysis = {}
        
        for dim in range(min(n_dimensions, 10)):  # Analyze top 10 dimensions
            dim_values = latent_vectors[:, dim]
            
            # Group statistics
            group_stats = {}
            for label in unique_labels:
                mask = labels == label
                group_values = dim_values[mask]
                
                group_stats[label_names[label]] = {
                    'mean': np.mean(group_values),
                    'std': np.std(group_values),
                    'median': np.median(group_values),
                    'range': np.ptp(group_values)
                }
            
            # Discriminative power (F-statistic)
            group_data = [dim_values[labels == label] for label in unique_labels]
            f_stat = self._compute_f_statistic(group_data)
            
            # Variance explained by this dimension
            total_variance = np.var(dim_values)
            between_group_variance = self._compute_between_group_variance(
                dim_values, labels, unique_labels
            )
            
            dimension_analysis[f'dimension_{dim}'] = {
                'group_statistics': group_stats,
                'f_statistic': f_stat,
                'total_variance': total_variance,
                'between_group_variance': between_group_variance,
                'discriminative_ratio': between_group_variance / total_variance if total_variance > 0 else 0
            }
        
        return dimension_analysis
    
    def interpret_biomarkers(self,
                           waveforms: np.ndarray,
                           labels: np.ndarray,
                           label_names: Dict[int, str]) -> Dict[str, Any]:
        """Interpret clinical biomarkers across groups."""
        print("Interpreting clinical biomarkers...")
        
        unique_labels = np.unique(labels)
        biomarker_interpretation = {}
        
        # Extract biomarkers for all groups
        all_biomarkers = {}
        biomarker_names = ['tremor_power', 'peak_velocity', 'jerk_cost', 
                          'velocity_asymmetry', 'endpoint_variability']
        
        for label in unique_labels:
            group_waveforms = waveforms[labels == label]
            group_biomarkers = self._extract_biomarkers(group_waveforms)
            all_biomarkers[label_names[label]] = group_biomarkers
        
        # Compare biomarkers across groups
        for i, biomarker_name in enumerate(biomarker_names):
            biomarker_values = {}
            
            for group_name in all_biomarkers.keys():
                biomarker_values[group_name] = all_biomarkers[group_name][:, i]
            
            # Statistical comparison
            group_means = {name: np.mean(values) for name, values in biomarker_values.items()}
            group_stds = {name: np.std(values) for name, values in biomarker_values.items()}
            
            # Effect sizes between groups
            effect_sizes = {}
            group_names = list(group_means.keys())
            for i, group1 in enumerate(group_names):
                for group2 in group_names[i+1:]:
                    values1 = biomarker_values[group1]
                    values2 = biomarker_values[group2]
                    cohen_d = self._cohen_d(values1, values2)
                    effect_sizes[f"{group1}_vs_{group2}"] = cohen_d
            
            biomarker_interpretation[biomarker_name] = {
                'group_means': group_means,
                'group_stds': group_stds,
                'effect_sizes': effect_sizes,
                'clinical_interpretation': self._interpret_biomarker_clinically(
                    biomarker_name, group_means
                )
            }
        
        return biomarker_interpretation
    
    def _compute_velocity_saliency(self, group_waveforms: np.ndarray) -> np.ndarray:
        """Compute velocity-based saliency for a group."""
        velocity_saliency = []
        
        for waveform in group_waveforms:
            velocity = np.gradient(waveform)
            acceleration = np.gradient(velocity)
            
            # Saliency based on velocity profile characteristics
            velocity_magnitude = np.abs(velocity)
            acceleration_magnitude = np.abs(acceleration)
            
            # Combine velocity and acceleration information
            combined_saliency = velocity_magnitude + 0.5 * acceleration_magnitude
            velocity_saliency.append(combined_saliency)
        
        # Average across group
        return np.mean(velocity_saliency, axis=0)
    
    def _compute_f_statistic(self, group_data: List[np.ndarray]) -> float:
        """Compute F-statistic for group differences."""
        try:
            from scipy.stats import f_oneway
            return f_oneway(*group_data)[0]
        except:
            # Fallback calculation
            k = len(group_data)  # number of groups
            n_total = sum(len(group) for group in group_data)
            
            # Grand mean
            all_data = np.concatenate(group_data)
            grand_mean = np.mean(all_data)
            
            # Between-group sum of squares
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 
                           for group in group_data)
            
            # Within-group sum of squares
            ss_within = sum(np.sum((group - np.mean(group))**2) 
                          for group in group_data)
            
            # F-statistic
            if ss_within == 0 or n_total - k == 0:
                return 0
            
            ms_between = ss_between / (k - 1)
            ms_within = ss_within / (n_total - k)
            
            return ms_between / ms_within if ms_within > 0 else 0
    
    def _compute_between_group_variance(self,
                                      values: np.ndarray,
                                      labels: np.ndarray,
                                      unique_labels: np.ndarray) -> float:
        """Compute between-group variance."""
        grand_mean = np.mean(values)
        between_var = 0
        
        for label in unique_labels:
            group_values = values[labels == label]
            group_mean = np.mean(group_values)
            group_size = len(group_values)
            between_var += group_size * (group_mean - grand_mean)**2
        
        return between_var / len(values)
    
    def _extract_biomarkers(self, waveforms: np.ndarray) -> np.ndarray:
        """Extract clinical biomarkers from waveforms."""
        biomarkers = []
        
        for waveform in waveforms:
            velocity = np.gradient(waveform)
            acceleration = np.gradient(velocity)
            jerk = np.gradient(acceleration)
            
            # Tremor power (8-15 Hz)
            fft = np.fft.fft(velocity)
            freqs = np.fft.fftfreq(len(velocity), 1/200)
            tremor_mask = (np.abs(freqs) >= 8) & (np.abs(freqs) <= 15)
            tremor_power = np.sum(np.abs(fft[tremor_mask])**2)
            
            # Peak velocity
            peak_velocity = np.max(np.abs(velocity))
            
            # Jerk cost
            jerk_cost = np.sum(jerk**2)
            
            # Velocity asymmetry
            peak_idx = np.argmax(np.abs(velocity))
            if peak_idx > 0 and peak_idx < len(velocity) - 1:
                left_energy = np.sum(velocity[:peak_idx]**2)
                right_energy = np.sum(velocity[peak_idx:]**2)
                velocity_asymmetry = (left_energy - right_energy) / (left_energy + right_energy + 1e-8)
            else:
                velocity_asymmetry = 0
            
            # Endpoint variability
            endpoint_variability = np.std(waveform[-5:])
            
            biomarkers.append([tremor_power, peak_velocity, jerk_cost, 
                             velocity_asymmetry, endpoint_variability])
        
        return np.array(biomarkers)
    
    def _cohen_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        
        if n1 <= 1 or n2 <= 1:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _interpret_biomarker_clinically(self, 
                                      biomarker_name: str,
                                      group_means: Dict[str, float]) -> str:
        """Provide clinical interpretation of biomarker patterns."""
        interpretations = {
            'tremor_power': {
                'high': 'Increased tremor activity suggests dopaminergic dysfunction',
                'low': 'Reduced tremor power indicates better motor control'
            },
            'peak_velocity': {
                'high': 'Preserved peak velocity suggests intact burst neuron function',
                'low': 'Reduced peak velocity indicates brainstem pathology'
            },
            'jerk_cost': {
                'high': 'Increased jerk cost reflects movement smoothness impairment',
                'low': 'Lower jerk cost indicates preserved movement coordination'
            },
            'velocity_asymmetry': {
                'high': 'Asymmetric velocity profiles suggest impaired motor planning',
                'low': 'Symmetric profiles indicate preserved saccadic programming'
            },
            'endpoint_variability': {
                'high': 'Increased endpoint variability reflects cerebellar dysfunction',
                'low': 'Stable endpoints suggest preserved accuracy control'
            }
        }
        
        if biomarker_name not in interpretations:
            return "No clinical interpretation available"
        
        # Find group with highest mean
        max_group = max(group_means.items(), key=lambda x: x[1])
        min_group = min(group_means.items(), key=lambda x: x[1])
        
        interpretation = interpretations[biomarker_name]
        
        return (f"{max_group[0]} shows highest {biomarker_name} ({max_group[1]:.3f}): "
                f"{interpretation['high']}. "
                f"{min_group[0]} shows lowest values ({min_group[1]:.3f}): "
                f"{interpretation['low']}.")
