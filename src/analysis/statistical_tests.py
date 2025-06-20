"""
Statistical analysis framework for saccadic eye movement data.

Implements comprehensive statistical testing including effect sizes,
power analysis, and multiple comparison corrections.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, pearsonr, spearmanr
import warnings

# Optional import for power analysis
try:
    from statsmodels.stats.power import ttest_ind_solve_power
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for geometric saccade analysis.
    
    Provides effect size calculations, power analysis, medication effects,
    and multiple comparison corrections.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize statistical analyzer with configuration."""
        self.config = config or {}
        self.alpha = self.config.get('alpha', 0.05)
        self.correction_method = self.config.get('multiple_comparison_correction', 'fdr_bh')
        
    def analyze(self, results: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.
        
        Parameters
        ----------
        results : dict
            Analysis results from pipeline
        data : dict
            Original data dictionary
            
        Returns
        -------
        dict
            Statistical analysis results
        """
        print("Performing comprehensive statistical analysis...")
        
        statistical_results = {}
        
        # Effect size analysis
        if 'classification' in results:
            statistical_results['effect_sizes'] = self.calculate_effect_sizes(
                results['classification'], data
            )
        
        # Power analysis
        if 'classification' in results and STATSMODELS_AVAILABLE:
            statistical_results['power_analysis'] = self.perform_power_analysis(
                results['classification']
            )
        
        # Medication effects analysis
        statistical_results['medication_effects'] = self.analyze_medication_effects(data)
        
        # Group comparisons
        statistical_results['group_comparisons'] = self.perform_group_comparisons(data)
        
        return statistical_results
    
    def calculate_effect_sizes(self, classification_results: Dict, data: Dict) -> Dict[str, Any]:
        """Calculate Cohen's d effect sizes for pairwise comparisons."""
        print("Calculating effect sizes...")
        
        effect_sizes = {}
        labels = data['labels']
        label_names = data['label_names']
        unique_labels = np.unique(labels)
        
        # Calculate effect sizes between groups
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels[i+1:], i+1):
                pair_name = f"{label_names[label1]} vs {label_names[label2]}"
                
                # Get participant-level data for this comparison
                mask1 = labels == label1
                mask2 = labels == label2
                
                if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                    # Use waveform features for effect size calculation
                    group1_data = data['waveforms'][mask1]
                    group2_data = data['waveforms'][mask2]
                    
                    # Calculate Cohen's d for each feature
                    cohens_d_values = []
                    for feat_idx in range(group1_data.shape[1]):
                        d = self._cohen_d(group1_data[:, feat_idx], group2_data[:, feat_idx])
                        cohens_d_values.append(d)
                    
                    # Summary statistics
                    mean_d = np.mean(np.abs(cohens_d_values))
                    max_d = np.max(np.abs(cohens_d_values))
                    
                    effect_sizes[pair_name] = {
                        'mean_cohens_d': mean_d,
                        'max_cohens_d': max_d,
                        'effect_size_category': self._categorize_effect_size(mean_d),
                        'n_group1': np.sum(mask1),
                        'n_group2': np.sum(mask2)
                    }
        
        return effect_sizes
    
    def perform_power_analysis(self, classification_results: Dict) -> Dict[str, Any]:
        """Perform post-hoc power analysis."""
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available for power analysis'}
        
        print("Performing power analysis...")
        
        power_results = {}
        
        for pair_name, metrics in classification_results.get('metrics_summary', []):
            if isinstance(metrics, dict) and 'auc' in metrics:
                auc = metrics['auc']
                
                # Convert AUC to Cohen's d using normal approximation
                # d = sqrt(2) * Phi^(-1)(AUC) where Phi^(-1) is inverse normal CDF
                if auc > 0.5:
                    z_score = stats.norm.ppf(auc)
                    cohens_d = z_score * np.sqrt(2)
                else:
                    cohens_d = 0
                
                # Sample sizes (estimated from typical neuroscience studies)
                n1 = n2 = 15  # Typical group size in movement disorder studies
                
                try:
                    power = ttest_ind_solve_power(
                        effect_size=cohens_d,
                        nobs1=n1,
                        alpha=self.alpha,
                        power=None,
                        ratio=n2/n1
                    )
                    
                    power_results[pair_name] = {
                        'auc': auc,
                        'cohens_d': cohens_d,
                        'power': power,
                        'sample_size_per_group': n1,
                        'adequate_power': power >= 0.8
                    }
                except:
                    power_results[pair_name] = {
                        'auc': auc,
                        'cohens_d': cohens_d,
                        'power': np.nan,
                        'error': 'Power calculation failed'
                    }
        
        return power_results
    
    def analyze_medication_effects(self, data: Dict) -> Dict[str, Any]:
        """Analyze systematic effects of medication on biomarkers."""
        print("Analyzing medication effects...")
        
        labels = data['labels']
        waveforms = data['waveforms']
        
        # Compare De Novo PD (1) vs Medicated PD (4)
        denovo_mask = labels == 1
        medicated_mask = labels == 4
        
        if np.sum(denovo_mask) == 0 or np.sum(medicated_mask) == 0:
            return {'error': 'Insufficient data for medication effect analysis'}
        
        denovo_data = waveforms[denovo_mask]
        medicated_data = waveforms[medicated_mask]
        
        # Extract biomarker features for comparison
        biomarkers = self._extract_medication_biomarkers(denovo_data, medicated_data)
        
        medication_effects = {}
        
        for biomarker_name, (denovo_values, medicated_values) in biomarkers.items():
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = mannwhitneyu(
                denovo_values, medicated_values, alternative='two-sided'
            )
            
            # Effect size (Cohen's d)
            cohens_d = self._cohen_d(denovo_values, medicated_values)
            
            medication_effects[biomarker_name] = {
                'denovo_mean': np.mean(denovo_values),
                'denovo_std': np.std(denovo_values),
                'medicated_mean': np.mean(medicated_values),
                'medicated_std': np.std(medicated_values),
                'mann_whitney_u': statistic,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'effect_category': self._categorize_effect_size(abs(cohens_d))
            }
        
        # Apply multiple comparison correction
        p_values = [result['p_value'] for result in medication_effects.values()]
        if STATSMODELS_AVAILABLE and len(p_values) > 1:
            rejected, p_corrected, _, _ = multipletests(
                p_values, alpha=self.alpha, method=self.correction_method
            )
            
            for i, biomarker_name in enumerate(medication_effects.keys()):
                medication_effects[biomarker_name]['p_value_corrected'] = p_corrected[i]
                medication_effects[biomarker_name]['significant_corrected'] = rejected[i]
        
        return medication_effects
    
    def perform_group_comparisons(self, data: Dict) -> Dict[str, Any]:
        """Perform statistical comparisons between all groups."""
        print("Performing group comparisons...")
        
        labels = data['labels']
        waveforms = data['waveforms']
        label_names = data['label_names']
        unique_labels = np.unique(labels)
        
        # Overall group comparison (Kruskal-Wallis test)
        group_data = [waveforms[labels == label].flatten() for label in unique_labels]
        
        try:
            kw_statistic, kw_p_value = kruskal(*group_data)
        except:
            kw_statistic, kw_p_value = np.nan, np.nan
        
        # Pairwise comparisons
        pairwise_comparisons = {}
        
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels[i+1:], i+1):
                pair_name = f"{label_names[label1]} vs {label_names[label2]}"
                
                group1_data = waveforms[labels == label1].flatten()
                group2_data = waveforms[labels == label2].flatten()
                
                # Mann-Whitney U test
                try:
                    statistic, p_value = mannwhitneyu(
                        group1_data, group2_data, alternative='two-sided'
                    )
                    cohens_d = self._cohen_d(group1_data, group2_data)
                    
                    pairwise_comparisons[pair_name] = {
                        'mann_whitney_u': statistic,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'effect_category': self._categorize_effect_size(abs(cohens_d))
                    }
                except:
                    pairwise_comparisons[pair_name] = {
                        'error': 'Statistical test failed'
                    }
        
        return {
            'overall_test': {
                'kruskal_wallis_statistic': kw_statistic,
                'kruskal_wallis_p_value': kw_p_value,
                'significant': kw_p_value < self.alpha if not np.isnan(kw_p_value) else False
            },
            'pairwise_comparisons': pairwise_comparisons
        }
    
    def _extract_medication_biomarkers(self, denovo_data: np.ndarray, 
                                     medicated_data: np.ndarray) -> Dict[str, Tuple]:
        """Extract biomarker features for medication effect analysis."""
        biomarkers = {}
        
        def extract_biomarkers_from_group(data):
            tremor_power = []
            peak_velocity = []
            jerk_cost = []
            velocity_asymmetry = []
            endpoint_variability = []
            
            for waveform in data:
                velocity = np.gradient(waveform)
                acceleration = np.gradient(velocity)
                jerk = np.gradient(acceleration)
                
                # Tremor power (8-15 Hz band)
                fft = np.fft.fft(velocity)
                freqs = np.fft.fftfreq(len(velocity), 1/200)
                tremor_mask = (np.abs(freqs) >= 8) & (np.abs(freqs) <= 15)
                tremor_power.append(np.sum(np.abs(fft[tremor_mask])**2))
                
                # Peak velocity
                peak_velocity.append(np.max(np.abs(velocity)))
                
                # Jerk cost
                jerk_cost.append(np.sum(jerk**2))
                
                # Velocity asymmetry
                peak_idx = np.argmax(np.abs(velocity))
                if peak_idx > 0 and peak_idx < len(velocity) - 1:
                    left_energy = np.sum(velocity[:peak_idx]**2)
                    right_energy = np.sum(velocity[peak_idx:]**2)
                    asymmetry = (left_energy - right_energy) / (left_energy + right_energy + 1e-8)
                else:
                    asymmetry = 0
                velocity_asymmetry.append(asymmetry)
                
                # Endpoint variability
                endpoint_variability.append(np.std(waveform[-5:]))
            
            return (tremor_power, peak_velocity, jerk_cost, 
                   velocity_asymmetry, endpoint_variability)
        
        denovo_biomarkers = extract_biomarkers_from_group(denovo_data)
        medicated_biomarkers = extract_biomarkers_from_group(medicated_data)
        
        biomarker_names = ['tremor_power', 'peak_velocity', 'jerk_cost', 
                          'velocity_asymmetry', 'endpoint_variability']
        
        for i, name in enumerate(biomarker_names):
            biomarkers[name] = (denovo_biomarkers[i], medicated_biomarkers[i])
        
        return biomarkers
    
    def _cohen_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        
        if n1 <= 1 or n2 <= 1:
            return 0.0
        
        # Calculate means
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        # Calculate pooled standard deviation
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _categorize_effect_size(self, cohens_d: float) -> str:
        """Categorize effect size magnitude."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
