"""
Main geometric analysis pipeline for saccadic eye movements.

This module implements the complete analysis workflow including data preprocessing,
latent space construction, geometric analysis, and comprehensive validation.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

from ..utils.data_preprocessing import DataPreprocessor
from .latent_space import LatentSpaceAnalyzer
from .feature_extraction import SaccadeFeatureExtractor
from ..analysis.feature_scalpel import FeatureScalpelClassifier
from ..analysis.validation import ValidationFramework
from ..analysis.statistical_tests import StatisticalAnalyzer
from ..visualization.geometric_plots import GeometricVisualizer
from ..utils.topology_analysis import TopologyAnalyzer
from ..utils.graph_analysis import GraphAnalyzer

warnings.filterwarnings('ignore')


class GeometricAnalysisPipeline:
    """
    Complete pipeline for geometric analysis of saccadic eye movements.
    
    This class orchestrates the entire analysis workflow from raw saccade data
    to final results including classification, validation, and visualization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the geometric analysis pipeline.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters for the analysis. If None, uses defaults.
        """
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
            
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))
        self.latent_analyzer = LatentSpaceAnalyzer(self.config.get('latent_space', {}))
        self.feature_extractor = SaccadeFeatureExtractor(self.config.get('features', {}))
        self.classifier = FeatureScalpelClassifier(self.config.get('classification', {}))
        self.validator = ValidationFramework(self.config.get('validation', {}))
        self.statistical_analyzer = StatisticalAnalyzer(self.config.get('statistics', {}))
        self.visualizer = GeometricVisualizer(self.config.get('visualization', {}))
        self.topology_analyzer = TopologyAnalyzer(self.config.get('topology', {}))
        self.graph_analyzer = GraphAnalyzer(self.config.get('graph', {}))
        
        # Results storage
        self.results = {}
        
    def _load_default_config(self) -> Dict:
        """Load default configuration parameters."""
        return {
            'preprocessing': {
                'peak_normalize': True,
                'trim_points': 5,
                'quality_threshold': 0.1
            },
            'latent_space': {
                'variance_threshold': 0.95,
                'use_randomized_svd': True,
                'random_state': 42
            },
            'features': {
                'include_biomarkers': True,
                'include_spectral': True,
                'include_kinematic': True
            },
            'classification': {
                'cv_folds': 5,
                'random_state': 42,
                'use_class_weights': True
            },
            'validation': {
                'n_permutations': 1000,
                'run_ablation': True,
                'run_power_analysis': True
            },
            'statistics': {
                'alpha': 0.05,
                'multiple_comparison_correction': 'fdr_bh'
            },
            'visualization': {
                'figure_format': 'png',
                'dpi': 300,
                'style': 'publication'
            },
            'topology': {
                'max_dimension': 1,
                'distance_metric': 'euclidean'
            },
            'graph': {
                'n_neighbors': 10,
                'connectivity_threshold': 0.5
            }
        }
    
    def run_analysis(self, 
                    data: Dict[str, np.ndarray], 
                    output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete geometric analysis pipeline.
        
        Parameters
        ----------
        data : dict
            Dictionary containing saccade data with keys:
            - 'waveforms': array of shape (n_saccades, n_timepoints)
            - 'labels': array of shape (n_saccades,)
            - 'participant_ids': array of shape (n_saccades,)
            - 'tasks': array of shape (n_saccades,) 
            - 'label_names': dict mapping label codes to names
            
        output_dir : str, optional
            Directory to save results and figures
            
        Returns
        -------
        dict
            Complete analysis results including metrics, statistics, and paths
        """
        print("Starting geometric analysis pipeline...")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path("results")
            output_path.mkdir(exist_ok=True)
            
        # Step 1: Data preprocessing
        print("Step 1: Preprocessing data...")
        processed_data = self.preprocessor.process(data)
        self.results['preprocessing'] = processed_data['metadata']
        
        # Step 2: Feature extraction
        print("Step 2: Extracting features...")
        features = self.feature_extractor.extract_features(
            processed_data['waveforms'],
            processed_data['labels'],
            processed_data['tasks']
        )
        self.results['features'] = features['metadata']
        
        # Step 3: Latent space construction
        print("Step 3: Constructing latent space...")
        latent_results = self.latent_analyzer.analyze(features['saccade_features'])
        self.results['latent_space'] = latent_results
        
        # Step 4: Hierarchical aggregation to participant level
        print("Step 4: Aggregating to participant level...")
        participant_features = self._aggregate_to_participants(
            latent_results['latent_vectors'],
            features['biomarkers'],
            processed_data['participant_ids'],
            processed_data['tasks']
        )
        
        # Step 5: Feature Scalpel classification
        print("Step 5: Running Feature Scalpel classification...")
        classification_results = self.classifier.fit_predict(
            participant_features['comprehensive'],
            participant_features['biomarker_only'],
            participant_features['labels'],
            processed_data['label_names']
        )
        self.results['classification'] = classification_results
        
        # Step 6: Topology analysis
        print("Step 6: Analyzing topology...")
        topology_results = self.topology_analyzer.analyze(
            latent_results['latent_vectors'],
            processed_data['labels'],
            processed_data['label_names']
        )
        self.results['topology'] = topology_results
        
        # Step 7: Graph analysis
        print("Step 7: Analyzing graph structure...")
        graph_results = self.graph_analyzer.analyze(
            latent_results['latent_vectors'],
            processed_data['labels'],
            processed_data['label_names']
        )
        self.results['graph'] = graph_results
        
        # Step 8: Comprehensive validation
        print("Step 8: Running validation framework...")
        validation_results = self.validator.validate(
            participant_features,
            processed_data,
            features
        )
        self.results['validation'] = validation_results
        
        # Step 9: Statistical analysis
        print("Step 9: Performing statistical analysis...")
        statistical_results = self.statistical_analyzer.analyze(
            self.results,
            processed_data
        )
        self.results['statistics'] = statistical_results
        
        # Step 10: Generate visualizations
        print("Step 10: Generating visualizations...")
        visualization_paths = self._generate_visualizations(
            processed_data,
            latent_results,
            topology_results,
            graph_results,
            classification_results,
            output_path
        )
        self.results['visualizations'] = visualization_paths
        
        # Step 11: Save comprehensive results
        print("Step 11: Saving results...")
        self._save_results(output_path)
        
        print("Analysis complete!")
        return self.results
    
    def _aggregate_to_participants(self, 
                                 latent_vectors: np.ndarray,
                                 biomarkers: np.ndarray,
                                 participant_ids: np.ndarray,
                                 tasks: np.ndarray) -> Dict[str, np.ndarray]:
        """Aggregate saccade-level features to participant level."""
        unique_pids = np.unique(participant_ids)
        n_participants = len(unique_pids)
        
        # Initialize aggregated features
        comprehensive_features = []
        biomarker_features = []
        participant_labels = []
        
        for pid in unique_pids:
            pid_mask = participant_ids == pid
            pid_latent = latent_vectors[pid_mask]
            pid_biomarkers = biomarkers[pid_mask]
            pid_tasks = tasks[pid_mask]
            
            # Aggregate by task
            task_features = []
            for task in [0, 1]:  # antisaccades, prosaccades
                task_mask = pid_tasks == task
                if np.any(task_mask):
                    task_latent = pid_latent[task_mask]
                    # Statistical summaries
                    means = np.mean(task_latent, axis=0)
                    stds = np.std(task_latent, axis=0)
                    quantiles = np.percentile(task_latent, [10, 25, 50, 75, 90], axis=0)
                    task_features.extend([means, stds, quantiles.flatten()])
                else:
                    # Fill with zeros if task not present
                    n_dims = latent_vectors.shape[1]
                    task_features.extend([
                        np.zeros(n_dims),
                        np.zeros(n_dims), 
                        np.zeros(n_dims * 5)
                    ])
            
            # Combine task features
            comprehensive_feat = np.concatenate(task_features)
            comprehensive_features.append(comprehensive_feat)
            
            # Biomarker-only features (simplified)
            biomarker_feat = np.mean(pid_biomarkers, axis=0)
            biomarker_features.append(biomarker_feat)
            
            # Get participant label (assuming consistent within participant)
            participant_labels.append(processed_data['labels'][pid_mask][0])
        
        return {
            'comprehensive': np.array(comprehensive_features),
            'biomarker_only': np.array(biomarker_features),
            'labels': np.array(participant_labels),
            'participant_ids': unique_pids
        }
    
    def _generate_visualizations(self, 
                               data: Dict,
                               latent_results: Dict,
                               topology_results: Dict,
                               graph_results: Dict,
                               classification_results: Dict,
                               output_path: Path) -> Dict[str, str]:
        """Generate all visualization figures."""
        viz_path = output_path / "figures"
        viz_path.mkdir(exist_ok=True)
        
        paths = {}
        
        # Figure 1: Geometric latent space
        paths['geometry'] = self.visualizer.plot_latent_space(
            latent_results['embedding'],
            data['labels'],
            data['label_names'],
            str(viz_path / "figure1_geometry.png")
        )
        
        # Figure 2: Topology analysis
        paths['topology'] = self.visualizer.plot_topology(
            topology_results,
            str(viz_path / "figure2_topology.png")
        )
        
        # Figure 3: Graph analysis
        paths['graph'] = self.visualizer.plot_graph_analysis(
            graph_results,
            str(viz_path / "figure3_graph.png")
        )
        
        # Figure 4: Feature attribution
        paths['attribution'] = self.visualizer.plot_feature_attribution(
            latent_results['attributions'],
            str(viz_path / "figure4_attribution.png")
        )
        
        # Figure 5: Classification performance
        paths['classification'] = self.visualizer.plot_classification_results(
            classification_results,
            str(viz_path / "figure5_classification.png")
        )
        
        return paths
    
    def _save_results(self, output_path: Path):
        """Save comprehensive results to JSON and CSV files."""
        results_path = output_path / "results"
        results_path.mkdir(exist_ok=True)
        
        # Save main results as JSON
        with open(results_path / "comprehensive_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save classification metrics as CSV
        if 'classification' in self.results:
            metrics_df = pd.DataFrame(self.results['classification']['metrics'])
            metrics_df.to_csv(results_path / "classification_metrics.csv", index=False)
        
        # Save statistical test results
        if 'statistics' in self.results:
            stats_df = pd.DataFrame(self.results['statistics'])
            stats_df.to_csv(results_path / "statistical_tests.csv", index=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key results."""
        if not self.results:
            return {"error": "No analysis has been run yet"}
        
        summary = {
            "n_saccades": self.results.get('preprocessing', {}).get('n_saccades', 0),
            "n_participants": self.results.get('preprocessing', {}).get('n_participants', 0),
            "latent_dimensions": self.results.get('latent_space', {}).get('n_components', 0),
            "variance_explained": self.results.get('latent_space', {}).get('variance_explained', 0),
            "mean_auc": self.results.get('classification', {}).get('mean_auc', 0),
            "validation_passed": self.results.get('validation', {}).get('all_tests_passed', False)
        }
        
        return summary 