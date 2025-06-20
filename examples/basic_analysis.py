#!/usr/bin/env python3
"""
Basic example of geometric analysis for saccadic eye movements.

This script demonstrates how to use the geometric analysis pipeline
for analyzing saccadic eye movement data and generating publication-quality results.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.pipeline import GeometricAnalysisPipeline
from utils.data_loader import SaccadeDataLoader
import yaml


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "analysis_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_synthetic_data(n_participants: int = 50, n_saccades_per_participant: int = 100) -> dict:
    """
    Create synthetic saccade data for demonstration purposes.
    
    In practice, replace this with your actual data loading function.
    """
    print("Creating synthetic saccade data for demonstration...")
    
    # Generate synthetic saccade waveforms (30 timepoints each)
    n_timepoints = 30
    total_saccades = n_participants * n_saccades_per_participant
    
    # Create different patterns for each group
    waveforms = []
    labels = []
    participant_ids = []
    tasks = []
    
    # Group patterns (simplified for demonstration)
    group_patterns = {
        1: {'amplitude': 1.0, 'noise': 0.1, 'asymmetry': 0.0},  # De Novo PD
        4: {'amplitude': 0.8, 'noise': 0.15, 'asymmetry': 0.1}, # Medicated PD  
        5: {'amplitude': 0.6, 'noise': 0.2, 'asymmetry': 0.2},  # PSP
        6: {'amplitude': 1.0, 'noise': 0.05, 'asymmetry': 0.0}  # Healthy Controls
    }
    
    groups = [1, 4, 5, 6]
    participants_per_group = n_participants // len(groups)
    
    pid = 0
    for group in groups:
        pattern = group_patterns[group]
        
        for p in range(participants_per_group):
            for s in range(n_saccades_per_participant):
                # Generate basic saccade template
                t = np.linspace(0, 1, n_timepoints)
                
                # Asymmetric velocity profile
                peak_time = 0.3 + pattern['asymmetry']
                waveform = pattern['amplitude'] * np.exp(-((t - peak_time) / 0.2)**2)
                
                # Add noise
                waveform += np.random.normal(0, pattern['noise'], n_timepoints)
                
                # Peak normalize
                if np.max(np.abs(waveform)) > 0:
                    waveform = waveform / np.max(np.abs(waveform))
                
                waveforms.append(waveform)
                labels.append(group)
                participant_ids.append(pid)
                tasks.append(s % 2)  # Alternate between tasks
            
            pid += 1
    
    # Convert to numpy arrays
    data = {
        'waveforms': np.array(waveforms),
        'labels': np.array(labels),
        'participant_ids': np.array(participant_ids),
        'tasks': np.array(tasks),
        'label_names': {
            1: 'De Novo PD',
            4: 'Medicated PD', 
            5: 'PSP',
            6: 'HC'
        }
    }
    
    print(f"Generated {len(waveforms)} saccades from {n_participants} participants")
    print(f"Groups: {[f'{name} (n={np.sum(np.array(labels)==code)})' for code, name in data['label_names'].items()]}")
    
    return data


def main():
    """Run the complete geometric analysis pipeline."""
    print("=== Geometric Analysis of Saccadic Eye Movements ===\n")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    # Create or load data
    print("Loading saccade data...")
    # For demonstration, we create synthetic data
    # In practice, replace this with:
    # data = SaccadeDataLoader.load_from_files(data_paths)
    data = create_synthetic_data(n_participants=50, n_saccades_per_participant=100)
    
    # Initialize pipeline
    print("Initializing geometric analysis pipeline...")
    pipeline = GeometricAnalysisPipeline(config)
    
    # Run complete analysis
    print("Running complete analysis pipeline...")
    output_dir = "results/basic_analysis"
    results = pipeline.run_analysis(data, output_dir=output_dir)
    
    # Display summary
    print("\n=== Analysis Summary ===")
    summary = pipeline.get_summary()
    
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Key results
    if 'classification' in results:
        print(f"\nClassification Results:")
        print(f"Mean AUC: {results['classification']['mean_auc']:.3f}")
        print(f"Standard deviation: {results['classification']['std_auc']:.3f}")
        
        print(f"\nPairwise AUCs:")
        for metric in results['classification']['metrics_summary']:
            print(f"  {metric['comparison']}: {metric['auc']:.3f} ({metric['strategy']} features)")
    
    # Validation results
    if 'validation' in results:
        validation = results['validation']
        print(f"\nValidation Results:")
        print(f"All tests passed: {validation.get('all_tests_passed', 'Unknown')}")
        
        if 'ablation' in validation:
            print(f"Most important feature group: {validation['ablation'].get('most_important', 'Unknown')}")
        
        if 'permutation' in validation:
            print(f"Permutation test p-values: all < {validation['permutation'].get('max_pvalue', 'Unknown')}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Figures saved to: {output_dir}/figures")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main() 