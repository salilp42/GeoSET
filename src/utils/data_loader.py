"""Data loading utilities for saccadic eye movement analysis."""

import numpy as np
import joblib
from typing import Dict, Any, List, Union
from pathlib import Path


class SaccadeDataLoader:
    """Utility class for loading saccadic eye movement data."""
    
    @staticmethod
    def load_from_joblib(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load saccade data from joblib file."""
        data = joblib.load(filepath)
        
        if isinstance(data, tuple):
            # Handle tuple format (waveforms, labels, participant_ids)
            waveforms, labels, participant_ids = data
            return {
                'waveforms': waveforms,
                'labels': labels,
                'participant_ids': participant_ids,
                'tasks': np.zeros(len(waveforms)),  # Default task assignment
                'label_names': {1: 'Group1', 4: 'Group2', 5: 'Group3', 6: 'Group4'}
            }
        elif isinstance(data, dict):
            return data
        else:
            raise ValueError("Unsupported data format")
    
    @staticmethod
    def load_from_files(data_paths: List[str]) -> Dict[str, Any]:
        """Load and combine data from multiple files."""
        all_waveforms = []
        all_labels = []
        all_pids = []
        all_tasks = []
        
        for i, path in enumerate(data_paths):
            data = SaccadeDataLoader.load_from_joblib(path)
            all_waveforms.append(data['waveforms'])
            all_labels.append(data['labels'])
            all_pids.append(data['participant_ids'])
            all_tasks.append(np.full(len(data['waveforms']), i))
        
        return {
            'waveforms': np.vstack(all_waveforms),
            'labels': np.hstack(all_labels),
            'participant_ids': np.hstack(all_pids),
            'tasks': np.hstack(all_tasks),
            'label_names': {1: 'De Novo PD', 4: 'Medicated PD', 5: 'PSP', 6: 'HC'}
        }
