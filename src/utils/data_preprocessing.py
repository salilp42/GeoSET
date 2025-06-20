"""Data preprocessing utilities for saccadic eye movement analysis."""

import numpy as np
from typing import Dict, Any, Optional


class DataPreprocessor:
    """Preprocessor for saccadic eye movement data."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.peak_normalize = self.config.get('peak_normalize', True)
        self.trim_points = self.config.get('trim_points', 5)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw saccade data."""
        processed_data = data.copy()
        
        # Peak normalize waveforms
        if self.peak_normalize:
            normalized_waveforms = []
            for waveform in data['waveforms']:
                peak = np.max(np.abs(waveform))
                if peak > 1e-6:
                    normalized_waveforms.append(waveform / peak)
                else:
                    normalized_waveforms.append(waveform)
            processed_data['waveforms'] = np.array(normalized_waveforms)
        
        # Add metadata
        processed_data['metadata'] = {
            'n_saccades': len(data['waveforms']),
            'n_participants': len(np.unique(data['participant_ids'])),
            'preprocessing_applied': {
                'peak_normalize': self.peak_normalize,
                'trim_points': self.trim_points
            }
        }
        
        return processed_data
