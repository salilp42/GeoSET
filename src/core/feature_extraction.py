"""
Feature extraction module for saccadic eye movements.

This module implements comprehensive feature extraction including temporal,
spectral, kinematic, and clinical biomarker features.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Optional imports
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

warnings.filterwarnings('ignore')


class SaccadeFeatureExtractor:
    """
    Comprehensive feature extractor for saccadic eye movements.
    
    Extracts temporal, spectral, kinematic, and clinical biomarker features
    from saccadic waveforms for geometric analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the feature extractor with configuration."""
        self.config = config or {}
        self.include_biomarkers = self.config.get('include_biomarkers', True)
        self.include_spectral = self.config.get('include_spectral', True)
        self.include_kinematic = self.config.get('include_kinematic', True)
        
    def extract_features(self, 
                        waveforms: np.ndarray,
                        labels: np.ndarray,
                        tasks: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive features from saccade waveforms.
        
        Parameters
        ----------
        waveforms : array-like of shape (n_saccades, n_timepoints)
            Saccadic waveforms
        labels : array-like of shape (n_saccades,)
            Group labels
        tasks : array-like of shape (n_saccades,)
            Task indicators (0=antisaccade, 1=prosaccade)
            
        Returns
        -------
        dict
            Dictionary containing extracted features and metadata
        """
        print(f"Extracting features from {len(waveforms)} saccades...")
        
        # Extract different feature types
        temporal_features = self._extract_temporal_features(waveforms)
        
        if self.include_kinematic:
            kinematic_features = self._extract_kinematic_features(waveforms)
        else:
            kinematic_features = np.array([])
            
        if self.include_spectral:
            spectral_features = self._extract_spectral_features(waveforms)
        else:
            spectral_features = np.array([])
            
        if self.include_biomarkers:
            biomarker_features = self._extract_biomarker_features(waveforms, labels, tasks)
        else:
            biomarker_features = np.array([])
        
        # Combine all features
        feature_list = [temporal_features]
        if kinematic_features.size > 0:
            feature_list.append(kinematic_features)
        if spectral_features.size > 0:
            feature_list.append(spectral_features)
        if biomarker_features.size > 0:
            feature_list.append(biomarker_features)
            
        saccade_features = np.hstack(feature_list)
        
        return {
            'saccade_features': saccade_features,
            'biomarkers': biomarker_features if biomarker_features.size > 0 else temporal_features,
            'temporal': temporal_features,
            'kinematic': kinematic_features,
            'spectral': spectral_features,
            'metadata': {
                'n_features': saccade_features.shape[1],
                'n_temporal': temporal_features.shape[1],
                'n_kinematic': kinematic_features.shape[1] if kinematic_features.size > 0 else 0,
                'n_spectral': spectral_features.shape[1] if spectral_features.size > 0 else 0,
                'n_biomarkers': biomarker_features.shape[1] if biomarker_features.size > 0 else 0
            }
        }
    
    def _extract_temporal_features(self, waveforms: np.ndarray) -> np.ndarray:
        """Extract temporal features from position and velocity."""
        features = []
        
        for waveform in waveforms:
            # Position features (normalized)
            features.extend(waveform)
            
            # Velocity features
            velocity = np.gradient(waveform)
            features.extend(velocity)
            
        return np.array(features).reshape(len(waveforms), -1)
    
    def _extract_kinematic_features(self, waveforms: np.ndarray) -> np.ndarray:
        """Extract kinematic features (peaks, timing, etc.)."""
        features = []
        
        for waveform in waveforms:
            velocity = np.gradient(waveform)
            acceleration = np.gradient(velocity)
            
            # Peak features
            peak_velocity = np.max(np.abs(velocity))
            peak_acceleration = np.max(np.abs(acceleration))
            
            # Timing features
            peak_time = np.argmax(np.abs(velocity)) / len(waveform)
            
            # Asymmetry
            peak_idx = np.argmax(np.abs(velocity))
            left_energy = np.sum(velocity[:peak_idx]**2) if peak_idx > 0 else 0
            right_energy = np.sum(velocity[peak_idx:]**2) if peak_idx < len(velocity) else 0
            asymmetry = (left_energy - right_energy) / (left_energy + right_energy + 1e-8)
            
            features.append([peak_velocity, peak_acceleration, peak_time, asymmetry])
            
        return np.array(features)
    
    def _extract_spectral_features(self, waveforms: np.ndarray) -> np.ndarray:
        """Extract spectral features using FFT and wavelets."""
        features = []
        
        for waveform in waveforms:
            velocity = np.gradient(waveform)
            
            # FFT features
            fft = np.fft.fft(velocity)
            power_spectrum = np.abs(fft)**2
            
            # Frequency bands (assuming 200 Hz sampling, 30 points = 150ms)
            # Tremor band: 8-15 Hz
            freqs = np.fft.fftfreq(len(velocity), 1/200)  # 200 Hz sampling
            tremor_mask = (np.abs(freqs) >= 8) & (np.abs(freqs) <= 15)
            tremor_power = np.sum(power_spectrum[tremor_mask])
            
            spectral_feats = [tremor_power]
            
            # Wavelet features if available
            if PYWAVELETS_AVAILABLE:
                try:
                    coeffs = pywt.wavedec(velocity, 'db4', level=3)
                    # Energy at different scales
                    for coeff in coeffs[1:3]:  # Detail coefficients at levels 2-3
                        spectral_feats.append(np.sum(coeff**2))
                except:
                    # Fallback if wavelet decomposition fails
                    spectral_feats.extend([0, 0])
            else:
                spectral_feats.extend([0, 0])
                
            features.append(spectral_feats)
            
        return np.array(features)
    
    def _extract_biomarker_features(self, 
                                  waveforms: np.ndarray,
                                  labels: np.ndarray,
                                  tasks: np.ndarray) -> np.ndarray:
        """Extract clinical biomarker features."""
        features = []
        
        for i, waveform in enumerate(waveforms):
            velocity = np.gradient(waveform)
            acceleration = np.gradient(velocity)
            jerk = np.gradient(acceleration)
            
            # Jerk cost (movement smoothness)
            jerk_cost = np.sum(jerk**2)
            
            # Velocity asymmetry
            peak_idx = np.argmax(np.abs(velocity))
            if peak_idx > 0 and peak_idx < len(velocity) - 1:
                left_energy = np.sum(velocity[:peak_idx]**2)
                right_energy = np.sum(velocity[peak_idx:]**2)
                velocity_asymmetry = (left_energy - right_energy) / (left_energy + right_energy + 1e-8)
            else:
                velocity_asymmetry = 0
                
            # Endpoint variability (last 5 points)
            endpoint_variability = np.std(waveform[-5:])
            
            # Peak velocity
            peak_velocity = np.max(np.abs(velocity))
            
            # Velocity profile kurtosis
            velocity_kurtosis = self._kurtosis(velocity)
            
            biomarker_feats = [
                jerk_cost,
                velocity_asymmetry, 
                endpoint_variability,
                peak_velocity,
                velocity_kurtosis
            ]
            
            features.append(biomarker_feats)
            
        return np.array(features)
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Compute kurtosis of a signal."""
        x_centered = x - np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.mean((x_centered / std)**4) - 3
