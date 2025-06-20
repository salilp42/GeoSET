"""
Core geometric analysis framework for saccadic eye movements.

This module provides the main pipeline and core functionality for analyzing
saccadic eye movement patterns using geometric and topological methods.
"""

from .pipeline import GeometricAnalysisPipeline
from .latent_space import LatentSpaceAnalyzer
from .feature_extraction import SaccadeFeatureExtractor

__all__ = [
    'GeometricAnalysisPipeline',
    'LatentSpaceAnalyzer', 
    'SaccadeFeatureExtractor'
] 