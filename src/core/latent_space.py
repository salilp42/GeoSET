"""
Latent space analysis for saccadic eye movements.

This module implements PCA autoencoding for dimensionality reduction and
supervised UMAP for visualization and geometric analysis.
"""

import numpy as np
import warnings
from typing import Dict, Optional, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional imports with fallbacks
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Using PCA for 2D projection.")

warnings.filterwarnings('ignore')


class LatentSpaceAnalyzer:
    """
    Analyzer for constructing and analyzing latent spaces from saccade features.
    
    Uses PCA autoencoding for dimensionality reduction with configurable variance
    threshold, and supervised UMAP for 2D visualization when available.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the latent space analyzer.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters. Defaults include:
            - variance_threshold: 0.95 (95% variance retention)
            - use_randomized_svd: True (for large datasets)
            - random_state: 42
            - umap_params: dict of UMAP parameters
        """
        self.config = config or {}
        self.variance_threshold = self.config.get('variance_threshold', 0.95)
        self.use_randomized_svd = self.config.get('use_randomized_svd', True)
        self.random_state = self.config.get('random_state', 42)
        
        # UMAP parameters
        self.umap_params = self.config.get('umap_params', {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'random_state': 42
        })
        
        # Fitted components
        self.scaler = None
        self.pca = None
        self.umap_reducer = None
        self.is_fitted = False
        
    def analyze(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform complete latent space analysis.
        
        Parameters
        ----------
        features : array-like of shape (n_samples, n_features)
            Input feature matrix
        labels : array-like of shape (n_samples,), optional
            Labels for supervised UMAP projection
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'latent_vectors': Latent space representations
            - 'embedding': 2D embedding for visualization
            - 'reconstruction_errors': Per-sample reconstruction errors
            - 'variance_explained': Cumulative variance explained
            - 'n_components': Number of latent dimensions
            - 'metadata': Analysis metadata
        """
        print(f"Analyzing latent space for {features.shape[0]} samples with {features.shape[1]} features...")
        
        # Step 1: Standardize features
        latent_vectors, reconstruction_errors = self._fit_autoencoder(features)
        
        # Step 2: Create 2D embedding
        embedding = self._create_embedding(latent_vectors, labels)
        
        # Step 3: Compute attribution (simplified feature importance)
        attributions = self._compute_attributions(features, latent_vectors)
        
        results = {
            'latent_vectors': latent_vectors,
            'embedding': embedding,
            'reconstruction_errors': reconstruction_errors,
            'attributions': attributions,
            'variance_explained': self.pca.explained_variance_ratio_.cumsum()[-1],
            'n_components': self.pca.n_components_,
            'metadata': {
                'input_shape': features.shape,
                'latent_shape': latent_vectors.shape,
                'variance_threshold': self.variance_threshold,
                'method': 'PCA autoencoding',
                'embedding_method': 'UMAP' if UMAP_AVAILABLE and labels is not None else 'PCA'
            }
        }
        
        self.is_fitted = True
        return results
    
    def _fit_autoencoder(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit PCA autoencoder for dimensionality reduction.
        
        Parameters
        ----------
        features : array-like
            Input features
            
        Returns
        -------
        tuple
            (latent_vectors, reconstruction_errors)
        """
        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Determine PCA solver based on data size and configuration
        n_samples, n_features = features_scaled.shape
        
        if self.use_randomized_svd and n_samples > 10000:
            # For large datasets, first determine number of components needed
            if isinstance(self.variance_threshold, float) and self.variance_threshold < 1.0:
                # Quick estimation of components needed
                pca_temp = PCA(n_components=min(50, n_features), svd_solver="randomized", random_state=self.random_state)
                pca_temp.fit(features_scaled)
                cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumvar >= self.variance_threshold) + 1
                n_components = min(n_components, n_features)
            else:
                n_components = self.variance_threshold
                
            self.pca = PCA(n_components=n_components, svd_solver="randomized", random_state=self.random_state)
            print(f"Using randomized PCA with {n_components} components for efficiency")
        else:
            self.pca = PCA(n_components=self.variance_threshold, svd_solver="full")
            
        # Fit and transform
        latent_vectors = self.pca.fit_transform(features_scaled)
        
        # Compute reconstruction
        features_reconstructed = self.pca.inverse_transform(latent_vectors)
        features_reconstructed = self.scaler.inverse_transform(features_reconstructed)
        
        # Reconstruction errors
        reconstruction_errors = np.mean((features - features_reconstructed)**2, axis=1)
        
        print(f"PCA autoencoding: {features.shape[1]} → {latent_vectors.shape[1]} dimensions")
        print(f"Variance explained: {self.pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
        
        return latent_vectors, reconstruction_errors
    
    def _create_embedding(self, latent_vectors: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create 2D embedding for visualization.
        
        Parameters
        ----------
        latent_vectors : array-like
            Latent space representations
        labels : array-like, optional
            Labels for supervised embedding
            
        Returns
        -------
        array-like
            2D embedding coordinates
        """
        if UMAP_AVAILABLE and labels is not None:
            # Supervised UMAP for better class separation
            self.umap_reducer = umap.UMAP(
                n_components=2,
                **self.umap_params
            )
            embedding = self.umap_reducer.fit_transform(latent_vectors, y=labels)
            print("Created supervised UMAP embedding")
        else:
            # Fallback to PCA for 2D projection
            pca_2d = PCA(n_components=2, random_state=self.random_state)
            embedding = pca_2d.fit_transform(latent_vectors)
            print("Created PCA 2D embedding (UMAP not available or no labels)")
            
        return embedding
    
    def _compute_attributions(self, features: np.ndarray, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Compute feature attributions (simplified importance scores).
        
        Parameters
        ----------
        features : array-like
            Original features
        latent_vectors : array-like
            Latent representations
            
        Returns
        -------
        array-like
            Attribution scores for each feature
        """
        # Simple attribution: correlation between original features and latent components
        attributions = np.abs(np.corrcoef(features.T, latent_vectors.T)[:features.shape[1], features.shape[1]:])
        
        # Average attribution across latent dimensions
        mean_attributions = np.mean(attributions, axis=1)
        
        return mean_attributions
    
    def transform(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Transform new data using fitted components.
        
        Parameters
        ----------
        features : array-like
            New feature data
            
        Returns
        -------
        dict
            Transformed data including latent vectors and embedding
        """
        if not self.is_fitted:
            raise ValueError("Analyzer must be fitted before transform")
            
        # Transform to latent space
        features_scaled = self.scaler.transform(features)
        latent_vectors = self.pca.transform(features_scaled)
        
        # Transform to 2D embedding
        if self.umap_reducer is not None:
            embedding = self.umap_reducer.transform(latent_vectors)
        else:
            # For PCA fallback, would need to store the 2D PCA
            embedding = latent_vectors[:, :2]  # Use first 2 components
            
        # Reconstruction errors
        features_reconstructed = self.pca.inverse_transform(latent_vectors)
        features_reconstructed = self.scaler.inverse_transform(features_reconstructed)
        reconstruction_errors = np.mean((features - features_reconstructed)**2, axis=1)
        
        return {
            'latent_vectors': latent_vectors,
            'embedding': embedding,
            'reconstruction_errors': reconstruction_errors
        }
    
    def get_component_importance(self, feature_names: Optional[list] = None) -> Dict[str, np.ndarray]:
        """
        Get importance of original features for each latent component.
        
        Parameters
        ----------
        feature_names : list, optional
            Names of original features
            
        Returns
        -------
        dict
            Component loadings and feature importance
        """
        if not self.is_fitted:
            raise ValueError("Analyzer must be fitted first")
            
        # PCA components (loadings)
        components = self.pca.components_
        
        # Feature importance (absolute loadings weighted by explained variance)
        feature_importance = np.abs(components).T @ self.pca.explained_variance_ratio_
        
        results = {
            'components': components,
            'feature_importance': feature_importance,
            'explained_variance_ratio': self.pca.explained_variance_ratio_
        }
        
        if feature_names:
            results['feature_names'] = feature_names
            
        return results
    
    def get_reconstruction_quality(self) -> Dict[str, float]:
        """
        Get metrics for reconstruction quality.
        
        Returns
        -------
        dict
            Quality metrics including variance explained and reconstruction error stats
        """
        if not self.is_fitted:
            raise ValueError("Analyzer must be fitted first")
            
        return {
            'total_variance_explained': self.pca.explained_variance_ratio_.sum(),
            'n_components': self.pca.n_components_,
            'compression_ratio': len(self.pca.explained_variance_ratio_) / self.pca.n_components_
        } 