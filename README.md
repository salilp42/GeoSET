# Geometric Analysis of Saccadic Eye Movements

A comprehensive framework for analyzing saccadic eye movement patterns using geometric and topological methods to identify neurological biomarkers.

## Overview

This repository implements a geometric framework that transforms saccadic waveforms into interpretable latent spaces where different neurological conditions occupy distinct regions. The approach combines dimensionality reduction, supervised manifold learning, and comprehensive statistical validation to achieve reliable classification while maintaining biological interpretability.

## Key Features

- **Geometric Framework**: PCA autoencoding with supervised UMAP projection for latent space analysis
- **Feature Scalpel Strategy**: Adaptive feature selection based on classification difficulty
- **Comprehensive Validation**: Ablation studies, permutation testing, and effect size analysis
- **Interpretability**: Direct mapping between geometric patterns and temporal features
- **Statistical Analysis**: Power analysis, medication effects, and cross-validation
- **Publication-Quality Visualization**: Automated figure generation with publication standards

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, scikit-learn
- matplotlib, seaborn
- Optional: UMAP, PyTorch, LightGBM for enhanced functionality

### Setup

```bash
git clone https://github.com/salilp42/GeoSET.git
cd GeoSET
pip install -r requirements.txt
```

## Usage

### Basic Analysis

```python
from src.core.pipeline import GeometricAnalysisPipeline
from src.utils.data_loader import load_saccade_data

# Load your saccade data
data = load_saccade_data('path/to/data')

# Initialize pipeline
pipeline = GeometricAnalysisPipeline()

# Run complete analysis
results = pipeline.run_analysis(data)
```

### Feature Scalpel Classification

```python
from src.analysis.feature_scalpel import FeatureScalpelClassifier

classifier = FeatureScalpelClassifier()
performance = classifier.fit_predict(features, labels)
```

### Geometric Visualization

```python
from src.visualization.geometric_plots import generate_latent_space_plot

generate_latent_space_plot(latent_vectors, labels, output_path='figures/')
```

## Repository Structure

```
saccadic-geometry-analysis/
├── src/
│   ├── core/                 # Core analysis pipeline
│   ├── analysis/             # Statistical and classification methods
│   ├── visualization/        # Plotting and figure generation
│   └── utils/               # Utilities and helper functions
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── config/                  # Configuration files
├── examples/                # Usage examples
└── requirements.txt         # Dependencies
```

## Core Components

### Geometric Analysis (`src/core/`)
- `pipeline.py`: Main analysis pipeline
- `latent_space.py`: PCA autoencoding and manifold learning
- `feature_extraction.py`: Biomarker computation and feature engineering

### Statistical Analysis (`src/analysis/`)
- `feature_scalpel.py`: Adaptive classification framework
- `validation.py`: Cross-validation and permutation testing
- `statistical_tests.py`: Effect sizes and power analysis
- `interpretability.py`: Saliency mapping and attribution

### Visualization (`src/visualization/`)
- `geometric_plots.py`: Latent space and topology visualization
- `classification_plots.py`: ROC curves and performance metrics
- `interpretability_plots.py`: Feature importance and temporal patterns

### Utilities (`src/utils/`)
- `data_preprocessing.py`: Data harmonization and quality control
- `topology_analysis.py`: Distance distributions and clustering
- `graph_analysis.py`: Network construction and analysis
- `biomarkers.py`: Clinical biomarker extraction

## Configuration

Modify `config/analysis_config.yaml` to adjust:
- PCA variance threshold
- UMAP parameters
- Cross-validation settings
- Feature selection criteria
- Output formats

## Examples

See `examples/` directory for:
- Basic geometric analysis workflow
- Feature Scalpel classification
- Comprehensive validation pipeline
- Custom visualization examples

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{geoset,
  author = {Patel, Salil},
  title = {GeoSET: Geometric Analysis of Saccadic Eye Movements},
  url = {https://github.com/salilp42/GeoSET},
  year = {2025}
}
```

## Contact

For questions or collaboration opportunities, please open an issue or contact the author. 