"""
SickleScope: Python Genomics Analysis Package

A comprehensive package for sickle cell disease variant analysis providing instant 
genetic risk assessment with advanced visualizations and machine learning predictions.

Key Features:
- Genetic variant detection and classification
- Risk scoring with weighted algorithms  
- Modifier gene analysis
- Machine learning severity prediction
- Interactive visualizations
- Command-line interface and Python API

Example Usage:
    >>> from sickle_scope import SickleAnalyser
    >>> analyser = SickleAnalyser()
    >>> results = analyser.analyse_csv('variants.csv')
    >>> analyser.generate_plots(results, 'output/')

Classes:
    SickleAnalyser: Main analysis engine for variant classification and risk scoring
    SickleVisualiser: Advanced plotting and visualization framework
    SeverityPredictor: Machine learning model for severity prediction
    
Constants:
    __version__: Package version string
    __author__: Package author information
"""

__version__ = "0.1.0"
__author__ = "SickleScope Development Team"
__email__ = "support@sicklescope.org"
__license__ = "MIT"
__description__ = "Python Genomics Analysis Package for Sickle Cell Disease"

# Import main classes for convenient access
from .analyser import SickleAnalyser
from .visualiser import SickleVisualiser  
from .ml_models import SeverityPredictor

# Define public API
__all__ = [
    'SickleAnalyser',
    'SickleVisualiser', 
    'SeverityPredictor',
    '__version__',
    '__author__',
    '__description__'
]

# Package-level configuration
import warnings
import logging

# Configure warnings for genomics analysis
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Package metadata for compatibility
PACKAGE_INFO = {
    'name': 'sickle-scope',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'license': __license__,
    'python_requires': '>=3.9',
    'keywords': ['genomics', 'sickle-cell', 'genetics', 'bioinformatics', 'analysis'],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ]
}