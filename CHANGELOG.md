# Changelog

All notable changes to SickleScope will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 11-09-2025

### Added
- Initial release of SickleScope genomics analysis package
- Core genetic variant analysis engine with HBB gene focus
- Command-line interface with comprehensive error handling
- Python API for programmatic access
- Machine learning severity prediction using Random Forest
- Interactive and static visualisation framework
- Risk scoring algorithm with weighted pathogenic variant assessment
- Support for CSV, TSV, and Excel input formats
- Comprehensive test suite with >80% coverage
- Jupyter notebook tutorials and examples
- Built-in HBB variants database with pathogenic and protective variants
- Population comparison and statistical analysis
- Interactive Plotly visualisations
- HTML report generation
- Memory optimisation and performance monitoring
- Input validation and data quality checks

### Features
- **Analysis Pipeline**: Genetic variant detection, classification, and risk assessment
- **Visualisations**: Risk dashboards, chromosomal mapping, severity predictions
- **Machine Learning**: Random Forest model for severity category prediction
- **CLI Tools**: Full command-line interface with validation and info commands
- **Python API**: Seamless integration into existing bioinformatics workflows
- **Data Support**: CSV, TSV, Excel formats with comprehensive validation

### Dependencies
- Python 3.9+
- pandas, numpy, matplotlib, seaborn, plotly
- scikit-learn, scipy, click, rich
- jupyter (optional for notebooks)

### Documentation
- Comprehensive README with installation and usage guides
- API documentation with detailed function descriptions
- Tutorial notebooks with step-by-step examples
- Advanced analysis workflow examples