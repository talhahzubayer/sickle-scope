# SickleScope: Python Genomics Analysis Package
SickleScope is a Python package for sickle cell disease variant analysis that provides instant genetic risk assessment with visualisations. Built to simplify genetic variant analysis Built without navigating complex pipelines.

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Option 1: Install from PyPI (Recommended)
```bash
pip install sickle-scope
```

### Option 2: Development Installation
```bash
# Clone the repository
git clone https://github.com/talhahzubayer/sickle-scope.git
cd sickle-scope

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Option 3: Direct Installation from Source
```bash
# Download and install directly
pip install git+https://github.com/talhahzubayer/sickle-scope.git
```

### Verify Installation
```bash
# Check if installation was successful
python -c "import sickle_scope; print('SickleScope installed successfully!')"

# View package information
python -m sickle_scope.cli info
```

### Troubleshooting Installation

#### Common Issues

**Python Version Error**
```bash
# Check Python version (requires 3.9+)
python --version

# On some systems, use python3
python3 --version
```

**Permission Errors**
```bash
# Install for current user only
pip install --user sickle-scope

# Or use virtual environment (recommended)
python -m venv sickle_env
source sickle_env/bin/activate  # Windows: sickle_env\Scripts\activate
pip install sickle-scope
```

**Dependency Conflicts**
```bash
# Create clean environment
python -m venv clean_env
source clean_env/bin/activate
pip install --upgrade pip
pip install sickle-scope
```

**Missing System Dependencies**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# macOS with Homebrew
brew install python

# Windows: Download Python from python.org
```

## Quick Start

```bash
# Analyse variants
sickle-analyse input.csv --output results/

# Generate comprehensive report
sickle-analyse input.csv --report --plot

# Quick validation
sickle-analyse validate input.csv
```

## Architecture

### Package Structure
```
sickle-scope/
├── sickle_scope/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── analyser.py            # Core analysis engine
│   ├── visualiser.py          # Plotting functions
│   ├── ml_models.py           # Machine learning models
│   ├── utils.py               # Performance optimisation utilities
│   └── data/                  # Reference databases
│       └── hbb_variants.json  # Curated HBB pathogenic variants and modifiers
├── notebooks/
│   ├── tutorial.ipynb      # Step-by-step guide
│   ├── examples.ipynb      # Sample analyses
│   └── advanced.ipynb      # Deep-dive analysis
├── tests/
│   ├── __init__.py
│   ├── test_analyser.py    # Core analysis tests
│   ├── test_cli.py         # CLI interface tests
│   ├── test_integration.py # Integration tests
│   ├── test_visualiser.py  # Visualisation tests
│   └── sample_data/        # Test input files
│       ├── test_variants.csv
│       ├── hbb_variants.csv
│       └── invalid_data.csv
├── results/                # Analysis outputs (created automatically)
│   ├── sickle_analysis.csv # Analysis results
│   ├── sickle_report.html  # HTML report
│   └── plots/              # Visualisation plots
├── run_tests.py            # Test runner script
├── setup.py
├── requirements.txt
└── README.md
```

### Core Dependencies
- **Data Processing**: pandas, numpy, pysam (optional)
- **Machine Learning**: scikit-learn, scipy
- **Visualisation**: matplotlib, seaborn, plotly
- **CLI Framework**: click, rich

## Features

### Dual Interface Design
- **CLI**: Perfect for automation and batch processing
- **Python API**: Seamless integration into existing workflows
- **Jupyter Notebooks**: Interactive exploration and learning

### Analysis Pipeline
- Genetic variant detection and classification
- Risk scoring with weighted algorithms
- Modifier gene analysis
- Severity prediction using machine learning
- Population comparison and statistics

### Visualisation Suite
- Risk score dashboards with gauge-style displays
- Chromosomal variant position mapping
- Genotype distribution charts
- Severity prediction with confidence intervals
- Interactive Plotly visualisations for Jupyter

## Usage Examples

### Command Line Interface
```bash
# Basic analysis (saves to current directory)
python -m sickle_scope.cli analyse variants.csv

# Organised output with reports and plots
python -m sickle_scope.cli analyse variants.csv \
  --output results/ \
  --report \
  --plot \
  --verbose

# Validate input data before analysis
python -m sickle_scope.cli validate variants.csv

# Get package information
python -m sickle_scope.cli info

# Full workflow with ML severity prediction
python -m sickle_scope.cli analyse tests/sample_data/hbb_variants.csv \
  --output my_analysis/ \
  --report \
  --plot \
  --predict-severity \
  --verbose \
  --config custom_params.json

# Interactive visualisation mode
python -m sickle_scope.cli analyse variants.csv \
  --output results/ \
  --interactive-plots \
  --population-compare \
  --manhattan-plot
```

### Output Directory Structure
When using `--output results/`, SickleScope creates an organised directory structure:
```
results/
├── sickle_analysis.csv        # Main results file with variant classifications
├── severity_predictions.csv   # ML severity predictions
├── sickle_report.html         # Comprehensive HTML report (--report flag)
├── interactive_dashboard.html # Interactive Plotly dashboard
└── plots/                     # Visualisation directory (--plot flag)
    ├── risk_score_plot.png
    ├── variant_distribution.png
    ├── severity_prediction.png      # ML model outputs
    ├── population_comparison.png    # Population analysis plots
    ├── manhattan_plot.html          # Interactive Manhattan-style plot
    └── interactive/                 # Interactive Plotly visualisations
        ├── risk_dashboard.html
        ├── variant_explorer.html
        └── severity_heatmap.html
```

### Python API
```python
from sickle_scope import SickleAnalyser
from sickle_scope.ml_models import SeverityPredictor
from sickle_scope.visualiser import InteractiveVisualiser

# Initialise analyser
analyser = SickleAnalyser()

# Load and analyse data
results = analyser.analyse_csv('variants.csv')

# Generate basic visualisations
analyser.plot_risk_score(results)
analyser.plot_variant_distribution(results)

# Machine Learning - Severity Prediction
predictor = SeverityPredictor()
severity_predictions = predictor.predict_severity(results)
predictor.plot_severity_prediction(severity_predictions)

# Advanced Interactive Visualisations
interactive_visual = InteractiveVisualiser()
interactive_visual.create_plotly_dashboard(results, severity_predictions)
interactive_visual.plot_population_comparison(results)
interactive_visual.create_manhattan_style_plot(results)

# Export comprehensive results
results.to_csv('sickle_analysis.csv')
severity_predictions.to_csv('severity_predictions.csv')
```

### Risk Scoring Algorithm
```python
def calculate_risk_score(variants):
    """
    Weighted risk scoring based on:
    - HBB gene variants (60% weight)
    - BCL11A modifiers (20% weight)
    - Other modifiers (20% weight)
    """
    hbb_score = assess_hbb_variants(variants)
    modifier_score = assess_modifiers(variants)
    return (hbb_score * 0.6) + (modifier_score * 0.4)
```

## Data Input Requirements

### Supported Formats
- CSV with variant data
- TSV (tab-separated)
- Excel files (.xlsx)
- VCF file support (optional)

### Required Columns
```python
required_columns = [
    'chromosome',    # e.g., '11', 'chr11'
    'position',      # genomic position
    'ref_allele',    # reference nucleotide
    'alt_allele',    # alternate nucleotide
    'genotype'       # 0/0, 0/1, 1/1
]
```

## Machine Learning Components

### Severity Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: Genetic variants + population data
- **Training Data**: Literature-derived phenotype correlations
- **Output**: Severity categories (Mild, Moderate, Severe)

```python
from sklearn.ensemble import RandomForestClassifier

def train_severity_model(training_data):
    features = extract_features(training_data)
    labels = training_data['severity_category']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(features, labels)
    return model
```

### Built-in Reference Databases
```python
reference_data = {
    'hbb_variants': 'data/hbb_variants.json'  # Primary reference database
}
```

The `hbb_variants.json` file contains a comprehensive collection of:
- **Pathogenic HBB variants**: Including HbS (rs334), HbC, HbE and other clinically significant variants
- **Protective modifiers**: BCL11A, KLF1, and other genetic factors that modify disease severity
- **Population frequencies**: Allele frequencies across different populations (gnomAD, 1000 Genomes)
- **Clinical annotations**: HGVS nomenclature, amino acid changes, pathogenicity scores
- **Metadata**: Reference genome (GRCh38), data sources (ClinVar, OMIM, dbSNP), last updated in 8th September 2025

This database enables the package to work offline and provides standardised variant classification without requiring external API calls.

## Interactive Notebooks

### Tutorial.ipynb
Step-by-step learning guide with:
- Data loading and preprocessing
- Variant analysis workflow
- Visualisation creation
- Results interpretation

### Examples.ipynb
Pre-loaded sample datasets demonstrating:
- Multiple analysis scenarios
- Different data formats
- Advanced visualisation techniques

### Advanced.ipynb
Complex analysis workflows including:
- Custom risk algorithms
- Population comparison studies  
- Machine learning model training and validation
- Interactive Plotly dashboard creation
- Severity prediction model optimisation
- Advanced statistical analysis with interactive visualisations

## Development Roadmap (14-Day Sprint)

### Day 1: Project Foundation
- [x] Set up Python development environment (Python 3.9+, pip, virtualenv)
- [x] Create GitHub repository with initial structure
- [x] Initialise Python package structure with `__init__.py` files
- [x] Set up basic `setup.py` and `requirements.txt`

### Day 2: CLI Framework
- [x] Install and explore Click framework for CLI
- [x] Create basic `cli.py` with argument parsing
- [x] Implement `--help` documentation and basic commands
- [x] Test CLI installation with `pip install -e .`

### Day 3: Data Processing Pipeline
- [x] Create `analyser.py` with pandas CSV reading functions
- [x] Implement data validation (required columns, data types)
- [x] Build variant filtering functions for chromosome 11 (HBB region)
- [x] Create sample CSV files with test data

### Day 4: Risk Scoring Algorithm
- [x] Research and compile HBB pathogenic variants database
- [x] Implement basic variant classification (pathogenic/benign)
- [x] Create risk scoring function with weighted algorithm
- [x] Test algorithm with known variants

### Day 5: Visualisation Framework
- [x] Set up matplotlib and seaborn for plotting
- [x] Create basic risk score visualisation (gauge plot)
- [x] Implement variant distribution plots
- [x] Test visualisations with sample data

### Day 6: Jupyter Tutorial Notebook
- [x] Create comprehensive `tutorial.ipynb`
- [x] Add interactive examples with real data
- [x] Include step-by-step analysis explanations
- [x] Test notebook execution from start to finish

### Day 7: Testing & Bug Fixes
- [x] Write unit tests for all core functions
- [x] Create integration tests with sample data
- [x] Fix any bugs discovered during testing
- [x] Validate test coverage >80%

### Day 8: Machine Learning Integration
- [x] Implement Random Forest model for severity prediction
- [x] Create training dataset from literature sources
- [x] Add model validation and cross-validation
- [x] Test prediction accuracy

### Day 9: Advanced Visualisations
- [x] Create interactive Plotly visualisations for Jupyter
- [x] Implement advanced statistical plots (Manhattan plot style)
- [x] Add population comparison features
- [x] Test all visualisations in notebook environment

### Day 10: Documentation & Polish
- [x] Write comprehensive README with installation guide
- [x] Document all functions with detailed docstrings
- [x] Create API documentation
- [x] Optimise code performance and memory usage

### Day 11: Package Finalisation
- [x] Add comprehensive error handling and user-friendly messages
- [x] Create pip-installable package structure
- [x] Test package installation on fresh environment
- [x] Validate all CLI commands work correctly

### Day 12: Example Notebooks
- [x] Create `examples.ipynb` with multiple use cases
- [x] Build `advanced.ipynb` for complex analysis workflows
- [x] Test all notebooks run without errors
- [x] Add clear documentation within notebooks

### Day 13: Demo & Presentation
- [x] Record comprehensive demo video
- [x] Create presentation slides explaining the project
- [x] Prepare technical documentation for portfolio
- [x] Test complete workflow from installation to results

### Day 14: Final Polish & Release
- [x] Final code cleanup and commenting
- [x] Complete all documentation gaps
- [x] Create release checklist and version tagging
- [x] Prepare GitHub repository for public release

### Prerequisites
- Python 3.9+
- pip package manager
- virtualenv (recommended)

### Development Setup
```bash
# Clone repository
git clone https://github.com/talhahzubayer/sickle-scope.git
cd sickle-scope

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

### Testing
```bash
# Run unit tests
python -m pytest tests/test_analyser.py

# Run integration tests with sample data
python -m pytest tests/ --integration

# Check code coverage
pytest --cov=sickle_scope tests/
```