# SickleScope: Python Genomics Analysis Package
SickleScope is a Python package for sickle cell disease variant analysis that provides instant genetic risk assessment with visualisations. Built to simplify genetic variant analysis Built without navigating complex pipelines.

## Quick Start

```bash
# Install package
pip install sickle-scope

# Analyse variants
sickle-analyse input.csv --output results/

# Generate comprehensive report
sickle-analyse input.csv --report --plot
```

## Architecture

### Package Structure
```
sickle-scope/
├── sickle_scope/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── analyser.py         # Core analysis engine
│   ├── visualiser.py       # Plotting functions
│   ├── data/               # Reference databases
│   └── utils.py            # Helper functions
├── notebooks/
│   ├── tutorial.ipynb      # Step-by-step guide
│   ├── examples.ipynb      # Sample analyses
│   └── advanced.ipynb      # Deep-dive analysis
├── tests/
│   ├── test_analyser.py
│   └── sample_data/        # Test input files
│       ├── test_variants.csv
│       ├── hbb_variants.csv
│       └── invalid_data.csv
├── results/                # Analysis outputs (created automatically)
│   ├── sickle_analysis.csv # Analysis results
│   ├── sickle_report.html  # HTML report
│   └── plots/              # Visualisation plots
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

# Full workflow example
python -m sickle_scope.cli analyse tests/sample_data/hbb_variants.csv \
  --output my_analysis/ \
  --report \
  --plot \
  --verbose \
  --config custom_params.json
```

### Output Directory Structure
When using `--output results/`, SickleScope creates an organised directory structure:
```
results/
├── sickle_analysis.csv     # Main results file with variant classifications
├── sickle_report.html      # Comprehensive HTML report (--report flag)
└── plots/                  # Visualisation directory (--plot flag)
    ├── risk_score_plot.png
    ├── variant_distribution.png
    └── severity_prediction.png
```

### Python API
```python
from sickle_scope import SickleAnalyser

# Initialise analyser
analyser = SickleAnalyser()

# Load and analyse data
results = analyser.analyse_csv('variants.csv')

# Generate visualisations
analyser.plot_risk_score(results)
analyser.plot_variant_distribution(results)

# Export results
results.to_csv('sickle_analysis.csv')
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
    'hbb_variants': 'data/hbb_pathogenic_variants.json',
    'modifiers': 'data/scd_modifiers.json',
    'population_freq': 'data/gnomad_scd_subset.json',
    'clinical_studies': 'data/phenotype_correlations.json'
}
```

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
- Machine learning model training

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
- [ ] Research and compile HBB pathogenic variants database
- [ ] Implement basic variant classification (pathogenic/benign)
- [ ] Create risk scoring function with weighted algorithm
- [ ] Test algorithm with known variants

### Day 5: Visualisation Framework
- [ ] Set up matplotlib and seaborn for plotting
- [ ] Create basic risk score visualisation (gauge plot)
- [ ] Implement variant distribution plots
- [ ] Test visualisations with sample data

### Day 6: Jupyter Tutorial Notebook
- [ ] Create comprehensive `tutorial.ipynb`
- [ ] Add interactive examples with real data
- [ ] Include step-by-step analysis explanations
- [ ] Test notebook execution from start to finish

### Day 7: Testing & Bug Fixes
- [ ] Write unit tests for all core functions
- [ ] Create integration tests with sample data
- [ ] Fix any bugs discovered during testing
- [ ] Validate test coverage >80%

### Day 8: Machine Learning Integration
- [ ] Implement Random Forest model for severity prediction
- [ ] Create training dataset from literature sources
- [ ] Add model validation and cross-validation
- [ ] Test prediction accuracy

### Day 9: Advanced Visualisations
- [ ] Create interactive Plotly visualisations for Jupyter
- [ ] Implement advanced statistical plots (Manhattan plot style)
- [ ] Add population comparison features
- [ ] Test all visualisations in notebook environment

### Day 10: Documentation & Polish
- [ ] Write comprehensive README with installation guide
- [ ] Document all functions with detailed docstrings
- [ ] Create API documentation
- [ ] Optimise code performance and memory usage

### Day 11: Package Finalisation
- [ ] Add comprehensive error handling and user-friendly messages
- [ ] Create pip-installable package structure
- [ ] Test package installation on fresh environment
- [ ] Validate all CLI commands work correctly

### Day 12: Example Notebooks
- [ ] Create `examples.ipynb` with multiple use cases
- [ ] Build `advanced.ipynb` for complex analysis workflows
- [ ] Test all notebooks run without errors
- [ ] Add clear documentation within notebooks

### Day 13: Demo & Presentation
- [ ] Record comprehensive demo video
- [ ] Create presentation slides explaining the project
- [ ] Prepare technical documentation for portfolio
- [ ] Test complete workflow from installation to results

### Day 14: Final Polish & Release
- [ ] Final code cleanup and commenting
- [ ] Complete all documentation gaps
- [ ] Create release checklist and version tagging
- [ ] Prepare GitHub repository for public release

### Prerequisites
- Python 3.9+
- pip package manager
- virtualenv (recommended)

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/sickle-scope.git
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