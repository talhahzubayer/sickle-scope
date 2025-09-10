# SickleScope API Documentation

Complete reference for the SickleScope Python Genomics Analysis Package.

## Table of Contents

1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [SickleAnalyser](#sickleanalyser)
4. [SickleVisualiser](#sicklevisualiser)
5. [SeverityPredictor](#severitypredictor)
6. [Data Structures](#data-structures)
7. [Configuration](#configuration)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

## Overview

SickleScope provides a comprehensive Python API for analysing genetic variants related to sickle cell disease. The package consists of three main classes:

- **SickleAnalyser**: Core analysis engine for variant classification and risk scoring
- **SickleVisualiser**: Advanced plotting and visualisation framework  
- **SeverityPredictor**: Machine learning model for severity prediction

## Core Classes

### Quick Start

```python
from sickle_scope import SickleAnalyser, SickleVisualiser, SeverityPredictor

# Initialize analyser
analyser = SickleAnalyser(verbose=True, enable_ml=True)

# Analyze variants
results = analyser.analyse_csv('variants.csv')

# Generate visualisations
visualiser = SickleVisualiser()
visualiser.create_comprehensive_report_plots(results, 'output/')

# Get ML predictions
ml_info = analyser.get_ml_model_info()
```

## SickleAnalyser

Main class for analysing genetic variants related to sickle cell disease.

### Constructor

```python
SickleAnalyser(verbose=False, enable_ml=True)
```

**Parameters:**
- `verbose` (bool): Enable verbose logging. Default: False
- `enable_ml` (bool): Enable machine learning severity prediction. Default: True

**Attributes:**
- `required_columns` (List[str]): Required columns for input data
- `hbb_region` (Dict): HBB gene region coordinates
- `config` (Dict): Configuration settings
- `hbb_variants_db` (Dict): Loaded variant database
- `ml_predictor` (SeverityPredictor): ML prediction model
- `ml_trained` (bool): Whether ML model is trained and ready

### Methods

#### analyse_file(file_path)

Analyze genetic variants from input file.

```python
results = analyser.analyse_file('variants.csv')
```

**Parameters:**
- `file_path` (Union[str, Path]): Path to input file (CSV, TSV, or Excel)

**Returns:**
- `pd.DataFrame`: Analysis results with risk scores and classifications

**Raises:**
- `ValueError`: If input validation fails
- `FileNotFoundError`: If input file doesn't exist

#### analyse_csv(csv_path)

Convenience method for analysing CSV files.

```python
results = analyser.analyse_csv('variants.csv')
```

**Parameters:**
- `csv_path` (Union[str, Path]): Path to CSV file

**Returns:**
- `pd.DataFrame`: Analysis results

#### validate_input(file_path)

Validate input file format and required columns.

```python
is_valid, messages = analyser.validate_input('variants.csv')
```

**Parameters:**
- `file_path` (Union[str, Path]): Path to input file

**Returns:**
- `Tuple[bool, List[str]]`: (is_valid, error_messages)

#### load_config(config_path)

Load configuration from JSON file.

```python
analyser.load_config('config.json')
```

**Parameters:**
- `config_path` (Union[str, Path]): Path to configuration file

#### generate_report(results, output_path)

Generate HTML report of analysis results.

```python
analyser.generate_report(results, 'report.html')
```

**Parameters:**
- `results` (pd.DataFrame): Analysis results DataFrame
- `output_path` (Union[str, Path]): Path to save HTML report

#### generate_plots(results, output_dir)

Generate visualisation plots using the SickleVisualiser.

```python
plot_paths = analyser.generate_plots(results, 'plots/')
```

**Parameters:**
- `results` (pd.DataFrame): Analysis results DataFrame
- `output_dir` (Union[str, Path]): Directory to save plots

**Returns:**
- `List[Path]`: List of generated plot file paths

#### get_ml_model_info()

Get information about the ML model.

```python
ml_info = analyser.get_ml_model_info()
```

**Returns:**
- `Dict`: ML model information and status

#### retrain_ml_model()

Retrain the ML model with current database.

```python
training_results = analyser.retrain_ml_model()
```

**Returns:**
- `Dict`: Training results and model performance metrics

### Analysis Results Structure

The analysis results DataFrame contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `chromosome` | str | Chromosome identifier |
| `position` | int | Genomic position |
| `ref_allele` | str | Reference nucleotide |
| `alt_allele` | str | Alternative nucleotide |
| `genotype` | str | Genotype (0/0, 0/1, 1/1) |
| `variant_classification` | str | Variant classification (pathogenic, benign, uncertain, modifier) |
| `variant_id` | str | Variant identifier from database |
| `variant_name` | str | Human-readable variant name |
| `is_pathogenic` | bool | Whether variant is pathogenic |
| `is_modifier` | bool | Whether variant is a modifier |
| `risk_score` | float | Calculated risk score (0-100) |
| `severity_category` | str | Predicted severity category |
| `severity_description` | str | Human-readable severity description |
| `clinical_management` | str | Recommended clinical management |
| `ml_predicted_severity` | str | ML-predicted severity (if enabled) |
| `ml_confidence_score` | float | ML prediction confidence (if enabled) |

## SickleVisualiser

Advanced plotting and visualisation framework for genetic variant analysis results.

### Constructor

```python
SickleVisualiser(style='default', figsize=(10, 8))
```

**Parameters:**
- `style` (str): Matplotlib style ('default', 'seaborn', 'ggplot'). Default: 'default'
- `figsize` (Tuple[int, int]): Default figure size. Default: (10, 8)

### Methods

#### create_comprehensive_report_plots(results, output_dir)

Generate complete set of analysis plots.

```python
plot_paths = visualiser.create_comprehensive_report_plots(results, 'plots/')
```

**Parameters:**
- `results` (pd.DataFrame): Analysis results DataFrame
- `output_dir` (Union[str, Path]): Directory to save plots

**Returns:**
- `List[Path]`: List of generated plot file paths

#### plot_risk_score_distribution(results, save_path=None)

Create risk score distribution plot.

```python
fig = visualiser.plot_risk_score_distribution(results, 'risk_dist.png')
```

**Parameters:**
- `results` (pd.DataFrame): Analysis results
- `save_path` (Optional[Union[str, Path]]): Path to save plot

**Returns:**
- `matplotlib.figure.Figure`: Plot figure

#### plot_variant_classification_pie(results, save_path=None)

Create pie chart of variant classifications.

```python
fig = visualiser.plot_variant_classification_pie(results, 'classifications.png')
```

#### plot_severity_heatmap(results, save_path=None)

Create heatmap of severity predictions.

```python
fig = visualiser.plot_severity_heatmap(results, 'severity.png')
```

#### create_interactive_dashboard(results, output_path=None)

Create interactive Plotly dashboard.

```python
dashboard_path = visualiser.create_interactive_dashboard(results, 'dashboard.html')
```

**Parameters:**
- `results` (pd.DataFrame): Analysis results
- `output_path` (Optional[Union[str, Path]]): Path to save dashboard

**Returns:**
- `Union[str, go.Figure]`: Dashboard path or Plotly figure

#### generate_summary_statistics(results)

Generate summary statistics for analysis results.

```python
stats = visualiser.generate_summary_statistics(results)
```

**Parameters:**
- `results` (pd.DataFrame): Analysis results

**Returns:**
- `Dict`: Summary statistics including variant counts and scores

## SeverityPredictor

Machine learning model for severity prediction using Random Forest.

### Constructor

```python
SeverityPredictor(verbose=False)
```

**Parameters:**
- `verbose` (bool): Enable verbose logging. Default: False

### Methods

#### initialise_model(hbb_variants_db)

Initialize and train the ML model.

```python
training_results = predictor.initialise_model(variants_database)
```

**Parameters:**
- `hbb_variants_db` (Dict): HBB variants database for training

**Returns:**
- `Dict`: Training results and performance metrics

#### predict_severity(results_df)

Predict severity for analysis results.

```python
predictions = predictor.predict_severity(results_df)
```

**Parameters:**
- `results_df` (pd.DataFrame): Analysis results DataFrame

**Returns:**
- `pd.DataFrame`: Predictions with confidence scores

#### get_model_info()

Get model information and feature importance.

```python
model_info = predictor.get_model_info()
```

**Returns:**
- `Dict`: Model information, performance metrics, and feature importance

## Data Structures

### Input Data Requirements

Input files must contain the following columns:

```python
required_columns = [
    'chromosome',    # e.g., '11', 'chr11'
    'position',      # genomic position (integer)
    'ref_allele',    # reference nucleotide (A, T, G, C)
    'alt_allele',    # alternative nucleotide (A, T, G, C)
    'genotype'       # 0/0, 0/1, 1/1 (or 0|0, 0|1, 1|1)
]
```

### Supported File Formats

- **CSV**: Comma-separated values (`.csv`)
- **TSV**: Tab-separated values (`.tsv`) 
- **Excel**: Excel files (`.xlsx`)

### Example Input Data

```csv
chromosome,position,ref_allele,alt_allele,genotype
11,5248232,A,T,0/1
11,5248201,G,A,1/1
11,5227002,T,C,0/0
```

## Configuration

### Configuration File Format

Configuration files use JSON format:

```json
{
    "risk_weights": {
        "hbb_variants": 0.6,
        "modifier_genes": 0.4
    },
    "severity_thresholds": {
        "mild": 0.3,
        "moderate": 0.7,
        "severe": 1.0
    },
    "ml_parameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
}
```

### Default Configuration

```python
default_config = {
    'risk_weights': {
        'hbb_variants': 0.6,
        'modifier_genes': 0.4
    },
    'severity_thresholds': {
        'mild': 0.3,
        'moderate': 0.7,
        'severe': 1.0
    }
}
```

## Error Handling

### Common Exceptions

#### ValueError
Raised for invalid input data or configuration:

```python
try:
    results = analyser.analyse_file('invalid_file.csv')
except ValueError as e:
    print(f"Input validation failed: {e}")
```

#### FileNotFoundError
Raised when input files don't exist:

```python
try:
    results = analyser.analyse_file('missing_file.csv')
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

#### RuntimeError
Raised for ML model issues:

```python
try:
    predictor.initialise_model(invalid_db)
except RuntimeError as e:
    print(f"ML training failed: {e}")
```

### Best Practices

1. **Always validate input** before analysis:
   ```python
   is_valid, messages = analyser.validate_input(file_path)
   if not is_valid:
       print("Validation errors:", messages)
   ```

2. **Handle ML model failures gracefully**:
   ```python
   analyser = SickleAnalyser(enable_ml=True)
   ml_info = analyser.get_ml_model_info()
   if ml_info['status'] != 'trained':
       print("ML model not available, using rule-based predictions")
   ```

3. **Use try-except for file operations**:
   ```python
   try:
       results = analyser.analyse_file(file_path)
   except Exception as e:
       logging.error(f"Analysis failed: {e}")
   ```

## Examples

### Basic Analysis Workflow

```python
from sickle_scope import SickleAnalyser, SickleVisualiser
from pathlib import Path

# Initialize components
analyser = SickleAnalyser(verbose=True, enable_ml=True)
visualiser = SickleVisualiser()

# Validate input
is_valid, messages = analyser.validate_input('variants.csv')
if not is_valid:
    print("Validation errors:", messages)
    exit(1)

# Perform analysis
results = analyser.analyse_file('variants.csv')

# Save results
results.to_csv('analysis_results.csv', index=False)

# Generate report
analyser.generate_report(results, 'report.html')

# Create visualisations
output_dir = Path('plots')
plot_paths = visualiser.create_comprehensive_report_plots(results, output_dir)

print(f"Analysis complete. Generated {len(plot_paths)} plots.")
```

### Custom Configuration

```python
import json
from sickle_scope import SickleAnalyser

# Create custom configuration
custom_config = {
    "risk_weights": {
        "hbb_variants": 0.7,
        "modifier_genes": 0.3
    },
    "severity_thresholds": {
        "mild": 0.25,
        "moderate": 0.65,
        "severe": 1.0
    }
}

# Save configuration
with open('custom_config.json', 'w') as f:
    json.dump(custom_config, f, indent=2)

# Use custom configuration
analyser = SickleAnalyser()
analyser.load_config('custom_config.json')
results = analyser.analyse_file('variants.csv')
```

### Interactive Dashboard Creation

```python
from sickle_scope import SickleAnalyser, SickleVisualiser

# Analyze data
analyser = SickleAnalyser(enable_ml=True)
results = analyser.analyse_file('variants.csv')

# Create interactive dashboard
visualiser = SickleVisualiser()
dashboard_path = visualiser.create_interactive_dashboard(
    results, 
    'interactive_dashboard.html'
)

print(f"Interactive dashboard saved to: {dashboard_path}")
```

### Machine Learning Model Information

```python
from sickle_scope import SickleAnalyser

analyser = SickleAnalyser(enable_ml=True)

# Get ML model information
ml_info = analyser.get_ml_model_info()

if ml_info['status'] == 'trained':
    print(f"Model type: {ml_info['model_type']}")
    print(f"Feature count: {ml_info['feature_count']}")
    print(f"Accuracy: {ml_info.get('cv_mean_accuracy', 'N/A')}")
    
    # Print feature importance
    if 'feature_importance' in ml_info:
        print("\nTop 5 Important Features:")
        importance = ml_info['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features[:5]:
            print(f"  {feature}: {score:.3f}")
else:
    print(f"ML model status: {ml_info['status']}")
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from sickle_scope import SickleAnalyser

analyser = SickleAnalyser(verbose=True)
input_dir = Path('input_files')
output_dir = Path('batch_results')
output_dir.mkdir(exist_ok=True)

# Process all CSV files in directory
for csv_file in input_dir.glob('*.csv'):
    try:
        print(f"Processing: {csv_file.name}")
        
        # Validate and analyse
        is_valid, messages = analyser.validate_input(csv_file)
        if not is_valid:
            print(f"Skipping {csv_file.name}: {messages}")
            continue
            
        results = analyser.analyse_file(csv_file)
        
        # Save results
        output_file = output_dir / f"{csv_file.stem}_results.csv"
        results.to_csv(output_file, index=False)
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")

print("Batch processing complete.")
```

---

For more examples and advanced usage, see the Jupyter notebooks in the `notebooks/` directory.