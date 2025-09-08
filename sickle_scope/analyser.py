"""
SickleScope Genetic Variant Analyser

Core analysis engine for sickle cell disease variant analysis and risk assessment.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import warnings


class SickleAnalyser:
    """Main class for analysing genetic variants related to sickle cell disease."""
    
    def __init__(self, verbose: bool = False):
        """Initialise the SickleScope analyser.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.required_columns = [
            'chromosome', 'position', 'ref_allele', 'alt_allele', 'genotype'
        ]
        
        # HBB gene region (chromosome 11: 5,246,696-5,250,625)
        self.hbb_region = {
            'chromosome': '11',
            'start': 5246696,
            'end': 5250625
        }
        
        self.config = self._load_default_config()
        
        if self.verbose:
            print("SickleAnalyser initialised successfully")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration settings."""
        return {
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
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Update default config with user settings
            self.config.update(user_config)
            
            if self.verbose:
                print(f"Configuration loaded from {config_path}")
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load config from {config_path}: {e}")
    
    def validate_input(self, file_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate input file format and required columns.
        
        Args:
            file_path: Path to input file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        messages = []
        file_path = Path(file_path)
        
        # Check file exists
        if not file_path.exists():
            messages.append(f"File does not exist: {file_path}")
            return False, messages
        
        # Check file extension
        if file_path.suffix.lower() not in ['.csv', '.tsv', '.xlsx']:
            messages.append(f"Unsupported file format: {file_path.suffix}")
            return False, messages
        
        try:
            # Try to read the file
            df = self._read_file(file_path)
            
            # Check required columns
            missing_columns = set(self.required_columns) - set(df.columns)
            if missing_columns:
                messages.append(f"Missing required columns: {missing_columns}")
            
            # Check for empty dataframe
            if df.empty:
                messages.append("File contains no data rows")
            
            # Check data types and values
            if 'chromosome' in df.columns:
                # Standardise chromosome format
                invalid_chroms = df[~df['chromosome'].astype(str).str.match(r'^(chr)?[0-9XY]+$', case=False)]
                if not invalid_chroms.empty:
                    messages.append(f"Invalid chromosome values found: {invalid_chroms['chromosome'].unique()}")
            
            if 'position' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['position']):
                    messages.append("Position column must be numeric")
                elif (df['position'] <= 0).any():
                    messages.append("Position values must be positive integers")
            
            if 'genotype' in df.columns:
                valid_genotypes = {'0/0', '0/1', '1/1', '0|0', '0|1', '1|1'}
                invalid_genotypes = df[~df['genotype'].isin(valid_genotypes)]
                if not invalid_genotypes.empty:
                    unique_invalid = invalid_genotypes['genotype'].unique()
                    messages.append(f"Invalid genotype formats: {unique_invalid}")
                    
        except Exception as e:
            messages.append(f"Error reading file: {str(e)}")
        
        return len(messages) == 0, messages
    
    def _read_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Read data file based on extension.
        
        Args:
            file_path: Path to input file
            
        Returns:
            DataFrame containing the data
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.tsv':
            return pd.read_csv(file_path, sep='\t')
        elif file_path.suffix.lower() == '.xlsx':
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
    def _standardise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise data format and chromosome naming.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Standardised DataFrame
        """
        df = df.copy()
        
        # Standardise chromosome format (remove 'chr' prefix)
        df['chromosome'] = df['chromosome'].astype(str).str.replace('chr', '', case=False)
        
        # Ensure position is integer
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        
        # Standardise genotype format (use '/' separator)
        df['genotype'] = df['genotype'].str.replace('|', '/', regex=False)
        
        return df
    
    def _filter_hbb_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter variants in the HBB gene region (chromosome 11).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame containing only HBB region variants
        """
        hbb_mask = (
            (df['chromosome'] == self.hbb_region['chromosome']) &
            (df['position'] >= self.hbb_region['start']) &
            (df['position'] <= self.hbb_region['end'])
        )
        
        return df[hbb_mask].copy()
    
    def _classify_variant(self, variant: pd.Series) -> str:
        """Classify a single variant as pathogenic, benign, or uncertain.
        
        Args:
            variant: Single variant data
            
        Returns:
            Classification string
        """
        # Simple classification based on known pathogenic positions
        # In a real implementation, this would use a comprehensive database
        known_pathogenic_positions = {
            5248232,  # HbS (Glu6Val)
            5248158,  # HbC (Glu6Lys)
            5247359,  # HbE (Glu26Lys)
        }
        
        if variant['position'] in known_pathogenic_positions:
            return 'pathogenic'
        elif variant['genotype'] in ['0/0']:
            return 'benign'
        else:
            return 'uncertain'
        
    def _calculate_risk_score(self, variant: pd.Series) -> float:
        """Calculate risk score for a variant.
        
        Args:
            variant: Single variant data
            
        Returns:
            Risk score between 0 and 1
        """
        base_score = 0.0
        
        # Score based on variant classification
        if variant['variant_classification'] == 'pathogenic':
            base_score = 0.8
        elif variant['variant_classification'] == 'uncertain':
            base_score = 0.3
        
        # Adjust based on genotype
        genotype_multipliers = {
            '1/1': 1.0,  # Homozygous
            '0/1': 0.6,  # Heterozygous
            '0/0': 0.0   # Wild type
        }
        
        multiplier = genotype_multipliers.get(variant['genotype'], 0.5)
        
        return min(base_score * multiplier, 1.0)
    
    def _predict_severity(self, risk_score: float) -> str:
        """Predict disease severity based on risk score.
        
        Args:
            risk_score: Calculated risk score
            
        Returns:
            Severity prediction string
        """
        thresholds = self.config['severity_thresholds']
        
        if risk_score < thresholds['mild']:
            return 'mild'
        elif risk_score < thresholds['moderate']:
            return 'moderate'
        else:
            return 'severe'
    
    def analyse_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Analyse genetic variants from input file.
        
        Args:
            file_path: Path to input file
            
        Returns:
            DataFrame with analysis results
        """
        if self.verbose:
            print(f"Reading data from: {file_path}")
        
        # Read and validate data
        df = self._read_file(file_path)
        is_valid, messages = self.validate_input(file_path)
        
        if not is_valid:
            raise ValueError(f"Input validation failed: {messages}")
        
        # Standardise data format
        df = self._standardise_data(df)
        
        # Filter for HBB region variants
        hbb_variants = self._filter_hbb_variants(df)
        
        if self.verbose:
            print(f"Found {len(hbb_variants)} variants in HBB region")
        
        # Classify variants
        df['variant_classification'] = df.apply(self._classify_variant, axis=1)
        df['is_pathogenic'] = df['variant_classification'] == 'pathogenic'
        
        # Calculate risk scores
        df['risk_score'] = df.apply(self._calculate_risk_score, axis=1)
        
        # Predict severity
        df['severity_prediction'] = df['risk_score'].apply(self._predict_severity)
        
        if self.verbose:
            pathogenic_count = df['is_pathogenic'].sum()
            print(f"Identified {pathogenic_count} potentially pathogenic variants")
        
        return df
    
    
    def generate_report(self, results: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Generate HTML report of analysis results.
        
        Args:
            results: Analysis results DataFrame
            output_path: Path to save HTML report
        """
        # Basic HTML report generation
        report_html = f"""
        <html>
        <head><title>SickleScope Analysis Report</title></head>
        <body>
        <h1>SickleScope Analysis Report</h1>
        <h2>Summary</h2>
        <p>Total variants analysed: {len(results)}</p>
        <p>Pathogenic variants: {results['is_pathogenic'].sum()}</p>
        <p>Severe risk variants: {(results['severity_prediction'] == 'severe').sum()}</p>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(report_html)
    
    def generate_plots(self, results: pd.DataFrame, output_dir: Union[str, Path]) -> None:
        """Generate visualisation plots.
        
        Args:
            results: Analysis results DataFrame
            output_dir: Directory to save plots
        """
        # Placeholder for plot generation
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # This would contain actual plotting code using matplotlib/seaborn
        if self.verbose:
            print(f"Plots would be generated in: {output_dir}")
    
    def analyse_csv(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """Convenience method for analysing CSV files.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Analysis results DataFrame
        """
        return self.analyse_file(csv_path)