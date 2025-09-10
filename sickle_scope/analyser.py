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
from functools import lru_cache
from .ml_models import SeverityPredictor
from .utils import performance_monitor, optimise_dataframe, validate_input_size, memory_usage


class SickleAnalyser:
    """Main class for analysing genetic variants related to sickle cell disease."""
    
    def __init__(self, verbose: bool = False, enable_ml: bool = True):
        """Initialise the SickleScope analyser.
        
        Args:
            verbose: Enable verbose logging
            enable_ml: Enable machine learning severity prediction
        """
        self.verbose = verbose
        self.enable_ml = enable_ml
        self.required_columns = [
            'chromosome', 'position', 'ref_allele', 'alt_allele', 'genotype'
        ]
        
        # HBB gene region (chromosome 11: expanded to include all known variants)
        self.hbb_region = {
            'chromosome': '11',
            'start': 5200000,  # Expanded range to capture all HBB variants
            'end': 5280000
        }
        
        self.config = self._load_default_config()
        self.hbb_variants_db = self._load_hbb_variants_database()
        
        # Initialise ML predictor
        self.ml_predictor = None
        self.ml_trained = False
        if self.enable_ml and self.hbb_variants_db:
            self._initialise_ml_predictor()
        
        if self.verbose:
            print("SickleAnalyser initialised successfully")
            if self.enable_ml and self.ml_trained:
                print("Machine learning severity predictor ready")
    
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
    
    def _load_hbb_variants_database(self) -> Dict:
        """Load HBB variants database from JSON file.
        
        Returns:
            Dictionary containing variant database
        """
        try:
            # Get path to the data directory relative to this file
            current_dir = Path(__file__).parent
            db_path = current_dir / 'data' / 'hbb_variants.json'
            
            with open(db_path, 'r', encoding='utf-8') as f:
                db = json.load(f)
            
            if self.verbose:
                pathogenic_count = len(db.get('pathogenic_variants', {}))
                modifier_count = sum(len(modifiers) for modifiers in db.get('modifier_variants', {}).values())
                print(f"Loaded HBB database: {pathogenic_count} pathogenic variants, {modifier_count} modifiers")
                
            return db
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load HBB variants database: {e}")
            return {}
    
    def _initialise_ml_predictor(self) -> None:
        """Initialise and train the ML severity predictor."""
        try:
            self.ml_predictor = SeverityPredictor(verbose=self.verbose)
            training_results = self.ml_predictor.initialise_model(self.hbb_variants_db)
            self.ml_trained = True
            
            if self.verbose:
                print(f"ML model trained with {training_results['cv_mean_accuracy']:.3f} CV accuracy")
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not initialise ML predictor: {e}")
            self.ml_trained = False
    
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
        # Optimise: Avoid copy if not necessary, use vectorised operations
        # Work on copy to avoid modifying original
        df = df.copy()
        
        # Vectorised chromosome standardisation (more efficient than string operations)
        df['chromosome'] = df['chromosome'].astype(str).str.replace('chr', '', case=False, regex=False)
        
        # More efficient numeric conversion with downcast
        df['position'] = pd.to_numeric(df['position'], errors='coerce', downcast='integer')
        
        # Vectorised genotype standardisation
        df['genotype'] = df['genotype'].astype(str).str.replace('|', '/', regex=False)
        
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
    
    @lru_cache(maxsize=1000)
    def _classify_variant_cached(self, chrom: str, pos: int, ref: str, alt: str, genotype: str) -> Dict:
        """Cached variant classification for performance optimisation."""
        # Create a temporary variant object for classification
        variant_data = {
            'chromosome': chrom,
            'position': pos, 
            'ref_allele': ref,
            'alt_allele': alt,
            'genotype': genotype
        }
        return self._classify_variant_internal(variant_data)
    
    def _classify_variant(self, variant: pd.Series) -> Dict:
        """Classify a single variant as pathogenic, benign, or uncertain.
        
        Args:
            variant: Single variant data
            
        Returns:
            Dictionary with classification details
        """
        # Use cached classification for better performance
        return self._classify_variant_cached(
            str(variant['chromosome']),
            int(variant['position']),
            str(variant['ref_allele']),
            str(variant['alt_allele']),
            str(variant['genotype'])
        )
    
    def _classify_variant_internal(self, variant: Dict) -> Dict:
        """Internal variant classification logic.
        
        Args:
            variant: Variant data dictionary
            
        Returns:
            Dictionary with classification details
        """
        # Check against pathogenic variants database
        pathogenic_variants = self.hbb_variants_db.get('pathogenic_variants', {})
        
        for variant_id, var_data in pathogenic_variants.items():
            if (variant['chromosome'] == str(var_data['chromosome']) and 
                variant['position'] == var_data['position'] and
                variant['ref_allele'] == var_data['ref'] and
                variant['alt_allele'] == var_data['alt']):
                
                return {
                    'classification': 'pathogenic',
                    'variant_id': variant_id,
                    'variant_name': var_data['name'],
                    'severity_score': var_data['severity_score'],
                    'disease': var_data.get('disease', 'unknown'),
                    'mechanism': var_data.get('mechanism', 'unknown')
                }
        
        # Check modifier variants
        modifier_variants = self.hbb_variants_db.get('modifier_variants', {})
        for gene, variants in modifier_variants.items():
            for variant_id, var_data in variants.items():
                if (variant['chromosome'] == str(var_data['chromosome']) and 
                    variant['position'] == var_data['position'] and
                    variant['ref_allele'] == var_data['ref'] and
                    variant['alt_allele'] == var_data['alt']):
                    
                    return {
                        'classification': 'modifier',
                        'variant_id': variant_id,
                        'variant_name': var_data['name'],
                        'modifier_score': var_data['modifier_score'],
                        'effect': var_data['effect'],
                        'gene': gene
                    }
        
        # Default classification for unknown variants
        if variant['genotype'] == '0/0':
            return {
                'classification': 'benign',
                'variant_id': 'unknown',
                'variant_name': 'wild_type',
                'severity_score': 0
            }
        else:
            return {
                'classification': 'uncertain',
                'variant_id': 'unknown',
                'variant_name': 'novel_variant',
                'severity_score': 2
            }
        
    def _calculate_risk_score(self, variant_row: pd.Series, variant_classification: Dict) -> float:
        """Calculate risk score for a variant using weighted algorithm.
        
        Args:
            variant_row: Single variant data row
            variant_classification: Classification details from _classify_variant
            
        Returns:
            Risk score between 0 and 100
        """
        # Get scoring algorithm weights from database
        scoring_algorithm = self.hbb_variants_db.get('scoring_algorithm', {})
        weights = scoring_algorithm.get('weights', {
            'primary_mutation': 0.60,
            'bcl11a_modifiers': 0.20,
            'hbs1l_myb_modifiers': 0.10,
            'other_modifiers': 0.10
        })
        max_score = scoring_algorithm.get('max_score', 100)
        
        base_score = 0.0
        
        # Score based on variant classification and severity
        if variant_classification['classification'] == 'pathogenic':
            severity_score = variant_classification.get('severity_score', 5)
            base_score = severity_score * weights.get('primary_mutation', 0.60)
            
        elif variant_classification['classification'] == 'modifier':
            # Modifier variants contribute negatively (protective effect)
            modifier_score = abs(variant_classification.get('modifier_score', -2))
            gene = variant_classification.get('gene', 'other')
            
            if gene == 'BCL11A':
                base_score = modifier_score * weights.get('bcl11a_modifiers', 0.20)
            elif gene == 'HBS1L_MYB':
                base_score = modifier_score * weights.get('hbs1l_myb_modifiers', 0.10)
            else:
                base_score = modifier_score * weights.get('other_modifiers', 0.10)
                
        elif variant_classification['classification'] == 'uncertain':
            base_score = variant_classification.get('severity_score', 2) * weights.get('primary_mutation', 0.60)
        
        # Adjust based on genotype (zygosity impact)
        genotype_multipliers = {
            '1/1': 1.0,  # Homozygous - full impact
            '0/1': 0.5,  # Heterozygous - reduced impact
            '0/0': 0.0   # Wild type - no impact
        }
        
        multiplier = genotype_multipliers.get(variant_row['genotype'], 0.25)
        final_score = base_score * multiplier
        
        # Handle protective modifiers (negative scores become protective)
        if variant_classification['classification'] == 'modifier':
            modifier_score_raw = variant_classification.get('modifier_score', -2)
            if modifier_score_raw < 0:
                # Protective modifier - subtract from risk
                final_score = modifier_score_raw * multiplier  # This will be negative
        
        return min(max(final_score, -20), max_score)  # Cap between -20 and max_score
    
    def _predict_severity(self, risk_score: float) -> Dict:
        """Predict disease severity based on risk score.
        
        Args:
            risk_score: Calculated risk score (0-100 scale)
            
        Returns:
            Dictionary with severity prediction and details
        """
        # Get severity categories from database
        severity_categories = self.hbb_variants_db.get('severity_categories', {
            'minimal_risk': {'score_range': [0, 20]},
            'carrier_status': {'score_range': [20, 40]},
            'moderate_risk': {'score_range': [40, 70]},
            'high_risk': {'score_range': [70, 100]}
        })
        
        for category, data in severity_categories.items():
            score_range = data['score_range']
            if score_range[0] <= risk_score <= score_range[1]:
                return {
                    'severity_category': category,
                    'description': data.get('description', category.replace('_', ' ')),
                    'management': data.get('management', 'Consult healthcare provider'),
                    'monitoring': data.get('monitoring', 'Regular check-ups'),
                    'score': risk_score
                }
        
        # Handle negative scores (protective modifiers)
        if risk_score < 0:
            return {
                'severity_category': 'protective_factors',
                'description': 'Protective genetic factors present',
                'management': 'Beneficial genetic profile',
                'monitoring': 'Standard care',
                'score': risk_score
            }
        
        # Fallback for scores outside defined ranges
        return {
            'severity_category': 'unknown',
            'description': 'Unable to determine severity',
            'management': 'Consult genetic counselor',
            'monitoring': 'Follow up recommended',
            'score': risk_score
        }
    
    @performance_monitor
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
        
        # Check input size for performance optimisation
        if not validate_input_size(df, max_size_mb=200.0):
            if self.verbose:
                print(f"Processing large dataset ({memory_usage():.1f}MB memory usage)")
        
        # Standardise data format
        df = self._standardise_data(df)
        
        # Optimise DataFrame memory usage
        df = optimise_dataframe(df)
        
        # Filter for HBB region variants (for reporting only)
        hbb_variants = self._filter_hbb_variants(df)
        
        if self.verbose:
            print(f"Found {len(hbb_variants)} variants in HBB region")
            print(f"Total variants for analysis: {len(df)}")
        
        # Optimise: Vectorised classification and scoring
        # Pre-allocate arrays for better memory efficiency
        n_variants = len(df)
        classifications = [None] * n_variants
        risk_scores = np.zeros(n_variants, dtype=np.float32)
        
        # Batch classify variants for better performance
        for idx in range(n_variants):
            row = df.iloc[idx]
            classification = self._classify_variant(row)
            classifications[idx] = classification
            risk_scores[idx] = self._calculate_risk_score(row, classification)
        
        # Vectorized column assignment (more memory efficient)
        df['variant_classification'] = [c['classification'] for c in classifications]
        df['variant_id'] = [c.get('variant_id', 'unknown') for c in classifications]
        df['variant_name'] = [c.get('variant_name', 'unknown') for c in classifications]
        df['is_pathogenic'] = df['variant_classification'] == 'pathogenic'
        df['is_modifier'] = df['variant_classification'] == 'modifier'
        df['risk_score'] = risk_scores
        
        # Vectorized severity prediction
        severity_data = [self._predict_severity(score) for score in risk_scores]
        df['severity_category'] = [s['severity_category'] for s in severity_data]
        df['severity_description'] = [s['description'] for s in severity_data]
        df['clinical_management'] = [s['management'] for s in severity_data]
        
        # Add ML predictions if enabled and model is trained
        if self.enable_ml and self.ml_trained:
            try:
                ml_predictions = self.ml_predictor.predict_severity(df)
                df['ml_predicted_severity'] = ml_predictions['predicted_severity']
                df['ml_confidence_score'] = ml_predictions['confidence_score']
                
                # Add probability columns for ML predictions
                for col in ml_predictions.columns:
                    if col.startswith('prob_'):
                        df[f'ml_{col}'] = ml_predictions[col]
                
                if self.verbose:
                    print("Added ML severity predictions")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: ML prediction failed: {e}")
        
        if self.verbose:
            pathogenic_count = df['is_pathogenic'].sum()
            print(f"Identified {pathogenic_count} potentially pathogenic variants")
        
        return df
    
    def get_ml_model_info(self) -> Dict:
        """Get information about the ML model.
        
        Returns:
            Dictionary with ML model information
        """
        if not self.enable_ml:
            return {'status': 'disabled'}
        
        if not self.ml_trained or self.ml_predictor is None:
            return {'status': 'not_trained'}
        
        return self.ml_predictor.get_model_info()
    
    def retrain_ml_model(self) -> Dict:
        """Retrain the ML model with current database.
        
        Returns:
            Training results dictionary
        """
        if not self.enable_ml:
            raise ValueError("ML is disabled for this analyser instance")
        
        if not self.hbb_variants_db:
            raise ValueError("No HBB variants database available for training")
        
        self._initialise_ml_predictor()
        return self.get_ml_model_info()
    
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
        <p>Severe risk variants: {(results['severity_category'] == 'high_risk').sum()}</p>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(report_html)
    
    def generate_plots(self, results: pd.DataFrame, output_dir: Union[str, Path]):
        """Generate visualisation plots using the SickleVisualiser.
        
        Args:
            results: Analysis results DataFrame
            output_dir: Directory to save plots
        """
        from .visualiser import SickleVisualiser
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"Generating visualisations in: {output_dir}")
        
        # Initialise visualiser
        visualiser = SickleVisualiser()
        
        # Generate comprehensive plots
        plot_paths = visualiser.create_comprehensive_report_plots(results, output_dir)
        
        # Generate summary statistics
        stats = visualiser.generate_summary_statistics(results)
        
        if self.verbose:
            print(f"Generated {len(plot_paths)} visualisation plots")
            print(f"Summary: {stats['pathogenic_variants']} pathogenic, {stats['modifier_variants']} modifier variants")
            print(f"Mean risk score: {stats['mean_risk_score']:.2f}")
        
        return plot_paths
    
    def analyse_csv(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """Convenience method for analysing CSV files.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Analysis results DataFrame
        """
        return self.analyse_file(csv_path)