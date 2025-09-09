"""
Unit tests for SickleScope Genetic Variant Analyser.

Tests cover all core functionality including data validation, variant classification,
risk scoring, and severity prediction.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sickle_scope.analyser import SickleAnalyser


class TestSickleAnalyser:
    """Test suite for SickleAnalyser class."""
    
    @pytest.fixture
    def analyser(self):
        """Create a SickleAnalyser instance for testing."""
        return SickleAnalyser(verbose=False)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample variant data for testing."""
        return pd.DataFrame({
            'chromosome': ['11', '11', '11', '12'],
            'position': [5248232, 5248158, 5247359, 1000000],
            'ref_allele': ['A', 'G', 'G', 'C'],
            'alt_allele': ['T', 'A', 'A', 'T'],
            'genotype': ['0/1', '1/1', '0/1', '0/0']
        })
    
    @pytest.fixture
    def invalid_data(self):
        """Create invalid variant data for testing validation."""
        return pd.DataFrame({
            'chromosome': ['invalid', '11'],
            'position': [-100, 5248232],
            'ref_allele': ['A', 'G'],
            'alt_allele': ['T', 'A'],
            'genotype': ['invalid', '0/1']
        })
    
    def test_initialization(self):
        """Test SickleAnalyser initialization."""
        # Test default initialization
        analyser = SickleAnalyser()
        assert analyser.verbose == False
        assert len(analyser.required_columns) == 5
        assert 'chromosome' in analyser.required_columns
        
        # Test verbose initialization
        verbose_analyser = SickleAnalyser(verbose=True)
        assert verbose_analyser.verbose == True
        
        # Test HBB region configuration
        assert analyser.hbb_region['chromosome'] == '11'
        assert analyser.hbb_region['start'] == 5200000
        assert analyser.hbb_region['end'] == 5280000
    
    def test_load_default_config(self, analyser):
        """Test default configuration loading."""
        config = analyser.config
        
        assert 'risk_weights' in config
        assert 'severity_thresholds' in config
        assert config['risk_weights']['hbb_variants'] == 0.6
        assert config['risk_weights']['modifier_genes'] == 0.4
    
    def test_standardise_data(self, analyser, sample_data):
        """Test data standardisation functionality."""
        # Add chromosome with 'chr' prefix
        test_data = sample_data.copy()
        test_data.loc[0, 'chromosome'] = 'chr11'
        test_data.loc[1, 'genotype'] = '0|1'  # Test pipe separator
        
        standardised = analyser._standardise_data(test_data)
        
        # Check chromosome standardisation
        assert standardised.loc[0, 'chromosome'] == '11'
        
        # Check genotype standardisation
        assert standardised.loc[1, 'genotype'] == '0/1'
        
        # Check position is numeric
        assert pd.api.types.is_numeric_dtype(standardised['position'])
    
    def test_filter_hbb_variants(self, analyser, sample_data):
        """Test filtering of variants in HBB region."""
        hbb_variants = analyser._filter_hbb_variants(sample_data)
        
        # Should keep chromosome 11 variants within range
        expected_count = len(sample_data[sample_data['chromosome'] == '11'])
        assert len(hbb_variants) == expected_count
        
        # Should exclude chromosome 12 variants
        assert not any(hbb_variants['chromosome'] == '12')
    
    def test_classify_variant_pathogenic(self, analyser):
        """Test variant classification for pathogenic variants."""
        # Mock the HBB variants database
        analyser.hbb_variants_db = {
            'pathogenic_variants': {
                'HbS': {
                    'chromosome': 11,
                    'position': 5248232,
                    'ref': 'A',
                    'alt': 'T',
                    'name': 'HbS',
                    'severity_score': 8,
                    'disease': 'sickle_cell_disease'
                }
            }
        }
        
        variant = pd.Series({
            'chromosome': '11',
            'position': 5248232,
            'ref_allele': 'A',
            'alt_allele': 'T',
            'genotype': '0/1'
        })
        
        classification = analyser._classify_variant(variant)
        
        assert classification['classification'] == 'pathogenic'
        assert classification['variant_name'] == 'HbS'
        assert classification['severity_score'] == 8
    
    def test_classify_variant_benign(self, analyser):
        """Test variant classification for benign variants."""
        analyser.hbb_variants_db = {'pathogenic_variants': {}, 'modifier_variants': {}}
        
        variant = pd.Series({
            'chromosome': '11',
            'position': 5248232,
            'ref_allele': 'A',
            'alt_allele': 'T',
            'genotype': '0/0'  # Wild type
        })
        
        classification = analyser._classify_variant(variant)
        
        assert classification['classification'] == 'benign'
        assert classification['variant_name'] == 'wild_type'
        assert classification['severity_score'] == 0
    
    def test_classify_variant_uncertain(self, analyser):
        """Test variant classification for uncertain variants."""
        analyser.hbb_variants_db = {'pathogenic_variants': {}, 'modifier_variants': {}}
        
        variant = pd.Series({
            'chromosome': '11',
            'position': 5248232,
            'ref_allele': 'A',
            'alt_allele': 'T',
            'genotype': '0/1'  # Novel variant
        })
        
        classification = analyser._classify_variant(variant)
        
        assert classification['classification'] == 'uncertain'
        assert classification['variant_name'] == 'novel_variant'
        assert classification['severity_score'] == 2
    
    def test_calculate_risk_score_pathogenic(self, analyser):
        """Test risk score calculation for pathogenic variants."""
        # Mock scoring algorithm in database
        analyser.hbb_variants_db = {
            'scoring_algorithm': {
                'weights': {
                    'primary_mutation': 0.6,
                    'bcl11a_modifiers': 0.2,
                    'hbs1l_myb_modifiers': 0.1,
                    'other_modifiers': 0.1
                },
                'max_score': 100
            }
        }
        
        variant_row = pd.Series({'genotype': '1/1'})  # Homozygous
        classification = {
            'classification': 'pathogenic',
            'severity_score': 8
        }
        
        risk_score = analyser._calculate_risk_score(variant_row, classification)
        
        # Should be severity_score * primary_weight * genotype_multiplier
        expected_score = 8 * 0.6 * 1.0  # 4.8
        assert abs(risk_score - expected_score) < 0.01
    
    def test_calculate_risk_score_heterozygous(self, analyser):
        """Test risk score calculation for heterozygous variants."""
        analyser.hbb_variants_db = {
            'scoring_algorithm': {
                'weights': {'primary_mutation': 0.6},
                'max_score': 100
            }
        }
        
        variant_row = pd.Series({'genotype': '0/1'})  # Heterozygous
        classification = {
            'classification': 'pathogenic',
            'severity_score': 8
        }
        
        risk_score = analyser._calculate_risk_score(variant_row, classification)
        
        # Should be severity_score * primary_weight * heterozygous_multiplier
        expected_score = 8 * 0.6 * 0.5  # 2.4
        assert abs(risk_score - expected_score) < 0.01
    
    def test_predict_severity(self, analyser):
        """Test severity prediction based on risk scores."""
        # Mock severity categories
        analyser.hbb_variants_db = {
            'severity_categories': {
                'minimal_risk': {'score_range': [0, 20]},
                'carrier_status': {'score_range': [20, 40]},
                'moderate_risk': {'score_range': [40, 70]},
                'high_risk': {'score_range': [70, 100]}
            }
        }
        
        # Test different risk score ranges
        test_cases = [
            (10, 'minimal_risk'),
            (30, 'carrier_status'),
            (60, 'moderate_risk'),
            (90, 'high_risk'),
            (-5, 'protective_factors')
        ]
        
        for risk_score, expected_category in test_cases:
            prediction = analyser._predict_severity(risk_score)
            assert prediction['severity_category'] == expected_category
            assert prediction['score'] == risk_score
    
    def test_validate_input_valid_file(self, analyser, sample_data):
        """Test input validation with valid data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            is_valid, messages = analyser.validate_input(temp_path)
            assert is_valid == True
            assert len(messages) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_validate_input_missing_file(self, analyser):
        """Test input validation with missing file."""
        is_valid, messages = analyser.validate_input('nonexistent.csv')
        
        assert is_valid == False
        assert len(messages) > 0
        assert 'does not exist' in messages[0]
    
    def test_validate_input_invalid_format(self, analyser):
        """Test input validation with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            is_valid, messages = analyser.validate_input(temp_path)
            assert is_valid == False
            assert any('Unsupported file format' in msg for msg in messages)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_input_missing_columns(self, analyser):
        """Test input validation with missing required columns."""
        incomplete_data = pd.DataFrame({
            'chromosome': ['11'],
            'position': [5248232]
            # Missing ref_allele, alt_allele, genotype
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            is_valid, messages = analyser.validate_input(temp_path)
            assert is_valid == False
            assert any('Missing required columns' in msg for msg in messages)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_input_invalid_data(self, analyser, invalid_data):
        """Test input validation with invalid data values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            is_valid, messages = analyser.validate_input(temp_path)
            assert is_valid == False
            # Should catch invalid chromosome and genotype
            assert len(messages) >= 2
        finally:
            Path(temp_path).unlink()
    
    def test_read_file_csv(self, analyser, sample_data):
        """Test reading CSV files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            df = analyser._read_file(temp_path)
            assert len(df) == len(sample_data)
            assert list(df.columns) == list(sample_data.columns)
        finally:
            Path(temp_path).unlink()
    
    def test_read_file_tsv(self, analyser, sample_data):
        """Test reading TSV files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            sample_data.to_csv(f.name, sep='\t', index=False)
            temp_path = f.name
        
        try:
            df = analyser._read_file(temp_path)
            assert len(df) == len(sample_data)
            assert list(df.columns) == list(sample_data.columns)
        finally:
            Path(temp_path).unlink()
    
    def test_read_file_unsupported(self, analyser):
        """Test reading unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            with pytest.raises(ValueError, match="Unsupported file format"):
                analyser._read_file(f.name)
    
    @patch('sickle_scope.analyser.SickleAnalyser._load_hbb_variants_database')
    def test_analyse_file_integration(self, mock_load_db, analyser, sample_data):
        """Integration test for complete file analysis."""
        # Mock the database
        mock_load_db.return_value = {
            'pathogenic_variants': {},
            'modifier_variants': {},
            'scoring_algorithm': {'weights': {'primary_mutation': 0.6}, 'max_score': 100},
            'severity_categories': {
                'minimal_risk': {'score_range': [0, 20]},
                'high_risk': {'score_range': [70, 100]}
            }
        }
        analyser.hbb_variants_db = mock_load_db.return_value
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            results = analyser.analyse_file(temp_path)
            
            # Check results structure
            assert len(results) == len(sample_data)
            expected_columns = [
                'chromosome', 'position', 'ref_allele', 'alt_allele', 'genotype',
                'variant_classification', 'variant_id', 'variant_name', 
                'is_pathogenic', 'is_modifier', 'risk_score', 'severity_category',
                'severity_description', 'clinical_management'
            ]
            
            for col in expected_columns:
                assert col in results.columns
            
            # Check data types
            assert pd.api.types.is_numeric_dtype(results['risk_score'])
            assert pd.api.types.is_bool_dtype(results['is_pathogenic'])
            assert pd.api.types.is_bool_dtype(results['is_modifier'])
            
        finally:
            Path(temp_path).unlink()
    
    def test_analyse_csv_convenience_method(self, analyser, sample_data):
        """Test the convenience method for CSV analysis."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Mock the database to avoid file dependencies
            analyser.hbb_variants_db = {
                'pathogenic_variants': {},
                'modifier_variants': {},
                'scoring_algorithm': {'weights': {'primary_mutation': 0.6}, 'max_score': 100},
                'severity_categories': {'minimal_risk': {'score_range': [0, 20]}}
            }
            
            results = analyser.analyse_csv(temp_path)
            assert len(results) > 0
            assert 'risk_score' in results.columns
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_config_valid(self, analyser):
        """Test loading valid configuration file."""
        config_data = {
            'risk_weights': {
                'hbb_variants': 0.7,
                'modifier_genes': 0.3
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            analyser.load_config(temp_path)
            assert analyser.config['risk_weights']['hbb_variants'] == 0.7
            assert analyser.config['risk_weights']['modifier_genes'] == 0.3
        finally:
            Path(temp_path).unlink()
    
    def test_load_config_invalid_file(self, analyser):
        """Test loading invalid configuration file."""
        # Should not raise exception, just issue warning
        original_config = analyser.config.copy()
        analyser.load_config('nonexistent_config.json')
        
        # Config should remain unchanged
        assert analyser.config == original_config


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframes."""
        analyser = SickleAnalyser()
        empty_df = pd.DataFrame(columns=['chromosome', 'position', 'ref_allele', 'alt_allele', 'genotype'])
        
        # Should handle empty data gracefully
        standardised = analyser._standardise_data(empty_df)
        assert len(standardised) == 0
        
        filtered = analyser._filter_hbb_variants(empty_df)
        assert len(filtered) == 0
    
    def test_extreme_risk_scores(self):
        """Test risk score calculation with extreme values."""
        analyser = SickleAnalyser()
        analyser.hbb_variants_db = {
            'scoring_algorithm': {
                'weights': {'primary_mutation': 0.6},
                'max_score': 100
            }
        }
        
        # Test very high severity score
        variant_row = pd.Series({'genotype': '1/1'})
        classification = {'classification': 'pathogenic', 'severity_score': 1000}
        
        risk_score = analyser._calculate_risk_score(variant_row, classification)
        assert risk_score <= 100  # Should be capped at max_score
        
        # Test negative modifier score
        classification = {
            'classification': 'modifier', 
            'modifier_score': -10,
            'gene': 'BCL11A'
        }
        
        risk_score = analyser._calculate_risk_score(variant_row, classification)
        assert risk_score >= -20  # Should be capped at minimum
    
    def test_malformed_genotypes(self):
        """Test handling of malformed genotype data."""
        analyser = SickleAnalyser()
        
        # Test various genotype formats
        test_data = pd.DataFrame({
            'chromosome': ['11'] * 5,
            'position': [5248232] * 5,
            'ref_allele': ['A'] * 5,
            'alt_allele': ['T'] * 5,
            'genotype': ['0/1', '0|1', '1/1', '0/0', 'invalid']
        })
        
        standardised = analyser._standardise_data(test_data)
        
        # Should convert pipe separators to forward slash
        assert standardised.loc[1, 'genotype'] == '0/1'
        
        # Invalid genotypes should remain as is (to be caught in validation)
        assert standardised.loc[4, 'genotype'] == 'invalid'


if __name__ == '__main__':
    pytest.main([__file__])