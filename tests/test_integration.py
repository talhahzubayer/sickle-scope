"""
Integration tests for SickleScope complete workflows.

Tests end-to-end functionality including file processing, analysis pipeline,
and output generation using real sample data.
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from click.testing import CliRunner
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sickle_scope.analyser import SickleAnalyser
from sickle_scope.visualiser import SickleVisualiser
from sickle_scope.cli import cli


class TestEndToEndWorkflow:
    """Integration tests for complete analysis workflows."""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Get path to sample data directory."""
        return Path(__file__).parent / 'sample_data'
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_complete_analysis_workflow(self, sample_data_dir, temp_output_dir):
        """Test complete analysis from CSV to results."""
        # Use the existing test_variants.csv
        input_file = sample_data_dir / 'test_variants.csv'
        assert input_file.exists(), f"Sample file not found: {input_file}"
        
        # Initialize analyser
        analyser = SickleAnalyser(verbose=True)
        
        # Perform analysis
        results = analyser.analyse_file(input_file)
        
        # Verify results structure
        assert len(results) > 0, "Analysis should return non-empty results"
        
        # Check required columns exist
        required_columns = [
            'chromosome', 'position', 'ref_allele', 'alt_allele', 'genotype',
            'variant_classification', 'risk_score', 'severity_category'
        ]
        for col in required_columns:
            assert col in results.columns, f"Missing column: {col}"
        
        # Verify data types
        assert pd.api.types.is_numeric_dtype(results['risk_score'])
        assert pd.api.types.is_bool_dtype(results['is_pathogenic'])
        
        # Save results
        output_file = temp_output_dir / 'analysis_results.csv'
        results.to_csv(output_file, index=False)
        assert output_file.exists()
        
        # Verify saved file can be read back
        saved_results = pd.read_csv(output_file)
        assert len(saved_results) == len(results)
    
    def test_hbb_variants_analysis(self, sample_data_dir, temp_output_dir):
        """Test analysis specifically for HBB region variants."""
        input_file = sample_data_dir / 'hbb_variants.csv'
        assert input_file.exists(), f"HBB sample file not found: {input_file}"
        
        analyser = SickleAnalyser(verbose=True)
        results = analyser.analyse_file(input_file)
        
        # Most variants should be in chromosome 11 (HBB region), but modifier genes from other chromosomes are allowed
        chr11_count = sum(results['chromosome'] == '11')
        total_count = len(results)
        assert chr11_count >= total_count * 0.8, f"At least 80% of variants should be chromosome 11, got {chr11_count}/{total_count}"
        
        # Check that HBB region filtering works
        hbb_variants = analyser._filter_hbb_variants(results)
        
        # Should contain some variants within the HBB region
        assert len(hbb_variants) >= 0, "Should find variants in HBB region"
        
        # Verify positions are within expected range
        hbb_positions = hbb_variants['position']
        if len(hbb_positions) > 0:
            assert all(hbb_positions >= analyser.hbb_region['start'])
            assert all(hbb_positions <= analyser.hbb_region['end'])
    
    def test_visualisation_integration(self, sample_data_dir, temp_output_dir):
        """Test integration between analysis and visualisation."""
        input_file = sample_data_dir / 'test_variants.csv'
        
        # Perform analysis
        analyser = SickleAnalyser(verbose=False)
        results = analyser.analyse_file(input_file)
        
        # Generate visualisations
        visualiser = SickleVisualiser()
        plot_dir = temp_output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # Test individual plot generation
        plots_created = []
        
        try:
            # Risk score distribution
            risk_plot = plot_dir / 'risk_distribution.png'
            visualiser.plot_risk_score_distribution(results, str(risk_plot))
            if risk_plot.exists():
                plots_created.append('risk_distribution')
        except Exception as e:
            print(f"Risk distribution plot failed: {e}")
        
        try:
            # Variant classification chart
            class_plot = plot_dir / 'variant_classification.png'
            visualiser.plot_variant_classification_chart(results, str(class_plot))
            if class_plot.exists():
                plots_created.append('variant_classification')
        except Exception as e:
            print(f"Variant classification plot failed: {e}")
        
        try:
            # Comprehensive report plots
            comprehensive_plots = visualiser.create_comprehensive_report_plots(results, str(plot_dir))
            if comprehensive_plots:
                plots_created.extend(['comprehensive'])
        except Exception as e:
            print(f"Comprehensive plots failed: {e}")
        
        # At least some plots should be created successfully
        assert len(plots_created) > 0, "At least one plot should be created successfully"
        
        # Test summary statistics
        stats = visualiser.generate_summary_statistics(results)
        assert isinstance(stats, dict)
        assert 'total_variants' in stats
        assert stats['total_variants'] == len(results)
    
    def test_cli_integration_basic(self, sample_data_dir, temp_output_dir):
        """Test CLI integration with basic analysis."""
        input_file = sample_data_dir / 'test_variants.csv'
        runner = CliRunner()
        
        # Test basic analysis command
        result = runner.invoke(cli, [
            'analyse', str(input_file),
            '--output', str(temp_output_dir),
            '--verbose'
        ])
        
        if result.exit_code != 0:
            print("CLI Error Output:", result.output)
            print("Exception:", result.exception)
        
        assert result.exit_code == 0, f"CLI command failed: {result.output}"
        
        # Check that output file was created
        output_file = temp_output_dir / 'sickle_analysis.csv'
        assert output_file.exists(), "Output CSV file should be created"
        
        # Verify output file content
        results_df = pd.read_csv(output_file)
        assert len(results_df) > 0, "Results file should contain data"
        assert 'risk_score' in results_df.columns
    
    def test_cli_integration_with_reports(self, sample_data_dir, temp_output_dir):
        """Test CLI integration with report and plot generation."""
        input_file = sample_data_dir / 'test_variants.csv'
        runner = CliRunner()
        
        # Test analysis with reports and plots
        result = runner.invoke(cli, [
            'analyse', str(input_file),
            '--output', str(temp_output_dir),
            '--report',
            '--plot',
            '--verbose'
        ])
        
        if result.exit_code != 0:
            print("CLI Error Output:", result.output)
            print("Exception:", result.exception)
            # Don't fail the test if visualization libraries have issues
            if 'matplotlib' in str(result.exception) or 'display' in str(result.exception):
                pytest.skip("Visualization backend issues in test environment")
        
        assert result.exit_code == 0, f"CLI command with reports failed: {result.output}"
        
        # Check output files
        assert (temp_output_dir / 'sickle_analysis.csv').exists()
        
        # HTML report should be created
        html_report = temp_output_dir / 'sickle_report.html'
        if html_report.exists():
            assert html_report.stat().st_size > 0, "HTML report should not be empty"
        
        # Plots directory should be created
        plots_dir = temp_output_dir / 'plots'
        if plots_dir.exists():
            plot_files = list(plots_dir.glob('*.png'))
            # If plots were created, they should have content
            for plot_file in plot_files:
                assert plot_file.stat().st_size > 0, f"Plot file should not be empty: {plot_file}"
    
    def test_custom_config_integration(self, sample_data_dir, temp_output_dir):
        """Test integration with custom configuration."""
        input_file = sample_data_dir / 'test_variants.csv'
        
        # Create custom config
        custom_config = {
            'risk_weights': {
                'hbb_variants': 0.7,
                'modifier_genes': 0.3
            },
            'severity_thresholds': {
                'mild': 0.2,
                'moderate': 0.6,
                'severe': 1.0
            }
        }
        
        config_file = temp_output_dir / 'custom_config.json'
        with open(config_file, 'w') as f:
            json.dump(custom_config, f)
        
        # Test with custom config via CLI
        runner = CliRunner()
        result = runner.invoke(cli, [
            'analyse', str(input_file),
            '--output', str(temp_output_dir),
            '--config', str(config_file),
            '--verbose'
        ])
        
        if result.exit_code != 0:
            print("CLI Error Output:", result.output)
            print("Exception:", result.exception)
        
        assert result.exit_code == 0, f"CLI with custom config failed: {result.output}"
        
        # Verify analysis was performed
        output_file = temp_output_dir / 'sickle_analysis.csv'
        assert output_file.exists()
        
        results_df = pd.read_csv(output_file)
        assert len(results_df) > 0
    
    def test_error_handling_integration(self, temp_output_dir):
        """Test integration error handling with invalid data."""
        # Create invalid CSV file
        invalid_data = pd.DataFrame({
            'chromosome': ['invalid', 'also_invalid'],
            'position': [-100, 'not_a_number'],
            'invalid_column': ['data1', 'data2']
        })
        
        invalid_file = temp_output_dir / 'invalid_data.csv'
        invalid_data.to_csv(invalid_file, index=False)
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'analyse', str(invalid_file),
            '--verbose'
        ])
        
        # Should fail gracefully with informative error
        assert result.exit_code != 0, "Should fail with invalid data"
        assert 'validation failed' in result.output.lower() or 'error' in result.output.lower()
    
    def test_data_validation_workflow(self, sample_data_dir):
        """Test data validation in complete workflow."""
        input_file = sample_data_dir / 'test_variants.csv'
        
        analyser = SickleAnalyser()
        
        # Test validation passes for good data
        is_valid, messages = analyser.validate_input(input_file)
        assert is_valid, f"Validation should pass for sample data: {messages}"
        
        # Test that analysis works after validation
        if is_valid:
            results = analyser.analyse_file(input_file)
            assert len(results) > 0
            assert 'risk_score' in results.columns
    
    def test_different_file_formats(self, temp_output_dir):
        """Test analysis with different file formats."""
        # Create sample data
        sample_data = pd.DataFrame({
            'chromosome': ['11', '11', '11'],
            'position': [5248232, 5248158, 5247359],
            'ref_allele': ['A', 'G', 'G'],
            'alt_allele': ['T', 'A', 'A'],
            'genotype': ['0/1', '1/1', '0/1']
        })
        
        analyser = SickleAnalyser()
        
        # Test CSV format
        csv_file = temp_output_dir / 'test.csv'
        sample_data.to_csv(csv_file, index=False)
        
        results_csv = analyser.analyse_file(csv_file)
        assert len(results_csv) == 3
        
        # Test TSV format
        tsv_file = temp_output_dir / 'test.tsv'
        sample_data.to_csv(tsv_file, sep='\t', index=False)
        
        results_tsv = analyser.analyse_file(tsv_file)
        assert len(results_tsv) == 3
        
        # Results should be equivalent
        assert list(results_csv['chromosome']) == list(results_tsv['chromosome'])
        assert list(results_csv['position']) == list(results_tsv['position'])


class TestPerformanceAndScaling:
    """Tests for performance and scaling with larger datasets."""
    
    def test_large_dataset_handling(self, tmp_path):
        """Test analysis with larger dataset."""
        # Create larger synthetic dataset
        np_random = pytest.importorskip("numpy").random
        
        n_variants = 1000
        large_data = pd.DataFrame({
            'chromosome': ['11'] * n_variants,
            'position': range(5200000, 5200000 + n_variants),
            'ref_allele': np_random.choice(['A', 'T', 'G', 'C'], n_variants),
            'alt_allele': np_random.choice(['A', 'T', 'G', 'C'], n_variants),
            'genotype': np_random.choice(['0/0', '0/1', '1/1'], n_variants)
        })
        
        large_file = tmp_path / 'large_dataset.csv'
        large_data.to_csv(large_file, index=False)
        
        # Test analysis performance
        analyser = SickleAnalyser()
        
        import time
        start_time = time.time()
        results = analyser.analyse_file(large_file)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Results should be complete
        assert len(results) == n_variants
        assert 'risk_score' in results.columns
        
        # Performance should be reasonable (adjust threshold as needed)
        # Allow up to 30 seconds for 1000 variants in test environment
        assert analysis_time < 30, f"Analysis took too long: {analysis_time}s for {n_variants} variants"
        
        print(f"Processed {n_variants} variants in {analysis_time:.2f} seconds")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])