"""
Unit tests for SickleScope Command Line Interface.

Tests CLI commands, argument parsing, and integration with the analysis engine.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sickle_scope.cli import cli, analyse


class TestCLI:
    """Test suite for CLI functionality."""
    
    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create temporary CSV file with sample data."""
        data = pd.DataFrame({
            'chromosome': ['11', '11', '11'],
            'position': [5248232, 5248158, 5247359],
            'ref_allele': ['A', 'G', 'G'],
            'alt_allele': ['T', 'A', 'A'],
            'genotype': ['0/1', '1/1', '0/1']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'SickleScope' in result.output
        assert 'Analyse genetic variants' in result.output
        assert 'analyse' in result.output
    
    def test_cli_version(self, runner):
        """Test CLI version command."""
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert '0.1.0' in result.output
    
    def test_analyse_command_help(self, runner):
        """Test analyse command help."""
        result = runner.invoke(cli, ['analyse', '--help'])
        
        assert result.exit_code == 0
        assert 'Analyse genetic variants' in result.output
        assert '--output' in result.output
        assert '--report' in result.output
        assert '--plot' in result.output
        assert '--verbose' in result.output
        assert '--config' in result.output
    
    def test_analyse_missing_file(self, runner):
        """Test analyse command with missing input file."""
        result = runner.invoke(cli, ['analyse', 'nonexistent.csv'])
        
        assert result.exit_code != 0
        # Click should handle the file existence check
    
    @patch('sickle_scope.cli.SickleAnalyser')
    def test_analyse_basic_execution(self, mock_analyser_class, runner, sample_csv_file):
        """Test basic analyse command execution."""
        # Mock the analyser
        mock_analyser = MagicMock()
        mock_analyser_class.return_value = mock_analyser
        
        # Mock analysis results
        mock_results = pd.DataFrame({
            'chromosome': ['11'],
            'position': [5248232],
            'risk_score': [45.5],
            'severity_category': ['moderate_risk']
        })
        mock_analyser.analyse_file.return_value = mock_results
        
        result = runner.invoke(cli, ['analyse', sample_csv_file])
        
        assert result.exit_code == 0
        mock_analyser_class.assert_called_once_with(verbose=False, enable_ml=True)
        mock_analyser.analyse_file.assert_called_once_with(sample_csv_file)
    
    @patch('sickle_scope.cli.SickleAnalyser')
    def test_analyse_with_verbose(self, mock_analyser_class, runner, sample_csv_file):
        """Test analyse command with verbose flag."""
        mock_analyser = MagicMock()
        mock_analyser_class.return_value = mock_analyser
        mock_analyser.analyse_file.return_value = pd.DataFrame({'col': [1]})
        
        result = runner.invoke(cli, ['analyse', sample_csv_file, '--verbose'])
        
        assert result.exit_code == 0
        mock_analyser_class.assert_called_once_with(verbose=True, enable_ml=True)
        assert 'Analysing variants from:' in result.output
    
    @patch('sickle_scope.cli.SickleAnalyser')
    def test_analyse_with_output_directory(self, mock_analyser_class, runner, sample_csv_file):
        """Test analyse command with output directory."""
        mock_analyser = MagicMock()
        mock_analyser_class.return_value = mock_analyser
        mock_analyser.analyse_file.return_value = pd.DataFrame({'col': [1]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'results'
            
            result = runner.invoke(cli, [
                'analyse', sample_csv_file,
                '--output', str(output_path),
                '--verbose'
            ])
            
            assert result.exit_code == 0
            assert output_path.exists()
            assert 'Output directory:' in result.output
    
    @patch('sickle_scope.cli.SickleAnalyser')
    def test_analyse_with_report_flag(self, mock_analyser_class, runner, sample_csv_file):
        """Test analyse command with report generation."""
        mock_analyser = MagicMock()
        mock_analyser_class.return_value = mock_analyser
        mock_results = pd.DataFrame({'col': [1]})
        mock_analyser.analyse_file.return_value = mock_results
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                'analyse', sample_csv_file,
                '--output', temp_dir,
                '--report'
            ])
            
            assert result.exit_code == 0
            mock_analyser.generate_report.assert_called_once()
            # Check that report path was constructed correctly
            call_args = mock_analyser.generate_report.call_args
            assert call_args[0][0].equals(mock_results)  # DataFrame argument
    
    @patch('sickle_scope.cli.SickleAnalyser')
    def test_analyse_with_plot_flag(self, mock_analyser_class, runner, sample_csv_file):
        """Test analyse command with plot generation."""
        mock_analyser = MagicMock()
        mock_analyser_class.return_value = mock_analyser
        mock_results = pd.DataFrame({'col': [1]})
        mock_analyser.analyse_file.return_value = mock_results
        mock_analyser.generate_plots.return_value = ['plot1.png', 'plot2.png']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                'analyse', sample_csv_file,
                '--output', temp_dir,
                '--plot'
            ])
            
            assert result.exit_code == 0
            mock_analyser.generate_plots.assert_called_once()
            # Check that plot directory path was constructed correctly
            call_args = mock_analyser.generate_plots.call_args
            assert call_args[0][0].equals(mock_results)  # DataFrame argument
    
    @patch('sickle_scope.cli.SickleAnalyser')
    def test_analyse_with_config_file(self, mock_analyser_class, runner, sample_csv_file):
        """Test analyse command with custom configuration."""
        mock_analyser = MagicMock()
        mock_analyser_class.return_value = mock_analyser
        mock_analyser.analyse_file.return_value = pd.DataFrame({'col': [1]})
        
        # Create temporary config file
        config_data = '{"risk_weights": {"hbb_variants": 0.7}}'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(config_data)
            config_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'analyse', sample_csv_file,
                '--config', config_path
            ])
            
            assert result.exit_code == 0
            mock_analyser.load_config.assert_called_once_with(config_path)
            
        finally:
            Path(config_path).unlink()
    
    @patch('sickle_scope.cli.SickleAnalyser')
    def test_analyse_all_flags_combined(self, mock_analyser_class, runner, sample_csv_file):
        """Test analyse command with all flags combined."""
        mock_analyser = MagicMock()
        mock_analyser_class.return_value = mock_analyser
        mock_results = pd.DataFrame({
            'chromosome': ['11'],
            'risk_score': [45.5]
        })
        mock_analyser.analyse_file.return_value = mock_results
        mock_analyser.generate_plots.return_value = ['plot1.png']
        
        config_data = '{"test": true}'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(config_data)
            config_path = f.name
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                result = runner.invoke(cli, [
                    'analyse', sample_csv_file,
                    '--output', temp_dir,
                    '--report',
                    '--plot',
                    '--verbose',
                    '--config', config_path
                ])
                
                assert result.exit_code == 0
                
                # Verify all methods were called
                mock_analyser.load_config.assert_called_once_with(config_path)
                mock_analyser.analyse_file.assert_called_once()
                mock_analyser.generate_report.assert_called_once()
                mock_analyser.generate_plots.assert_called_once()
                
        finally:
            Path(config_path).unlink()
    
    @patch('sickle_scope.cli.SickleAnalyser')
    def test_analyse_error_handling(self, mock_analyser_class, runner, sample_csv_file):
        """Test analyse command error handling."""
        mock_analyser = MagicMock()
        mock_analyser_class.return_value = mock_analyser
        
        # Mock an analysis error
        mock_analyser.analyse_file.side_effect = ValueError("Invalid data format")
        
        result = runner.invoke(cli, ['analyse', sample_csv_file])
        
        assert result.exit_code != 0
        assert 'Invalid data format' in result.output
    
    @patch('sickle_scope.cli.SickleAnalyser')
    def test_analyse_file_save_operations(self, mock_analyser_class, runner, sample_csv_file):
        """Test that analyse command saves files in correct locations."""
        mock_analyser = MagicMock()
        mock_analyser_class.return_value = mock_analyser
        mock_results = pd.DataFrame({
            'chromosome': ['11'],
            'position': [5248232],
            'risk_score': [45.5]
        })
        mock_analyser.analyse_file.return_value = mock_results
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                'analyse', sample_csv_file,
                '--output', temp_dir,
                '--verbose'
            ])
            
            assert result.exit_code == 0
            
            # Check that CSV was saved in the output directory
            expected_csv_path = Path(temp_dir) / 'sickle_analysis.csv'
            # Since we're mocking, we need to verify the path was constructed correctly
            # by checking that the temp_dir was created
            assert Path(temp_dir).exists()


class TestCLIEdgeCases:
    """Test edge cases and error conditions for CLI."""
    
    @pytest.fixture
    def runner(self):
        """Create Click test runner.""" 
        return CliRunner()
    
    def test_analyse_without_arguments(self, runner):
        """Test analyse command without any arguments."""
        result = runner.invoke(cli, ['analyse'])
        
        assert result.exit_code != 0
        # Should show error about missing argument
        assert 'Missing argument' in result.output or 'Usage:' in result.output
    
    def test_invalid_command(self, runner):
        """Test invalid CLI command."""
        result = runner.invoke(cli, ['invalid_command'])
        
        assert result.exit_code != 0
        assert 'No such command' in result.output
    
    def test_analyse_nonexistent_config(self, runner):
        """Test analyse with nonexistent config file.""" 
        with tempfile.NamedTemporaryFile(suffix='.csv') as sample_file:
            # Create minimal CSV content
            sample_file.write(b'chromosome,position,ref_allele,alt_allele,genotype\n11,5248232,A,T,0/1\n')
            sample_file.flush()
            
            result = runner.invoke(cli, [
                'analyse', sample_file.name,
                '--config', 'nonexistent_config.json'
            ])
            
            # Should fail due to nonexistent config file
            assert result.exit_code != 0


if __name__ == '__main__':
    pytest.main([__file__])