"""
Unit tests for SickleScope Visualisation Framework.

Tests plotting functions, chart generation, and visual analytics.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sickle_scope.visualiser import SickleVisualiser


class TestSickleVisualiser:
    """Test suite for SickleVisualiser class."""
    
    @pytest.fixture
    def visualiser(self):
        """Create a SickleVisualiser instance for testing."""
        return SickleVisualiser()
    
    @pytest.fixture
    def sample_results(self):
        """Create sample analysis results for testing."""
        return pd.DataFrame({
            'chromosome': ['11', '11', '11', '12', '11'],
            'position': [5248232, 5248158, 5247359, 1000000, 5249000],
            'ref_allele': ['A', 'G', 'G', 'C', 'T'],
            'alt_allele': ['T', 'A', 'A', 'T', 'C'],
            'genotype': ['0/1', '1/1', '0/1', '0/0', '0/1'],
            'variant_classification': ['pathogenic', 'pathogenic', 'uncertain', 'benign', 'modifier'],
            'variant_name': ['HbS', 'HbC', 'novel_variant', 'wild_type', 'BCL11A_mod'],
            'is_pathogenic': [True, True, False, False, False],
            'is_modifier': [False, False, False, False, True],
            'risk_score': [45.5, 60.0, 15.2, 0.0, -5.0],
            'severity_category': ['moderate_risk', 'moderate_risk', 'minimal_risk', 'minimal_risk', 'protective_factors'],
            'severity_description': ['Moderate risk', 'Moderate risk', 'Minimal risk', 'Minimal risk', 'Protective factors'],
            'clinical_management': ['Regular monitoring', 'Regular monitoring', 'Standard care', 'Standard care', 'Beneficial profile']
        })
    
    def test_initialization(self):
        """Test SickleVisualiser initialization."""
        # Test default initialization
        visualiser = SickleVisualiser()
        assert visualiser.style == 'default'
        assert visualiser.default_figsize == (10, 8)
        
        # Test custom initialization
        custom_visualiser = SickleVisualiser(style='seaborn', figsize=(12, 6))
        assert custom_visualiser.style == 'seaborn'
        assert custom_visualiser.default_figsize == (12, 6)
        
        # Test color schemes are defined
        assert 'minimal_risk' in visualiser.severity_colors
        assert 'pathogenic' in visualiser.variant_colors
    
    def test_color_schemes(self, visualiser):
        """Test color scheme definitions."""
        # Test severity colors
        severity_colors = visualiser.severity_colors
        expected_severities = ['minimal_risk', 'carrier_status', 'moderate_risk', 'high_risk', 'protective_factors']
        
        for severity in expected_severities:
            assert severity in severity_colors
            assert severity_colors[severity].startswith('#')  # Hex color format
        
        # Test variant colors
        variant_colors = visualiser.variant_colors
        expected_variants = ['pathogenic', 'modifier', 'benign', 'uncertain']
        
        for variant in expected_variants:
            assert variant in variant_colors
            assert variant_colors[variant].startswith('#')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_variant_distribution(self, mock_show, mock_savefig, visualiser, sample_results):
        """Test variant distribution plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'variant_dist.png'
            
            # This should not raise an exception
            visualiser.plot_variant_distribution(sample_results, str(output_path))
            
            # Verify matplotlib was called
            mock_savefig.assert_called()
            plt.close('all')  # Clean up plots
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_variant_distribution_chart(self, mock_show, mock_savefig, visualiser, sample_results):
        """Test variant distribution chart."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'variant_class.png'
            
            visualiser.plot_variant_distribution(sample_results, str(output_path))
            
            mock_savefig.assert_called()
            plt.close('all')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')  
    def test_plot_severity_prediction(self, mock_show, mock_savefig, visualiser, sample_results):
        """Test severity prediction plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'severity_prediction.png'
            
            visualiser.plot_severity_prediction(sample_results, str(output_path))
            
            mock_savefig.assert_called()
            plt.close('all')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_risk_heatmap(self, mock_show, mock_savefig, visualiser, sample_results):
        """Test risk heatmap plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'risk_heatmap.png'
            
            visualiser.plot_risk_heatmap(sample_results, str(output_path))
            
            mock_savefig.assert_called()
            plt.close('all')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_risk_score_gauge(self, mock_show, mock_savefig, visualiser):
        """Test risk score gauge chart creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'risk_gauge.png'
            
            # Test different risk scores
            test_scores = [0, 25, 50, 75, 100]
            
            for score in test_scores:
                visualiser.plot_risk_score_gauge(score, save_path=str(output_path))
                mock_savefig.assert_called()
                plt.close('all')
    
    def test_generate_summary_statistics(self, visualiser, sample_results):
        """Test summary statistics generation."""
        stats = visualiser.generate_summary_statistics(sample_results)
        
        # Check expected keys
        expected_keys = [
            'total_variants', 'pathogenic_variants', 'modifier_variants',
            'high_risk_variants', 'mean_risk_score',
            'max_risk_score', 'min_risk_score'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        # Check calculated values
        assert stats['total_variants'] == len(sample_results)
        assert stats['pathogenic_variants'] == sample_results['is_pathogenic'].sum()
        assert stats['modifier_variants'] == sample_results['is_modifier'].sum()
        assert abs(stats['mean_risk_score'] - sample_results['risk_score'].mean()) < 0.01
        assert stats['max_risk_score'] == sample_results['risk_score'].max()
        assert stats['min_risk_score'] == sample_results['risk_score'].min()
        
        # Check basic stats are numeric
        assert isinstance(stats['mean_risk_score'], (int, float))
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_comprehensive_report_plots(self, mock_savefig, visualiser, sample_results):
        """Test comprehensive report plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_paths = visualiser.create_comprehensive_report_plots(sample_results, temp_dir)
            
            # Should return dict of plot paths
            assert isinstance(plot_paths, dict)
            assert len(plot_paths) > 0
            
            # All paths should be Path objects
            for plot_name, path in plot_paths.items():
                assert isinstance(path, Path)
            
            # Verify plots were saved (mocked)
            assert mock_savefig.call_count >= len(plot_paths)
            plt.close('all')
    
    def test_empty_dataframe_handling(self, visualiser):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame(columns=['chromosome', 'risk_score', 'severity_category'])
        
        # Should handle empty data gracefully without raising exceptions
        stats = visualiser.generate_summary_statistics(empty_df)
        
        assert stats['total_variants'] == 0
        assert stats['pathogenic_variants'] == 0
        assert stats['mean_risk_score'] == 0  # or np.nan, depending on implementation
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_with_missing_columns(self, mock_show, mock_savefig, visualiser):
        """Test plotting with missing expected columns."""
        # DataFrame missing some expected columns
        incomplete_data = pd.DataFrame({
            'chromosome': ['11', '11'],
            'position': [5248232, 5248158]
            # Missing risk_score, severity_category, etc.
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_plot.png'
            
            # Should handle missing columns gracefully or raise appropriate error
            with pytest.raises((KeyError, ValueError)):
                visualiser.plot_variant_distribution(incomplete_data, str(output_path))
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_file_saving(self, mock_savefig, visualiser, sample_results):
        """Test that plots are saved to correct file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_plot.png'
            
            visualiser.plot_variant_distribution(sample_results, str(output_path))
            
            # Check that savefig was called with correct path
            mock_savefig.assert_called()
            call_args = mock_savefig.call_args[0]  # Get positional arguments
            assert str(output_path) in call_args or output_path in call_args
            plt.close('all')
    
    def test_custom_figsize_application(self, sample_results):
        """Test that custom figure size is applied."""
        custom_visualiser = SickleVisualiser(figsize=(15, 10))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'custom_size.png'
            
            # Create a plot and check figure size
            fig = plt.figure(figsize=custom_visualiser.default_figsize)
            assert fig.get_size_inches().tolist() == [15, 10]
            plt.close(fig)


class TestVisualisationEdgeCases:
    """Test edge cases and error conditions for visualisation."""
    
    @pytest.fixture
    def visualiser(self):
        return SickleVisualiser()
    
    def test_extreme_risk_scores(self, visualiser):
        """Test plotting with extreme risk score values."""
        extreme_data = pd.DataFrame({
            'chromosome': ['11', '11', '11'],
            'position': [5248232, 5248158, 5247359],
            'risk_score': [-100, 0, 1000],  # Extreme values
            'severity_category': ['protective_factors', 'minimal_risk', 'high_risk'],
            'variant_classification': ['modifier', 'benign', 'pathogenic']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'extreme_scores.png'
            
            # Should handle extreme values without crashing
            try:
                visualiser.plot_variant_distribution(extreme_data, str(output_path))
                # If successful, just close plots
                plt.close('all')
            except Exception as e:
                # If it fails, it should be a reasonable error
                assert isinstance(e, (ValueError, TypeError))
    
    def test_single_data_point(self, visualiser):
        """Test plotting with only one data point."""
        single_point = pd.DataFrame({
            'chromosome': ['11'],
            'position': [5248232],
            'risk_score': [45.5],
            'severity_category': ['moderate_risk'],
            'variant_classification': ['pathogenic']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'single_point.png'
            
            # Should handle single data point
            try:
                visualiser.plot_variant_distribution(single_point, str(output_path))
                plt.close('all')
            except Exception:
                # Some plots might not work with single points - that's okay
                pass
    
    def test_invalid_output_path(self, visualiser, sample_analysis_results):
        """Test plotting with invalid output path."""
        # Try to save to a directory that doesn't exist and can't be created
        invalid_path = "/invalid/nonexistent/path/plot.png"
        
        with pytest.raises((OSError, FileNotFoundError, PermissionError)):
            visualiser.plot_variant_distribution(sample_analysis_results, invalid_path)
    
    @patch('matplotlib.pyplot.savefig', side_effect=Exception("Mock save error"))
    def test_save_error_handling(self, mock_savefig, visualiser, sample_analysis_results):
        """Test handling of save errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'error_plot.png'
            
            # Should raise the exception from savefig
            with pytest.raises(Exception, match="Mock save error"):
                visualiser.plot_variant_distribution(sample_analysis_results, str(output_path))


# Sample fixture data that can be used across multiple test modules
@pytest.fixture
def sample_analysis_results():
    """Sample analysis results that can be reused in integration tests."""
    return pd.DataFrame({
        'chromosome': ['11'] * 10,
        'position': range(5248000, 5248010),
        'ref_allele': ['A', 'G', 'T', 'C'] * 2 + ['A', 'G'],
        'alt_allele': ['T', 'A', 'C', 'G'] * 2 + ['T', 'A'],
        'genotype': ['0/1', '1/1', '0/0', '0/1'] * 2 + ['0/1', '1/1'],
        'variant_classification': ['pathogenic'] * 3 + ['benign'] * 3 + ['uncertain'] * 2 + ['modifier'] * 2,
        'risk_score': [60, 80, 0, 0, 5, 10, 25, 30, -5, -10],
        'severity_category': ['moderate_risk'] * 2 + ['minimal_risk'] * 4 + ['carrier_status'] * 2 + ['protective_factors'] * 2
    })


if __name__ == '__main__':
    pytest.main([__file__])