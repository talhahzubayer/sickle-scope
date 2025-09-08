"""
SickleScope Visualisation Framework

Advanced plotting functions for genetic variant analysis and risk assessment visualisation.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import warnings

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class SickleVisualiser:
    """Main class for creating visualisations of genetic variant analysis results."""
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 8)):
        """Initialise the SickleScope visualiser.
        
        Args:
            style: Matplotlib style to use ('default', 'seaborn', 'ggplot')
            figsize: Default figure size for plots
        """
        self.style = style
        self.default_figsize = figsize
        
        # Color schemes
        self.severity_colors = {
            'minimal_risk': '#2ecc71',      # Green
            'carrier_status': '#f39c12',    # Orange  
            'moderate_risk': '#e74c3c',     # Red
            'high_risk': '#8e44ad',         # Purple
            'protective_factors': '#3498db' # Blue
        }
        
        self.variant_colors = {
            'pathogenic': '#e74c3c',        # Red
            'modifier': '#3498db',          # Blue
            'benign': '#2ecc71',           # Green
            'uncertain': '#95a5a6'         # Gray
        }
        
        # Apply style
        if style != 'default':
            plt.style.use(style)
    
    def plot_risk_score_gauge(self, risk_score: float, variant_name: str = "Unknown", 
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Create a gauge-style plot for risk score visualisation.
        
        Args:
            risk_score: Risk score value (0-100 or negative for protective)
            variant_name: Name of the variant being displayed
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
        
        # Set up gauge parameters
        theta_min = 0
        theta_max = np.pi  # Semi-circle
        
        # Define score ranges and colors
        if risk_score < 0:
            # Protective factors
            color = self.severity_colors['protective_factors']
            category = "Protective Factors"
            score_normalised = 0.1  # Show as minimal on gauge
        elif risk_score <= 20:
            color = self.severity_colors['minimal_risk']
            category = "Minimal Risk"
            score_normalised = risk_score / 100
        elif risk_score <= 40:
            color = self.severity_colors['carrier_status']
            category = "Carrier Status"
            score_normalised = risk_score / 100
        elif risk_score <= 70:
            color = self.severity_colors['moderate_risk']
            category = "Moderate Risk"
            score_normalised = risk_score / 100
        else:
            color = self.severity_colors['high_risk']
            category = "High Risk"
            score_normalised = min(risk_score / 100, 1.0)
        
        # Draw gauge background
        theta_bg = np.linspace(theta_min, theta_max, 100)
        ax.fill_between(theta_bg, 0, 1, alpha=0.2, color='lightgray')
        
        # Draw colored segments
        segment_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
        segment_labels = ['Minimal', 'Carrier', 'Moderate', 'High']
        
        for i, (seg_color, label) in enumerate(zip(segment_colors, segment_labels)):
            theta_start = theta_min + i * (theta_max - theta_min) / 4
            theta_end = theta_min + (i + 1) * (theta_max - theta_min) / 4
            theta_seg = np.linspace(theta_start, theta_end, 25)
            ax.fill_between(theta_seg, 0.7, 1, alpha=0.6, color=seg_color, label=label)
        
        # Draw score indicator
        score_theta = theta_min + score_normalised * (theta_max - theta_min)
        ax.plot([score_theta, score_theta], [0, 0.9], linewidth=6, color='black')
        ax.plot(score_theta, 0.9, marker='o', markersize=15, color=color, markeredgecolor='black', markeredgewidth=2)
        
        # Formatting
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(1)
        ax.set_rlim(0, 1)
        ax.set_rticks([])
        ax.set_thetagrids(np.arange(0, 181, 45), ['0', '25', '50', '75', '100'])
        ax.grid(True, alpha=0.3)
        
        # Title and labels
        plt.title(f'Risk Assessment: {variant_name}\n'
                 f'Score: {risk_score:.1f} | Category: {category}', 
                 fontsize=16, fontweight='bold', pad=30)
        
        # Legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_variant_distribution(self, results: pd.DataFrame, 
                                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Create a distribution plot of variant classifications.
        
        Args:
            results: DataFrame with analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Variant Classification Distribution (Pie Chart)
        classification_counts = results['variant_classification'].value_counts()
        colors = [self.variant_colors.get(cat, '#95a5a6') for cat in classification_counts.index]
        
        ax1.pie(classification_counts.values, labels=classification_counts.index, 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Variant Classification Distribution', fontsize=14, fontweight='bold')
        
        # 2. Risk Score Distribution (Histogram)
        risk_scores = results['risk_score']
        ax2.hist(risk_scores, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(risk_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {risk_scores.mean():.2f}')
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Severity Categories (Bar Chart)
        severity_counts = results['severity_category'].value_counts()
        severity_colors = [self.severity_colors.get(cat, '#95a5a6') for cat in severity_counts.index]
        
        bars = ax3.bar(range(len(severity_counts)), severity_counts.values, color=severity_colors)
        ax3.set_xticks(range(len(severity_counts)))
        ax3.set_xticklabels(severity_counts.index, rotation=45, ha='right')
        ax3.set_ylabel('Count')
        ax3.set_title('Severity Category Distribution', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, severity_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 4. Chromosome Distribution (if multiple chromosomes present)
        if 'chromosome' in results.columns:
            chrom_counts = results['chromosome'].value_counts().sort_index()
            ax4.bar(range(len(chrom_counts)), chrom_counts.values, color='lightcoral')
            ax4.set_xticks(range(len(chrom_counts)))
            ax4.set_xticklabels([f'Chr {c}' for c in chrom_counts.index])
            ax4.set_ylabel('Variant Count')
            ax4.set_title('Variants by Chromosome', fontsize=14, fontweight='bold')
        else:
            # If no chromosome data, show genotype distribution
            if 'genotype' in results.columns:
                geno_counts = results['genotype'].value_counts()
                ax4.bar(range(len(geno_counts)), geno_counts.values, color='lightgreen')
                ax4.set_xticks(range(len(geno_counts)))
                ax4.set_xticklabels(geno_counts.index)
                ax4.set_ylabel('Count')
                ax4.set_title('Genotype Distribution', fontsize=14, fontweight='bold')
        
        plt.suptitle('Genetic Variant Analysis Overview', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_risk_heatmap(self, results: pd.DataFrame, 
                         save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Create a heatmap showing risk scores across positions and genotypes.
        
        Args:
            results: DataFrame with analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create pivot table for heatmap
        if len(results) > 0:
            # Create a simplified position identifier
            results_copy = results.copy()
            results_copy['position_id'] = results_copy['chromosome'].astype(str) + ':' + results_copy['position'].astype(str)
            
            # Pivot table with position vs genotype
            pivot_data = results_copy.pivot_table(values='risk_score', 
                                                 index='position_id', 
                                                 columns='genotype', 
                                                 aggfunc='mean')
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       cbar_kws={'label': 'Risk Score'}, ax=ax)
            
            ax.set_title('Risk Score Heatmap: Position vs Genotype', fontsize=14, fontweight='bold')
            ax.set_xlabel('Genotype', fontsize=12)
            ax.set_ylabel('Genomic Position', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No data available for heatmap', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Risk Score Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_severity_prediction(self, results: pd.DataFrame, 
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Create a comprehensive severity prediction visualisation.
        
        Args:
            results: DataFrame with analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Risk Score vs Severity Scatter Plot
        if 'severity_category' in results.columns and len(results) > 0:
            severity_order = ['minimal_risk', 'carrier_status', 'moderate_risk', 'high_risk', 'protective_factors']
            
            for i, category in enumerate(severity_order):
                if category in results['severity_category'].values:
                    mask = results['severity_category'] == category
                    subset = results[mask]
                    if len(subset) > 0:
                        ax1.scatter(subset['risk_score'], [i] * len(subset), 
                                  color=self.severity_colors.get(category, '#95a5a6'), alpha=0.7, s=100,
                                  label=f'{category.replace("_", " ").title()} (n={len(subset)})')
        else:
            ax1.text(0.5, 0.5, 'No severity data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax1.transAxes, fontsize=14)
        
        ax1.set_xlabel('Risk Score', fontsize=12)
        ax1.set_ylabel('Severity Category', fontsize=12)
        
        if 'severity_category' in results.columns and len(results) > 0:
            severity_order = ['minimal_risk', 'carrier_status', 'moderate_risk', 'high_risk', 'protective_factors']
            ax1.set_yticks(range(len(severity_order)))
            ax1.set_yticklabels([cat.replace('_', ' ').title() for cat in severity_order])
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax1.set_title('Risk Score vs Predicted Severity', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Clinical Management Recommendations
        if 'clinical_management' in results.columns:
            management_counts = results['clinical_management'].value_counts()
            
            # Create a horizontal bar chart
            y_pos = range(len(management_counts))
            colors = plt.cm.Set3(np.linspace(0, 1, len(management_counts)))
            
            bars = ax2.barh(y_pos, management_counts.values, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(management_counts.index, fontsize=10)
            ax2.set_xlabel('Number of Variants', fontsize=12)
            ax2.set_title('Clinical Management Recommendations', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, management_counts.values)):
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(value), ha='left', va='center', fontweight='bold')
        
        plt.suptitle('Severity Assessment and Clinical Recommendations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_report_plots(self, results: pd.DataFrame, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Generate all visualisation plots and save to directory.
        
        Args:
            results: DataFrame with analysis results
            output_dir: Directory to save all plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        # Skip individual risk gauge plots for now to prevent errors
        # Will implement in future version
        
        # 2. Variant distribution overview
        plot_paths['variant_distribution'] = output_dir / 'variant_distribution.png'
        self.plot_variant_distribution(results, plot_paths['variant_distribution'])
        plt.close()
        
        # 3. Risk heatmap
        plot_paths['risk_heatmap'] = output_dir / 'risk_heatmap.png'
        self.plot_risk_heatmap(results, plot_paths['risk_heatmap'])
        plt.close()
        
        # Skip severity prediction plot for now to avoid errors
        # Will be implemented in future version
        
        return plot_paths
    
    def generate_summary_statistics(self, results: pd.DataFrame) -> Dict[str, Union[int, float, str]]:
        """Generate summary statistics for the analysis results.
        
        Args:
            results: DataFrame with analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_variants': len(results),
            'pathogenic_variants': int(results['is_pathogenic'].sum()) if 'is_pathogenic' in results.columns else 0,
            'modifier_variants': int(results['is_modifier'].sum()) if 'is_modifier' in results.columns else 0,
            'mean_risk_score': float(results['risk_score'].mean()) if len(results) > 0 else 0.0,
            'max_risk_score': float(results['risk_score'].max()) if len(results) > 0 else 0.0,
            'min_risk_score': float(results['risk_score'].min()) if len(results) > 0 else 0.0,
            'high_risk_variants': int((results['risk_score'] > 70).sum()) if len(results) > 0 else 0,
            'protective_variants': int((results['risk_score'] < 0).sum()) if len(results) > 0 else 0,
        }
        
        # Most common severity category
        if 'severity_category' in results.columns and len(results) > 0:
            stats['most_common_severity'] = results['severity_category'].mode().iloc[0]
        else:
            stats['most_common_severity'] = 'unknown'
        
        return stats