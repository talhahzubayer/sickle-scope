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

# Interactive plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

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
        
        # ML prediction colors
        self.ml_severity_colors = {
            'normal': '#2ecc71',      # Green
            'mild': '#f39c12',        # Orange
            'moderate': '#e67e22',    # Dark orange  
            'severe': '#e74c3c'       # Red
        }
        
        # Apply style with error handling for deprecated styles
        if style != 'default':
            try:
                plt.style.use(style)
            except OSError:
                # Fallback to default if style is not available (e.g., 'seaborn' in newer matplotlib)
                plt.style.use('default')
    
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
    
    def plot_ml_predictions(self, results: pd.DataFrame, 
                           save_path: Optional[Union[str, Path]] = None,
                           figsize: Optional[Tuple[int, int]] = None) -> None:
        """Create visualisations for ML predictions.
        
        Args:
            results: DataFrame with analysis results including ML predictions
            save_path: Path to save the plot (optional)
            figsize: Figure size (optional)
        """
        if 'ml_predicted_severity' not in results.columns:
            print("Warning: No ML predictions found in results")
            return
        
        figsize = figsize or self.default_figsize
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ML Severity Distribution
        ml_counts = results['ml_predicted_severity'].value_counts()
        colors = [self.ml_severity_colors.get(severity, '#95a5a6') for severity in ml_counts.index]
        
        ax1.pie(ml_counts.values, labels=ml_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('ML Predicted Severity Distribution', fontsize=14, fontweight='bold')
        
        # 2. Confidence Score Distribution
        if 'ml_confidence_score' in results.columns:
            ax2.hist(results['ml_confidence_score'], bins=20, alpha=0.7, color='#3498db', edgecolor='black')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('ML Prediction Confidence Distribution', fontsize=14, fontweight='bold')
            ax2.axvline(results['ml_confidence_score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results["ml_confidence_score"].mean():.3f}')
            ax2.legend()
        
        # 3. Confidence vs Risk Score
        if 'ml_confidence_score' in results.columns:
            scatter = ax3.scatter(results['risk_score'], results['ml_confidence_score'], 
                                c=results['ml_predicted_severity'].map(self.ml_severity_colors),
                                alpha=0.6, s=50)
            ax3.set_xlabel('Risk Score')
            ax3.set_ylabel('ML Confidence Score')
            ax3.set_title('Risk Score vs ML Confidence', fontsize=14, fontweight='bold')
            
            # Add legend for severity colors
            handles = [plt.scatter([], [], c=color, label=severity) 
                      for severity, color in self.ml_severity_colors.items() 
                      if severity in results['ml_predicted_severity'].values]
            ax3.legend(handles=handles, title='ML Predicted Severity')
        
        # 4. ML Probability Heatmap
        prob_columns = [col for col in results.columns if col.startswith('ml_prob_')]
        if prob_columns:
            prob_data = results[prob_columns].T
            prob_data.index = [col.replace('ml_prob_', '') for col in prob_data.index]
            
            sns.heatmap(prob_data.iloc[:, :min(10, len(prob_data.columns))], 
                       annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
            ax4.set_title('ML Probability Heatmap (First 10 Variants)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Variant Index')
            ax4.set_ylabel('Severity Category')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    def plot_prediction_comparison(self, results: pd.DataFrame,
                                  save_path: Optional[Union[str, Path]] = None,
                                  figsize: Optional[Tuple[int, int]] = None) -> None:
        """Compare rule-based and ML predictions.
        
        Args:
            results: DataFrame with analysis results
            save_path: Path to save the plot (optional)
            figsize: Figure size (optional)
        """
        if 'ml_predicted_severity' not in results.columns:
            print("Warning: No ML predictions found in results")
            return
        
        figsize = figsize or (15, 10)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Side-by-side comparison
        rule_counts = results['severity_category'].value_counts()
        ml_counts = results['ml_predicted_severity'].value_counts()
        
        all_categories = sorted(set(rule_counts.index) | set(ml_counts.index))
        rule_values = [rule_counts.get(cat, 0) for cat in all_categories]
        ml_values = [ml_counts.get(cat, 0) for cat in all_categories]
        
        x = np.arange(len(all_categories))
        width = 0.35
        
        ax1.bar(x - width/2, rule_values, width, label='Rule-based', alpha=0.8, color='#3498db')
        ax1.bar(x + width/2, ml_values, width, label='ML Predictions', alpha=0.8, color='#e74c3c')
        
        ax1.set_xlabel('Severity Category')
        ax1.set_ylabel('Count')
        ax1.set_title('Rule-based vs ML Predictions', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_categories, rotation=45)
        ax1.legend()
        
        # 2. Agreement matrix (simplified)
        agreement = (results['severity_category'] == results['ml_predicted_severity'])
        agreement_pct = agreement.mean() * 100
        disagreement_pct = 100 - agreement_pct
        
        ax2.pie([agreement_pct, disagreement_pct], labels=['Agreement', 'Disagreement'], 
                colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Rule-based vs ML Agreement', fontsize=14, fontweight='bold')
        
        # 3. Risk Score vs ML Confidence colored by agreement
        if 'ml_confidence_score' in results.columns:
            colors = ['green' if agree else 'red' for agree in agreement]
            
            ax3.scatter(results['risk_score'], results['ml_confidence_score'], 
                       c=colors, alpha=0.6, s=50)
            ax3.set_xlabel('Risk Score')
            ax3.set_ylabel('ML Confidence Score')
            ax3.set_title('Risk vs Confidence (Green=Agreement, Red=Disagreement)', fontsize=12)
            
            # Add agreement statistics
            ax3.text(0.05, 0.95, f'Agreement: {agreement_pct:.1f}%', 
                    transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white'))
        
        # 4. Confidence distribution by agreement
        if 'ml_confidence_score' in results.columns:
            agree_conf = results[agreement]['ml_confidence_score']
            disagree_conf = results[~agreement]['ml_confidence_score']
            
            if len(agree_conf) > 0 and len(disagree_conf) > 0:
                ax4.hist([agree_conf, disagree_conf], bins=15, alpha=0.7, 
                        label=['Agreement', 'Disagreement'], color=['green', 'red'])
            elif len(agree_conf) > 0:
                ax4.hist(agree_conf, bins=15, alpha=0.7, 
                        label='Agreement', color='green')
            else:
                ax4.hist(disagree_conf, bins=15, alpha=0.7, 
                        label='Disagreement', color='red')
                        
            ax4.set_xlabel('ML Confidence Score')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Confidence Distribution by Agreement', fontsize=14, fontweight='bold')
            ax4.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
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
        
        # 4. Severity prediction plot
        plot_paths['severity_prediction'] = output_dir / 'severity_prediction.png'
        self.plot_severity_prediction(results, plot_paths['severity_prediction'])
        plt.close()
        
        # 5. ML predictions (if available)
        if 'ml_predicted_severity' in results.columns:
            plot_paths['ml_predictions'] = output_dir / 'ml_predictions.png'
            self.plot_ml_predictions(results, plot_paths['ml_predictions'])
            plt.close()
            
            # 6. Prediction comparison (if ML predictions available)
            plot_paths['prediction_comparison'] = output_dir / 'prediction_comparison.png'
            self.plot_prediction_comparison(results, plot_paths['prediction_comparison'])
            plt.close()
        
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
        
        # ML-specific statistics
        if 'ml_predicted_severity' in results.columns and len(results) > 0:
            stats['ml_predictions_available'] = True
            stats['ml_most_common_severity'] = results['ml_predicted_severity'].mode().iloc[0]
            
            if 'ml_confidence_score' in results.columns:
                stats['mean_ml_confidence'] = float(results['ml_confidence_score'].mean())
                stats['min_ml_confidence'] = float(results['ml_confidence_score'].min())
                stats['max_ml_confidence'] = float(results['ml_confidence_score'].max())
            
            # Agreement between rule-based and ML predictions
            if 'severity_category' in results.columns:
                agreement = (results['severity_category'] == results['ml_predicted_severity'])
                stats['prediction_agreement_rate'] = float(agreement.mean())
                stats['prediction_disagreements'] = int((~agreement).sum())
        else:
            stats['ml_predictions_available'] = False
        
        return stats
    
    # ==================== INTERACTIVE PLOTLY VISUALISATIONS ====================
    
    def plot_interactive_risk_gauge(self, risk_score: float, variant_name: str = "Unknown") -> go.Figure:
        """Create an interactive Plotly gauge plot for risk score visualisation.
        
        Args:
            risk_score: Risk score value (0-100 or negative for protective)
            variant_name: Name of the variant being displayed
            
        Returns:
            Plotly figure object
        """
        # Determine category and color
        if risk_score < 0:
            category = "Protective Factors"
            color = "#3498db"
            score_display = 0  # Show as minimal on gauge
        elif risk_score <= 20:
            category = "Minimal Risk"
            color = "#2ecc71"
            score_display = risk_score
        elif risk_score <= 40:
            category = "Carrier Status"
            color = "#f39c12"
            score_display = risk_score
        elif risk_score <= 70:
            category = "Moderate Risk"
            color = "#e74c3c"
            score_display = risk_score
        else:
            category = "High Risk"
            color = "#8e44ad"
            score_display = min(risk_score, 100)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score_display,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Risk Assessment: {variant_name}<br>Category: {category}"},
            delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#d5f4e6'},
                    {'range': [20, 40], 'color': '#ffeaa7'},
                    {'range': [40, 70], 'color': '#fab1a0'},
                    {'range': [70, 100], 'color': '#e17055'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            height=500,
            font={'color': "darkblue", 'family': "Arial"},
            paper_bgcolor="white"
        )
        
        return fig
    
    def plot_interactive_variant_manhattan(self, results: pd.DataFrame) -> go.Figure:
        """Create an interactive Manhattan-style plot for variant positions.
        
        Args:
            results: DataFrame with analysis results
            
        Returns:
            Plotly figure object
        """
        if len(results) == 0:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No data available for Manhattan plot",
                showarrow=False,
                xref="paper", yref="paper",
                font=dict(size=16)
            )
            return fig
        
        # Prepare data
        plot_data = results.copy()
        plot_data['neg_log_pvalue'] = -np.log10(np.maximum(1 - plot_data['risk_score']/100, 1e-10))
        plot_data['chromosome_num'] = plot_data['chromosome'].astype(str).str.replace('chr', '').astype(int, errors='ignore')
        
        # Create color map for chromosomes
        unique_chroms = sorted(plot_data['chromosome_num'].unique())
        colors = px.colors.qualitative.Set3[:len(unique_chroms)]
        color_map = dict(zip(unique_chroms, colors))
        
        fig = go.Figure()
        
        for chrom in unique_chroms:
            chrom_data = plot_data[plot_data['chromosome_num'] == chrom]
            
            fig.add_trace(go.Scatter(
                x=chrom_data['position'],
                y=chrom_data['neg_log_pvalue'],
                mode='markers',
                marker=dict(
                    color=color_map[chrom],
                    size=8,
                    opacity=0.7,
                    line=dict(width=0.5, color='black')
                ),
                text=chrom_data.apply(lambda row: 
                    f"Position: {row['position']}<br>" +
                    f"Risk Score: {row['risk_score']:.2f}<br>" +
                    f"Classification: {row.get('variant_classification', 'N/A')}<br>" +
                    f"Genotype: {row.get('genotype', 'N/A')}", axis=1),
                hovertemplate='<b>Chr %{customdata}</b><br>%{text}<extra></extra>',
                customdata=[chrom] * len(chrom_data),
                name=f'Chr {chrom}',
                showlegend=True
            ))
        
        fig.update_layout(
            title='Variant Manhattan Plot: Genomic Positions vs Risk Significance',
            xaxis_title='Genomic Position',
            yaxis_title='-log10(1 - Risk Score Percentile)',
            hovermode='closest',
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add significance threshold line
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red",
                      annotation_text="Significance Threshold (p=0.05)")
        
        return fig
    
    def plot_interactive_risk_distribution(self, results: pd.DataFrame) -> go.Figure:
        """Create interactive distribution plots for risk scores and classifications.
        
        Args:
            results: DataFrame with analysis results
            
        Returns:
            Plotly figure object
        """
        if len(results) == 0:
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="No data available", showarrow=False)
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Score Distribution', 'Variant Classification', 
                          'Severity Categories', 'Genotype Distribution'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Risk Score Histogram
        fig.add_trace(
            go.Histogram(
                x=results['risk_score'], 
                nbinsx=20, 
                name='Risk Score',
                marker_color='skyblue',
                opacity=0.7,
                hovertemplate='Risk Score: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add mean line
        mean_risk = results['risk_score'].mean()
        fig.add_vline(x=mean_risk, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_risk:.2f}", row=1, col=1)
        
        # 2. Variant Classification Pie Chart
        classification_counts = results['variant_classification'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=classification_counts.index,
                values=classification_counts.values,
                name="Classification",
                marker_colors=[self.variant_colors.get(cat, '#95a5a6') for cat in classification_counts.index],
                hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Severity Categories Bar Chart
        if 'severity_category' in results.columns:
            severity_counts = results['severity_category'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=severity_counts.index,
                    y=severity_counts.values,
                    name='Severity',
                    marker_color=[self.severity_colors.get(cat, '#95a5a6') for cat in severity_counts.index],
                    hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Genotype Distribution
        if 'genotype' in results.columns:
            genotype_counts = results['genotype'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=genotype_counts.index,
                    y=genotype_counts.values,
                    name='Genotype',
                    marker_color='lightgreen',
                    hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Comprehensive Variant Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def plot_interactive_risk_heatmap(self, results: pd.DataFrame) -> go.Figure:
        """Create an interactive heatmap of risk scores across positions and genotypes.
        
        Args:
            results: DataFrame with analysis results
            
        Returns:
            Plotly figure object
        """
        if len(results) == 0:
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="No data available for heatmap", showarrow=False)
            return fig
        
        # Prepare data
        results_copy = results.copy()
        results_copy['position_id'] = results_copy['chromosome'].astype(str) + ':' + results_copy['position'].astype(str)
        
        # Create pivot table
        pivot_data = results_copy.pivot_table(
            values='risk_score', 
            index='position_id', 
            columns='genotype', 
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Risk Score"),
            hovertemplate='Position: %{y}<br>Genotype: %{x}<br>Risk Score: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Risk Score Heatmap: Position vs Genotype',
            xaxis_title='Genotype',
            yaxis_title='Genomic Position',
            height=600
        )
        
        return fig
    
    def plot_interactive_ml_predictions(self, results: pd.DataFrame) -> go.Figure:
        """Create interactive visualisations for ML predictions with confidence scores.
        
        Args:
            results: DataFrame with analysis results including ML predictions
            
        Returns:
            Plotly figure object
        """
        if 'ml_predicted_severity' not in results.columns:
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="No ML predictions available", showarrow=False)
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ML Severity Predictions', 'Confidence vs Risk Score', 
                          'Confidence Distribution', 'Prediction Probabilities'),
            specs=[[{"type": "pie"}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. ML Severity Distribution
        ml_counts = results['ml_predicted_severity'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=ml_counts.index,
                values=ml_counts.values,
                name="ML Predictions",
                marker_colors=[self.ml_severity_colors.get(severity, '#95a5a6') for severity in ml_counts.index],
                hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Confidence vs Risk Score Scatter
        if 'ml_confidence_score' in results.columns:
            fig.add_trace(
                go.Scatter(
                    x=results['risk_score'],
                    y=results['ml_confidence_score'],
                    mode='markers',
                    marker=dict(
                        color=[self.ml_severity_colors.get(severity, '#95a5a6') 
                              for severity in results['ml_predicted_severity']],
                        size=8,
                        opacity=0.7
                    ),
                    text=results['ml_predicted_severity'],
                    hovertemplate='Risk Score: %{x}<br>Confidence: %{y:.3f}<br>Prediction: %{text}<extra></extra>',
                    name='Predictions'
                ),
                row=1, col=2
            )
            
            # 3. Confidence Distribution
            fig.add_trace(
                go.Histogram(
                    x=results['ml_confidence_score'],
                    nbinsx=20,
                    name='Confidence',
                    marker_color='#3498db',
                    opacity=0.7,
                    hovertemplate='Confidence: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Probability Heatmap (first 10 variants)
        prob_columns = [col for col in results.columns if col.startswith('ml_prob_')]
        if prob_columns and len(results) > 0:
            prob_data = results[prob_columns].head(10).T
            prob_data.index = [col.replace('ml_prob_', '') for col in prob_data.index]
            
            fig.add_trace(
                go.Heatmap(
                    z=prob_data.values,
                    x=[f'Variant {i+1}' for i in range(min(10, len(results)))],
                    y=prob_data.index,
                    colorscale='YlOrRd',
                    name='Probabilities',
                    hovertemplate='Variant: %{x}<br>Severity: %{y}<br>Probability: %{z:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Interactive ML Predictions Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def plot_interactive_population_comparison(self, results: pd.DataFrame, 
                                             population_column: str = 'population') -> go.Figure:
        """Create interactive population comparison visualisations.
        
        Args:
            results: DataFrame with analysis results
            population_column: Column name containing population data
            
        Returns:
            Plotly figure object
        """
        if population_column not in results.columns:
            # Create synthetic population data for demonstration
            populations = ['African', 'European', 'Asian', 'Mixed']
            results = results.copy()
            results[population_column] = np.random.choice(populations, len(results))
        
        if len(results) == 0:
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="No data available for population comparison", showarrow=False)
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Score by Population', 'Severity Distribution by Population',
                          'Variant Classification by Population', 'Population Demographics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        populations = results[population_column].unique()
        colors = px.colors.qualitative.Set2[:len(populations)]
        
        # 1. Risk Score Box Plot by Population
        for i, pop in enumerate(populations):
            pop_data = results[results[population_column] == pop]
            fig.add_trace(
                go.Box(
                    y=pop_data['risk_score'],
                    name=pop,
                    marker_color=colors[i],
                    boxpoints='outliers',
                    hovertemplate='Population: %{x}<br>Risk Score: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Severity Distribution by Population
        if 'severity_category' in results.columns:
            severity_pop_counts = results.groupby([population_column, 'severity_category']).size().unstack(fill_value=0)
            
            for severity in severity_pop_counts.columns:
                fig.add_trace(
                    go.Bar(
                        x=severity_pop_counts.index,
                        y=severity_pop_counts[severity],
                        name=severity,
                        marker_color=self.severity_colors.get(severity, '#95a5a6'),
                        hovertemplate='Population: %{x}<br>Severity: ' + severity + '<br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # 3. Variant Classification by Population  
        if 'variant_classification' in results.columns:
            class_pop_counts = results.groupby([population_column, 'variant_classification']).size().unstack(fill_value=0)
            
            for classification in class_pop_counts.columns:
                fig.add_trace(
                    go.Bar(
                        x=class_pop_counts.index,
                        y=class_pop_counts[classification],
                        name=classification,
                        marker_color=self.variant_colors.get(classification, '#95a5a6'),
                        hovertemplate='Population: %{x}<br>Classification: ' + classification + '<br>Count: %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # 4. Population Demographics
        pop_counts = results[population_column].value_counts()
        fig.add_trace(
            go.Pie(
                labels=pop_counts.index,
                values=pop_counts.values,
                name="Population",
                marker_colors=colors,
                hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Population Comparison Analysis",
            showlegend=True,
            height=800,
            barmode='stack'
        )
        
        return fig
    
    def create_interactive_dashboard(self, results: pd.DataFrame) -> go.Figure:
        """Create a comprehensive interactive dashboard combining multiple visualisations.
        
        Args:
            results: DataFrame with analysis results
            
        Returns:
            Plotly figure object with multiple subplots
        """
        if len(results) == 0:
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="No data available for dashboard", showarrow=False)
            return fig
        
        # Calculate summary statistics
        stats = self.generate_summary_statistics(results)
        
        # Create main dashboard figure
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Risk Score Overview', 'Variant Classifications', 'ML Predictions',
                'Genomic Positions', 'Severity Distribution', 'Confidence Scores',
                'Population Analysis', 'Risk Heatmap', 'Summary Statistics'
            ),
            specs=[
                [{"secondary_y": False}, {"type": "pie"}, {"type": "pie"}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "heatmap"}, {"secondary_y": False}]
            ]
        )
        
        # 1. Risk Score Distribution
        fig.add_trace(
            go.Histogram(x=results['risk_score'], nbinsx=15, name='Risk Scores', 
                        marker_color='skyblue', opacity=0.7),
            row=1, col=1
        )
        
        # 2. Variant Classifications
        class_counts = results['variant_classification'].value_counts()
        fig.add_trace(
            go.Pie(labels=class_counts.index, values=class_counts.values, name="Classifications"),
            row=1, col=2
        )
        
        # 3. ML Predictions (if available)
        if 'ml_predicted_severity' in results.columns:
            ml_counts = results['ml_predicted_severity'].value_counts()
            fig.add_trace(
                go.Pie(labels=ml_counts.index, values=ml_counts.values, name="ML Predictions"),
                row=1, col=3
            )
        
        # 4. Genomic Positions Scatter
        fig.add_trace(
            go.Scatter(
                x=results['position'], 
                y=results['risk_score'],
                mode='markers',
                marker=dict(color=results['risk_score'], colorscale='RdYlBu_r', size=6),
                name='Variants'
            ),
            row=2, col=1
        )
        
        # 5. Severity Distribution
        if 'severity_category' in results.columns:
            sev_counts = results['severity_category'].value_counts()
            fig.add_trace(
                go.Bar(x=sev_counts.index, y=sev_counts.values, name='Severity', 
                      marker_color='lightcoral'),
                row=2, col=2
            )
        
        # 6. Confidence Scores (if available)
        if 'ml_confidence_score' in results.columns:
            fig.add_trace(
                go.Histogram(x=results['ml_confidence_score'], nbinsx=15, name='Confidence', 
                           marker_color='lightgreen', opacity=0.7),
                row=2, col=3
            )
        
        # Add summary text
        summary_text = f"""
        <b>Analysis Summary:</b><br>
        Total Variants: {stats['total_variants']}<br>
        Pathogenic: {stats['pathogenic_variants']}<br>
        Mean Risk Score: {stats['mean_risk_score']:.2f}<br>
        High Risk Variants: {stats['high_risk_variants']}<br>
        Most Common Severity: {stats['most_common_severity']}
        """
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.85, y=0.15,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        fig.update_layout(
            title_text="SickleScope Interactive Analysis Dashboard",
            showlegend=False,
            height=1200,
            template="plotly_white"
        )
        
        return fig
    
    def plot_advanced_statistical_plots(self, results: pd.DataFrame) -> go.Figure:
        """Create advanced statistical plots including QQ plots, violin plots, and correlation matrices.
        
        Args:
            results: DataFrame with analysis results
            
        Returns:
            Plotly figure object with advanced statistical visualisations
        """
        if len(results) == 0:
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="No data available for statistical plots", showarrow=False)
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Score QQ Plot', 'Violin Plot by Severity', 
                          'Position Density Plot', 'Statistical Summary'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. QQ Plot for Risk Scores (against normal distribution)
        risk_scores = results['risk_score'].sort_values()
        n = len(risk_scores)
        theoretical_quantiles = np.linspace(0.01, 0.99, n)
        theoretical_values = np.percentile(np.random.normal(risk_scores.mean(), risk_scores.std(), 10000), 
                                         theoretical_quantiles * 100)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_values,
                y=risk_scores.values,
                mode='markers',
                marker=dict(color='blue', size=4, opacity=0.6),
                name='QQ Plot',
                hovertemplate='Theoretical: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add reference line for perfect normal distribution
        min_val, max_val = min(theoretical_values.min(), risk_scores.min()), max(theoretical_values.max(), risk_scores.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Normal Reference',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Violin Plot by Severity Category
        if 'severity_category' in results.columns:
            severity_categories = results['severity_category'].unique()
            for i, category in enumerate(severity_categories):
                category_data = results[results['severity_category'] == category]
                fig.add_trace(
                    go.Violin(
                        y=category_data['risk_score'],
                        name=category,
                        box_visible=True,
                        meanline_visible=True,
                        points='outliers',
                        hovertemplate=f'Severity: {category}<br>Risk Score: %{{y}}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # 3. Position Density Plot (similar to Manhattan but as density)
        if 'chromosome' in results.columns and 'position' in results.columns:
            # Create a density-like plot using histogram2d
            chromosomes = results['chromosome'].unique()
            
            for chrom in chromosomes:
                chrom_data = results[results['chromosome'] == chrom]
                if len(chrom_data) > 1:
                    fig.add_trace(
                        go.Histogram2d(
                            x=chrom_data['position'],
                            y=chrom_data['risk_score'],
                            colorscale='Viridis',
                            nbinsx=20,
                            nbinsy=15,
                            name=f'Chr {chrom} Density',
                            hovertemplate='Position: %{x}<br>Risk Score: %{y}<br>Count: %{z}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    break  # Show only first chromosome to avoid overcrowding
        
        # 4. Statistical Summary Box
        stats = self.generate_summary_statistics(results)
        
        # Create a table-like visualisation using annotations
        summary_stats = [
            f"Total Variants: {stats['total_variants']}",
            f"Mean Risk: {stats['mean_risk_score']:.2f}",
            f"Std Dev: {results['risk_score'].std():.2f}",
            f"Median Risk: {results['risk_score'].median():.2f}",
            f"IQR: {results['risk_score'].quantile(0.75) - results['risk_score'].quantile(0.25):.2f}",
            f"Skewness: {results['risk_score'].skew():.3f}",
            f"Kurtosis: {results['risk_score'].kurtosis():.3f}"
        ]
        
        # Create a bar chart showing key statistics
        stat_names = ['Mean', 'Median', 'Q25', 'Q75', 'Min', 'Max']
        stat_values = [
            results['risk_score'].mean(),
            results['risk_score'].median(), 
            results['risk_score'].quantile(0.25),
            results['risk_score'].quantile(0.75),
            results['risk_score'].min(),
            results['risk_score'].max()
        ]
        
        fig.add_trace(
            go.Bar(
                x=stat_names,
                y=stat_values,
                marker_color='lightblue',
                name='Statistics',
                hovertemplate='%{x}: %{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Advanced Statistical Analysis Dashboard",
            showlegend=True,
            height=800
        )
        
        # Update subplot titles and axes
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=1)
        fig.update_yaxes(title_text="Observed Risk Scores", row=1, col=1)
        
        fig.update_xaxes(title_text="Severity Category", row=1, col=2)
        fig.update_yaxes(title_text="Risk Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Genomic Position", row=2, col=1)
        fig.update_yaxes(title_text="Risk Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Statistic", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        return fig
    
    def plot_correlation_analysis(self, results: pd.DataFrame) -> go.Figure:
        """Create correlation analysis plots for numerical features.
        
        Args:
            results: DataFrame with analysis results
            
        Returns:
            Plotly figure object showing correlations
        """
        if len(results) == 0:
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="No data available for correlation analysis", showarrow=False)
            return fig
        
        # Select numerical columns
        numerical_cols = results.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(x=0.5, y=0.5, text="Insufficient numerical columns for correlation analysis", showarrow=False)
            return fig
        
        # Calculate correlation matrix
        corr_matrix = results[numerical_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        # Add correlation values as text annotations
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.index)):
                fig.add_annotation(
                    x=corr_matrix.columns[i],
                    y=corr_matrix.index[j],
                    text=f"{corr_matrix.iloc[j, i]:.3f}",
                    showarrow=False,
                    font=dict(color="white" if abs(corr_matrix.iloc[j, i]) > 0.5 else "black")
                )
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            xaxis_title='Features',
            yaxis_title='Features',
            height=600,
            width=600
        )
        
        return fig