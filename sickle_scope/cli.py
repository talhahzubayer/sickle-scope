"""
SickleScope Command Line Interface

This module provides the CLI for the SickleScope genomics analysis package.
"""

import click
import os
from pathlib import Path
from .analyser import SickleAnalyser


@click.group()
@click.version_option(version="0.1.0")
@click.pass_context
def cli(ctx):
    """SickleScope: Python Genomics Analysis Package for Sickle Cell Disease
    
    Analyse genetic variants and assess risk for sickle cell disease with
    instant genetic risk assessment and visualisations.
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.option('--report', is_flag=True, help='Generate comprehensive HTML report')
@click.option('--plot', is_flag=True, help='Generate visualisation plots')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--no-ml', is_flag=True, help='Disable machine learning predictions')
def analyse(input_file, output, report, plot, verbose, config, no_ml):
    """Analyse genetic variants for sickle cell disease risk assessment.
    
    INPUT_FILE: CSV, TSV, or Excel file containing variant data
    """
    if verbose:
        click.echo(f"Analysing variants from: {input_file}")
    
    # Create output directory if specified
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            click.echo(f"Output directory: {output_path}")
    
    try:
        # Initialise the analyser
        enable_ml = not no_ml  # Invert the flag
        analyser = SickleAnalyser(verbose=verbose, enable_ml=enable_ml)
        
        if verbose and enable_ml:
            ml_info = analyser.get_ml_model_info()
            if ml_info['status'] == 'trained':
                click.echo("[OK] Machine learning model ready for severity prediction")
            else:
                click.echo("[WARNING] ML model not available, using rule-based predictions only")
        
        # Load configuration if provided
        if config:
            analyser.load_config(config)
            if verbose:
                click.echo(f"Loaded configuration from: {config}")
        
        # Perform analysis
        click.echo("Starting variant analysis...")
        results = analyser.analyse_file(input_file)
        
        # Save results
        output_file = output_path / "sickle_analysis.csv" if output else "sickle_analysis.csv"
        results.to_csv(output_file, index=False)
        click.echo(f"Analysis complete. Results saved to: {output_file}")
        
        # Generate report if requested
        if report:
            report_file = output_path / "sickle_report.html" if output else "sickle_report.html"
            analyser.generate_report(results, report_file)
            click.echo(f"Report generated: {report_file}")
        
        # Generate plots if requested
        if plot:
            plot_dir = output_path / "plots" if output else Path("plots")
            plot_dir.mkdir(exist_ok=True)
            analyser.generate_plots(results, plot_dir)
            click.echo(f"Plots saved to: {plot_dir}")
            
    except Exception as e:
        click.echo(f"Error during analysis: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
def validate(input_file):
    """Validate input file format and required columns.
    
    INPUT_FILE: File to validate for SickleScope analysis
    """
    try:
        analyser = SickleAnalyser()
        is_valid, messages = analyser.validate_input(input_file)
        
        if is_valid:
            click.echo(f" File validation passed: {input_file}")
        else:
            click.echo(f" File validation failed: {input_file}")
            for message in messages:
                click.echo(f"  - {message}")
            
    except Exception as e:
        click.echo(f"Error during validation: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
def info():
    """Display information about SickleScope analysis capabilities."""
    info_text = """
SickleScope Analysis Information:

Supported File Formats:
  " CSV (Comma-separated values)
  " TSV (Tab-separated values) 
  " Excel (.xlsx files)

Required Columns:
  " chromosome    - Chromosome identifier (e.g., '11', 'chr11')
  " position      - Genomic position
  " ref_allele    - Reference nucleotide
  " alt_allele    - Alternative nucleotide
  " genotype      - Genotype format (0/0, 0/1, 1/1)

Analysis Features:
  " Genetic variant detection and classification
  " Risk scoring with weighted algorithms
  " Modifier gene analysis
  " Severity prediction using machine learning
  " Population comparison and statistics

Visualisation Options:
  " Risk score dashboards
  " Chromosomal variant position mapping
  " Genotype distribution charts
  " Severity prediction with confidence intervals
    """
    click.echo(info_text)


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed model information')
def ml_info(verbose):
    """Display machine learning model information and status."""
    click.echo("SickleScope Machine Learning Model Information")
    click.echo("=" * 50)
    
    try:
        # Initialize analyser to get ML info
        analyser = SickleAnalyser(verbose=False, enable_ml=True)
        ml_info = analyser.get_ml_model_info()
        
        status = ml_info.get('status', 'unknown')
        click.echo(f"Model Status: {status}")
        
        if status == 'trained':
            click.echo(f"[OK] ML model is trained and ready")
            click.echo(f"Model Type: {ml_info.get('model_type', 'unknown')}")
            click.echo(f"Feature Count: {ml_info.get('feature_count', 0)}")
            
            severity_categories = ml_info.get('severity_categories', [])
            if severity_categories:
                click.echo(f"Prediction Categories: {', '.join(severity_categories)}")
            
            if verbose and 'feature_importance' in ml_info:
                click.echo("\nFeature Importance:")
                importance = ml_info['feature_importance']
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, score in sorted_features:
                    click.echo(f"  {feature}: {score:.3f}")
                    
        elif status == 'disabled':
            click.echo("[WARNING] ML predictions are disabled")
        elif status == 'not_trained':
            click.echo("[WARNING] ML model is not trained")
        else:
            click.echo("[ERROR] ML model status unknown")
            
    except Exception as e:
        click.echo(f"Error getting ML info: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    cli()