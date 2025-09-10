"""
SickleScope Command Line Interface

This module provides the CLI for the SickleScope genomics analysis package.
"""

import click
import os
import sys
import pandas as pd
from pathlib import Path
from .analyser import SickleAnalyser
import traceback


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
        click.echo("[INFO] Starting variant analysis...")
        results = analyser.analyse_file(input_file)
        
        # Display analysis summary
        total_variants = len(results)
        if 'classification' in results.columns:
            pathogenic_variants = len(results[results['classification'] == 'Pathogenic'])
        else:
            pathogenic_variants = 0
        click.echo(f"\n[SUCCESS] Analysis completed successfully!")
        click.echo(f"[INFO] Processed {total_variants} variants")
        if pathogenic_variants > 0:
            click.echo(f"[WARNING] Found {pathogenic_variants} pathogenic variants")
        else:
            click.echo("[SUCCESS] No pathogenic variants detected")
        
        # Save results
        output_file = output_path / "sickle_analysis.csv" if output else "sickle_analysis.csv"
        results.to_csv(output_file, index=False)
        click.echo(f"\n[OUTPUT] Results saved to: {output_file}")
        
        # Generate report if requested
        if report:
            report_file = output_path / "sickle_report.html" if output else "sickle_report.html"
            click.echo("\n[REPORT] Generating comprehensive report...")
            analyser.generate_report(results, report_file)
            click.echo(f"[OUTPUT] HTML report generated: {report_file}")
        
        # Generate plots if requested
        if plot:
            plot_dir = output_path / "plots" if output else Path("plots")
            plot_dir.mkdir(exist_ok=True)
            click.echo("\n[PLOTS] Generating visualisation plots...")
            analyser.generate_plots(results, plot_dir)
            click.echo(f"[OUTPUT] Visualisation plots saved to: {plot_dir}")
            
        click.echo(f"\n[COMPLETE] SickleScope analysis finished successfully!")
            
    except FileNotFoundError as e:
        click.echo(f"[ERROR] File not found: {input_file}", err=True)
        click.echo("Please check that the file path is correct and the file exists.", err=True)
        sys.exit(1)
    except PermissionError as e:
        click.echo(f"[ERROR] Permission denied: Cannot access {input_file}", err=True)
        click.echo("Please check file permissions or run with appropriate privileges.", err=True)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        click.echo(f"[ERROR] Empty file: {input_file} contains no data", err=True)
        click.echo("Please provide a file with variant data.", err=True)
        sys.exit(1)
    except pd.errors.ParserError as e:
        click.echo(f"[ERROR] File parsing error: {input_file} is not in a valid format", err=True)
        click.echo("Supported formats: CSV, TSV, Excel (.xlsx)", err=True)
        click.echo(f"Parser error: {str(e)}", err=True)
        sys.exit(1)
    except KeyError as e:
        click.echo(f"[ERROR] Missing required column: {str(e)}", err=True)
        click.echo("Required columns: chromosome, position, ref_allele, alt_allele, genotype", err=True)
        click.echo("Use 'sickle-analyse validate <file>' to check your data format.", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"[ERROR] Data validation error: {str(e)}", err=True)
        click.echo("Please check your data format and try again.", err=True)
        sys.exit(1)
    except MemoryError:
        click.echo("[ERROR] Out of memory: File is too large to process", err=True)
        click.echo("Try processing a smaller subset of your data.", err=True)
        sys.exit(1)
    except ImportError as e:
        click.echo(f"[ERROR] Missing dependency: {str(e)}", err=True)
        click.echo("Run 'pip install sickle-scope[dev]' to install all dependencies.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] Unexpected error during analysis: {str(e)}", err=True)
        if verbose:
            click.echo("\n=== Debug Information ===", err=True)
            traceback.print_exc()
            click.echo("========================\n", err=True)
        click.echo("If this error persists, please report it at:", err=True)
        click.echo("https://github.com/talhahzubayer/sickle-scope/issues", err=True)
        sys.exit(1)


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
            
    except FileNotFoundError:
        click.echo(f"[ERROR] File not found: {input_file}", err=True)
        click.echo("Please check that the file path is correct and the file exists.", err=True)
        sys.exit(1)
    except PermissionError:
        click.echo(f"[ERROR] Permission denied: Cannot access {input_file}", err=True)
        click.echo("Please check file permissions.", err=True)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        click.echo(f"[ERROR] Empty file: {input_file} contains no data", err=True)
        sys.exit(1)
    except pd.errors.ParserError as e:
        click.echo(f"[ERROR] File parsing error: Invalid format", err=True)
        click.echo("Supported formats: CSV, TSV, Excel (.xlsx)", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] Validation error: {str(e)}", err=True)
        click.echo("Please check your file format and try again.", err=True)
        sys.exit(1)


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
            
    except ImportError as e:
        click.echo(f"[ERROR] Missing ML dependencies: {str(e)}", err=True)
        click.echo("Install with: pip install sickle-scope[dev]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] Error getting ML info: {str(e)}", err=True)
        if verbose:
            click.echo("\n=== Debug Information ===", err=True)
            traceback.print_exc()
            click.echo("========================\n", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()