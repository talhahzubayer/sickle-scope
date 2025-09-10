"""Setup configuration for SickleScope package."""

from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # Basic information
    name="sickle-scope",
    version="0.1.0",
    author="Talhah Zubayer", 
    author_email="talhahzubayer101@gmail.com",
    description="A genomic analysis tool for sickle cell disease variants",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    
    # Project URLs
    url="https://github.com/talhahzubayer/sickle-scope",
    project_urls={
        "Bug Reports": "https://github.com/talhahzubayer/sickle-scope/issues",
        "Source": "https://github.com/talhahzubayer/sickle-scope",
    },
    
    # Package configuration
    packages=find_packages(),
    package_data={
        "sickle_scope": ["data/*.json", "data/*.csv"],  # Include data files
    },
    
    # Classifiers help users find the project
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=[
        "click>=8.0.0",        # For CLI
        "pandas>=1.3.0",       # For data processing
        "numpy>=1.21.0",       # For numerical operations
        "matplotlib>=3.4.0",   # For plotting
        "seaborn>=0.11.0",     # For statistical visualisations
        "plotly>=5.0.0",       # For interactive visualisations
        "scipy>=1.7.0",        # For scientific computing
        "scikit-learn>=1.0.0", # For machine learning
        "joblib>=1.0.0",       # For model persistence
    ],
    
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    
    # Create command-line script
    entry_points={
        "console_scripts": [
            "sickle-analyse=sickle_scope.cli:cli",
        ],
    },
)