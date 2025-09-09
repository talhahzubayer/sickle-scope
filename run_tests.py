#!/usr/bin/env python3
"""
Test runner script for SickleScope project.

This script provides different testing modes for development and CI/CD.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode


def main():
    """Main test runner function."""
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    if len(sys.argv) == 1:
        print("SickleScope Test Runner")
        print("Usage: python run_tests.py [mode]")
        print("")
        print("Available modes:")
        print("  fast    - Run core analyser and CLI tests only (recommended for development)")
        print("  full    - Run all tests including integration and visualisation")
        print("  coverage - Run core tests with detailed coverage report")
        print("  ci      - Run tests suitable for CI/CD (fast, no visualisation)")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == 'fast':
        # Fast testing for development - core functionality only
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/test_analyser.py',
            'tests/test_cli.py',
            '-v', '--tb=short'
        ]
        return run_command(cmd)
    
    elif mode == 'coverage':
        # Coverage testing - core functionality with coverage report
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/test_analyser.py',
            'tests/test_cli.py',
            '--cov=sickle_scope',
            '--cov-report=term-missing',
            '--cov-report=html',
            '-v'
        ]
        return run_command(cmd)
    
    elif mode == 'full':
        # Full test suite including integration and visualisation
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '--cov=sickle_scope',
            '--cov-report=term-missing',
            '-v'
        ]
        return run_command(cmd)
    
    elif mode == 'ci':
        # CI/CD friendly testing - no GUI dependencies
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/test_analyser.py',
            'tests/test_cli.py',
            'tests/test_integration.py',
            '--cov=sickle_scope',
            '--cov-report=xml',
            '--tb=short',
            '-v'
        ]
        return run_command(cmd)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'python run_tests.py' to see available modes.")
        return 1


if __name__ == '__main__':
    import os
    sys.exit(main())