#!/usr/bin/env python3
"""Test runner script for CI/CD and local testing."""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in {description}:")
        print(result.stderr)
        return False
    
    print(f"SUCCESS: {description}")
    if result.stdout.strip():
        print("Output:", result.stdout[:500])  # Print first 500 chars
    return True


def main():
    """Run all tests and checks."""
    success = True
    
    # 1. Run black formatting check
    if not run_command("black --check src/ tests/", "Black formatting check"):
        print("Formatting issues found. Run 'black src/ tests/' to fix.")
        success = False
    
    # 2. Run flake8 linting
    if not run_command("flake8 src/ tests/ --max-line-length=88", "Flake8 linting"):
        success = False
    
    # 3. Run mypy type checking
    if not run_command("mypy src/ --ignore-missing-imports", "Mypy type checking"):
        success = False
    
    # 4. Run pytest with coverage
    if not run_command(
        "pytest tests/ -v --cov=src/ --cov-report=term-missing", 
        "Pytest with coverage"
    ):
        success = False
    
    # 5. Run the actual Task 1 script (if data exists)
    if os.path.exists("data/raw/complaints.csv"):
        print("\n" + "="*60)
        print("Testing Task 1 script with actual data...")
        print("="*60)
        
        try:
            # Import and test
            sys.path.append('src')
            from data_preprocessing import DataLoader
            
            loader = DataLoader()
            df = loader.load_raw_complaints()
            print(f"✓ Successfully loaded {len(df)} complaints")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            success = False
    
    # Final result
    print("\n" + "="*60)
    if success:
        print("✓ ALL CHECKS PASSED")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
