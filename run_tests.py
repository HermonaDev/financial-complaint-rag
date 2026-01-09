#!/usr/bin/env python3
"""Test runner script for CI/CD and local testing."""

import subprocess
import sys
import platform
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    # Use appropriate shell for the OS
    if platform.system() == "Windows":
        # Use cmd.exe on Windows
        result = subprocess.run(command, shell=True, capture_output=True, text=True, executable='cmd.exe')
    else:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in {description}:")
        print(f"Return code: {result.returncode}")
        if result.stderr.strip():
            print("Stderr:", result.stderr[:500])
        return False
    
    print(f"SUCCESS: {description}")
    if result.stdout.strip():
        print("Output:", result.stdout[:500])  # Print first 500 chars
    return True


def main():
    """Run all tests and checks."""
    success = True
    
    # 1. Run black formatting check
    print("\n" + "="*60)
    print("STEP 1: Checking code formatting with black...")
    print("="*60)
    
    # First format the code
    format_result = subprocess.run("black src/ tests/", shell=True, capture_output=True, text=True)
    if format_result.returncode != 0:
        print(f"Black formatting failed: {format_result.stderr[:200]}")
        success = False
    else:
        print("✓ Code formatted with black")
    
    # 2. Run flake8 linting
    print("\n" + "="*60)
    print("STEP 2: Running flake8 linting...")
    print("="*60)
    
    flake8_result = subprocess.run(
        (
            "flake8 src/ tests/ --max-line-length=88 --count"
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    
    if flake8_result.returncode != 0:
        print("Flake8 found issues:")
        if flake8_result.stdout:
            print(flake8_result.stdout)
        success = False
    else:
        print("✓ No flake8 issues found")
    
    # 3. Run mypy type checking
    print("\n" + "="*60)
    print("STEP 3: Running mypy type checking...")
    print("="*60)
    
    mypy_result = subprocess.run(
        (
            "mypy src/ --ignore-missing-imports"
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    
    if mypy_result.returncode != 0:
        print("Mypy found type issues:")
        if mypy_result.stdout:
            print(mypy_result.stdout[:1000])  # Limit output
        success = False
    else:
        print("✓ No mypy type issues found")
    
    # 4. Run pytest
    print("\n" + "="*60)
    print("STEP 4: Running pytest...")
    print("="*60)
    
    pytest_result = subprocess.run(
        (
            "pytest tests/ -v"
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    
    if pytest_result.returncode != 0:
        print("Pytest failures:")
        if pytest_result.stdout:
            # Extract just the failure summary
            lines = pytest_result.stdout.split('\n')
            for line in lines:
                if 'FAILED' in line or 'ERROR' in line or 'failed' in line.lower():
                    print(line)
        success = False
    else:
        print("✓ All pytest tests passed")
    
    # 5. Test data loading
    print("\n" + "="*60)
    print("STEP 5: Testing data loading...")
    print("="*60)
    
    try:
        sys.path.append('src')
        from data_preprocessing import DataLoader
        
        loader = DataLoader()
        
        # Check if data exists
        data_path = Path("data/raw/complaints.csv")
        if data_path.exists():
            df = loader.load_raw_complaints()
            print(f"✓ Successfully loaded {len(df):,} complaints")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        else:
            print("⚠ Data file not found: data/raw/complaints.csv")
            print("  (This is expected if data hasn't been downloaded yet)")
            
    except Exception as e:
        print(f"✗ Error testing data loading: {e}")
        success = False
    
    # Final result
    print("\n" + "="*60)
    if success:
        print("✓ ALL CHECKS PASSED")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Review issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
