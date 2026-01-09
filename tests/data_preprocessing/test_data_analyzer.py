"""Tests for DataAnalyzer class."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import pytest  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from data_preprocessing.data_analyzer import DataAnalyzer  # noqa: E402


class TestDataAnalyzer:
    """Test DataAnalyzer functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        data = {
            "Product": np.random.choice(
                ["Credit card", "Personal loan", "Savings account"], size=100
            ),
            "Consumer complaint narrative": [
                "Test complaint " * np.random.randint(1, 10) for _ in range(100)
            ],
            "Date received": pd.date_range("2023-01-01", periods=100, freq="D"),
        }

        # Add some missing values
        for i in range(5):
            data["Consumer complaint narrative"][i] = np.nan

        return pd.DataFrame(data)

    def test_analyze_product_distribution(self, sample_data, tmp_path):
        """Test product distribution analysis."""
        analyzer = DataAnalyzer(output_dir=tmp_path)

        result = analyzer.analyze_product_distribution(sample_data, "Product")

        assert "counts" in result
        assert "percentages" in result
        assert "plot_path" in result

        # Check that plot was saved
        assert result["plot_path"].exists()

        # Check counts
        counts = result["counts"]
        assert sum(counts.values()) == len(sample_data)

    def test_analyze_narrative_lengths(self, sample_data, tmp_path):
        """Test narrative length analysis."""
        analyzer = DataAnalyzer(output_dir=tmp_path)

        result = analyzer.analyze_narrative_lengths(
            sample_data, "Consumer complaint narrative"
        )

        assert "mean" in result
        assert "median" in result
        assert "min" in result
        assert "max" in result

        # Check statistics are reasonable
        assert result["mean"] > 0
        assert result["min"] >= 0
        assert result["max"] >= result["mean"]

    def test_generate_summary_report(self, sample_data, tmp_path):
        """Test comprehensive summary report generation."""
        analyzer = DataAnalyzer(output_dir=tmp_path)

        report = analyzer.generate_summary_report(
            sample_data,
            product_col="Product",
            narrative_col="Consumer complaint narrative",
        )

        assert "total_records" in report
        assert "product_distribution" in report
        assert "narrative_analysis" in report
        assert "missing_values" in report

        assert report["total_records"] == len(sample_data)

        # Check nested structures
        assert "counts" in report["product_distribution"]
        assert "mean" in report["narrative_analysis"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
