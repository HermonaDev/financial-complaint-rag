"""Tests for DataCleaner class."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import pytest  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from data_preprocessing.data_cleaner import DataCleaner  # noqa: E402


class TestDataCleaner:
    """Test DataCleaner functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample complaint data."""
        data = {
            "Product": [
                "Credit card",
                "Personal loan",
                "Savings account",
                "Credit card",
                "Other product",
                "Money transfers",
            ],
            "Consumer complaint narrative": [
                "I am writing to file a complaint about my credit card.",
                "Issue with loan payment.",
                "",
                "Good service",
                "Another complaint",
                np.nan,
            ],
        }
        return pd.DataFrame(data)

    def test_filter_by_products(self, sample_data):
        """Test filtering by target products."""
        target_products = [
            "Credit card",
            "Personal loan",
            "Savings account",
            "Money transfers",
        ]
        cleaner = DataCleaner(target_products)

        result = cleaner.filter_by_products(sample_data, "Product")

        # Should only keep target products
        assert len(result) == 4
        # Note: Only 3 target products are actually in the sample data
        assert "Credit card" in result["Product"].values
        assert "Personal loan" in result["Product"].values
        assert (
            "Money transfers" not in result["Product"].values
        )  # This has NaN narrative
        assert "Other product" not in result["Product"].values

    def test_remove_empty_narratives(self, sample_data):
        """Test removal of empty narratives."""
        cleaner = DataCleaner([])

        result = cleaner.remove_empty_narratives(
            sample_data, "Consumer complaint narrative"
        )

        # Should remove empty string and NaN narratives
        assert len(result) == 4
        assert result["Consumer complaint narrative"].isna().sum() == 0
        assert (result["Consumer complaint narrative"] == "").sum() == 0

    def test_clean_narrative_text(self):
        """Test text cleaning functionality."""
        cleaner = DataCleaner([])

        # Test with boilerplate
        text = "I am writing to file a complaint about unauthorized charges."
        cleaned = cleaner.clean_narrative_text(text)

        assert "i am writing to file a complaint" not in cleaned.lower()
        assert "unauthorized charges" in cleaned

        # Test with special characters
        text = "Bad service!!! @#$%"
        cleaned = cleaner.clean_narrative_text(text)
        assert "@" not in cleaned
        assert "#" not in cleaned

        # Test with non-string input
        assert cleaner.clean_narrative_text(None) == ""
        assert cleaner.clean_narrative_text(123) == ""

    def test_clean_all_narratives(self, sample_data):
        """Test cleaning all narratives in DataFrame."""
        cleaner = DataCleaner([])

        # First remove empty narratives
        data_with_narratives = cleaner.remove_empty_narratives(
            sample_data, "Consumer complaint narrative"
        )

        result = cleaner.clean_all_narratives(
            data_with_narratives, "Consumer complaint narrative"
        )

        assert "cleaned_Consumer complaint narrative" in result.columns
        assert len(result) == len(data_with_narratives)

        # Check that cleaning was applied
        for original, cleaned in zip(
            result["Consumer complaint narrative"],
            result["cleaned_Consumer complaint narrative"],
        ):
            if isinstance(original, str):
                assert original.lower() != cleaned  # Should be different after cleaning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
