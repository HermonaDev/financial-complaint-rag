"""Tests for DataLoader class."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import pytest  # noqa: E402
import pandas as pd  # noqa: E402
from unittest.mock import patch  # noqa: E402

from data_preprocessing.data_loader import DataLoader  # noqa: E402


class TestDataLoader:
    """Test DataLoader functionality."""

    def test_init_with_config(self, tmp_path):
        """Test initialization with configuration."""
        # Create a test config file
        config_content = """
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        loader = DataLoader(str(config_file))

        assert loader.raw_data_path == Path("data/raw")
        assert loader.processed_data_path == Path("data/processed")

    def test_load_raw_complaints_file_not_found(self, tmp_path):
        """Test loading when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            loader = DataLoader()
            loader.raw_data_path = tmp_path  # Mock path
            loader.load_raw_complaints("nonexistent.csv")

    @patch("pandas.read_csv")
    def test_load_raw_complaints_success(self, mock_read_csv, tmp_path):
        """Test successful loading of complaints."""
        # Mock the DataFrame
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_read_csv.return_value = mock_df

        loader = DataLoader()
        loader.raw_data_path = tmp_path

        # Create test file
        test_file = tmp_path / "complaints.csv"
        test_file.write_text("test,data\n1,2\n")

        result = loader.load_raw_complaints("complaints.csv")

        assert len(result) == 3
        mock_read_csv.assert_called_once_with(
            str(tmp_path / "complaints.csv"), low_memory=False
        )

    def test_save_processed_data(self, tmp_path):
        """Test saving processed data."""
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

        loader = DataLoader()
        loader.processed_data_path = tmp_path

        output_path = loader.save_processed_data(test_df, "test_output.csv")

        assert output_path.exists()
        assert output_path.name == "test_output.csv"

        # Verify content
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["A", "B"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
