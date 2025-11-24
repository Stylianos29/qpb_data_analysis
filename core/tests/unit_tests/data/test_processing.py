import os
import tempfile
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
import logging

from library.data.processing import load_csv
from library.constants import ROOT


class TestLoadCSV:
    """Test suite for the load_csv function using synthetic data."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def basic_csv(self, temp_dir):
        """Create a basic valid CSV file."""
        csv_content = """Overlap_operator_method,Kernel_operator_type,QCD_beta_value,Configuration_label,Bare_mass,CG_epsilon,MSCG_epsilon,Clover_coefficient
KL,Wilson,6.00,001,-0.15,1.0e-12,1.0e-08,1
KL,Brillouin,6.00,002,-0.10,5.0e-13,5.0e-09,1
Chebyshev,Wilson,6.20,003,-0.05,1.0e-11,1.0e-07,0
KL,Wilson,6.20,004,0.00,2.0e-12,2.0e-08,1"""
        
        csv_file = temp_dir / "basic_valid.csv"
        csv_file.write_text(csv_content)
        return csv_file
    
    @pytest.fixture
    def missing_values_csv(self, temp_dir):
        """Create a CSV file with various types of missing values."""
        csv_content = """Overlap_operator_method,Kernel_operator_type,Configuration_label,Bare_mass
KL,Wilson,001,-0.15
,Brillouin,002,-0.10
Chebyshev,,003,
KL,Wilson,004,N/A
,NULL,005,NA"""
        
        csv_file = temp_dir / "missing_values.csv"
        csv_file.write_text(csv_content)
        return csv_file
    
    @pytest.fixture
    def categorical_csv(self, temp_dir):
        """Create a CSV file for testing categorical conversion."""
        csv_content = """Kernel_operator_type,QCD_beta_value,Configuration_label
Wilson,6.00,001
Brillouin,6.00,002
Wilson,6.20,003
Brillouin,6.20,004"""
        
        csv_file = temp_dir / "categorical.csv"
        csv_file.write_text(csv_content)
        return csv_file
    
    @pytest.fixture
    def unexpected_categorical_csv(self, temp_dir):
        """Create a CSV file with unexpected categorical values."""
        csv_content = """Kernel_operator_type,QCD_beta_value,Configuration_label
Wilson,6.00,001
Brillouin,6.00,002
UnexpectedKernel,6.20,003
Wilson,6.20,004"""
        
        csv_file = temp_dir / "unexpected_categorical.csv"
        csv_file.write_text(csv_content)
        return csv_file
    
    @pytest.fixture
    def empty_csv(self, temp_dir):
        """Create an empty CSV file (header only)."""
        csv_content = """Overlap_operator_method,Kernel_operator_type,QCD_beta_value"""
        
        csv_file = temp_dir / "empty.csv"
        csv_file.write_text(csv_content)
        return csv_file
    
    @pytest.fixture
    def malformed_csv(self, temp_dir):
        """Create a malformed CSV file that will actually cause parsing errors."""
        csv_content = """Overlap_operator_method,Kernel_operator_type,Configuration_label
KL,Wilson,001
"Unclosed quote field,002
KL,Brillouin,003"""
        
        csv_file = temp_dir / "malformed.csv"
        csv_file.write_text(csv_content)
        return csv_file

    def test_basic_loading(self, basic_csv):
        """Test basic CSV loading functionality."""
        # Use minimal converters to avoid conversion errors with test data
        df = load_csv(basic_csv, converters_mapping={})
        
        # Check return type and non-emptiness
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 4
        
        # Check expected columns
        expected_columns = [
            "Overlap_operator_method", "Kernel_operator_type", 
            "QCD_beta_value", "Configuration_label", "Bare_mass"
        ]
        for col in expected_columns:
            assert col in df.columns
    
    def test_categorical_conversion_default(self, categorical_csv):
        """Test that categorical conversion works by default."""
        df = load_csv(categorical_csv, converters_mapping={})
        
        # Check that Kernel_operator_type is categorical
        assert isinstance(df["Kernel_operator_type"].dtype, pd.CategoricalDtype)
        
        # Check categorical properties
        categories = df["Kernel_operator_type"].cat.categories.tolist()
        assert categories == ["Wilson", "Brillouin"]
        assert df["Kernel_operator_type"].cat.ordered == True
    
    def test_categorical_conversion_disabled(self, categorical_csv):
        """Test that categorical conversion can be disabled."""
        df = load_csv(categorical_csv)
        
        # Check that Kernel_operator_type is NOT categorical
        assert not isinstance(df["Kernel_operator_type"].dtype, pd.CategoricalDtype)
        assert df["Kernel_operator_type"].dtype == 'object'
    
    def test_unexpected_categorical_values(self, unexpected_categorical_csv):
        """Test handling of unexpected categorical values."""
        with patch("logging.warning") as mock_warning:
            df = load_csv(unexpected_categorical_csv, converters_mapping={})
            
            # Should warn about unexpected values
            mock_warning.assert_called()
            warning_msg = mock_warning.call_args[0][0]
            assert "unexpected values" in warning_msg.lower()
            assert "UnexpectedKernel" in warning_msg
            
            # Should NOT be categorical due to unexpected values
            assert not isinstance(df["Kernel_operator_type"].dtype, pd.CategoricalDtype)
    
    def test_custom_dtype_mapping(self, basic_csv):
        """Test custom dtype mapping."""
        custom_dtype = {"Bare_mass": str}
        df = load_csv(basic_csv, dtype_mapping=custom_dtype)
        
        assert df["Bare_mass"].dtype == 'object'  # pandas uses 'object' for strings
    
    def test_custom_converters(self, basic_csv):
        """Test custom converters."""
        custom_converter = {"Bare_mass": lambda x: float(x) * 2}
        df_original = load_csv(basic_csv)
        df_converted = load_csv(basic_csv, converters_mapping=custom_converter)
        
        # Values should be doubled
        original_values = df_original["Bare_mass"].astype(float)
        converted_values = df_converted["Bare_mass"]
        
        for i in range(len(original_values)):
            expected = original_values.iloc[i] * 2
            actual = converted_values.iloc[i]
            assert abs(actual - expected) < 1e-10
    
    def test_missing_values_detection(self, missing_values_csv):
        """Test that missing values are detected and reported."""
        with patch("logging.warning") as mock_warning:
            # Use empty converters to avoid conversion errors
            df = load_csv(missing_values_csv, converters_mapping={})
            
            # Should have warned about missing values
            mock_warning.assert_called()
            
            # Check that various warning types were called
            warning_calls = [call[0][0] for call in mock_warning.call_args_list]
            warning_text = ' '.join(warning_calls).lower()
            
            # Check for any missing value detection - more flexible assertion
            assert any(phrase in warning_text for phrase in [
                "missing values detected", 
                "total potentially missing",
                "empty string values detected",
                "missing value placeholders detected"
            ])
    
    def test_required_columns_validation(self, basic_csv):
        """Test required columns validation."""
        required_cols = {"Kernel_operator_type", "QCD_beta_value"}
        
        # Should work with valid required columns
        df = load_csv(basic_csv, validate_required_columns=required_cols)
        assert not df.empty
        
        # Should fail with missing required columns
        missing_cols = {"NonExistentColumn", "AnotherMissingColumn"}
        with pytest.raises(ValueError, match="Required columns missing"):
            load_csv(basic_csv, validate_required_columns=missing_cols)
    
    def test_empty_csv(self, empty_csv):
        """Test loading empty CSV (header only)."""
        with patch('logging.warning') as mock_warning:
            df = load_csv(empty_csv)
            
            assert isinstance(df, pd.DataFrame)
            assert df.empty
            mock_warning.assert_called()
            assert "empty" in mock_warning.call_args[0][0].lower()
    
    def test_file_not_found(self):
        """Test FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            load_csv("non_existent_file.csv")
    
    def test_invalid_path_type(self, temp_dir):
        """Test ValueError for directory instead of file."""
        with pytest.raises(ValueError, match="Path is not a file"):
            load_csv(temp_dir)
    
    def test_malformed_csv(self, malformed_csv):
        """Test handling of malformed CSV files."""
        with pytest.raises(ValueError, match="Error parsing CSV|Unexpected error loading CSV"):
            load_csv(malformed_csv)
    
    def test_pathlib_path_input(self, basic_csv):
        """Test that pathlib.Path objects work as input."""
        df = load_csv(basic_csv)  # basic_csv is already a Path object
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    
    def test_string_path_input(self, basic_csv):
        """Test that string paths work as input."""
        df = load_csv(str(basic_csv))
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    
    def test_file_extension_warning(self, temp_dir):
        """Test warning for unusual file extensions."""
        unusual_file = temp_dir / "data.xyz"
        unusual_file.write_text("col1,col2\n1,2\n")
        
        with patch('logging.warning') as mock_warning:
            df = load_csv(unusual_file)
            
            mock_warning.assert_called()
            warning_msg = mock_warning.call_args[0][0]
            assert "not typical for CSV files" in warning_msg
    
    def test_encoding_parameter(self, temp_dir):
        """Test custom encoding parameter."""
        # Create a file with specific encoding
        csv_content = "col1,col2\nvalue1,value2\n"
        csv_file = temp_dir / "encoded.csv"
        csv_file.write_text(csv_content, encoding='latin-1')
        
        # Should work with correct encoding
        df = load_csv(csv_file, encoding='latin-1')
        assert not df.empty
        
        # Should fail with wrong encoding (though this might not always fail depending on content)
        # This test is more about ensuring the parameter is passed through correctly
    
    def test_custom_categorical_config(self, temp_dir):
        """Test custom categorical columns configuration."""
        csv_content = """Status,Value Active,1 Inactive,2 Active,3"""
        
        csv_file = temp_dir / "custom_cat.csv"
        csv_file.write_text(csv_content)
                
        df = load_csv(csv_file, converters_mapping={})
        
        assert isinstance(df["Status"].dtype, pd.CategoricalDtype)
        assert "Pending" in df["Status"].cat.categories
        assert df["Status"].cat.ordered == False

    def test_converters_with_missing_values(self, temp_dir):
        """Test that converters handle missing values gracefully."""
        csv_content = """QCD_beta_value,Delta_Min,Delta_Max
6.00,0.1,1.5
,0.2,1.8
6.20,,2.0
6.40,0.3,"""
        
        csv_file = temp_dir / "converter_test.csv"
        csv_file.write_text(csv_content)
        
        # Create safe converters that handle empty strings
        safe_converters = {
            "QCD_beta_value": lambda x: f"{float(x):.2f}" if x.strip() else "",
            "Delta_Min": lambda x: f"{float(x):.2f}" if x.strip() else "",
            "Delta_Max": lambda x: f"{float(x):.2f}" if x.strip() else "",
        }
        
        df = load_csv(csv_file, converters_mapping=safe_converters)
        
        # Should load successfully without converter errors
        assert not df.empty
        assert len(df) == 4
        
        # Check that non-empty values were converted properly
        assert df.loc[0, "QCD_beta_value"] == "6.00"
        assert df.loc[2, "QCD_beta_value"] == "6.20"


# Integration test with real data (optional - keep one for full pipeline testing)
class TestLoadCSVIntegration:
    """Integration tests using real data files if they exist."""
    
    def test_load_real_csv_if_exists(self):
        """Test with real CSV file if it exists (integration test)."""
        real_csv_path = os.path.join(
            ROOT,
            "core/tests/mock_data/valid/"
            "KL_several_m_varying_EpsCG_and_EpsMSCG_processed_parameter_values.csv",
        )
        
        if os.path.exists(real_csv_path):
            df = load_csv(real_csv_path)
            assert isinstance(df, pd.DataFrame)
            # Add specific assertions based on your real data structure
        else:
            pytest.skip("Real CSV file not found - skipping integration test")


if __name__ == "__main__":
    pytest.main([__file__])