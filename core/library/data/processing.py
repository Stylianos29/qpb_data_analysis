import logging
from pathlib import Path
from typing import cast
from typing import Dict, Callable, Optional, Union, Set, Any

import pandas as pd

from ..constants import DTYPE_MAPPING, CONVERTERS_MAPPING


def load_csv(
    input_csv_file_path: Union[str, Path],
    dtype_mapping: Optional[Dict[str, Any]] = None,
    converters_mapping: Optional[Dict[str, Callable]] = None,
    validate_required_columns: Optional[Set[str]] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame with robust error handling
    and optional filtering of dtypes and converters based on the CSV
    file's header.

    Parameters:
    -----------
    input_csv_file_path : str or Path
        Path to the input CSV file.
    dtype_mapping : dict, optional
        A dictionary mapping column names to data types. Defaults to
        DTYPE_MAPPING from constants.
    converters_mapping : dict, optional
        A dictionary mapping column names to converter functions.
        Defaults to CONVERTERS_MAPPING from constants.
    categorical_columns : dict, optional
        A dictionary mapping column names to categorical configuration:
        {column_name: {'categories': [list], 'ordered': bool}} Example:
        {'Kernel_operator_type': {'categories': ['Wilson', 'Brillouin'],
        'ordered': True}}
    validate_required_columns : set, optional
        Set of column names that must be present in the CSV.
    encoding : str, optional
        File encoding. Defaults to 'utf-8'.
    apply_categorical : bool, optional
        Whether to apply categorical data types. Defaults to True.

    Returns:
    --------
    pd.DataFrame
        The loaded DataFrame with the specified dtypes and converters
        applied.

    Raises:
    -------
    FileNotFoundError
        If the CSV file doesn't exist.
    ValueError
        If required columns are missing or if the file is not a valid
        CSV.
    pd.errors.EmptyDataError
        If the CSV file is empty.
    UnicodeDecodeError
        If there are encoding issues.
    """

    # Convert to Path object for better handling
    file_path = Path(input_csv_file_path)

    # Validate file existence
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if file_path.suffix.lower() not in [".csv", ".txt"]:
        logging.warning(
            f"File extension '{file_path.suffix}' is not typical for CSV files"
        )

    # Set default mappings if not provided
    if dtype_mapping is None:
        dtype_mapping = DTYPE_MAPPING.copy()

    if converters_mapping is None:
        converters_mapping = CONVERTERS_MAPPING.copy()

    try:
        # Read header more robustly using pandas
        try:
            header_df = pd.read_csv(file_path, nrows=0, encoding=encoding)
            csv_header = set(header_df.columns)
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"CSV file is empty: {file_path}")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Encoding error reading {file_path}. "
                "Try a different encoding parameter.",
            )

        # Validate required columns if specified
        if validate_required_columns:
            missing_columns = validate_required_columns - csv_header
            if missing_columns:
                raise ValueError(
                    f"Required columns missing from CSV: {missing_columns}. "
                )

        # Filter mappings based on the header
        filtered_dtype_mapping = {
            field: dtype
            for field, dtype in dtype_mapping.items()
            if field in csv_header
        }

        filtered_converters_mapping = {
            field: converter
            for field, converter in converters_mapping.items()
            if field in csv_header
        }

        # Load the CSV file with the filtered mappings
        try:
            dataframe = pd.read_csv(
                file_path,
                dtype=cast(Any, filtered_dtype_mapping),
                converters=filtered_converters_mapping,
                encoding=encoding,
            )
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading CSV {file_path}: {e}")

        # Validate that dataframe is not empty
        if dataframe.empty:
            logging.warning(f"Loaded DataFrame is empty: {file_path}")
            return dataframe

        # Check for missing values
        _check_missing_values(dataframe, file_path)

        logging.info(
            f"Successfully loaded CSV with shape {dataframe.shape}: {file_path}"
        )
        return dataframe

    except Exception as e:
        logging.error(f"Failed to load CSV file {file_path}: {e}")
        raise


def apply_categorical_dtypes(
    dataframe: pd.DataFrame, categorical_config: Dict[str, Dict[str, Union[list, bool]]]
) -> pd.DataFrame:
    """
    Apply categorical data types to DataFrame columns after all
    transformations are complete.

    This should be called AFTER data transformations (e.g., "Standard" → "Wilson")
    to ensure values match expected categories.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        DataFrame to apply categorical types to
    categorical_config : dict
        Configuration mapping column names to category definitions:
        {column_name: {'categories': [list], 'ordered': bool}}
        Example: {'Kernel_operator_type': {'categories': ['Wilson', 'Brillouin'],
                                           'ordered': False}}

    Returns:
    --------
    pd.DataFrame
        DataFrame with categorical types applied where configured

    Example:
    --------
    >>> config = {
    ...     'Kernel_operator_type': {
    ...         'categories': ['Wilson', 'Brillouin'],
    ...         'ordered': False
    ...     }
    ... }
    >>> df = apply_categorical_dtypes(df, config)
    """
    for column_name, config in categorical_config.items():
        if column_name in dataframe.columns:
            try:
                dataframe[column_name] = pd.Categorical(
                    dataframe[column_name],
                    categories=config["categories"],
                    ordered=cast(bool, config.get("ordered", False)),
                )
                logging.info(f"Applied categorical type to column '{column_name}'")
            except Exception as e:
                logging.warning(
                    f"Could not apply categorical type to '{column_name}': {e}"
                )

    return dataframe


def _check_missing_values(dataframe: pd.DataFrame, file_path: Path) -> None:
    """
    Check for and report missing values in the DataFrame.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The DataFrame to check for missing values.
    file_path : Path
        The file path for logging context.
    """
    total_cells = dataframe.size

    # Check for standard missing values (NaN, None, etc.)
    missing_mask = dataframe.isna()
    total_missing = missing_mask.sum().sum()

    # Check for empty strings (often considered missing in CSV context)
    empty_string_mask = dataframe.astype(str).eq("")
    total_empty_strings = empty_string_mask.sum().sum()

    # Check for common missing value placeholders
    common_na_values = {"N/A", "NA", "NULL", "null", "None", "none", "#N/A", "#NULL!"}
    placeholder_mask = dataframe.astype(str).isin(common_na_values)
    total_placeholders = placeholder_mask.sum().sum()

    # Report findings
    if total_missing > 0:
        missing_percentage = (total_missing / total_cells) * 100
        logging.warning(
            f"Missing values detected in {file_path}: {total_missing} cells "
            f"({missing_percentage:.2f}% of total data)"
        )

        # Report per-column missing values
        columns_with_missing = missing_mask.sum()
        columns_with_missing = columns_with_missing[columns_with_missing > 0]
        if not columns_with_missing.empty:
            for col, count in columns_with_missing.items():
                col_percentage = (count / len(dataframe)) * 100
                logging.warning(
                    f"  Column '{col}': {count} missing values "
                    f"({col_percentage:.1f}% of column)"
                )

    if total_empty_strings > 0:
        empty_percentage = (total_empty_strings / total_cells) * 100
        logging.warning(
            f"Empty string values detected in {file_path}: {total_empty_strings} "
            f"cells ({empty_percentage:.2f}% of total data)"
        )

    if total_placeholders > 0:
        placeholder_percentage = (total_placeholders / total_cells) * 100
        logging.warning(
            f"Missing value placeholders detected in {file_path}: "
            f"{total_placeholders} cells ({placeholder_percentage:.2f}% of total data) "
            f"containing: {common_na_values}"
        )

    # Summary message if any missing data found
    total_problematic = total_missing + total_empty_strings + total_placeholders
    if total_problematic > 0:
        logging.warning(
            f"Total potentially missing/problematic values: {total_problematic} "
            f"({(total_problematic / total_cells) * 100:.2f}% of all data)"
        )
