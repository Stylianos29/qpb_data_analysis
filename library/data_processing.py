import pandas as pd


def get_fields_with_multiple_values(df, excluded_fields=None) -> list:
    """
    Retrieves a list of column names from the given Pandas DataFrame that
    contain more than one unique value, excluding any columns specified in the
    'excluded_fields' set.

    Parameters:
    
    - df (pd.DataFrame): The Pandas DataFrame to analyze.
    - excluded_fields (set, optional): A set of column names to exclude from the
    results. If not provided or is empty, no columns will be excluded. Default
    is None.

    Returns:
    
    - List of column names that have more than one unique value,
    excluding any specified in 'excluded_fields'.
    
    Usage:
    
    - get_fields_with_multiple_values(df, {"Filename", "Plaquette"}) -
    get_fields_with_multiple_values(df)
    """

    if excluded_fields is None:
        excluded_fields = set()  # Default to an empty set if not provided

    # Filter columns with more than one unique value and exclude specified
    # fields
    varying_fields = [col for col in df.columns 
                      if df[col].nunique() > 1 and col not in excluded_fields]

    return varying_fields


def get_fields_with_unique_values(df: pd.DataFrame) -> dict:
    """
    This function returns a dictionary where the keys are the column names 
    from the DataFrame, and the values are the single unique value in the column 
    for those columns that contain only a single unique value.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe whose columns are to be analyzed.

    Returns:
    --------
    dict
        A dictionary with column names as keys and the single unique value 
        as values for those columns that have only one unique value.
    
    Example:
    --------
    df = pd.DataFrame({
        'col1': [1, 1, 1, 1],
        'col2': [1, 1, 1, 1],
        'col3': [5, 6, 7, 8]
    })
    result = get_fields_with_unique_values(df)
    print(result)
    # Output: {'col1': 1, 'col2': 1}
    """
    single_unique_value_columns = {}

    # Iterate through each column
    for column in df.columns:
        unique_values = df[column].unique()

        # Only add columns where there is a single unique value
        if len(unique_values) == 1:
            # Store the unique value directly, not as a list
            single_unique_value_columns[column] = unique_values[0]

    return single_unique_value_columns
