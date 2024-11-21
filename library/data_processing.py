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
