"""
DataFrame analysis and manipulation tools for distinguishing between tunable
parameters and output quantities.

This module provides classes for analyzing and manipulating Pandas DataFrames in
the context of scientific data analysis, with special focus on categorizing
columns as either tunable parameters (inputs) or output quantities (results).

Classes:
    DataFrameAnalyzer: Main public class for DataFrame analysis and
    manipulation.
        Provides methods for filtering, grouping, adding derived columns, and
        managing DataFrame state through context managers.

    _DataFrameInspector: Private base class for read-only DataFrame inspection.
        Not intended for direct use.

Example:
    >>> import pandas as pd
    >>> from library.data.analyzer import DataFrameAnalyzer
    >>> 
    >>> df = pd.DataFrame({
    ...     'temperature': [300, 310, 320],
    ...     'pressure': [1.0, 1.5, 2.0],
    ...     'yield': [0.85, 0.90, 0.92]
    ... })
    >>> 
    >>> analyzer = DataFrameAnalyzer(df)
    >>> print(analyzer.list_of_multivalued_tunable_parameter_names)
    ['temperature', 'pressure']
    >>> 
    >>> # Use context manager for temporary modifications
    >>> with analyzer:
    ...     analyzer.restrict_dataframe("temperature > 305")
    ...     print(len(analyzer.dataframe))  # 2 rows
    >>> print(len(analyzer.dataframe))  # 3 rows (restored)

Dependencies:
    - pandas: For DataFrame operations
    - library.constants: For TUNABLE_PARAMETER_NAMES_LIST
"""

from typing import Optional

import pandas as pd

from ..constants import TUNABLE_PARAMETER_NAMES_LIST


class _DataFrameInspector:
    """
    Private base class for read-only DataFrame inspection and analysis.

    This class provides methods to analyze DataFrame structure and content
    without modifying the data. It categorizes columns into tunable parameters
    and output quantities, and identifies single-valued vs multi-valued columns.

    This class should not be instantiated directly. Use DataFrameAnalyzer
    instead.

    Attributes:
        - dataframe (pd.DataFrame): The DataFrame to inspect (not copied).
        - list_of_dataframe_column_names (list): All column names.
        - list_of_tunable_parameter_names_from_dataframe (list): Columns
          identified as tunable parameters.
        - list_of_output_quantity_names_from_dataframe (list): Columns
          identified as output quantities.
        - unique_value_columns_dictionary (dict): Single-valued columns and
          their values.
        - multivalued_columns_count_dictionary (dict): Multi-valued columns and
          their counts.
        - list_of_single_valued_column_names (list): Names of single-valued
          columns.
        - list_of_multivalued_column_names (list): Names of multi-valued
          columns.
        - list_of_single_valued_tunable_parameter_names (list): Single-valued
          tunable parameters.
        - list_of_multivalued_tunable_parameter_names (list): Multi-valued
          tunable parameters.
        - list_of_single_valued_output_quantity_names (list): Single-valued
          output quantities.
        - list_of_multivalued_output_quantity_names (list): Multi-valued output
          quantities.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the inspector with a DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to inspect.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Store reference to the dataframe (no copy)
        self.dataframe = dataframe

        # Initialize column categories
        self._update_column_categories()

    def _update_column_categories(self):
        """
        Update and categorize column names into various lists and dictionaries.

        This internal method identifies:
        - Tunable parameters vs output quantities
        - Columns with single vs multiple unique values
        - Intersections of these categories
        """
        self.list_of_dataframe_column_names = self.dataframe.columns.tolist()

        # Extract tunable parameter names
        self.list_of_tunable_parameter_names_from_dataframe = [
            column_name
            for column_name in self.list_of_dataframe_column_names
            if column_name in TUNABLE_PARAMETER_NAMES_LIST
        ]

        # Extract output quantity names
        self.list_of_output_quantity_names_from_dataframe = [
            column_name
            for column_name in self.list_of_dataframe_column_names
            if column_name not in TUNABLE_PARAMETER_NAMES_LIST
        ]

        # Get single and multi-valued columns
        self.unique_value_columns_dictionary = self._unique_value_columns()
        self.multivalued_columns_count_dictionary = self._multivalued_columns_count()

        # Extract column name lists
        self.list_of_single_valued_column_names = list(
            self.unique_value_columns_dictionary.keys()
        )
        self.list_of_multivalued_column_names = list(
            self.multivalued_columns_count_dictionary.keys()
        )

        # Create intersection lists
        self.list_of_single_valued_tunable_parameter_names = list(
            set(self.list_of_tunable_parameter_names_from_dataframe)
            & set(self.list_of_single_valued_column_names)
        )

        self.list_of_multivalued_tunable_parameter_names = list(
            set(self.list_of_tunable_parameter_names_from_dataframe)
            & set(self.list_of_multivalued_column_names)
        )

        self.list_of_single_valued_output_quantity_names = list(
            set(self.list_of_output_quantity_names_from_dataframe)
            & set(self.list_of_single_valued_column_names)
        )

        self.list_of_multivalued_output_quantity_names = list(
            set(self.list_of_output_quantity_names_from_dataframe)
            & set(self.list_of_multivalued_column_names)
        )

    def _unique_value_columns(self) -> dict:
        """
        Identify columns that have only a single unique value.

        Returns:
            dict: Column names as keys, their unique values as values.
        """
        unique_value_columns_dictionary = {}
        for column in self.dataframe.columns:
            unique_values = self.dataframe[column].unique()
            if len(unique_values) == 1:
                unique_value_columns_dictionary[column] = unique_values[0]

        return unique_value_columns_dictionary

    def _multivalued_columns_count(self) -> dict:
        """
        Identify columns with multiple unique values and count them.

        Returns:
            dict: Column names as keys, counts of unique values as values.
        """
        multivalued_columns_count_dictionary = {}
        for column in self.dataframe.columns:
            unique_values_count = self.dataframe[column].nunique()
            if unique_values_count > 1:
                multivalued_columns_count_dictionary[column] = unique_values_count

        return multivalued_columns_count_dictionary

    def unique_values(self, column_name: str) -> list:
        """
        Return sorted list of unique values for the specified column.

        Args:
            column_name (str): The name of the column to analyze.

        Returns:
            list: Sorted list of unique values in the column.

        Raises:
            ValueError: If the column doesn't exist in the DataFrame.
        """
        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        # Get unique values and sort them
        unique_values = sorted(self.dataframe[column_name].unique())

        # Convert numpy types to native Python types for cleaner usage
        unique_values_python = [
            item.item() if hasattr(item, "item") else item for item in unique_values
        ]

        return unique_values_python


class DataFrameAnalyzer(_DataFrameInspector):
    """
    A class for analyzing and manipulating Pandas DataFrames with a focus on
    distinguishing between tunable parameters and output quantities.

    This class extends _DataFrameInspector to add data manipulation capabilities
    including filtering, column addition, and state management. It maintains
    both an original and working copy of the DataFrame.

    The class supports two ways to manage state:
    - Context manager: For temporary modifications that auto-revert
    - restore_original_dataframe(): For manual reset when not using context
      manager

    Attributes:
        - original_dataframe (pd.DataFrame): Immutable copy of the original
          input DataFrame.
        - dataframe (pd.DataFrame): Working DataFrame that can be filtered and
          manipulated.
        - All attributes from _DataFrameInspector
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataFrameAnalyzer with a DataFrame.

        Creates both an original (immutable) copy and a working copy of the
        input DataFrame.

        Args:
            - dataframe (pd.DataFrame): The input DataFrame to analyze.

        Raises:
            - TypeError: If the input is not a Pandas DataFrame.
        """
        # Store a copy of the original DataFrame (immutable reference)
        self.original_dataframe = dataframe.copy()

        # Initialize the parent class with a working copy
        super().__init__(dataframe.copy())

        # For group_by method - will be computed on demand
        self._filter_out_parameters_list = None

    @property
    def reduced_multivalued_tunable_parameter_names_list(self) -> list:
        """
        Get the list of multivalued tunable parameters after filtering.

        This property is computed based on the last call to
        group_by_multivalued_tunable_parameters(). If the method hasn't been
        called yet, returns all multivalued tunable parameters.

        Returns:
            - list: Sorted list of multivalued tunable parameter names after
              filtering.
        """
        if self._filter_out_parameters_list is None:
            return sorted(self.list_of_multivalued_tunable_parameter_names.copy())

        return sorted(
            [
                item
                for item in self.list_of_multivalued_tunable_parameter_names
                if item not in self._filter_out_parameters_list
            ]
        )

    def __enter__(self):
        """
        Enter context manager - store current DataFrame state on a stack.

        This method implements the context manager protocol, allowing temporary
        modifications to the DataFrame that are automatically reverted when the
        context ends. It maintains a stack of DataFrame states to support nested
        context managers.

        The state is preserved by creating a deep copy of the current DataFrame
        and pushing it onto a stack. When exiting the context, the most recent
        state is restored by popping from the stack.

        Usage:
            # Single context
            with analyzer:
                analyzer.restrict_dataframe("column > 5")
                # DataFrame is modified here
            # DataFrame is automatically restored here

            # Nested contexts
            with analyzer:
                analyzer.restrict_dataframe("column > 5")
                with analyzer:
                    analyzer.restrict_dataframe("other_column == 'value'")
                    # Inner context modifications
                # Restored to outer context state
            # Restored to original state

        Returns:
            DataFrameAnalyzer: Returns self for use in the context manager.
        """
        if not hasattr(self, "_context_stack"):
            self._context_stack = []
        self._context_stack.append(self.dataframe.copy())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager - restore DataFrame state from the stack.

        This method implements the context manager protocol's exit handling. It:
        1. Restores the DataFrame to its state before the context was entered
        2. Updates all column categories to reflect the restored state
        3. Maintains proper exception handling

        The restoration process pops the most recent DataFrame state from the
        stack and restores it, ensuring nested contexts work correctly. Column
        categories are then recomputed to maintain consistency.

        Args:
            - exc_type: The type of any exception that occurred (or None)
            - exc_val: The instance of any exception that occurred (or None)
            - exc_tb: The traceback of any exception that occurred (or None)

        Returns:
            bool: Always returns False, allowing exceptions to propagate
                 normally rather than being suppressed.

        Note:
            The method handles the case where no context stack exists yet,
            making it safe to use even if __enter__ was never called.
        """
        if hasattr(self, "_context_stack") and self._context_stack:
            self.dataframe = self._context_stack.pop()
        self._update_column_categories()
        # Don't suppress exceptions
        return False

    def group_by_multivalued_tunable_parameters(
        self,
        filter_out_parameters_list: Optional[list] = None,
        verbose: bool = False,
    ):
        """
        Group the DataFrame by multivalued tunable parameters, optionally
        excluding some parameters.

        This method allows for flexible grouping of data based on tunable
        parameters that have multiple unique values. Parameters can be excluded
        from the grouping if they are not relevant to the current analysis.

        Side Effects:
            - Updates the internal state used by the
              reduced_multivalued_tunable_parameter_names_list property

        Args:
            - filter_out_parameters_list (list, optional): List of parameter
              names to exclude from the grouping operation. Default is None
              (include all multivalued parameters).
            - verbose (bool, optional): If True, prints the reduced parameter
              list. Default is False.

        Returns:
            - pandas.core.groupby.DataFrameGroupBy: A GroupBy object created by
              grouping on the specified multivalued tunable parameters.

        Raises:
            - TypeError: If filter_out_parameters_list is not a list.
            - ValueError: If filter_out_parameters_list contains invalid
              parameter names.

        Example:
            # Group by all multivalued tunable parameters groups =
            analyzer.group_by_multivalued_tunable_parameters()

            # Group by all except 'temperature' and 'pressure' groups =
            analyzer.group_by_multivalued_tunable_parameters(
                filter_out_parameters_list=['temperature', 'pressure']
            )
        """
        if filter_out_parameters_list is not None:
            if not isinstance(filter_out_parameters_list, list):
                raise TypeError("'filter_out_parameters_list' must be a list!")

            invalid_parameters = [
                param
                for param in filter_out_parameters_list
                if param not in self.list_of_multivalued_tunable_parameter_names
            ]
            if invalid_parameters:
                raise ValueError(
                    f"Invalid parameters in filter_out_parameters_list: "
                    f"{invalid_parameters}. Must be from: "
                    f"{self.list_of_multivalued_tunable_parameter_names}."
                )

        # Store for property computation
        self._filter_out_parameters_list = filter_out_parameters_list

        # Get the reduced list via property
        grouping_columns = self.reduced_multivalued_tunable_parameter_names_list

        if verbose:
            print("List of reduced multivalued tunable parameter names:")
            print(grouping_columns)

        # Create groupby object
        if grouping_columns:
            return self.dataframe.groupby(grouping_columns, observed=True)
        else:
            # Group by a dummy value to return the entire dataframe
            return self.dataframe.groupby(lambda _: "Dummy", observed=True)

    def restrict_data(self, condition=None, filter_func=None):
        """
        Restricts the DataFrame to rows that satisfy given conditions.

        This method provides two ways to filter the DataFrame:
        1. Using a string condition with Pandas query syntax
        2. Using a custom filter function for more complex filtering logic

        Side Effects:
            - Modifies the working dataframe in-place
            - Updates all column categorization attributes

        Args:
            - condition (str, optional): A string condition to filter rows using
              Pandas query syntax. E.g., "MSCG_epsilon >= 1e-06 and
              Kernel_operator_type == 'Wilson'"
            - filter_func (callable, optional): A function that takes a
              DataFrame and returns a boolean Series for more complex filtering.

        Returns:
            - self: Returns the analyzer instance for method chaining.

        Raises:
            - ValueError: If neither condition nor filter_func is provided.
            - TypeError: If condition is not a string or filter_func is not
              callable.

        Examples:
            # Using a query string analyzer.restrict_dataframe("parameter1 > 1
            and parameter2 == 'B'")

            # Using a filter function def complex_filter(df):
                return ((df["param1"] >= 2) & (df["param2"] == "B"))
            analyzer.restrict_dataframe(filter_func=complex_filter)

            # Method chaining analyzer.restrict_dataframe("param >
            5").add_derived_column(...)
        """
        if condition is None and filter_func is None:
            raise ValueError("Either condition or filter_func must be provided")

        if condition is not None and not isinstance(condition, str):
            raise TypeError("condition must be a string")

        if filter_func is not None and not callable(filter_func):
            raise TypeError("filter_func must be callable")

        try:
            if condition is not None:
                self.dataframe = self.dataframe.query(condition)
            elif filter_func is not None:
                mask = filter_func(self.dataframe)
                if not isinstance(mask, pd.Series) or mask.dtype != bool:
                    raise TypeError("filter_func must return a boolean Series")
                self.dataframe = self.dataframe[mask]

            # Update column categories after filtering
            self._update_column_categories()

        except Exception as e:
            raise ValueError(f"Failed to apply filter: {e}")

        return self  # For method chaining

    def add_derived_column(
        self, new_column_name, derivation_function=None, expression=None
    ):
        """
        Add a new column to the DataFrame derived from existing columns.

        This method provides two ways to create derived columns: 1. Using a
        derivation function that operates on the dataframe 2. Using a string
        expression that will be evaluated

        Side Effects:
            - Adds a new column to the working dataframe
            - Updates all column categorization attributes
            - If the operation fails, the dataframe is restored to its previous
              state

        Args:
            - new_column_name (str): Name for the new column to be added.
            - derivation_function (callable, optional): A function that takes
              the dataframe as input and returns a Series with values for the
              new column.
            - expression (str, optional): A string expression using column names
              that will be evaluated to create the new column.

        Returns:
            - self: Returns the analyzer instance for method chaining.

        Raises:
            - ValueError: If neither derivation_function nor expression is
              provided, or if the column name already exists, or if the
              operation fails.
            - TypeError: If the types are incorrect.

        Examples:
            # Using a derivation function analyzer.add_derived_column(
                "pressure_kPa", derivation_function=lambda df: df["pressure"] *
                10
            )

            # Using a string expression analyzer.add_derived_column(
                "total_energy", expression="kinetic_energy + potential_energy"
            )
        """
        if new_column_name in self.list_of_dataframe_column_names:
            raise ValueError(
                f"Column '{new_column_name}' already exists in the DataFrame."
            )

        if derivation_function is None and expression is None:
            raise ValueError(
                "Either derivation_function or expression must be provided."
            )

        try:
            if derivation_function is not None:
                self.dataframe[new_column_name] = derivation_function(self.dataframe)
            elif expression is not None:
                self.dataframe[new_column_name] = self.dataframe.eval(expression)

            # Update column categories to include the new column
            self._update_column_categories()

        except Exception as e:
            # Clean up if anything goes wrong
            if new_column_name in self.dataframe.columns:
                self.dataframe.drop(columns=[new_column_name], inplace=True)

            error_type = type(e).__name__
            raise ValueError(f"Failed to create derived column: {error_type}: {e}")

        return self  # For method chaining

    def restore_original_data(self):
        """
        Reset the working DataFrame to the original, unfiltered state.

        This method replaces the current working DataFrame with a fresh copy of
        the original DataFrame that was provided when the object was created. It
        also updates all the column categories to reflect the original data.

        This is useful when you want to start a new analysis from scratch after
        applying various filters or transformations, especially in interactive
        environments like Jupyter notebooks.

        Side Effects:
            - Replaces the working dataframe with a copy of the original
            - Updates all column categorization attributes
            - Resets any internal state from group_by operations

        Returns:
            - self: Returns the analyzer instance for method chaining.

        Example:
            analyzer.restrict_dataframe("param > 5")
            analyzer.add_derived_column("new_col", expression="col1 * 2")
            # ... do some analysis ... analyzer.restore_original_dataframe()  #
            Back to original state
        """
        self.dataframe = self.original_dataframe.copy()
        self._update_column_categories()
        self._filter_out_parameters_list = None  # Reset grouping state

        return self  # For method chaining

    # Backward compatibility aliases
    def restrict_dataframe(self, condition=None, filter_func=None):
        """Deprecated: Use restrict_data() instead."""
        import warnings
        warnings.warn("restrict_dataframe() is deprecated. Use restrict_data() instead.", 
                    DeprecationWarning, stacklevel=2)
        return self.restrict_data(condition=condition, filter_func=filter_func)

    def restore_original_dataframe(self):
        """Deprecated: Use restore_original_data() instead."""
        import warnings
        warnings.warn("restore_original_dataframe() is deprecated. Use restore_original_data() instead.", 
                    DeprecationWarning, stacklevel=2)
        return self.restore_original_data()