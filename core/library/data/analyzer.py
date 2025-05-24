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
        self.unique_value_columns_dictionary = self._get_unique_value_columns()
        self.multivalued_columns_count_dictionary = (
            self._get_multivalued_columns_count()
        )

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

    def _get_unique_value_columns(self) -> dict:
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

    def _get_multivalued_columns_count(self) -> dict:
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

    def column_unique_values(self, column_name: str) -> list:
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


class DataFrameAnalyzer:
    """
    A class for analyzing and manipulating Pandas DataFrames with a focus on
    distinguishing between tunable parameters and output quantities.

    This class provides functionality to categorize DataFrame columns, identify
    columns with single vs. multiple unique values, group data by parameters,
    and filter the DataFrame based on various conditions.

    Attributes:
        - original_dataframe (pd.DataFrame): A copy of the original input
          DataFrame.
        - dataframe (pd.DataFrame): The working DataFrame that can be filtered
          and manipulated.
        - list_of_dataframe_column_names (list): List of all column names in the
          DataFrame.
        - list_of_tunable_parameter_names_from_dataframe (list): Columns
          identified as tunable parameters.
        - list_of_output_quantity_names_from_dataframe (list): Columns
          identified as output quantities.
        - unique_value_columns_dictionary (dict): Columns with single unique
          values and their values.
        - multivalued_columns_count_dictionary (dict): Columns with multiple
          unique values and counts.
        - list_of_single_valued_column_names (list): Column names with only one
          unique value.
        - list_of_multivalued_column_names (list): Column names with multiple
          unique values.
        - list_of_single_valued_tunable_parameter_names (list): Tunable
          parameters with one unique value.
        - list_of_multivalued_tunable_parameter_names (list): Tunable parameters
          with multiple values.
        - list_of_single_valued_output_quantity_names (list): Output quantities
          with one unique value.
        - list_of_multivalued_output_quantity_names (list): Output quantities
          with multiple values.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataFrameAnalyzer with a DataFrame.

        We divide parameters into two categories:
        1. Tunable parameters: These are parameters that can be adjusted by the
           user.
        2. Output quantities: These are the results of calculations.

        The primary purpose of this class is to distinguish between tunable
        parameters and output quantities, and to construct DataFrame groups
        based on tunable parameters or subsets of them, according to user needs.

        Args:
            dataframe (pd.DataFrame): The input DataFrame to analyze.

        Raises:
            TypeError: If the input is not a Pandas DataFrame.
        """
        # Verify that the input is a Pandas DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame")

        # Store a copy of the original DataFrame
        self.original_dataframe = dataframe.copy()

        # Initialize the dataframe attribute
        self.dataframe = self.original_dataframe.copy()

        # Initialize all lists of dataframe column name categories
        self._update_column_categories()

    def _update_column_categories(self):
        """
        Private method: Update and categorize column names from the dataframe
        into various lists and dictionaries.

        This internal method processes dataframe columns to identify:
        - Tunable parameters vs output quantities
        - Columns with single vs multiple unique values
        - Intersections of these categories

        The method updates multiple class attributes that categorize the columns
        in different ways for later use in analysis and filtering operations.

        Note: This method is called automatically during initialization and
        after operations that modify the dataframe structure. It should not be
        called directly by users.
        """
        self.list_of_dataframe_column_names = self.dataframe.columns.tolist()

        # Extract list of tunable parameter names
        self.list_of_tunable_parameter_names_from_dataframe = [
            column_name
            for column_name in self.list_of_dataframe_column_names
            if column_name in TUNABLE_PARAMETER_NAMES_LIST
        ]

        # Extract list of output quantity names
        self.list_of_output_quantity_names_from_dataframe = [
            column_name
            for column_name in self.list_of_dataframe_column_names
            # TODO: I have to decide which method is better
            # if column_name in constants.OUTPUT_QUANTITY_NAMES_LIST
            if column_name not in TUNABLE_PARAMETER_NAMES_LIST
        ]

        # Get dictionary of columns with single unique values and their values
        self.unique_value_columns_dictionary = self._get_unique_value_columns()

        # Get dictionary of multivalued columns with number of unique values
        self.multivalued_columns_count_dictionary = (
            self._get_multivalued_columns_count()
        )

        # Extract list of column names with single unique values
        self.list_of_single_valued_column_names = list(
            self.unique_value_columns_dictionary.keys()
        )

        # Extract list of column names with multiple unique values
        self.list_of_multivalued_column_names = list(
            self.multivalued_columns_count_dictionary.keys()
        )

        # Create list of tunable parameters with single unique values
        self.list_of_single_valued_tunable_parameter_names = list(
            set(self.list_of_tunable_parameter_names_from_dataframe)
            & set(self.list_of_single_valued_column_names)
        )

        # Create list of tunable parameters with multiple unique values
        self.list_of_multivalued_tunable_parameter_names = list(
            set(self.list_of_tunable_parameter_names_from_dataframe)
            & set(self.list_of_multivalued_column_names)
        )

        # Create list of output quantities with single unique values
        self.list_of_single_valued_output_quantity_names = list(
            set(self.list_of_output_quantity_names_from_dataframe)
            & set(self.list_of_single_valued_column_names)
        )

        # Create list of output quantities with multiple unique values
        self.list_of_multivalued_output_quantity_names = list(
            set(self.list_of_output_quantity_names_from_dataframe)
            & set(self.list_of_multivalued_column_names)
        )

    def _get_unique_value_columns(self) -> dict:
        """
        Private helper method: Identify columns that have only a single unique
        value.

        Used internally to populate class attributes related to single-valued
        columns.

        Returns:
            dict: A dictionary where keys are column names with only one unique
            value, and values are the actual unique values for those columns.
        """
        unique_value_columns_dictionary = {}
        for column in self.dataframe.columns:
            unique_values = self.dataframe[column].unique()

            # Only add columns where there is a single unique value
            if len(unique_values) == 1:
                # Store the unique value directly, not as a list
                unique_value_columns_dictionary[column] = unique_values[0]

        return unique_value_columns_dictionary

    def _get_multivalued_columns_count(self) -> dict:
        """
        Private helper method: Identify columns that have multiple unique values
        and count them.

        Used internally to populate class attributes related to multi-valued
        columns.

        Returns:
            dict: A dictionary where keys are column names with multiple unique
            values, and values are the counts of unique values in those columns.
        """
        multivalued_columns_count_dictionary = {}
        for column in self.dataframe.columns:
            # Get the number of unique values in the column
            unique_values_count = self.dataframe[column].nunique()
            if unique_values_count > 1:
                # If there's more than one unique value, add it to dictionary
                multivalued_columns_count_dictionary[column] = unique_values_count

        return multivalued_columns_count_dictionary

    def __enter__(self):
        """Enter context manager - store current dataframe state."""
        self._context_dataframe = self.dataframe.copy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - restore original dataframe."""
        self.dataframe = self._context_dataframe
        self._update_column_categories()

    def get_unique_values(self, column_name, print_output: bool = False) -> list:
        """
        Print the count and list of unique values for a specified column.

        This method displays the total number of unique values in the specified column,
        followed by a list of those unique values sorted in ascending order.

        Args:
            column_name (str): The name of the column to analyze.

        Returns:
            None: Prints information to the console.

        Raises:
            ValueError: If the specified column doesn't exist in the DataFrame.
        """
        # Check if the column exists
        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        # Get unique values and sort them
        unique_values = sorted(self.dataframe[column_name].unique())
        unique_count = len(unique_values)

        # Convert numpy types to native Python types for cleaner display
        unique_values_python = [
            item.item() if hasattr(item, "item") else item for item in unique_values
        ]

        if print_output:
            # Print the information
            print(f"Column '{column_name}' has {unique_count} unique values:")
            print(unique_values_python)

        return unique_values_python

    def group_by_multivalued_tunable_parameters(
        self,
        filter_out_parameters_list: list = None,
        verbose: bool = False,
    ):
        """
        Group the DataFrame by multivalued tunable parameters, optionally
        excluding some parameters.

        This method allows for flexible grouping of data based on tunable
        parameters that have multiple unique values. Parameters can be excluded
        from the grouping if they are not relevant to the current analysis.

        Args:
            filter_out_parameters_list (list, optional): List of parameter names
            to exclude from the grouping operation. Default is None (include all
            multivalued parameters).

        Returns:
            pandas.core.groupby.DataFrameGroupBy: A GroupBy object created by
            grouping on the specified multivalued tunable parameters.

        Raises:
            TypeError: If filter_out_parameters_list is not a list. ValueError:
            If filter_out_parameters_list contains invalid parameter names.

        Example:
            # Group by all multivalued tunable parameters groups =
            analyzer.group_by_multivalued_tunable_parameters()

            # Group by all except 'temperature' and 'pressure'

            groups = analyzer.group_by_multivalued_tunable_parameters(
                filter_out_parameters_list=['temperature', 'pressure']
            )
        """
        # TODO: I need to reintroduce this statement carefully: or
        # "len(filter_out_parameters_list) > 0"
        if filter_out_parameters_list is not None:
            # Check if filter_out_parameters_list is a list
            if not isinstance(filter_out_parameters_list, list):
                raise TypeError("'filter_out_parameters_list' must be a list!")

            # Check if all elements in filter_out_parameters_list are valid
            # tunable parameters
            invalid_parameters = [
                param
                for param in filter_out_parameters_list
                if param not in self.list_of_multivalued_tunable_parameter_names
            ]
            if invalid_parameters:
                raise ValueError(
                    "Invalid parameters in filter_out_parameters_list: "
                    f"{invalid_parameters}. Must be from: "
                    f"{self.list_of_multivalued_tunable_parameter_names}."
                )

            # Create a reduced list of multivalued tunable parameters by
            # excluding the specified ones from the filter_out_parameters_list
            self.reduced_multivalued_tunable_parameter_names_list = [
                item
                for item in self.list_of_multivalued_tunable_parameter_names
                if item not in filter_out_parameters_list
            ]

        else:
            self.reduced_multivalued_tunable_parameter_names_list = (
                self.list_of_multivalued_tunable_parameter_names.copy()
            )

        self.reduced_multivalued_tunable_parameter_names_list = sorted(
            self.reduced_multivalued_tunable_parameter_names_list
        )

        if verbose:
            print("List of reduced multivalued tunable parameter names:")
            print(self.reduced_multivalued_tunable_parameter_names_list)

        # Create a groupby object based on the reduced list of multivalued
        # tunable parameters. If reduced list is empty, return entire dataframe
        if self.reduced_multivalued_tunable_parameter_names_list:
            return self.dataframe.groupby(
                self.reduced_multivalued_tunable_parameter_names_list, observed=True
            )
        else:
            # Group by a dummy value to return the entire dataframe
            return self.dataframe.groupby(lambda _: "Dummy", observed=True)

    def restrict_dataframe(self, condition=None, filter_func=None):
        """
        Restricts the DataFrame to rows that satisfy given conditions.

        This method provides two ways to filter the DataFrame: 1. Using a string
        condition with Pandas query syntax 2. Using a custom filter function for
        more complex filtering logic

        Args:
            - condition (str, optional): A string condition to filter rows using
              Pandas query syntax e.g., "MSCG_epsilon >= 1e-06 and
              Kernel_operator_type == 'Wilson'"
            - filter_func (callable, optional): A function that takes a
              DataFrame and returns a boolean Series for more complex filtering
              needs

        Returns:
            None: Modifies the DataFrameAnalyzer's DataFrame in-place.

        Raises:
            ValueError: If neither condition nor filter_func is provided, or if
            filtering fails.

        Examples:
            # Example 1: Using a query string for simpler conditions
            analyzer.restrict_dataframe(
                "(MSCG_epsilon >= 1e-06 and Kernel_operator_type == 'Wilson') "
                "or (MSCG_epsilon >= 1e-05 and Kernel_operator_type ==
                'Brillouin')"
            )

            # Example 2: Using a filter function for more complex conditions def
            complex_filter(df):
                return (
                    ((df["MSCG_epsilon"] >= 1e-06) & (df["Kernel_operator_type"]
                    == "Wilson")) | ((df["MSCG_epsilon"] >= 1e-05) &
                    (df["Kernel_operator_type"] == "Brillouin"))
                )
            analyzer.restrict_dataframe(filter_func=complex_filter)
        """
        if condition is None and filter_func is None:
            raise ValueError("Either condition or filter_func must be provided")

        if condition is not None and not isinstance(condition, str):
            raise TypeError("condition must be a string")

        if filter_func is not None and not callable(filter_func):
            raise TypeError("filter_func must be callable")

        try:
            if condition is not None:
                # Use `query` to filter the DataFrame based on the condition
                # string
                self.dataframe = self.dataframe.query(condition)
            elif filter_func is not None:
                # Apply the custom filter function for more complex filtering
                mask = filter_func(self.dataframe)
                self.dataframe = self.dataframe[mask]
            else:
                raise ValueError("Either condition or filter_func must be provided")

            # Recompute the column categories after filtering
            self._update_column_categories()

        except Exception as e:
            raise ValueError(f"Failed to apply filter: {e}")

        return self  # Return self for method chaining

    def add_derived_column(
        self, new_column_name, derivation_function=None, expression=None
    ):
        """
        Add a new column to the DataFrame derived from existing columns using
        either a function or a string expression.

        This method automatically updates column categories after adding the new
        column.

        Args:
            - new_column_name (str): Name for the new column to be added.
            - derivation_function (callable, optional): A function that takes
              the dataframe as input and returns a Series with values for the
              new column.
            - expression (str, optional): A string expression using column names
              that will be evaluated to create the new column (e.g., "pressure *
              10").

        Returns:
            None: Modifies the DataFrameAnalyzer's DataFrame in-place.

        Raises:
            ValueError: If neither derivation_function nor expression is
            provided,
                or if the specified column name already exists.

        Examples:
            # Example 1: Using a derivation function
            analyzer.add_derived_column(
                "pressure_kPa", derivation_function=lambda df: df["pressure"] *
                10
            )

            # Example 2: Using a string expression analyzer.add_derived_column(
                "total_energy", expression="kinetic_energy + potential_energy"
            )
        """
        try:
            # Check if the column name already exists
            if new_column_name in self.list_of_dataframe_column_names:
                raise ValueError(
                    f"Column '{new_column_name}' already exists in the DataFrame."
                )

            # Check that at least one method is provided
            if derivation_function is None and expression is None:
                raise ValueError(
                    "Either derivation_function or expression must be provided."
                )

            try:
                if derivation_function is not None:
                    # Apply the provided function to derive the new column
                    self.dataframe[new_column_name] = derivation_function(
                        self.dataframe
                    )
                elif expression is not None:
                    # Evaluate the expression string using pandas eval
                    self.dataframe[new_column_name] = self.dataframe.eval(expression)

                # Update column categories to include the new column
                self._update_column_categories()

            except Exception as e:
                # If anything goes wrong, clean up and raise the exception
                if new_column_name in self.dataframe.columns:
                    self.dataframe.drop(columns=[new_column_name], inplace=True)
                raise ValueError(f"Failed to create derived column: {e}")

        except Exception as e:
            # Clean up and provide more context in the error
            if new_column_name in self.dataframe.columns:
                self.dataframe.drop(columns=[new_column_name], inplace=True)

            error_type = type(e).__name__
            raise ValueError(f"Failed to create derived column: {error_type}: {e}")

    def restore_original_dataframe(self):
        """
        Reset the working DataFrame to the original, unfiltered state.

        This method replaces the current working DataFrame with a fresh copy of
        the original DataFrame that was provided when the object was created. It
        also updates all the column categories to reflect the original data.

        This is useful when you want to start a new analysis from scratch after
        applying various filters or transformations.

        Returns:
            None: Modifies the DataFrameAnalyzer's DataFrame in-place.
        """
        self.dataframe = self.original_dataframe.copy()
        self._update_column_categories()
