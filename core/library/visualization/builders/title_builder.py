from typing import Optional

from library.constants.data_types import PARAMETERS_WITH_EXPONENTIAL_FORMAT


class PlotTitleBuilder:
    """
    Constructs plot titles from metadata following specific formatting
    rules.

    This class handles the construction of informative plot titles by
    combining metadata from experimental parameters, with special
    handling for scientific notation, parameter exclusion, and title
    wrapping.

    The title building process:
    1. Optional leading substring
    2. Special overlap/kernel parameters with hierarchical formatting
    3. Remaining tunable parameters with proper labeling
    4. Automatic wrapping for long titles

    Examples:
        Simple title: "Temperature=300.00, Pressure=1.50"

        Complex title: "Chebyshev Wilson 50, Temperature 300.00,
        Pressure 1.50"

        Wrapped  title: "Chebyshev Wilson 50, Temperature
        300.00,\nPressure 1.50"
    """

    def __init__(
        self,
        title_labels: dict,
        title_number_format: str = ".2f",
        title_exponential_format: str = ".0e",
    ):
        """
        Initialize the title builder with label mappings and formatting
        options.

        Parameters:
        -----------
        title_labels : dict
            Maps parameter names to human-readable labels for titles.
            E.g., {"temperature": "Temperature (K)", "pressure":
            "Pressure (Pa)"}
        title_number_format : str, optional
            Format string for regular numeric values. Default is ".2f"
            for 2 decimal places. Examples: ".2f", ".3g", ".1e"
        title_exponential_format : str, optional
            Format string for parameters requiring exponential notation.
            Default is ".0e" for clean exponential display. Examples:
            ".0e", ".1e", ".2e"
        """
        self.title_labels = title_labels
        self.title_number_format = title_number_format
        self.title_exponential_format = title_exponential_format

    def build(
        self,
        metadata_dict: dict,
        tunable_params: list,
        excluded: Optional[set] = None,
        leading_substring: Optional[str] = None,
        title_from_columns: Optional[list] = None,
        wrapping_length: int = 80,
    ) -> str:
        """
        Build a plot title from metadata following the structured
        format.

        Parameters:
        -----------
        metadata_dict : dict
            Dictionary containing parameter values for this plot group.
            May include special keys like "Overlap_operator_method" and
            "Kernel_operator_type" which get special positioning.
        tunable_params : list
            List of all tunable parameter names that could appear in
            titles.
        excluded : set, optional
            Set of parameter names to exclude from the title (e.g.,
            parameters used for grouping or labeling that shouldn't
            appear in titles).
        leading_substring : str, optional
            Optional leading string to prepend to the title.
        title_from_columns : list, optional
            If provided, creates a simple title using only these
            columns, bypassing the complex title building logic.
        wrapping_length : int, optional
            Maximum length before wrapping. Set to 0 to disable
            wrapping. Default is 80 characters.

        Returns:
        --------
        str
            A formatted title string, potentially with newlines for
            wrapping.

        Example:
            >>> builder = PlotTitleBuilder({"temp": "Temperature (K)"})
            >>> metadata = {"Overlap_operator_method": "Chebyshev", "temp": 300}
            >>> builder.build(metadata, ["temp"])
            'Chebyshev, Temperature (K)=300.00'
        """
        # Shortcut for simple title from selected columns
        if title_from_columns:
            return self._build_simple_title(metadata_dict, title_from_columns)

        title_parts = []
        excluded = excluded or set()

        # Add leading substring
        if leading_substring:
            title_parts.append(leading_substring)

        # Handle special overlap/kernel logic
        title_parts.extend(self._overlap_kernel_parts(metadata_dict, excluded))

        # Add remaining tunable parameters
        title_parts.extend(
            self._parameter_parts(metadata_dict, tunable_params, excluded)
        )

        # Finalize title
        full_title = " ".join(title_parts).strip().rstrip(",")

        # Wrap if needed
        if wrapping_length and len(full_title) > wrapping_length:
            full_title = self._wrap_title(full_title, wrapping_length)

        return full_title

    def _build_simple_title(self, metadata_dict: dict, columns: list) -> str:
        """
        Build a simple title from selected columns only.

        This bypasses the complex title building logic and creates a
        straightforward title using only the specified columns.

        Parameters:
        -----------
        metadata_dict : dict
            Dictionary containing parameter values
        columns : list
            List of column names to include in the title

        Returns:
        --------
        str
            Simple comma-separated title
        """
        parts = []
        for col in columns:
            value = metadata_dict.get(col)
            if value is None:
                continue
            label = self.title_labels.get(col, col)
            formatted_value = self._format_value(col, value)

            if "Kernel_operator_type" in col:
                parts.append(f"{formatted_value} Kernel")
            else:
                parts.append(f"{label}={formatted_value}")
        return ", ".join(parts)

    def _overlap_kernel_parts(self, metadata: dict, excluded: set) -> list:
        """
        Extract overlap/kernel related title parts with special formatting.

        Handles the hierarchical structure:
        - Overlap method (Chebyshev/KL/Bare)
        - Kernel type (Wilson/Brillouin)
        - Method-specific parameters (Chebyshev terms, KL order)

        Parameters:
        -----------
        metadata : dict
            Dictionary containing parameter values
        excluded : set
            Set of parameters to exclude from title

        Returns:
        --------
        list
            List of formatted title parts for overlap/kernel parameters
        """
        parts = []
        overlap_method = metadata.get("Overlap_operator_method")
        kernel_type = metadata.get("Kernel_operator_type")

        # Build the title part independently for each parameter
        components = []

        # Add overlap method if present and not excluded
        if overlap_method and "Overlap_operator_method" not in excluded:
            components.append(str(overlap_method))

        # Add kernel type if present and not excluded (INDEPENDENT CHECK)
        if kernel_type and "Kernel_operator_type" not in excluded:
            components.append(str(kernel_type))

        # Add method-specific parameters only if overlap_method exists and is not excluded
        if overlap_method and "Overlap_operator_method" not in excluded:
            if overlap_method == "Chebyshev":
                terms = metadata.get("Number_of_Chebyshev_terms")
                if terms is not None and "Number_of_Chebyshev_terms" not in excluded:
                    components.append("N=" + str(terms))
            elif overlap_method in ["KL", "Neuberger", "Zolotarev"]:
                order = metadata.get("Rational_order")
                if order is not None and "Rational_order" not in excluded:
                    components.append("n=" + str(order))

        # Only add to parts if we have something to show
        if components:
            parts.append(" ".join(components) + ",")

        return parts

    def _parameter_parts(
        self, metadata: dict, tunable_params: list, excluded: set
    ) -> list:
        """
        Get title parts for remaining tunable parameters.

        Processes all tunable parameters except those already handled
        by the overlap/kernel logic or explicitly excluded.

        Parameters:
        -----------
        metadata : dict
            Dictionary containing parameter values
        tunable_params : list
            List of all tunable parameter names
        excluded : set
            Set of parameters to exclude

        Returns:
        --------
        list
            List of formatted parameter strings
        """
        parts = []

        # Add these special parameters to excluded set to avoid duplication
        special_params = {
            "Overlap_operator_method",
            "Kernel_operator_type",
            "Number_of_Chebyshev_terms",
            "KL_diagonal_order",
            "Rational_order",
        }

        for param_name in tunable_params:
            # Skip if in excluded set OR if it's a special parameter already handled
            if (
                param_name in excluded
                or param_name not in metadata
                or param_name in special_params
            ):
                continue

            value = metadata[param_name]
            label = self.title_labels.get(param_name, param_name)
            formatted_value = self._format_value(param_name, value)
            parts.append(f"{label}={formatted_value},")

        return parts

    def _format_value(self, param_name: str, value) -> str:
        """
        Format a value for display in title.

        Applies the configured number formatting to numeric values and
        converts other types to strings. Uses exponential formatting for
        parameters specified in PARAMETERS_WITH_EXPONENTIAL_FORMAT.
        Parameters defined as integers are formatted without decimal places.

        Parameters:
        -----------
        param_name : str
            Name of the parameter (used to determine formatting type)
        value : any
            The value to format

        Returns:
        --------
        str
            Formatted string representation
        """
        # Import here to avoid circular dependency
        from library.constants.data_types import PARAMETERS_OF_INTEGER_VALUE
        
        if isinstance(value, (int, float)):
            # Check if parameter is defined as integer type
            if param_name in PARAMETERS_OF_INTEGER_VALUE or isinstance(value, int):
                return str(int(value))
            
            # Float formatting
            if param_name in PARAMETERS_WITH_EXPONENTIAL_FORMAT:
                return format(value, self.title_exponential_format)
            return format(value, self.title_number_format)
        
        return str(value)

    def _wrap_title(self, title: str, max_length: int) -> str:
        """
        Wrap title at a comma nearest to max_length, or near the middle
        if not possible.

        Attempts to find a natural break point (comma) near the desired
        length. If no suitable comma is found, falls back to wrapping
        near the middle.

        Parameters:
        -----------
        title : str
            The title string to wrap
        max_length : int
            Target maximum length before wrapping

        Returns:
        --------
        str
            Title with newline inserted at the best wrap point
        """
        comma_positions = [pos for pos, char in enumerate(title) if char == ","]
        if not comma_positions:
            return title

        # Find the comma closest to max_length
        split_pos = min(comma_positions, key=lambda x: abs(x - max_length))

        # If max_length is too far from any comma, fall back to the middle
        if abs(split_pos - max_length) > max_length // 4:
            split_pos = min(comma_positions, key=lambda x: abs(x - len(title) // 2))

        return title[: split_pos + 1] + "\n" + title[split_pos + 1 :]

    def set_number_format(self, format_string: str) -> None:
        """
        Update the number formatting for future title builds.

        Parameters:
        -----------
        format_string : str
            New format string for regular numbers (e.g., ".3f", ".2e",
            ".4g")
        """
        self.title_number_format = format_string

    def set_exponential_format(self, format_string: str) -> None:
        """
        Update the exponential formatting for future title builds.

        Parameters:
        -----------
        format_string : str
            New format string for exponential numbers (e.g., ".0e",
            ".1e", ".2e")
        """
        self.title_exponential_format = format_string

    def update_labels(self, new_labels: dict) -> None:
        """
        Update or add new label mappings.

        Parameters:
        -----------
        new_labels : dict
            Dictionary of parameter_name -> label mappings to add/update
        """
        self.title_labels.update(new_labels)
