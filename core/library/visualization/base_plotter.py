import os
import shutil


class _PlotFileManager:
    """Handles all file system operations for plots."""

    def __init__(self, base_directory: str):
        if not os.path.isdir(base_directory):
            raise ValueError(f"Invalid plots directory: '{base_directory}'")
        self.base_directory = base_directory

    def prepare_subdirectory(
        self, subdir_name: str, clear_existing: bool = False
    ) -> str:
        """Create or clean a subdirectory for storing plots."""
        full_path = os.path.join(self.base_directory, subdir_name)
        os.makedirs(full_path, exist_ok=True)

        if clear_existing:
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        return full_path

    def plot_path(self, directory: str, filename: str) -> str:
        """Construct full path for a plot file."""
        return os.path.join(directory, f"{filename}.png")


class _PlotTitleBuilder:
    """Constructs plot titles from metadata."""

    def __init__(self, title_labels: dict, title_number_format: str = ".2f"):
        self.title_labels = title_labels
        self.title_number_format = title_number_format

    def build(
        self,
        metadata_dict: dict,
        tunable_params: list,
        excluded: set = None,
        leading_substring: str = None,
        title_from_columns: list = None,
        wrapping_length: int = 90,
    ) -> str:
        """Build a plot title from metadata."""

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
        """Build a simple title from selected columns."""
        parts = []
        for col in columns:
            value = metadata_dict.get(col)
            if value is None:
                continue
            label = self.title_labels.get(col, col)
            formatted_value = self._format_value(value)

            if "Kernel_operator_type" in col:
                parts.append(f"{formatted_value} Kernel")
            else:
                parts.append(f"{label}={formatted_value}")
        return ", ".join(parts)

    def _overlap_kernel_parts(self, metadata: dict, excluded: set) -> list:
        """Extract overlap/kernel related title parts."""
        parts = []
        overlap_method = metadata.get("Overlap_operator_method")
        kernel_type = metadata.get("Kernel_operator_type")

        if overlap_method and "Overlap_operator_method" not in excluded:
            temp = [str(overlap_method)]

            if kernel_type and "Kernel_operator_type" not in excluded:
                temp.append(str(kernel_type))

            if overlap_method == "Chebyshev":
                terms = metadata.get("Number_of_Chebyshev_terms")
                if terms is not None and "Number_of_Chebyshev_terms" not in excluded:
                    temp.append(str(terms))
            elif overlap_method == "KL":
                order = metadata.get("KL_diagonal_order")
                if order is not None and "KL_diagonal_order" not in excluded:
                    temp.append(str(order))

            parts.append(" ".join(temp) + ",")

        return parts

    def _parameter_parts(
        self, metadata: dict, tunable_params: list, excluded: set
    ) -> list:
        """Get title parts for tunable parameters."""
        parts = []

        # Add these special parameters to excluded set to avoid duplication
        special_params = {
            "Overlap_operator_method",
            "Kernel_operator_type",
            "Number_of_Chebyshev_terms",
            "KL_diagonal_order",
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
            formatted_value = self._format_value(value)
            parts.append(f"{label} {formatted_value},")

        return parts

    def _format_value(self, value) -> str:
        """Format a value for display in title."""
        if isinstance(value, (int, float)):
            return format(value, self.title_number_format)
        return str(value)

    def _wrap_title(self, title: str, max_length: int) -> str:
        """Wrap title at a comma near the middle."""
        comma_positions = [pos for pos, char in enumerate(title) if char == ","]
        if comma_positions:
            split_pos = min(comma_positions, key=lambda x: abs(x - len(title) // 2))
            return title[: split_pos + 1] + "\n" + title[split_pos + 1 :]
        return title


class _PlotFilenameBuilder:
    """
    Constructs plot filenames from metadata following a specific format pattern.
    
    The filename format is:
    [prefix][overlap]_[base_name]_[kernel]_[param1][value1]_[param2][value2]...[suffix]
    
    Where:
        - prefix: Optional "Combined_" or custom prefix
        - overlap: Overlap operator method (KL/Chebyshev/Bare) if present
        - base_name: Core filename (typically "y_var_Vs_x_var")
        - kernel: Kernel operator type (Brillouin/Wilson) if present
        - paramN/valueN: Parameter labels and sanitized values
        - suffix: Optional "_grouped_by_[grouping_var]" for grouped plots
    
    Example outputs:
        - "Chebyshev_energy_Vs_time_Wilson_T300_P1p5"
        - "Combined_KL_mass_Vs_volume_L32_grouped_by_temperature"
    """

    def __init__(self, filename_labels: dict):
        """
        Initialize with a mapping of parameter names to abbreviated labels.
        
        Parameters:
        -----------
        filename_labels : dict
            Maps parameter names to short labels for filenames.
            E.g., {"temperature": "T", "pressure": "P"}
        """
        self.filename_labels = filename_labels

    def build(
        self,
        metadata_dict: dict,
        base_name: str,
        multivalued_params: list,
        grouping_variable: str = None,
        include_combined_prefix: bool = False,
        custom_prefix: str = None,
    ) -> str:
        """
        Build a filename from metadata following the structured format.

        Parameters:
        -----------
            - metadata_dict : dict
                Dictionary containing values of tunable parameters for this plot
                group. May include special keys like "Overlap_operator_method"
                and "Kernel_operator_type" which get special positioning.
            - base_name : str
                Core filename, typically in format "y_var_Vs_x_var".
            - multivalued_params : list
                List of parameter names that vary in the data and should be
                included in the filename.
            - grouping_variable : str or list, optional
                Variable(s) used for grouping. Appends "_grouped_by_[var]"
                suffix. If list, joins with "_and_" (max 2 variables supported).
            - include_combined_prefix : bool, optional
                Whether to prepend "Combined_" to the filename. Typically True
                when grouping_variable is defined. Default is False.
            - custom_prefix : str, optional
                Custom prefix that overrides "Combined_". If provided, this is
                used instead of the default prefix logic.

        Returns:
        --------
        str:
            A formatted filename (without extension) following the pattern:
            [prefix][overlap]_[base_name]_[kernel]_[params]_[suffix]
            
        Example:
            >>> metadata = {"Overlap_operator_method": "Chebyshev", 
            ...            "temperature": 3.14, "pressure": 2.0}
            >>> builder.build(metadata, "energy_Vs_time", ["temperature"], 
            ...              grouping_variable="pressure")
            'energy_Vs_time_T3p14_grouped_by_pressure'
        """
        def sanitize(value):
            return (
                str(value)
                .replace(".", "p")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
            )

        # -- Build filename in parts
        filename_parts = []

        # 1. Handle overlap operator method
        overlap_method = metadata_dict.get("Overlap_operator_method")
        if overlap_method in {"KL", "Chebyshev", "Bare"}:
            filename_parts.append(overlap_method)
            metadata_dict = metadata_dict.copy()
            metadata_dict.pop("Overlap_operator_method", None)

        # 2. Add base name
        filename_parts.append(base_name)

        # 3. Handle operator kernel type
        kernel_type = metadata_dict.get("Kernel_operator_type")
        if kernel_type in {"Brillouin", "Wilson"}:
            filename_parts.append(kernel_type)
            metadata_dict = metadata_dict.copy()
            metadata_dict.pop("Kernel_operator_type", None)

        # 4. Add multivalued tunable parameter labels and values
        for param in multivalued_params:
            if param in metadata_dict:
                label = self.filename_labels.get(param, param)
                value = sanitize(metadata_dict[param])
                filename_parts.append(f"{label}{value}")

        # 5. Determine optional prefix
        if custom_prefix is not None:
            prefix = custom_prefix
        elif include_combined_prefix:
            prefix = "Combined_"
        else:
            prefix = ""

        # 6. Add grouping suffix
        suffix = ""
        if grouping_variable:
            if isinstance(grouping_variable, str):
                suffix = f"_grouped_by_{grouping_variable}"
            else:
                # Note: There is a restriction of no more than two grouping
                # variables
                suffix = "_grouped_by_" + "_and_".join(grouping_variable)

        return prefix + "_".join(filename_parts) + suffix

    def validate_filename(self, filename):
        # Additional helper method
        # TODO: Work out the eventuality the filename is too long
        return len(filename) < 255
