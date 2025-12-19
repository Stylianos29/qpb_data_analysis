from typing import Optional, Union, List

from library.constants import OVERLAP_OPERATOR_METHODS, KERNEL_OPERATOR_TYPES


class PlotFilenameBuilder:
    """
    Constructs structured plot filenames from metadata following a specific format pattern.

    The filename format follows this pattern:
    [prefix][overlap]_[base_name]_[kernel]_[param1][value1]_[param2][value2]...[suffix]

    Components:
    -----------
    - prefix: Optional "Combined_" or custom prefix for grouped plots
    - overlap: Overlap operator method (KL/Chebyshev/Bare) if present
    - base_name: Core filename (typically "y_var_Vs_x_var")
    - kernel: Kernel operator type (Brillouin/Wilson) if present
    - paramN/valueN: Parameter labels and sanitized values
    - suffix: Optional "_grouped_by_[grouping_var]" for grouped plots

    Examples:
    ---------
    Individual plots:
        - "Chebyshev_energy_Vs_time_Wilson_T300_P1p5"
        - "energy_Vs_time_T300p5_P2p0"

    Grouped plots:
        - "Combined_KL_mass_Vs_volume_L32_grouped_by_temperature"
        - "Chebyshev_energy_Vs_time_Wilson_grouped_by_pressure_and_lattice_size"

    Features:
    ---------
    - Automatic value sanitization (dots become 'p', removes special chars)
    - Smart parameter ordering (overlap → kernel → others)
    - Support for single and dual grouping variables
    - Customizable abbreviation mappings
    - Filename length validation
    """

    def __init__(self, filename_labels: dict):
        """
        Initialize with a mapping of parameter names to abbreviated labels.

        Parameters:
        -----------
        filename_labels : dict
            Maps parameter names to short labels for filenames.
            E.g., {"temperature": "T", "pressure": "P", "lattice_size": "L"}

        Example:
        --------
        >>> labels = {"temperature": "T", "pressure": "P"}
        >>> builder = PlotFilenameBuilder(labels)
        """
        self.filename_labels = filename_labels

        # Special parameters that get prioritized positioning
        self._special_overlap_methods = OVERLAP_OPERATOR_METHODS
        self._special_kernel_types = KERNEL_OPERATOR_TYPES

        # Maximum filename length (most filesystems support 255 chars)
        self._max_filename_length = 250  # Leave some buffer

    def build(
        self,
        metadata_dict: dict,
        base_name: str,
        multivalued_params: list,
        grouping_variable: Optional[Union[str, List[str]]] = None,
        include_combined_prefix: bool = False,
        custom_prefix: Optional[str] = None,
    ) -> str:
        """
        Build a filename from metadata following the structured format.

        Parameters:
        -----------
        metadata_dict : dict
            Dictionary containing values of tunable parameters for this plot
            group. May include special keys like "Overlap_operator_method"
            and "Kernel_operator_type" which get special positioning.
        base_name : str
            Core filename, typically in format "y_var_Vs_x_var".
        multivalued_params : list
            List of parameter names that vary in the data and should be
            included in the filename.
        grouping_variable : str or list, optional
            Variable(s) used for grouping. Appends "_grouped_by_[var]"
            suffix. If list, joins with "_and_" (max 2 variables supported).
        include_combined_prefix : bool, optional
            Whether to prepend "Combined_" to the filename. Typically True
            when grouping_variable is defined. Default is False.
        custom_prefix : str, optional
            Custom prefix that overrides "Combined_". If provided, this is
            used instead of the default prefix logic.

        Returns:
        --------
        str
            A formatted filename (without extension) following the pattern:
            [prefix][overlap]_[base_name]_[kernel]_[params]_[suffix]

        Raises:
        -------
        ValueError
            If the resulting filename would be too long or contain invalid characters.

        Example:
        --------
        >>> metadata = {"Overlap_operator_method": "Chebyshev",
        ...            "temperature": 3.14, "pressure": 2.0}
        >>> builder.build(metadata, "energy_Vs_time", ["temperature"],
        ...              grouping_variable="pressure")
        'energy_Vs_time_T3p14_grouped_by_pressure'
        """
        # Create a working copy to avoid modifying the original
        working_metadata = metadata_dict.copy()

        # Build filename components in order
        filename_parts = []

        # 1. Handle overlap operator method (gets special positioning)
        overlap_method = working_metadata.get("Overlap_operator_method")
        if overlap_method in self._special_overlap_methods:
            filename_parts.append(overlap_method)
            working_metadata.pop("Overlap_operator_method", None)

        # 2. Add base name (only if not empty)
        if base_name:
            filename_parts.append(base_name)

        # 3. Handle operator kernel type (gets special positioning)
        kernel_type = working_metadata.get("Kernel_operator_type")
        if kernel_type in self._special_kernel_types:
            filename_parts.append(kernel_type)
            working_metadata.pop("Kernel_operator_type", None)

        # 4. Add multivalued tunable parameter labels and values
        for param in multivalued_params:
            if param in working_metadata:
                label = self.filename_labels.get(param, param)
                sanitized_value = self._sanitize_value(working_metadata[param])
                filename_parts.append(f"{label}{sanitized_value}")

        # 5. Determine optional prefix
        prefix = self._determine_prefix(custom_prefix, include_combined_prefix)

        # 6. Add grouping suffix
        suffix = self._build_grouping_suffix(grouping_variable)

        # 7. Combine all parts
        filename = prefix + "_".join(filename_parts) + suffix

        # 8. Validate the resulting filename
        self._validate_filename(filename)

        return filename

    def _sanitize_value(self, value) -> str:
        """
        Sanitize a value for use in filenames.

        Converts problematic characters to filename-safe alternatives:
        - Dots (.) become 'p'
        - Removes commas, parentheses
        - Converts to string representation

        Parameters:
        -----------
        value : any
            The value to sanitize

        Returns:
        --------
        str
            Sanitized string safe for filenames

        Example:
        --------
        >>> builder._sanitize_value(3.14)
        '3p14'
        >>> builder._sanitize_value("test(1,2)")
        'test12'
        """
        return (
            str(value)
            .replace(".", "p")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")  # Added: spaces to underscores
            .replace("/", "_")  # Added: slashes to underscores
        )

    def _determine_prefix(
        self, custom_prefix: Optional[str], include_combined: bool
    ) -> str:
        """
        Determine the appropriate prefix for the filename.

        Parameters:
        -----------
        custom_prefix : str, optional
            Custom prefix that overrides default logic
        include_combined : bool
            Whether to use "Combined_" prefix

        Returns:
        --------
        str
            The prefix to use (may be empty string)
        """
        if custom_prefix is not None:
            return custom_prefix
        elif include_combined:
            return "Combined_"
        else:
            return ""

    def _build_grouping_suffix(
        self, grouping_variable: Optional[Union[str, List[str]]]
    ) -> str:
        """
        Build the grouping suffix for the filename.

        Parameters:
        -----------
        grouping_variable : str, list, or None
            Variable(s) used for grouping

        Returns:
        --------
        str
            Grouping suffix (may be empty string)

        Raises:
        -------
        ValueError
            If more than 2 grouping variables are provided
        """
        if not grouping_variable:
            return ""

        if isinstance(grouping_variable, str):
            return f"_grouped_by_{grouping_variable}"
        elif isinstance(grouping_variable, list):
            if len(grouping_variable) > 2:
                raise ValueError("Maximum of 2 grouping variables supported")
            return "_grouped_by_" + "_and_".join(grouping_variable)
        else:
            raise TypeError("grouping_variable must be str, list, or None")

    def _validate_filename(self, filename: str) -> None:
        """
        Validate the generated filename.

        Parameters:
        -----------
        filename : str
            The filename to validate

        Raises:
        -------
        ValueError
            If filename is too long or contains invalid characters
        """
        if len(filename) > self._max_filename_length:
            raise ValueError(
                f"Filename too long ({len(filename)} chars). "
                f"Maximum allowed: {self._max_filename_length}"
            )

        # Check for problematic characters that might remain
        invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
        for char in invalid_chars:
            if char in filename:
                raise ValueError(f"Invalid character '{char}' in filename: {filename}")

    def validate_filename(self, filename: str) -> bool:
        """
        Public method to validate a filename without raising exceptions.

        Parameters:
        -----------
        filename : str
            The filename to validate

        Returns:
        --------
        bool
            True if filename is valid, False otherwise
        """
        try:
            self._validate_filename(filename)
            return True
        except ValueError:
            return False

    def set_max_length(self, max_length: int) -> None:
        """
        Update the maximum allowed filename length.

        Parameters:
        -----------
        max_length : int
            New maximum length (should be less than filesystem limit)
        """
        if max_length <= 0:
            raise ValueError("Maximum length must be positive")
        self._max_filename_length = max_length

    def add_filename_labels(self, new_labels: dict) -> None:
        """
        Add or update filename label mappings.

        Parameters:
        -----------
        new_labels : dict
            Dictionary of parameter_name -> abbreviation mappings to add/update

        Example:
        --------
        >>> builder.add_filename_labels({"new_param": "NP", "temperature": "TEMP"})
        """
        self.filename_labels.update(new_labels)

    def get_filename_preview(
        self, metadata_dict: dict, base_name: str, multivalued_params: list, **kwargs
    ) -> dict:
        """
        Generate a filename and return detailed information about it.

        Useful for debugging filename generation or checking length before creation.

        Parameters:
        -----------
        Same as build() method

        Returns:
        --------
        dict
            Dictionary containing:
            - 'filename': The generated filename
            - 'length': Length of the filename
            - 'components': Dict breaking down the filename parts
            - 'valid': Whether the filename passes validation
        """
        try:
            filename = self.build(
                metadata_dict, base_name, multivalued_params, **kwargs
            )

            # Build component breakdown
            working_metadata = metadata_dict.copy()
            components = {
                "prefix": self._determine_prefix(
                    kwargs.get("custom_prefix"),
                    kwargs.get("include_combined_prefix", False),
                ),
                "overlap": working_metadata.get("Overlap_operator_method", ""),
                "base_name": base_name,
                "kernel": working_metadata.get("Kernel_operator_type", ""),
                "parameters": [
                    f"{self.filename_labels.get(p, p)}{self._sanitize_value(working_metadata.get(p, ''))}"
                    for p in multivalued_params
                    if p in working_metadata
                ],
                "suffix": self._build_grouping_suffix(kwargs.get("grouping_variable")),
            }

            return {
                "filename": filename,
                "length": len(filename),
                "components": components,
                "valid": True,
            }

        except ValueError as e:
            return {
                "filename": None,
                "length": 0,
                "components": {},
                "valid": False,
                "error": str(e),
            }
