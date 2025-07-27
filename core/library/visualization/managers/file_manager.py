import os
import shutil
from typing import Optional, List
from pathlib import Path


class PlotFileManager:
    """
    Handles all file system operations for plots with enhanced safety
    and format support.

    This class manages directory creation, file path construction, and
    cleanup operations for plot files. It provides safety measures for
    destructive operations and supports multiple output formats.

    Features:
    ---------
    - Multiple file format support (PNG, PDF, SVG, EPS, JPG)
    - Safe directory cleanup with confirmation
    - Robust error handling and validation
    - Path sanitization and validation
    - Configurable default file extensions

    Example:
    --------
    >>> manager = PlotFileManager("/path/to/plots", default_format="png")
    >>> subdir = manager.prepare_subdirectory("energy_plots")
    >>> path = manager.plot_path(subdir, "energy_vs_time", format="pdf")
    >>> # Returns: "/path/to/plots/energy_plots/energy_vs_time.pdf"
    """

    # Supported file formats with their characteristics
    SUPPORTED_FORMATS = {
        "png": {"extension": ".png", "description": "Portable Network Graphics"},
        "pdf": {"extension": ".pdf", "description": "Portable Document Format"},
        "svg": {"extension": ".svg", "description": "Scalable Vector Graphics"},
        "eps": {"extension": ".eps", "description": "Encapsulated PostScript"},
        "jpg": {"extension": ".jpg", "description": "JPEG Image"},
        "jpeg": {"extension": ".jpeg", "description": "JPEG Image"},
        "tiff": {"extension": ".tiff", "description": "Tagged Image File Format"},
        "ps": {"extension": ".ps", "description": "PostScript"},
    }

    def __init__(self, base_directory: str, default_format: str = "png"):
        """
        Initialize the file manager with a base directory and default format.

        Parameters:
        -----------
        base_directory : str
            Base directory path where all plots will be stored.
            Will be created if it doesn't exist.
        default_format : str, optional
            Default file format for plots. Must be one of the supported formats.
            Default is "png".

        Raises:
        -------
        ValueError
            If base_directory is invalid or default_format is not supported.
        OSError
            If directory cannot be created or accessed.
        """
        self._validate_format(default_format)
        self.default_format = default_format.lower()

        # Convert to Path object for better handling
        self.base_directory = Path(base_directory).resolve()

        # Create base directory if it doesn't exist
        self._ensure_directory_exists(self.base_directory)

        # Validate that we can write to the directory
        self._validate_directory_permissions(self.base_directory)

    def prepare_subdirectory(
        self, subdir_name: str, clear_existing: bool = False, confirm_clear: bool = True
    ) -> str:
        """
        Create or clean a subdirectory for storing plots.

        Parameters:
        -----------
        subdir_name : str
            Name of the subdirectory to create/prepare.
        clear_existing : bool, optional
            If True, remove all existing contents in the subdirectory.
            Default is False.
        confirm_clear : bool, optional
            If True and clear_existing is True, require explicit
            confirmation for destructive operations in interactive
            environments. Default is True.

        Returns:
        --------
        str
            Full path to the prepared subdirectory.

        Raises:
        -------
        ValueError
            If subdirectory name contains invalid characters.
        OSError
            If directory operations fail.
        RuntimeError
            If clear operation is cancelled by user.
        """
        # Validate and sanitize subdirectory name
        sanitized_name = self._sanitize_directory_name(subdir_name)
        full_path = self.base_directory / sanitized_name

        # Create directory if it doesn't exist
        self._ensure_directory_exists(full_path)

        # Handle clearing if requested
        if clear_existing:
            self._clear_directory_contents(full_path, confirm_clear)

        return str(full_path)

    def plot_path(
        self,
        directory: str,
        filename: str,
        format: Optional[str] = None,
        ensure_unique: bool = False,
    ) -> str:
        """
        Construct full path for a plot file with proper extension.

        Parameters:
        -----------
        directory : str
            Directory where the plot will be saved.
        filename : str
            Base filename (without extension).
        format : str, optional
            File format. If None, uses default format. Must be one of
            the supported formats.
        ensure_unique : bool, optional
            If True, append a number to make filename unique if it
            already exists. Default is False.

        Returns:
        --------
        str
            Full path to the plot file including proper extension.

        Raises:
        -------
        ValueError
            If format is not supported or filename contains invalid
            characters.
        """
        # Use default format if none specified
        if format is None:
            format = self.default_format
        else:
            self._validate_format(format)

        # Sanitize filename
        sanitized_filename = self._sanitize_filename(filename)

        # Get proper extension
        extension = self.SUPPORTED_FORMATS[format.lower()]["extension"]

        # Construct full path
        directory_path = Path(directory)
        base_path = directory_path / f"{sanitized_filename}{extension}"

        # Handle unique filename if requested
        if ensure_unique:
            base_path = self._make_unique_path(base_path)

        return str(base_path)

    def get_supported_formats(self) -> List[str]:
        """
        Get list of all supported file formats.

        Returns:
        --------
        list
            List of supported format names.
        """
        return list(self.SUPPORTED_FORMATS.keys())

    def get_format_info(self, format: str) -> dict:
        """
        Get information about a specific format.

        Parameters:
        -----------
        format : str
            Format name to query.

        Returns:
        --------
        dict
            Dictionary with format information including extension and
            description.

        Raises:
        -------
        ValueError
            If format is not supported.
        """
        self._validate_format(format)
        return self.SUPPORTED_FORMATS[format.lower()].copy()

    def set_default_format(self, format: str) -> None:
        """
        Change the default file format.

        Parameters:
        -----------
        format : str
            New default format. Must be one of the supported formats.

        Raises:
        -------
        ValueError
            If format is not supported.
        """
        self._validate_format(format)
        self.default_format = format.lower()

    def directory_exists(self, subdir_name: str) -> bool:
        """
        Check if a subdirectory exists.

        Parameters:
        -----------
        subdir_name : str
            Name of the subdirectory to check.

        Returns:
        --------
        bool
            True if directory exists, False otherwise.
        """
        sanitized_name = self._sanitize_directory_name(subdir_name)
        full_path = self.base_directory / sanitized_name
        return full_path.exists() and full_path.is_dir()

    def _validate_format(self, format: str) -> None:
        """Validate that format is supported."""
        if not isinstance(format, str):
            raise ValueError("Format must be a string")

        if format.lower() not in self.SUPPORTED_FORMATS:
            supported = ", ".join(self.SUPPORTED_FORMATS.keys())
            raise ValueError(
                f"Unsupported format '{format}'. Supported formats: {supported}"
            )

    def _ensure_directory_exists(self, directory_path: Path) -> None:
        """Create directory if it doesn't exist."""
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create directory '{directory_path}': {e}")

    def _validate_directory_permissions(self, directory_path: Path) -> None:
        """Validate that we can read and write to the directory."""
        if not os.access(directory_path, os.R_OK):
            raise OSError(f"Cannot read from directory '{directory_path}'")

        if not os.access(directory_path, os.W_OK):
            raise OSError(f"Cannot write to directory '{directory_path}'")

    def _sanitize_directory_name(self, name: str) -> str:
        """
        Sanitize directory name to be filesystem-safe.

        Parameters:
        -----------
        name : str
            Original directory name.

        Returns:
        --------
        str
            Sanitized directory name.

        Raises:
        -------
        ValueError
            If name is empty or contains only invalid characters.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Directory name cannot be empty")

        # Remove/replace problematic characters
        invalid_chars = '<>:"|?*'
        sanitized = name.strip()

        for char in invalid_chars:
            sanitized = sanitized.replace(char, "_")

        # Replace multiple consecutive spaces/underscores with single
        # underscore
        import re

        sanitized = re.sub(r"[_\s]+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        if not sanitized:
            raise ValueError(
                f"Directory name '{name}' contains only invalid characters"
            )

        return sanitized

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to be filesystem-safe.

        Parameters:
        -----------
        filename : str
            Original filename (without extension).

        Returns:
        --------
        str
            Sanitized filename.

        Raises:
        -------
        ValueError
            If filename is empty or contains only invalid characters.
        """
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Filename cannot be empty")

        # Remove/replace problematic characters (similar to directory
        # sanitization)
        invalid_chars = '<>:"|?*/'
        sanitized = filename.strip()

        for char in invalid_chars:
            sanitized = sanitized.replace(char, "_")

        # Replace multiple consecutive spaces/underscores with single
        # underscore
        import re

        sanitized = re.sub(r"[_\s]+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        if not sanitized:
            raise ValueError(f"Filename '{filename}' contains only invalid characters")

        return sanitized

    def _clear_directory_contents(self, directory_path: Path, confirm: bool) -> None:
        """
        Clear all contents of a directory with optional confirmation.

        Parameters:
        -----------
        directory_path : Path
            Directory to clear.
        confirm : bool
            Whether to ask for confirmation.

        Raises:
        -------
        RuntimeError
            If user cancels the operation.
        """
        # Check if directory has any contents
        try:
            contents = list(directory_path.iterdir())
        except OSError as e:
            raise OSError(f"Cannot access directory '{directory_path}': {e}")

        if not contents:
            return  # Nothing to clear

        # Ask for confirmation if requested and we're in an interactive
        # environment
        if confirm and self._is_interactive():
            response = (
                input(
                    f"Directory '{directory_path.name}' contains {len(contents)} items. "
                    f"Clear all contents? [y/N]: "
                )
                .strip()
                .lower()
            )

            if response not in ("y", "yes"):
                raise RuntimeError("Directory clear operation cancelled by user")

        # Clear the contents
        for item in contents:
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except OSError as e:
                raise OSError(f"Cannot remove '{item}': {e}")

    def _make_unique_path(self, base_path: Path) -> Path:
        """
        Generate a unique filename by appending a number if needed.

        Parameters:
        -----------
        base_path : Path
            Original path that might not be unique.

        Returns:
        --------
        Path
            Unique path (may be the same as input if already unique).
        """
        if not base_path.exists():
            return base_path

        # Extract parts for numbering
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        # Find a unique number
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def _is_interactive(self) -> bool:
        """
        Check if we're running in an interactive environment.

        Returns:
        --------
        bool
            True if interactive, False otherwise.
        """
        try:
            # Check if stdin is a terminal
            import sys

            return sys.stdin.isatty()
        except:
            return False
