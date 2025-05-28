"""
A minimal set of configurable Click validation callbacks for command-line
arguments.

This module provides exactly 4 validator functions that can be customized for
specific use cases through parameters. The validators handle both full file
paths and filename-only inputs, with appropriate validation behavior for each
case.

Usage Patterns:

1. Basic validation (any file/directory):
    @click.option("--input_file", callback=validate_input_file)
    
2. Extension-specific validation:
    from functools import partial
    validate_csv = partial(validate_input_file, extensions=['.csv'])
    @click.option("--csv_file", callback=validate_csv)
    
3. Format validation with custom checker:
    def check_hdf5(filepath):
        import h5py
        with h5py.File(filepath, 'r') as f: pass
    
    validate_hdf5 = partial(
                        validate_input_file,
                        extensions=['.h5'],
                        format_checker=check_hdf5
                        )
    @click.option("--hdf5_file", callback=validate_hdf5)

4. Filename-only validation (no existence check):
    # Will only check extensions, not file existence
    @click.option(
        "--output_name",
        callback=partial(validate_output_file, extensions=['.csv'])
        )

Key Features:
- Minimal API surface: Only 4 core functions to learn and maintain
- Configurable validation: Use functools.partial to create specialized
  validators
- Path flexibility: Handles both full paths and filename-only inputs
- Consistent error messages: All validation errors follow Click conventions
- No external dependencies: Uses only Click and standard library

Design Philosophy:
- Functions accept both full paths and filenames
- Full paths undergo complete validation (existence, format, etc.)
- Filenames undergo only structural validation (extensions)
- Output validators never create directories automatically (user responsibility)
- All functions are highly configurable through parameters
"""

import os
import click


def validate_input_directory(ctx, param, value, must_exist=True, readable=True):
    """
    Validate input directory path.

    Parameters:
        ctx: Click context
        param: Click parameter
        value: Directory path to validate
        must_exist: Whether directory must already exist (default: True)
        readable: Whether directory must be readable (default: True)

    Returns:
        str: Validated directory path

    Raises:
        click.BadParameter: If validation fails
    """
    if value is None:
        return None

    # Ensure value is a string (handle Click passing Option objects)
    if not isinstance(value, str):
        raise click.BadParameter(f"Expected string value, got {type(value).__name__}")

    # Custom validation with clear error messages
    if not os.path.exists(value):
        if must_exist:
            raise click.BadParameter(
                f"Input directory '{value}' does not exist. "
                "Please check the path and try again."
            )
    elif not os.path.isdir(value):
        raise click.BadParameter(
            f"Path '{value}' exists but is not a directory. "
            "Please provide a valid directory path."
        )
    elif readable and not os.access(value, os.R_OK):
        raise click.BadParameter(
            f"Directory '{value}' exists but is not readable. "
            "Please check permissions."
        )

    return os.path.abspath(value)


def validate_output_directory(ctx, param, value, check_parent_exists=False):
    """
    Validate output directory path (doesn't need to exist yet).

    Parameters:
        - ctx: Click context
        - param: Click parameter
        - value: Directory path to validate
        - check_parent_exists: Whether to verify parent directory exists
          (default: False)

    Returns:
        str: Validated directory path

    Raises:
        click.BadParameter: If validation fails or parent doesn't exist when
        required

    Note:
        This function never creates directories. Use check_parent_exists=True to
        ensure the parent directory exists, allowing the user to create it if
        needed.
    """
    if value is None:
        return None

    # Ensure value is a string (handle Click passing Option objects)
    if not isinstance(value, str):
        raise click.BadParameter(f"Expected string value, got {type(value).__name__}")

    # Check if path already exists and is not a directory
    if os.path.exists(value) and not os.path.isdir(value):
        raise click.BadParameter(
            f"Output path '{value}' already exists but is not a directory. "
            "Please choose a different path."
        )

    # Check if parent directory exists when required
    if check_parent_exists:
        parent_dir = os.path.dirname(value)
        if parent_dir and not os.path.exists(parent_dir):
            raise click.BadParameter(
                f"Parent directory '{parent_dir}' does not exist. "
                "Please create it first:\n"
                f"  mkdir -p '{parent_dir}'"
            )
        elif parent_dir and not os.path.isdir(parent_dir):
            raise click.BadParameter(
                f"Parent path '{parent_dir}' exists but is not a directory."
            )

    return os.path.abspath(value)


def validate_input_file(
    ctx, param, value, extensions=None, format_checker=None, readable=True
):
    """
    Validate input file path or filename with optional extension and format
    checking.

    Parameters:
        - ctx: Click context
        - param: Click parameter
        - value: File path or filename to validate
        - extensions: List of allowed extensions (e.g., ['.csv', '.h5']) or None
        - format_checker: Function to test file format, should raise exception
          if invalid
        - readable: Whether file must be readable when it's a full path
          (default: True)

    Returns:
        str: Validated file path or filename

    Raises:
        click.BadParameter: If validation fails

    Behavior:
        - Full paths (containing '/' or '\\') undergo complete validation
        - Filenames (no path separators) undergo only extension validation
        - Format checking only applies to full paths that exist

    Example:
        # Basic file validation
        callback=validate_input_file

        # CSV files only
        from functools import partial
        callback=partial(validate_input_file, extensions=['.csv'])

        # HDF5 with format checking
        def check_hdf5(filepath):
            import h5py
            with h5py.File(filepath, 'r') as f:
                pass

        callback=partial(
            validate_input_file,
            extensions=['.h5'],
            format_checker=check_hdf5
            )
    """
    if value is None:
        return None

    # Ensure value is a string (handle Click passing Option objects)
    if not isinstance(value, str):
        raise click.BadParameter(f"Expected string value, got {type(value).__name__}")

    # Determine if this is a full path or just a filename
    is_full_path = os.sep in value or ("\\" in value and os.name == "nt")

    if is_full_path:
        # Full path validation - check existence, readability, format
        if not os.path.exists(value):
            raise click.BadParameter(
                f"Input file '{value}' does not exist. "
                "Please check the path and try again."
            )
        elif not os.path.isfile(value):
            raise click.BadParameter(
                f"Path '{value}' exists but is not a file. "
                "Please provide a valid file path."
            )
        elif readable and not os.access(value, os.R_OK):
            raise click.BadParameter(
                f"File '{value}' exists but is not readable. "
                "Please check permissions."
            )

        validated_path = os.path.abspath(value)

        # Format validation (only for existing files)
        if format_checker:
            try:
                format_checker(validated_path)
            except Exception as e:
                file_type = "file"
                if extensions:
                    file_type = f"{extensions[0].replace('.', '').upper()} file"
                raise click.BadParameter(
                    f"File '{os.path.basename(validated_path)}' is not "
                    f"a valid {file_type}: {e}"
                )
    else:
        # Filename only - just basic validation, no existence check
        if not value.strip():
            raise click.BadParameter(
                "Filename cannot be empty or contain only whitespace."
            )
        validated_path = value.strip()

    # Extension validation (applies to both paths and filenames)
    if extensions:
        if not any(validated_path.lower().endswith(ext.lower()) for ext in extensions):
            ext_list = ", ".join(extensions)
            filename = (
                os.path.basename(validated_path) if is_full_path else validated_path
            )
            raise click.BadParameter(
                f"File '{filename}' must have one of these extensions: {ext_list}"
            )

    return validated_path


def validate_output_file(ctx, param, value, extensions=None, check_parent_exists=False):
    """
    Validate output file path or filename (doesn't need to exist yet).

    Parameters:

        - ctx: Click context
        - param: Click parameter
        - value: File path or filename to validate
        - extensions: List of allowed extensions (e.g., ['.csv', '.h5']) or None
        - check_parent_exists: Whether to verify parent directory exists for
          full paths

    Returns:
        str: Validated file path or filename

    Raises:
        click.BadParameter: If validation fails or parent directory doesn't
        exist when required

    Behavior:

        - Full paths (containing '/' or '\\') undergo path validation
        - Filenames (no path separators) undergo only extension validation
        - Never creates directories automatically

    Note:
        This function never creates directories. Use check_parent_exists=True to
        ensure the parent directory exists for full paths, allowing the user to
        create it if needed.

    Example:
        # Basic output file
        callback=validate_output_file

        # Must be CSV
        from functools import partial
        callback=partial(validate_output_file, extensions=['.csv'])

        # HDF5 with parent directory check
        callback=partial(
            validate_output_file,
            extensions=['.h5'],
            check_parent_exists=True
            )
    """
    if value is None:
        return None

    # Ensure value is a string (handle Click passing Option objects)
    if not isinstance(value, str):
        raise click.BadParameter(f"Expected string value, got {type(value).__name__}")

    # Determine if this is a full path or just a filename
    is_full_path = os.sep in value or ("\\" in value and os.name == "nt")

    if is_full_path:
        # Full path validation
        if os.path.exists(value) and not os.path.isfile(value):
            raise click.BadParameter(
                f"Output path '{value}' already exists but is not a file. "
                "Please choose a different path."
            )

        validated_path = os.path.abspath(value)

        # Check if parent directory exists (but don't create it)
        if check_parent_exists:
            parent_dir = os.path.dirname(validated_path)
            if parent_dir and not os.path.exists(parent_dir):
                raise click.BadParameter(
                    f"Parent directory '{parent_dir}' does not exist. "
                    "Please create it first:\n"
                    f"  mkdir -p '{parent_dir}'"
                )
            elif parent_dir and not os.path.isdir(parent_dir):
                raise click.BadParameter(
                    f"Parent path '{parent_dir}' exists but is not a directory."
                )
    else:
        # Filename only - just basic validation
        if not value.strip():
            raise click.BadParameter(
                "Output filename cannot be empty or contain only whitespace."
            )
        validated_path = value.strip()

    # Extension validation (applies to both paths and filenames)
    if extensions:
        if not any(validated_path.lower().endswith(ext.lower()) for ext in extensions):
            ext_list = ", ".join(extensions)
            filename = (
                os.path.basename(validated_path) if is_full_path else validated_path
            )
            raise click.BadParameter(
                f"Output filename '{filename}' must have one of "
                f"these extensions: {ext_list}"
            )

    return validated_path
