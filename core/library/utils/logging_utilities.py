"""
Enhanced Logging System for QPB Data Analysis
============================================

A flexible, configurable logging system that supports both scripts and
Jupyter notebooks with clear separation between console output and file
logging.
"""

import os
import sys
import logging
import textwrap
import inspect
from pathlib import Path
from typing import Optional, Union, Dict, Any
from enum import Enum
import json


class LogLevel(Enum):
    """Standard log levels with descriptive names."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LoggingConfig:
    """Configuration class for logging setup."""

    def __init__(
        self,
        # File logging options
        enable_file_logging: bool = True,
        file_log_level: LogLevel = LogLevel.DEBUG,
        log_directory: Optional[str] = None,
        log_filename: Optional[str] = None,
        # Console logging options
        enable_console_logging: bool = False,
        console_log_level: LogLevel = LogLevel.INFO,
        # Formatting options
        file_format: str = "%(asctime)s - %(levelname)s - %(message)s",
        console_format: str = "%(levelname)s: %(message)s",
        wrap_width: int = 80,
        # Behavior options
        auto_create_dirs: bool = True,
        clear_existing_handlers: bool = True,
        propagate: bool = False,
    ):
        self.enable_file_logging = enable_file_logging
        self.file_log_level = file_log_level
        self.log_directory = log_directory
        self.log_filename = log_filename

        self.enable_console_logging = enable_console_logging
        self.console_log_level = console_log_level

        self.file_format = file_format
        self.console_format = console_format
        self.wrap_width = wrap_width

        self.auto_create_dirs = auto_create_dirs
        self.clear_existing_handlers = clear_existing_handlers
        self.propagate = propagate

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoggingConfig":
        """Create config from dictionary."""
        # Convert string log levels to LogLevel enums
        if "file_log_level" in config_dict:
            config_dict["file_log_level"] = LogLevel[
                config_dict["file_log_level"].upper()
            ]
        if "console_log_level" in config_dict:
            config_dict["console_log_level"] = LogLevel[
                config_dict["console_log_level"].upper()
            ]

        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_path: str) -> "LoggingConfig":
        """Load config from JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class WrappedFormatter(logging.Formatter):
    """Custom formatter that wraps long messages."""

    def __init__(self, fmt: str, wrap_width: int = 80):
        super().__init__(fmt)
        self.wrap_width = wrap_width

    def format(self, record: logging.LogRecord) -> str:
        # Format the record first
        formatted = super().format(record)

        # Then wrap long lines
        if self.wrap_width > 0:
            lines = formatted.split("\n")
            wrapped_lines = []
            for line in lines:
                if len(line) <= self.wrap_width:
                    wrapped_lines.append(line)
                else:
                    wrapped_lines.extend(
                        textwrap.wrap(
                            line,
                            width=self.wrap_width,
                            subsequent_indent="    ",  # Indent continuation lines
                        )
                    )
            return "\n".join(wrapped_lines)

        return formatted


class QPBLogger:
    """
    Enhanced logging wrapper for QPB data analysis with flexible
    configuration.

    Features:
        - Separate console and file logging controls
        - Configurable log levels for each output
        - Support for both scripts and Jupyter notebooks
        - Automatic script name detection
        - Flexible formatting options
        - Context manager support
    """

    def __init__(
        self, config: Optional[LoggingConfig] = None, name: Optional[str] = None
    ):
        """
        Initialize the logger.

        Args:
            - config: LoggingConfig instance. If None, uses default
              configuration.
            - name: Logger name. If None, auto-detects from calling
              script.
        """
        self.config = config or LoggingConfig()
        self.name = name or self._detect_caller_name()
        self.logger = self._setup_logger()

    def _detect_caller_name(self) -> str:
        """Detect the name of the calling script or notebook."""
        try:
            # Get the calling frame (skip this method and __init__)
            frame = inspect.stack()[2]
            filename = frame.filename

            # Handle Jupyter notebooks
            if "<ipython-input-" in filename or "ipykernel_" in filename:
                return "jupyter_notebook"

            # Handle regular Python files
            return Path(filename).stem

        except (IndexError, AttributeError):
            return "unknown_script"

    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with configured handlers."""
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)  # Let handlers control levels

        # Clear existing handlers if requested
        if self.config.clear_existing_handlers:
            logger.handlers.clear()

        # Set propagation
        logger.propagate = self.config.propagate

        # Add file handler if enabled
        if self.config.enable_file_logging:
            file_handler = self._create_file_handler()
            if file_handler:
                logger.addHandler(file_handler)

        # Add console handler if enabled
        if self.config.enable_console_logging:
            console_handler = self._create_console_handler()
            logger.addHandler(console_handler)

        return logger

    def _create_file_handler(self) -> Optional[logging.FileHandler]:
        """Create and configure file handler."""
        # Determine log file path
        log_dir = self.config.log_directory or os.getcwd()
        log_filename = self.config.log_filename or f"{self.name}.log"
        log_path = Path(log_dir) / log_filename

        # Create directory if needed
        if self.config.auto_create_dirs:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Create handler
            handler = logging.FileHandler(log_path, mode="w")
            handler.setLevel(self.config.file_log_level.value)

            # Set formatter
            formatter = WrappedFormatter(
                self.config.file_format, self.config.wrap_width
            )
            handler.setFormatter(formatter)

            return handler

        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create log file {log_path}: {e}")
            return None

    def _create_console_handler(self) -> logging.StreamHandler:
        """Create and configure console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.config.console_log_level.value)

        # Set formatter
        formatter = WrappedFormatter(self.config.console_format, self.config.wrap_width)
        handler.setFormatter(formatter)

        return handler

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    # Convenient logging methods
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

    # Script lifecycle methods
    def log_script_start(self, extra_info: Optional[str] = None):
        """Log script execution start."""
        message = f"Script '{self.name}' execution initiated"
        if extra_info:
            message += f" - {extra_info}"
        self.info(message)

    def log_script_end(self, extra_info: Optional[str] = None):
        """Log script execution end."""
        message = f"Script '{self.name}' execution completed successfully"
        if extra_info:
            message += f" - {extra_info}"
        self.info(message)

    def log_script_error(self, error: Exception):
        """Log script execution error."""
        self.error(f"Script '{self.name}' failed with error: {error}")

    # Configuration management
    def update_config(self, **kwargs):
        """Update configuration and reinitialize logger."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Reinitialize logger with new config
        self.close()
        self.logger = self._setup_logger()

    def set_file_level(self, level: Union[LogLevel, str]):
        """Change file logging level at runtime."""
        if isinstance(level, str):
            level = LogLevel[level.upper()]

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(level.value)

    def set_console_level(self, level: Union[LogLevel, str]):
        """Change console logging level at runtime."""
        if isinstance(level, str):
            level = LogLevel[level.upper()]

        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(level.value)


# Convenience functions for common use cases
def create_script_logger(
    log_directory: Optional[str] = None,
    log_filename: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = False,
    verbose: bool = False,
) -> QPBLogger:
    """
    Create a logger configured for typical script usage.

    Args:
        - log_directory: Directory for log files
        - log_filename: Name of log file
        - enable_file_logging: Whether to log to file
        - enable_console_logging: Whether to log to console
        - verbose: If True, enables console logging at DEBUG level

    Returns:
        Configured QPBLogger instance
    """

    # Detect the actual calling script (not this function)
    caller_frame = inspect.stack()[1]
    script_name = os.path.basename(caller_frame.filename)

    config = LoggingConfig(
        enable_file_logging=enable_file_logging,
        log_directory=log_directory,
        log_filename=log_filename,
        enable_console_logging=enable_console_logging or verbose,
        console_log_level=LogLevel.DEBUG if verbose else LogLevel.INFO,
        file_format="%(asctime)s - %(levelname)s - %(message)s",
        console_format="%(levelname)s: %(message)s",
    )

    return QPBLogger(config, name=script_name)


def create_jupyter_logger(
    enable_console: bool = True,
    console_level: LogLevel = LogLevel.INFO,
    enable_file: bool = False,
    log_directory: Optional[str] = None,
) -> QPBLogger:
    """
    Create a logger configured for Jupyter notebook usage.

    Args:
        - enable_console: Whether to show logs in notebook output
        - console_level: Log level for console output
        - enable_file: Whether to also log to file
        - log_directory: Directory for log files (if file logging
          enabled)

    Returns:
        Configured QPBLogger instance
    """
    config = LoggingConfig(
        enable_console_logging=enable_console,
        console_log_level=console_level,
        console_format="[%(levelname)s] %(message)s",  # Simpler format for notebooks
        enable_file_logging=enable_file,
        log_directory=log_directory,
        wrap_width=100,  # Wider for notebook displays
    )

    return QPBLogger(config)


def create_minimal_logger(name: Optional[str] = None) -> QPBLogger:
    """Create a minimal logger that only logs to console."""
    config = LoggingConfig(
        enable_file_logging=False,
        enable_console_logging=True,
        console_format="%(message)s",  # Just the message
    )

    return QPBLogger(config, name=name)


# Example usage for migration from current LoggingWrapper:


class LoggingWrapper:
    """
    Backward-compatible wrapper for existing code.

    This maintains the same interface as your current LoggingWrapper
    while using the new system underneath.
    """

    def __init__(
        self, log_directory: str, log_filename: str, enable_logging: bool = True
    ):
        """Initialize with old-style parameters."""
        if enable_logging:
            self.logger = create_script_logger(
                log_directory=log_directory,
                log_filename=log_filename,
                enable_file_logging=True,
                enable_console_logging=False,
            )
        else:
            # Create a null logger that doesn't output anything
            config = LoggingConfig(
                enable_file_logging=False, enable_console_logging=False
            )
            self.logger = QPBLogger(config)

    def initiate_script_logging(self):
        """Log script execution start."""
        self.logger.log_script_start()

    def terminate_script_logging(self):
        """Log script execution termination."""
        self.logger.log_script_end()

    def info(self, message: str, to_console: bool = False):
        """Log info message with optional console output."""
        self.logger.info(message)
        if to_console:
            print(f"INFO: {message}")

    def warning(self, message: str, to_console: bool = False):
        """Log warning message with optional console output."""
        self.logger.warning(message)
        if to_console:
            print(f"WARNING: {message}")

    def error(self, message: str, to_console: bool = False):
        """Log error message with optional console output."""
        self.logger.error(message)
        if to_console:
            print(f"ERROR: {message}")

    def debug(self, message: str, to_console: bool = False):
        """Log debug message with optional console output."""
        self.logger.debug(message)
        if to_console:
            print(f"DEBUG: {message}")

    def critical(self, message: str, to_console: bool = False):
        """Log critical message with optional console output."""
        self.logger.critical(message)
        if to_console:
            print(f"CRITICAL: {message}")
