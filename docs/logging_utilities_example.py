"""
Usage Examples for Enhanced QPB Logging System
==============================================

Examples showing how to use the new logging system in different
scenarios.
"""

# =============================================================================
# Example 1: Drop-in replacement for existing scripts
# =============================================================================

# This works exactly like your current LoggingWrapper
from library.utils.logging_utilities import LoggingWrapper


def migrate_existing_script():
    """Example showing how existing scripts can use the new system with
    no changes."""

    # Your existing code works unchanged:
    logger = LoggingWrapper(
        log_directory="/path/to/logs", log_filename="my_script.log", enable_logging=True
    )

    logger.initiate_script_logging()
    logger.info("Processing data...")
    logger.warning("This is a warning", to_console=True)
    logger.terminate_script_logging()


# =============================================================================
# Example 2: Enhanced script logging with new features
# =============================================================================

from library.utils.logging_utilities import create_script_logger, LogLevel


def enhanced_script_logging():
    """Example using new features for better control."""

    # Create logger with both file and console output
    logger = create_script_logger(
        log_directory="./logs",
        log_filename="analysis.log",
        enable_file_logging=True,
        enable_console_logging=True,  # Also show logs in console
        verbose=False,  # Set to True for DEBUG level console output
    )

    logger.log_script_start("Processing jackknife analysis")

    try:
        logger.info("Loading HDF5 data...")
        # Your processing code here
        logger.debug("Detailed processing information")  # Only in file
        logger.info("Processing complete")  # In both file and console

        logger.log_script_end("All data processed successfully")

    except Exception as e:
        logger.log_script_error(e)
        raise

    finally:
        logger.close()


# =============================================================================
# Example 3: Click integration with separate verbose and logging
# controls
# =============================================================================

import click
from library.utils.logging_utilities import create_script_logger


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Verbose console output")
@click.option("--enable-logging", is_flag=True, help="Enable file logging")
@click.option("--log-dir", default="./logs", help="Log directory")
@click.option("--log-file", help="Log filename")
def better_cli_integration(verbose, enable_logging, log_dir, log_file):
    """
    Example showing clear separation between verbose console output and
    file logging.

    This resolves the confusion between verbose and logging options.
    """

    # Set up logging
    logger = create_script_logger(
        log_directory=log_dir if enable_logging else None,
        log_filename=log_file,
        enable_file_logging=enable_logging,
        enable_console_logging=False,  # Console handled separately
        verbose=False,
    )

    logger.log_script_start()

    # Verbose console output (independent of logging)
    if verbose:
        click.echo(f"Input file: input.h5")
        click.echo(f"Output directory: ./output")
        click.echo(f"Logging enabled: {enable_logging}")

    # Always log to file (if logging enabled)
    logger.info("Starting data processing")

    # Processing steps
    for i in range(5):
        if verbose:
            click.echo(f"Processing step {i+1}/5...")  # Console feedback

        logger.debug(f"Detailed step {i+1} information")  # Detailed file log

        # Simulate some work
        import time

        time.sleep(0.1)

    if verbose:
        click.echo("✓ Processing complete!")

    logger.log_script_end()


# =============================================================================
# Example 4: Jupyter notebook usage
# =============================================================================

from library.utils.logging_utilities import create_jupyter_logger


def jupyter_notebook_example():
    """Example for Jupyter notebook usage."""

    # Create logger optimized for notebooks
    logger = create_jupyter_logger(
        enable_console=True,  # Show logs in cell output
        console_level=LogLevel.INFO,  # Only show INFO and above
        enable_file=True,  # Also save to file
        log_directory="./notebook_logs",
    )

    logger.info("Starting notebook analysis")

    # Your analysis code here
    logger.debug("This won't show in notebook but will be in file")
    logger.info("This shows in notebook and file")
    logger.warning("Important warning shown prominently")

    logger.info("Analysis complete")


# =============================================================================
# Example 5: Configuration-driven logging
# =============================================================================

from library.utils.logging_utilities import QPBLogger, LoggingConfig, LogLevel


def config_driven_logging():
    """Example using configuration objects for complex setups."""

    # Create custom configuration
    config = LoggingConfig(
        # File logging
        enable_file_logging=True,
        file_log_level=LogLevel.DEBUG,
        log_directory="./detailed_logs",
        log_filename="detailed_analysis.log",
        # Console logging
        enable_console_logging=True,
        console_log_level=LogLevel.WARNING,  # Only warnings/errors to console
        # Formatting
        file_format="%(asctime)s [%(name)s] %(levelname)-8s: %(message)s",
        console_format="⚠️ %(levelname)s: %(message)s",
        wrap_width=120,
        # Behavior
        auto_create_dirs=True,
        clear_existing_handlers=True,
    )

    logger = QPBLogger(config, name="detailed_analysis")

    logger.debug("Detailed debug info")  # Only to file
    logger.info("General information")  # Only to file
    logger.warning("Important warning")  # To both file and console
    logger.error("Error occurred")  # To both file and console

    # Dynamic configuration changes
    logger.set_console_level(LogLevel.INFO)  # Now info messages show in console too
    logger.info("This now appears in console")


# =============================================================================
# Example 6: Context manager usage for automatic cleanup
# =============================================================================

from library.utils.logging_utilities import create_script_logger


def context_manager_example():
    """Example using context manager for automatic resource cleanup."""

    with create_script_logger(
        log_directory="./temp_logs",
        enable_file_logging=True,
        enable_console_logging=True,
    ) as logger:

        logger.log_script_start("Temporary analysis")

        try:
            logger.info("Doing some analysis...")
            # Your code here
            logger.info("Analysis successful")

        except Exception as e:
            logger.log_script_error(e)
            raise

    # Logger is automatically closed when exiting the with block


# =============================================================================
# Example 7: Configuration file usage
# =============================================================================

import json
from library.utils.logging_utilities import LoggingConfig, QPBLogger


def config_file_example():
    """Example using JSON configuration file."""

    # Create configuration file (save as logging_config.json)
    config_dict = {
        "enable_file_logging": True,
        "file_log_level": "DEBUG",
        "log_directory": "./logs",
        "log_filename": "from_config.log",
        "enable_console_logging": True,
        "console_log_level": "INFO",
        "file_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console_format": "[%(levelname)s] %(message)s",
        "wrap_width": 100,
    }

    # Save config to file
    with open("logging_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Load and use config
    config = LoggingConfig.from_file("logging_config.json")
    logger = QPBLogger(config, name="config_driven")

    logger.info("Logger configured from file")


# =============================================================================
# Example 8: Minimal logging for quick scripts
# =============================================================================

from library.utils.logging_utilities import create_minimal_logger


def minimal_logging_example():
    """Example for simple scripts that just need basic console
    output."""

    logger = create_minimal_logger("quick_script")

    logger.info("Starting quick analysis")
    logger.warning("Something to note")
    logger.error("An error occurred")

    # This just prints to console with minimal formatting: Starting
    # quick analysis Something to note
    # An error occurred


# =============================================================================
# Example 9: Multi-stage processing with different log levels
# =============================================================================


def multi_stage_processing():
    """Example showing different logging approaches for different
    processing stages."""

    logger = create_script_logger(
        log_directory="./processing_logs",
        enable_file_logging=True,
        enable_console_logging=True,
    )

    logger.log_script_start("Multi-stage processing")

    # Stage 1: Data loading (verbose logging)
    logger.set_console_level(LogLevel.DEBUG)
    logger.info("Stage 1: Loading data")
    logger.debug("Reading configuration files")
    logger.debug("Validating input parameters")
    logger.debug("Opening HDF5 files")

    # Stage 2: Processing (less verbose)
    logger.set_console_level(LogLevel.INFO)
    logger.info("Stage 2: Processing data")
    logger.debug("This debug info only goes to file now")
    logger.info("Processing 1000 configurations")

    # Stage 3: Critical results (minimal console output)
    logger.set_console_level(LogLevel.WARNING)
    logger.info("Stage 3: Generating results")
    logger.debug("Detailed calculations")
    logger.info("Results saved to output.h5")
    logger.warning("Review the following results carefully")

    logger.log_script_end()


if __name__ == "__main__":
    # Run examples
    print("Running logging examples...")

    # Uncomment to test different examples: migrate_existing_script()
    # enhanced_script_logging() jupyter_notebook_example()
    # config_driven_logging() context_manager_example()
    # minimal_logging_example() multi_stage_processing()
