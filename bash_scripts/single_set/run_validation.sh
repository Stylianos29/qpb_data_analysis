#!/bin/bash

################################################################################
# run_validation.sh - Validate raw QPB data files in a data set
#
# DESCRIPTION:
# Validates the integrity and naming conventions of raw QPB data files (.txt
# log files and .dat correlator files) within a specified data file set
# directory. This script performs comprehensive checks to ensure:
#
# - All required file types are present
# - File naming conventions are followed
# - File contents are valid and parseable
# - No duplicate or conflicting files exist
# - Files are accessible and readable
#
# The validation is performed by calling the Python validation utility
# (validate_qpb_data_files.py) which implements the detailed validation logic.
#
# VALIDATION CHECKS:
# - File existence and accessibility
# - File naming pattern compliance
# - File content structure validation
# - Duplicate file detection
# - Consistency checks across files
#
# OUTPUT:
# - Console output with validation results
# - Detailed log file in auxiliary directory
# - Exit code: 0 (success) or 1 (validation failed)
#
# USAGE:
#   ./run_validation.sh -i <data_files_set_directory> [options]
#
# For detailed usage information, run with -h or --help
#
################################################################################

# =============================================================================
# SECTION 1: BASIC PATH SETUP (Minimal, no side effects)
# =============================================================================

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# =============================================================================
# SECTION 2: LIBRARY PATH SETUP AND SOURCING
# =============================================================================

# Setup library path (allows override via environment variable)
if [[ -z "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]]; then
    LIBRARY_SCRIPTS_DIRECTORY_PATH="$(realpath "${SCRIPT_DIR}/../library")"
fi

# Validate library directory exists
if [[ ! -d "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]]; then
    echo "ERROR: Library scripts directory not found: $LIBRARY_SCRIPTS_DIRECTORY_PATH" >&2
    exit 1
fi

# Source all library functions
for library_script in "${LIBRARY_SCRIPTS_DIRECTORY_PATH}"/*.sh; do
    [[ -f "$library_script" ]] && source "$library_script"
done

# =============================================================================
# SECTION 3: CONSTANTS AND CONFIGURATION
# =============================================================================

# Python scripts directory
if [[ -z "$PYTHON_SCRIPTS_DIRECTORY" ]]; then
    PYTHON_SCRIPTS_DIRECTORY="$(realpath "${SCRIPT_DIR}/../../core/src")"
fi

# Python validation script
VALIDATION_SCRIPT="${PYTHON_SCRIPTS_DIRECTORY}/utils/validate_qpb_data_files.py"

# Script log filename
SCRIPT_LOG_FILENAME="${SCRIPT_NAME%.sh}.log"

# Auxiliary directory structure
AUXILIARY_DIR_NAME="auxiliary"

# =============================================================================
# SECTION 4: FUNCTION DEFINITIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME -i <data_files_set_directory> [options]

DESCRIPTION:
  Validates raw QPB data files in a specified data file set directory.
  Performs comprehensive checks on file naming, structure, and content.

REQUIRED ARGUMENTS:
  -i, --input_directory        Path to data file set directory to validate

OPTIONAL ARGUMENTS:
  -aux_dir, --auxiliary_directory    
                               Directory for log files (default: auto-detected)
  -log_name, --log_filename    Custom log filename (default: ${SCRIPT_LOG_FILENAME})
  --disable-cache              Disable timestamp-based caching (force validation)
  -h, --help                   Display this help message

VALIDATION PROCESS:
  1. Verifies data file set directory exists and is accessible
  2. Checks for presence of .txt log files (required)
  3. Validates file naming conventions
  4. Checks file contents for structural integrity
  5. Detects duplicate or conflicting files
  6. Generates detailed validation log

OUTPUT:
  - Console: Validation summary and results
  - Log file: Detailed validation report in auxiliary directory
  - Exit code: 0 (valid), 1 (validation failed or error)

AUXILIARY DIRECTORY:
  If not specified, the auxiliary directory is auto-detected:
  - If data set is under data_files/raw/, mirrors to data_files/processed/
  - Otherwise, uses parent directory of data file set

EXAMPLES:
  # Validate data set with auto-detected paths
  $SCRIPT_NAME -i ../data_files/raw/invert/my_experiment/

  # Validate with custom auxiliary directory
  $SCRIPT_NAME -i ../raw/experiment1/ -aux_dir ../logs/

  # Validate with custom log filename
  $SCRIPT_NAME -i ../raw/experiment1/ -log_name my_validation.log

  # Force validation (ignore cache)
  $SCRIPT_NAME -i ../raw/experiment1/ --disable-cache

NOTES:
  - Requires Python environment with qpb_data_analysis package
  - Validation script path: ${VALIDATION_SCRIPT}
  - Uses library functions from: ${LIBRARY_SCRIPTS_DIRECTORY_PATH}

EOF
    exit 0
}

function validate_prerequisites() {
    # Validate that all required prerequisites are available
    #
    # Checks for:
    # - Python scripts directory existence
    # - Python validation script existence
    #
    # Returns:
    #   0 - All prerequisites valid
    #   1 - Validation failed
    
    echo "Validating prerequisites..."
    
    # Check Python scripts directory
    if [[ ! -d "$PYTHON_SCRIPTS_DIRECTORY" ]]; then
        echo "ERROR: Python scripts directory not found: $PYTHON_SCRIPTS_DIRECTORY" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_error "Python scripts directory not found: $PYTHON_SCRIPTS_DIRECTORY"
        return 1
    fi
    echo "  ✓ Python scripts directory validated"
    
    # Check validation script exists
    if [[ ! -f "$VALIDATION_SCRIPT" ]]; then
        echo "ERROR: Python validation script not found: $VALIDATION_SCRIPT" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_error "Python validation script not found: $VALIDATION_SCRIPT"
        return 1
    fi
    echo "  ✓ Python validation script found"
    
    [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_info "Prerequisites validated successfully"
    return 0
}

function detect_auxiliary_directory() {
    # Auto-detect auxiliary directory based on data set directory location
    #
    # Logic:
    # - If data set is under data_files/raw/, mirrors to data_files/processed/
    # - Otherwise, uses parent directory of data file set
    #
    # Arguments:
    #   $1 - data_set_directory : Path to data file set directory
    #
    # Returns:
    #   Prints auxiliary directory path to stdout
    
    local data_set_dir="$1"
    local auxiliary_dir
    
    # Check if data set is under data_files/raw/
    if [[ "$data_set_dir" == *"/data_files/raw/"* ]]; then
        # Mirror structure in processed/
        local processed_dir="${data_set_dir/\/raw\//\/processed\/}"
        auxiliary_dir="${processed_dir}/${AUXILIARY_DIR_NAME}"
    else
        # Use parent directory
        auxiliary_dir="$(dirname "$data_set_dir")"
    fi
    
    echo "$auxiliary_dir"
}

function run_validation() {
    # Execute Python validation script on the data file set
    #
    # Arguments:
    #   $1 - data_set_directory      : Path to data file set
    #   $2 - auxiliary_directory     : Path to auxiliary files directory
    #
    # Returns:
    #   0 - Validation successful
    #   1 - Validation failed
    
    local data_set_dir="$1"
    local auxiliary_dir="$2"
    
    echo ""
    echo "Executing validation script..."
    
    log_info "Starting validation of data file set: $(basename "$data_set_dir")"
    log_info "Validation script: $VALIDATION_SCRIPT"
    
    # Build Python command
    local python_cmd="python \"$VALIDATION_SCRIPT\""
    python_cmd+=" --raw_data_files_set_directory_path \"$data_set_dir\""
    python_cmd+=" --enable_logging"
    python_cmd+=" --auxiliary_files_directory \"$auxiliary_dir\""
    
    log_info "Executing command: $python_cmd"
    
    # Execute validation
    if eval "$python_cmd"; then
        echo "  ✓ Validation completed successfully"
        log_info "Data file set validation successful"
        return 0
    else
        echo "ERROR: Validation failed" >&2
        log_error "Data file set validation failed"
        return 1
    fi
}

# =============================================================================
# SECTION 5: ARGUMENT PARSING
# =============================================================================

# Initialize variables with defaults
input_directory=""
auxiliary_directory=""
log_filename=""
enable_cache=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input_directory)
            input_directory="$2"
            shift 2
            ;;
        -aux_dir|--auxiliary_directory)
            auxiliary_directory="$2"
            shift 2
            ;;
        -log_name|--log_filename)
            log_filename="$2"
            shift 2
            ;;
        --disable-cache)
            enable_cache=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            echo "Use -h or --help for usage information" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# SECTION 6: HELP CHECK (Before any logging setup!)
# =============================================================================

# Note: Help is already handled in argument parsing above
# This section is here for clarity in the structure

# =============================================================================
# SECTION 7: INPUT VALIDATION AND LOGGING SETUP
# =============================================================================

# Validate required input
if [[ -z "$input_directory" ]]; then
    echo "ERROR: Input directory not specified. Use -i <directory>" >&2
    echo "Use -h or --help for usage information" >&2
    exit 1
fi

# Validate input directory exists
if [[ ! -d "$input_directory" ]]; then
    echo "ERROR: Input directory does not exist: $input_directory" >&2
    exit 1
fi

# Convert to absolute path
input_directory="$(realpath "$input_directory")"
input_dir_name="$(basename "$input_directory")"

# Auto-detect auxiliary directory if not specified
if [[ -z "$auxiliary_directory" ]]; then
    auxiliary_directory=$(detect_auxiliary_directory "$input_directory")
    echo "INFO: Auto-detected auxiliary directory: $(get_display_path "$auxiliary_directory")"
fi

# Ensure auxiliary directory exists
if [[ ! -d "$auxiliary_directory" ]]; then
    mkdir -p "$auxiliary_directory" || {
        echo "ERROR: Failed to create auxiliary directory: $auxiliary_directory" >&2
        exit 1
    }
    echo "INFO: Created auxiliary directory: $(get_display_path "$auxiliary_directory")"
fi
auxiliary_directory="$(realpath "$auxiliary_directory")"

# Set default log filename if not provided
if [[ -z "$log_filename" ]]; then
    log_filename="$SCRIPT_LOG_FILENAME"
else
    # Ensure .log extension
    if [[ "$log_filename" != *.log ]]; then
        log_filename="${log_filename}.log"
    fi
fi

# Setup logging infrastructure
export SCRIPT_TERMINATION_MESSAGE="\n\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION TERMINATED"

# Initialize logging
SCRIPT_LOG_FILE_PATH="${auxiliary_directory}/${log_filename}"
export SCRIPT_LOG_FILE_PATH

echo -e "\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION INITIATED\n" > "$SCRIPT_LOG_FILE_PATH"

log_info "=== RAW DATA FILES VALIDATION ==="
log_info "Script: $SCRIPT_NAME"
log_info "Data file set: $input_dir_name"
log_info "Input directory: $input_directory"
log_info "Auxiliary directory: $auxiliary_directory"
log_info "Log file: $SCRIPT_LOG_FILE_PATH"
log_info "Cache enabled: $enable_cache"

# =============================================================================
# SECTION 8: MAIN EXECUTION
# =============================================================================

# Display banner
echo ""
echo "==================================================================="
echo "   QPB DATA FILES VALIDATION"
echo "==================================================================="
echo "Data file set: $input_dir_name"
echo "Input:      $(get_display_path "$input_directory")"
echo "Auxiliary:  $(get_display_path "$auxiliary_directory")"
echo "Log file:   $(get_display_path "$SCRIPT_LOG_FILE_PATH")"
echo "==================================================================="

# Validate prerequisites
echo ""
echo "=== VALIDATING PREREQUISITES ==="
if ! validate_prerequisites; then
    echo "ERROR: Prerequisites validation failed" >&2
    log_error "Prerequisites validation failed"
    exit 1
fi

# Check for data files
echo ""
echo "=== CHECKING FOR DATA FILES ==="
if ! find "$input_directory" -maxdepth 1 -type f -name "*.txt" -print -quit | grep -q .; then
    echo "ERROR: No .txt log files found in data file set directory" >&2
    log_error "No .txt log files found in: $input_directory"
    exit 1
fi
echo "  ✓ Found .txt log files"
log_info "Data files detected in directory"

# Check for correlator files (optional)
if find "$input_directory" -maxdepth 1 -type f -name "*.dat" -print -quit | grep -q .; then
    echo "  ✓ Found .dat correlator files"
    log_info "Correlator files detected"
else
    echo "  ℹ No .dat correlator files found (optional)"
    log_info "No correlator files detected"
fi

# Execute validation
if ! run_validation "$input_directory" "$auxiliary_directory"; then
    echo ""
    echo "==================================================================="
    echo "   VALIDATION FAILED"
    echo "==================================================================="
    echo "Data file set: $input_dir_name"
    echo "Check log for details: $(get_display_path "$SCRIPT_LOG_FILE_PATH")"
    echo "==================================================================="
    
    log_error "=== VALIDATION FAILED ==="
    echo -e "$SCRIPT_TERMINATION_MESSAGE" >> "$SCRIPT_LOG_FILE_PATH"
    
    exit 1
fi

# Success
echo ""
echo "==================================================================="
echo "   VALIDATION SUCCESSFUL"
echo "==================================================================="
echo "Data file set: $input_dir_name"
echo "All data files validated successfully"
echo ""
echo "Output:"
echo "  Log file: $(get_display_path "$SCRIPT_LOG_FILE_PATH")"
echo "==================================================================="

log_info "=== VALIDATION COMPLETED SUCCESSFULLY ==="
log_info "All QPB data files validated successfully"
log_info "Data file set: $input_dir_name"

echo -e "$SCRIPT_TERMINATION_MESSAGE" >> "$SCRIPT_LOG_FILE_PATH"

exit 0
