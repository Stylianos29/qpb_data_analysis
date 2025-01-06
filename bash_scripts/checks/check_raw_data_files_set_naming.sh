#!/bin/bash

################################################################################
# validate_raw_data_files_set.sh
#
# Description: 
#
# Purpose:
#
# Usage:
#
# Flags:
#
# Note:
################################################################################

# CUSTOM FUNCTIONS DEFINITIONS

usage() {
    # Function to display usage information

    echo "Usage: $0 -p <data_files_set_directory>"
    echo "  -p, --path   Specify the directory containing raw files"
    exit 1
}

# ENVIRONMENT VARIABLES

CURRENT_SCRIPT_FULL_PATH=$(realpath "$0")
# Extract the current script's name from its full path
CURRENT_SCRIPT_NAME="$(basename "$CURRENT_SCRIPT_FULL_PATH")"
# Extract the current script's parent directory from its full path
CURRENT_SCRIPT_DIRECTORY="$(dirname "$CURRENT_SCRIPT_FULL_PATH")"
# Replace ".sh" with "_script.log" to create the log file name
SCRIPT_LOG_FILE_NAME=$(echo "$CURRENT_SCRIPT_NAME" | sed 's/\.sh$/_script.log/')
# Construct full path of library scripts directory if not set yet
if [ -z "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]; then
    LIBRARY_SCRIPTS_DIRECTORY_PATH=$(\
                            realpath "${CURRENT_SCRIPT_DIRECTORY}/../library")
    [[ ! -d "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]] \
                            && echo "Invalid library scripts path." && exit 1
fi

NON_INVERT_LOG_FILES_SUCCESS_FLAG="per stochastic source"
INVERT_LOG_FILES_SUCCESS_FLAG="CG done"
ERROR_FILES_FAILURE_FLAG="terminated"
NON_NUMERICAL_FAILURE_FLAG="= nan|= inf"

# Export script termination message to be used for finalizing logging
export SCRIPT_TERMINATION_MESSAGE="\n\t\t"$(echo "$CURRENT_SCRIPT_NAME" \
                    | tr '[:lower:]' '[:upper:]')" SCRIPT EXECUTION TERMINATED"

# SOURCE DEPENDENCIES

# Source all library scripts from "bash_scripts/library" using a loop avoiding
# this way name-specific sourcing and thus potential typos
for library_script in "${LIBRARY_SCRIPTS_DIRECTORY_PATH}"/*.sh;
do
    # Check if the current file in the loop is a regular file
    if [ -f "$library_script" ]; then
        source "$library_script"
    fi
done

# PARSE INPUT ARGUMENTS

data_files_set_directory=""
script_log_file_directory=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            data_files_set_directory="$2"
            shift 2
            ;;
        -l|--log)
            script_log_file_directory="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            ;;
    esac
done

# VALIDATE INPUT

# Ensure a data files set directory path is provided
if [ -z "$data_files_set_directory" ]; then
    echo "ERROR: No data files set directory path specified."
    usage
fi
# Verify the data files set directory exists
check_directory_exists "$data_files_set_directory"
data_files_set_directory_name=$(basename $data_files_set_directory)

# Check if a log directory is provided
if [ -z "$script_log_file_directory" ]; then
    # if not, then set it to the parent of the data files set directory
    script_log_file_directory=$(dirname $data_files_set_directory)
else
    # if it was provided, then check if it exists
    check_directory_exists "$script_log_file_directory"
fi

# INITIATE LOGGING

# Export log file path as a global variable to be used by custom functions
SCRIPT_LOG_FILE_PATH="${script_log_file_directory}/${SCRIPT_LOG_FILE_NAME}"
export SCRIPT_LOG_FILE_PATH

# Create or override a log file. Initiate logging
echo -e "\t\t"$(echo "$CURRENT_SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') \
                "SCRIPT EXECUTION INITIATED\n" > "$SCRIPT_LOG_FILE_PATH"

log_message="Initiate inspecting '${data_files_set_directory_name}' "
log_message+="data files set directory."
log "INFO" "$log_message"

# COUNT THE NUMBERS OF DATA FILES BY TYPE

# Create lists of filenames per file type
list_of_qpb_error_file_paths=($(\
                    find "$data_files_set_directory" -type f -name "*.err"))
list_of_qpb_log_file_paths=($(\
                    find "$data_files_set_directory" -type f -name "*.txt"))
list_of_qpb_correlators_file_paths=($(\
                    find "$data_files_set_directory" -type f -name "*.dat"))

# Check presence of at least one log file; exit if not
number_of_qpb_log_files=${#list_of_qpb_log_file_paths[@]}
if [[ $number_of_qpb_log_files -eq 0 ]]; then
    error_message="No qpb log files found in '$data_files_set_directory'. "
    error_message+="No validation process of the data files set can be "
    error_message+="performed without qpb log files present."
    termination_output "$error_message"
    exit 1
fi

# Check if any error files exist
error_files_present=false
number_of_qpb_error_files=${#list_of_qpb_error_file_paths[@]}
[[ ${number_of_qpb_error_files} -gt 0 ]] && error_files_present=true

# Check if any correlators files exist
correlators_files_present=false
number_of_qpb_correlators_files=${#list_of_qpb_correlators_file_paths[@]}
[[ ${number_of_qpb_correlators_files} -gt 0 ]] && correlators_files_present=true

# Report the number of data files in the directory
output_message="Directory contains ${number_of_qpb_log_files} qpb log files"
if [[ $error_files_present == true ]]; then
    output_message+=", ${number_of_qpb_error_files} qpb error files"
else
    output_message+=", no qpb error files"
fi
if [[ $correlators_files_present == true ]]; then
    output_message+=", and ${number_of_qpb_correlators_files} qpb correlators "
    output_message+="files."
else
    output_message+=", and no qpb correlators files."
fi
log "INFO" "$output_message"
echo -e "++ $output_message"

# IDENTIFY QPB MAIN PROGRAM TYPE: INVERT OR NON-INVERT

# Check if the majority of qpb log files contain one of the two "success" flags:
# "non-invert" or "invert." The remaining files are expected not to contain any
# success flags. If both types of flags are found, it indicates an inconsistency
# that must be resolved.
while true; do
    # Identify "non-invert" qpb log files based on the success flag.
    list_of_non_invert_qpb_log_files=()
    find_matching_qpb_log_files list_of_qpb_log_file_paths \
        "$NON_INVERT_LOG_FILES_SUCCESS_FLAG" "include" \
                                                list_of_non_invert_qpb_log_files
    number_of_non_invert_qpb_log_files=${#list_of_non_invert_qpb_log_files[@]}

    # Identify "invert" qpb log files based on the success flag.
    list_of_invert_qpb_log_files=()
    find_matching_qpb_log_files list_of_qpb_log_file_paths \
        "$INVERT_LOG_FILES_SUCCESS_FLAG" "include" list_of_invert_qpb_log_files
    number_of_invert_qpb_log_files=${#list_of_invert_qpb_log_files[@]}

    # Check for no valid qpb log file types in the directory at all.
    if [[ $number_of_non_invert_qpb_log_files -eq 0 && \
                                $number_of_invert_qpb_log_files -eq 0 ]]; then
        error_message="All data files were found to be corrupted. Validation "
        error_message+="process will stop immediately! It is recommended that "
        error_message+="new data files are provided for this data files set."
        termination_output "$error_message"
        exit 1
    # Check for conflicting qpb log file types in the directory.
    elif [[ $number_of_non_invert_qpb_log_files -gt 0 && \
                                $number_of_invert_qpb_log_files -gt 0 ]]; then
        # Warn the user and request resolution of the inconsistency.
        input_request_message="The directory contains qpb log files of both "
        input_request_message+="'invert' and 'non-invert' type! This is "
        input_request_message+="inconsistent and the validation process cannot "
        input_request_message+="proceed unless the situation is resolved. "
        log "WARNING" "$input_request_message"
        minority_type=""
        if [[ $number_of_non_invert_qpb_log_files -lt \
                                        $number_of_invert_qpb_log_files ]]; then
            minority_type="non-"
            list_of_minority_qpb_log_files=( \
                                    "${list_of_non_invert_qpb_log_files[@]}")
        else
            list_of_minority_qpb_log_files=( \
                                        "${list_of_invert_qpb_log_files[@]}")
        fi
        input_request_message+="Would you like to like to delete all the "
        input_request_message+="${minority_type}invert qpb data files? "
        input_request_message+="([yY]/nN) "
        
        while true; do
            read -p "++ $input_request_message" user_response
            # Convert the response to lowercase for easier comparison
            user_response=$(echo "$user_response" | tr '[:upper:]' '[:lower:]')
            # Handle the response, treating "Enter" as "yes"
            if [[ -z "$user_response" || "$user_response" == "y" || \
                                            "$user_response" == "yes" ]]; then
                # Delete all minority qpb log files along with any corresponding
                # error and correlators files
                for minority_qpb_log_file in \
                                    "${list_of_minority_qpb_log_files[@]}"; do
                    rm $minority_qpb_log_file
                    rm -f "${minority_qpb_log_file%.txt}.err"
                    rm -f "${minority_qpb_log_file%.txt}.dat"
                done
                log_message="All ${majority_type}invert qpb data files were "
                log_message+="deleted."
                log "INFO" "$log_message"
                echo -e "$log_message"
                
                # TODO: DRY this!!
                # Update lists of filenames per file type
                list_of_qpb_error_file_paths=($(\
                                    find "$data_files_set_directory" -type f -name "*.err"))
                list_of_qpb_log_file_paths=($(\
                                    find "$data_files_set_directory" -type f -name "*.txt"))
                list_of_qpb_correlators_file_paths=($(\
                                    find "$data_files_set_directory" -type f -name "*.dat"))
                # Update stored count of filenames per file type
                number_of_qpb_log_files=${#list_of_qpb_log_file_paths[@]}
                number_of_qpb_error_files=${#list_of_qpb_error_file_paths[@]}
                number_of_qpb_correlators_files=${#list_of_qpb_correlators_file_paths[@]}
                [[ ${number_of_qpb_error_files} -gt 0 ]] && error_files_present=true
                [[ ${number_of_qpb_correlators_files} -gt 0 ]] && correlators_files_present=true
                
                break
            elif [[ "$user_response" == "n" || "$user_response" == "no" ]]; then
                error_message="A data files set directory cannot contain qpb "
                error_message+="data files of both 'invert' and 'non-invert' "
                error_message+="type. It is recommended to separate data files "
                error_message+="of different types into distinct directories."
                termination_output "$error_message"
                exit 1
            else
                error_message="Invalid response. Please answer with 'y', "
                error_message+="'Y', 'yes', 'n', 'N', or 'no'."
                echo $error_message
            fi
        done
    else
        # No inconsistencies detected; break the loop.
        break  
    fi
done

# Conclusively determine the qpb main program type.
if [[ $number_of_non_invert_qpb_log_files -eq 0 ]]; then
    is_invert=true
    log_message="Data files were generated by an 'invert' qpb main program."
    log "INFO" "$log_message"
else
    is_invert=false
fi

# Warn if "invert" is found in the relative path of a non-invert data files set
# directory.
if [[ $is_invert == false && \
                    "${data_files_set_directory#*/raw/}" == *"invert"* ]]; then
    warning_message="The substring 'invert' appears in the path of a "
    warning_message+="non-invert data files set directory. This could be "
    warning_message+="misleading. Consider renaming the directory."
    log "WARNING" "$warning_message"
    echo "++ WARNING: ${warning_message}"
fi

# REMOVE INCONSISTENT CORRELATORS FILES

# Preemptively check for any qpb correlators files in a non-invert data files
# set directory. If such files are found, delete them along with their
# corresponding log and error files. This ensures consistency between the data
# files and the identified program type.
if [[ $is_invert == false && $correlators_files_present == true ]]; then

    # Ask user whether to delete inconsistent correlators files
    input_request_message="Correlators files have been detected in a non-invert"
    input_request_message+=" data files set directory. This is inconsistent "
    input_request_message+="with the program type. "
    log "WARNING" "$input_request_message"
    input_request_message+="Would you like to delete all correlators files and "
    input_request_message+="their associated log and error files? ([yY]/nN) " 
    while true; do
        read -p "++ $input_request_message" user_response
        # Convert the response to lowercase for easier comparison
        user_response=$(echo "$user_response" | tr '[:upper:]' '[:lower:]')
        # Handle the response, treating "Enter" as "yes"
        if [[ -z "$user_response" || "$user_response" == "y" || \
                                            "$user_response" == "yes" ]]; then
            # Remove correlators files and corresponding log and error files
            for qpb_correlators_file_path in \
                                "${list_of_qpb_correlators_file_paths[@]}"; do
                rm $qpb_correlators_file_path
                rm -f "${qpb_correlators_file_path%.dat}.txt"
                rm -f "${qpb_correlators_file_path%.dat}.err"
            done
            correlators_files_present=false
            log_message="All correlators files and their associated files "
            log_message+="were successfully deleted."
            log "INFO" "$log_message"
            echo -e "$log_message"
            break
        elif [[ "$user_response" == "n" || "$user_response" == "no" ]];
        then
            # Inform the user about the consequences of retaining the files
            error_message="Correlators files cannot coexist in a non-invert "
            error_message+="data files set directory. It is strongly "
            error_message+="recommended that all such files be removed to "
            error_message+="maintain consistency."
            termination_output "$error_message"
            exit 1
        else
            error_message="Invalid response. Please answer with 'y', "
            error_message+="'Y', 'yes', 'n', 'N', or 'no'."
            echo $error_message
        fi
    done
fi

# DELETE CORRUPTED QPB LOG FILES

if [[ $is_invert == false ]]; then
    list_of_valid_qpb_log_files=("${list_of_non_invert_qpb_log_files[@]}")
else
    list_of_valid_qpb_log_files=("${list_of_invert_qpb_log_files[@]}")
fi
number_of_valid_qpb_log_files=${#list_of_valid_qpb_log_files[@]}

if [[ $number_of_valid_qpb_log_files -eq $number_of_qpb_log_files ]]; then
    output_message="No corrupted qpb log files detected!"
    log "INFO" "$output_message"
    echo -e "++ $output_message"
elif [[ $number_of_valid_qpb_log_files -gt $number_of_qpb_log_files ]]; then
    error_message="For unknown reasons the number of valid qpb files is larger "
    error_message+="than the the number of qpb files itself! This is "
    error_message+="inconsistent and need to be investigated."
    termination_output "$error_message"
    exit 1
elif [[ $number_of_valid_qpb_log_files -lt $number_of_qpb_log_files ]]; then
    # List all corrupted qpb log files by excluding the valid ones
    list_of_corrupted_qpb_log_files=()
    for qpb_log_file in "${list_of_qpb_log_file_paths[@]}"; do
        if [[ ! " ${list_of_valid_qpb_log_files[*]} " =~ " ${qpb_log_file} " ]];
        then
            list_of_corrupted_qpb_log_files+=("$qpb_log_file")
        fi
    done
    # TODO: Use the "find_matching_qpb_log_files()" function
    # find_matching_qpb_log_files list_of_qpb_log_file_paths \
    #     "$NON_INVERT_LOG_FILES_SUCCESS_FLAG" "exclude" \
    #                                         list_of_corrupted_qpb_log_files
    number_of_corrupted_qpb_files=${#list_of_corrupted_qpb_log_files[@]}
    input_request_message="There are a total of ${number_of_corrupted_qpb_files} "
    input_request_message+="corrupted qpb log files in the directory. "
    log "WARNING" "$input_request_message"
    input_request_message+="Would you like to like to delete all corrupted qpb "
    input_request_message+="log files? ([yY]/nN) "
    while true; do
        read -p "++ $input_request_message" user_response
        # Convert the response to lowercase for easier comparison
        user_response=$(echo "$user_response" | tr '[:upper:]' '[:lower:]')
        # Handle the response, treating "Enter" as "yes"
        if [[ -z "$user_response" || "$user_response" == "y" || \
                                            "$user_response" == "yes" ]]; then
            # Remove all corrupted qpb log files and their corresponding error
            # and correlators files
            for corrupted_qpb_log_file in \
                                "${list_of_corrupted_qpb_log_files[@]}"; do
                rm $corrupted_qpb_log_file
                rm -f "${corrupted_qpb_log_file%.txt}.err"
                rm -f "${corrupted_qpb_log_file%.txt}.dat"
            done
            warning_message="All all corrupted qpb log files were deleted."
            log "WARNING" "$warning_message"
            echo -e "$warning_message"
            break
        elif [[ "$user_response" == "n" || "$user_response" == "no" ]];
        then
            warning_message="Validation process cannot continue with corrupted "
            warning_message+="qpb log files present in the directory. It is "
            warning_message+="highly recommended that they are removed."
            log "WARNING" "$warning_message"
            break
        else
            error_message="Invalid response. Please answer with 'y', "
            error_message+="'Y', 'yes', 'n', 'N', or 'no'."
            echo $error_message
        fi
    done
fi

# DELETE QPB LOG FILES CONTAINING NONNUMERICAL VALUES RESULTS

list_of_faulty_qpb_log_files=()
find_matching_qpb_log_files list_of_qpb_log_file_paths \
    "$NON_NUMERICAL_FAILURE_FLAG" "include" list_of_faulty_qpb_log_files

number_of_faulty_qpb_files=${#list_of_faulty_qpb_log_files[@]}
# If there are any faulty qpb log files then it is recommended they are deleted
if [[ ${number_of_faulty_qpb_files} -gt 0 ]]; then
    input_request_message="There are a total of ${number_of_faulty_qpb_files} "
    input_request_message+="faulty qpb log files in the directory containing "
    input_request_message+="nonnumerical values as calculated results. "
    log "WARNING" "$input_request_message"
    input_request_message+="Would you like to delete all these files? ([yY]/nN) "
    while true; do
        read -p "++ $input_request_message" user_response
        # Convert the response to lowercase for easier comparison
        user_response=$(echo "$user_response" | tr '[:upper:]' '[:lower:]')
        # Handle the response, treating "Enter" as "yes"
        if [[ -z "$user_response" || "$user_response" == "y" || \
                                            "$user_response" == "yes" ]]; then
            # Remove all faulty qpb log files and their corresponding error
            # and correlators file 
            for faulty_qpb_log_file in \
                                    "${list_of_faulty_qpb_log_files[@]}"; do
                rm $faulty_qpb_log_file
                rm -f "${faulty_qpb_log_file%.txt}.err"
                rm -f "${faulty_qpb_log_file%.txt}.dat"
            done
            warning_message="All all faulty qpb log files were deleted."
            log "WARNING" "$warning_message"
            echo -e "$warning_message"
            break
        elif [[ "$user_response" == "n" || "$user_response" == "no" ]];
        then
            warning_message="Please take into consideration that the directory "
            warning_message+="contains faulty qpb log files. It is recommended "
            warning_message+="that they are removed."
            log "WARNING" "$warning_message"
            break
        else
            error_message="Invalid response. Please answer with 'y', "
            error_message+="'Y', 'yes', 'n', 'N', or 'no'."
            echo $error_message
        fi
    done
fi

# TODO: Log lists of deleted corrupted and faulty qpb files.

# number_of_corrupted_qpb_log_files=${#list_of_corrupted_qpb_log_files[@]}
# if [[ $number_of_corrupted_qpb_log_files -eq 0 ]]; then
#     output_message="No corrupted or faulty qpb log files found!"
#     log "INFO" "$output_message"
#     echo -e "++ $output_message"
# else
#     warning_message="A total of ${number_of_corrupted_qpb_log_files} corrupted "
#     warning_message+="or faulty qpb log files found."
#     log "WARNING" "$warning_message"
#     echo "++ WARNING: ${warning_message}"

#     list_of_corrupted_qpb_error_files=()
#     list_of_corrupted_qpb_correlators_files=()

#     # Construct a detailed report of corrupted qpb log files
#     detailed_report_message="Detailed report of corrupted or faulty "
#     detailed_report_message+="data files:\n"
#     for corrupted_qpb_log_file in "${list_of_corrupted_qpb_log_files[@]}"; do
#         detailed_report_message+="-- $(basename $corrupted_qpb_log_file)\n"
#         # Corroborate corruption of log file using error file
#         if $error_files_present; then
#             qpb_error_file="${corrupted_qpb_log_file%.txt}.err"
#             list_of_corrupted_qpb_error_files+=("$qpb_error_file")
#             qpb_error_filename=$(basename $qpb_error_file)
#             # Check if the corresponding error file exists
#             if [[ ! -e "$qpb_error_file" ]]; then
#                 detailed_report_message+="  * No corresponding "
#                 detailed_report_message+="'$qpb_error_filename' error file "
#                 detailed_report_message+="found.\n"
#             fi
#             # Look for the qpb error failure flag inside the error file
#             if grep -iqE "terminated|failed" "$qpb_error_file"; then
#                 detailed_report_message+="  * Error file '$qpb_error_filename'"
#                 detailed_report_message+=" confirms log file corruption.\n"
#             fi
#         fi
#         # Corroborate corruption of log file using correlators file
#         if [[ $is_invert == true ]]; then
#             qpb_correlators_file="${corrupted_qpb_log_file%.txt}.dat"
#             list_of_corrupted_qpb_correlators_files+=("$qpb_correlators_file")
#             qpb_correlators_filename=$(basename $qpb_correlators_file)
#             # Check if the corresponding error file exists
#             if [[ ! -e "$qpb_correlators_file" ]]; then
#                 detailed_report_message+="  * No corresponding "
#                 detailed_report_message+="'$qpb_correlators_filename'"
#                 detailed_report_message+=" correlators file found.\n"
#             elif [[ ! -s "$qpb_correlators_file" ]]; then
#                 detailed_report_message+="  * Empty '$qpb_correlators_filename'"
#                 detailed_report_message+=" correlators file confirms "
#                 detailed_report_message+="log file corruption.\n"
#             fi
#         fi
#     done
#     log "INFO" "$detailed_report_message"

#     # Ask user for the detailed report to be printed on terminal
#     input_request_message="Would you like a detailed report of the corrupted "
#     input_request_message+="data files printed on terminal? ([yY]/nN) "
#     while true; do
#         read -p "++ $input_request_message" user_response
#         # Convert the response to lowercase for easier comparison
#         user_response=$(echo "$user_response" | tr '[:upper:]' '[:lower:]')
#         # Handle the response, treating "Enter" as "yes"
#         if [[ -z "$user_response" || "$user_response" == "y" || \
#                                             "$user_response" == "yes" ]]; then
#             echo -e "$detailed_report_message"
#             break
#         elif [[ "$user_response" == "n" || "$user_response" == "no" ]]; then
#             break
#         else
#             error_message="Invalid response. Please answer with 'y', 'Y', "
#             error_message+="'yes', 'n', 'N', or 'no'."
#             echo $error_message
#         fi
#     done
    
#     # Ask user whether to delete all corrupted or faulty qpb log files and
#     # their corresponding error and correlators files, if they exist
#     input_request_message="Would you like to remove all corrupted data "
#     input_request_message+="files? ([yY]/nN) "
#     while true; do
#         read -p "++ $input_request_message" user_response
#         # Convert the response to lowercase for easier comparison
#         user_response=$(echo "$user_response" | tr '[:upper:]' '[:lower:]')
#         # Handle the response, treating "Enter" as "yes"
#         if [[ -z "$user_response" || "$user_response" == "y" || \
#                                             "$user_response" == "yes" ]]; then
#             for corrupted_qpb_log_file in \
#                                         "${list_of_corrupted_qpb_log_files[@]}";
#             do
#                 rm $corrupted_qpb_log_file
#             done
#             for corrupted_qpb_error_file in \
#                                     "${list_of_corrupted_qpb_error_files[@]}";
#             do
#                 rm $corrupted_qpb_error_file
#             done
#             for corrupted_qpb_correlators_file in \
#                                 "${list_of_corrupted_qpb_correlators_files[@]}";
#             do
#                 rm $corrupted_qpb_correlators_file
#             done
#             log_message="All corrupted or faulty data files were deleted."
#             log "INFO" "$log_message"
#             echo -e "$log_message"
#             break
#         elif [[ "$user_response" == "n" || "$user_response" == "no" ]]; then
#             error_message="Validation process cannot proceed with corrupted "
#             error_message="data files present inside the data files set "
#             error_message+="directory. It recommended that corrupted or faulty "
#             error_message+="data files are removed from the directory."
#             termination_output "$error_message"
#             exit 1
#             break
#         else
#             error_message="Invalid response. Please answer with 'y', 'Y', "
#             error_message+="'yes', 'n', 'N', or 'no'."
#             echo $error_message
#         fi
#     done
# fi




# DELETE ALL ERROR FILES

if [[ $error_files_present == true ]]; then
    # Ask user whether to delete all the rest of the error files
    input_request_message="Would you like to delete all qpb error files of this" 
    input_request_message+=" directory? ([yY]/nN) "
    while true; do
        read -p "++ $input_request_message" user_response
        # Convert the response to lowercase for easier comparison
        user_response=$(echo "$user_response" | tr '[:upper:]' '[:lower:]')
        # Handle the response, treating "Enter" as "yes"
        if [[ -z "$user_response" || "$user_response" == "y" || \
                                            "$user_response" == "yes" ]]; then
            list_of_remaining_qpb_error_file_paths=($(\
                        find "$data_files_set_directory" -type f -name "*.err"))
            # for error_file in "${list_of_remaining_qpb_error_file_paths[@]}"; do
                # rm $error_file
            # done
            rm "${list_of_remaining_qpb_error_file_paths[@]}"
            log_message="All qpb error files were deleted."
            log "INFO" "$log_message"
            echo -e "$log_message"
            break
        elif [[ "$user_response" == "n" || "$user_response" == "no" ]];
        then
            break
        else
            error_message="Invalid response. Please answer with 'y', "
            error_message+="'Y', 'yes', 'n', 'N', or 'no'."
            echo $error_message
        fi
    done
fi

# DELETE ALL EMPTY CORRELATORS FILES AND THEIR CORRESPONDING LOG AND ERROR FILES

if [[ $correlators_files_present == true ]]; then
    # Update list of all correlators file in the case some were deleted before
    list_of_remaining_qpb_correlators_file_paths=($(\
                        find "$data_files_set_directory" -type f -name "*.dat"))
    # Look for empty correlators files
    list_of_empty_qpb_correlators_file_paths=()
    for correlators_file in \
                        "${list_of_remaining_qpb_correlators_file_paths[@]}"; do
        if [[ ! -s "$correlators_file" ]]; then
            list_of_empty_qpb_correlators_file_paths+=("$correlators_file")
        fi
    done
    # If there are any empty correlators files, ask user to delete them
    number_of_empty_qpb_correlators_files=${#list_of_empty_qpb_correlators_file_paths[@]}
    if [[ ${number_of_empty_qpb_correlators_files} -gt 0 ]]; then
        input_request_message="There are a total of "
        input_request_message+="${number_of_empty_qpb_correlators_files} empty "
        input_request_message+="qpb correlators files. "
        log "WARNING" "$input_request_message"
        input_request_message+="Would you like to delete all the empty" 
        input_request_message+=" qpb correlators files and their corresponding "
        input_request_message+="log and error files? ([yY]/nN) "
        while true; do
            read -p "++ $input_request_message" user_response
            # Convert the response to lowercase for easier comparison
            user_response=$(echo "$user_response" | tr '[:upper:]' '[:lower:]')
            # Handle the response, treating "Enter" as "yes"
            if [[ -z "$user_response" || "$user_response" == "y" || \
                                            "$user_response" == "yes" ]]; then
                for correlators_file in \
                            "${list_of_empty_qpb_correlators_file_paths[@]}"; do
                    rm $correlators_file
                    # Attempt to delete corresponding data file with no output
                    # if they do not exist
                    rm -f "${correlators_file%.dat}.txt"
                    rm -f "${correlators_file%.dat}.err"
                done
                log_message="All empty qpb correlators files and their "
                log_message+="corresponding log and error files were deleted."
                log "INFO" "$log_message"
                echo -e "$log_message"
                break
            elif [[ "$user_response" == "n" || "$user_response" == "no" ]];
            then
                error_message="Validation process cannot proceed with empty "
                error_message="correlators files present inside the data files "
                error_message+="set directory. It recommended these files "
                error_message+="along with their corresponding log and error "
                error_message+="files are removed from the directory."
                termination_output "$error_message"
                exit 1
                break
            else
                error_message="Invalid response. Please answer with 'y', "
                error_message+="'Y', 'yes', 'n', 'N', or 'no'."
                echo $error_message
            fi
        done
    fi
fi

# INVESTIGATE EXACT MATCHING BETWEEN QPB LOG AND CORRELATORS FILES

if [[ $is_invert == true ]]; then
    # Extract updated lists of qpb log and correlators files
    list_of_remaining_qpb_log_files=($(
        find "$data_files_set_directory" -type f -name "*.txt"))
    list_of_remaining_qpb_correlators_files=($(
        find "$data_files_set_directory" -type f -name "*.dat"))

    # Check for unpaired qpb log files
    list_of_unpaired_qpb_log_files=()
    for qpb_log_file in "${list_of_remaining_qpb_log_files[@]}"; do
        matching_qpb_correlators_file="${qpb_log_file%.txt}.dat"
        if [ ! -e "$matching_qpb_correlators_file" ]; then
            list_of_unpaired_qpb_log_files+=("$qpb_log_file")
        fi
    done

    # Check for unpaired qpb correlators files
    list_of_unpaired_qpb_correlators_files=()
    for qpb_correlators_file in "${list_of_remaining_qpb_correlators_files[@]}";
    do
        matching_qpb_log_file="${qpb_correlators_file%.dat}.txt"
        if [ ! -e "$matching_qpb_log_file" ]; then
            list_of_unpaired_qpb_correlators_files+=("$qpb_correlators_file")
        fi
    done

    # Ask the user whether to delete all unpaired files
    number_of_unpaired_qpb_log_files=${#list_of_unpaired_qpb_log_files[@]}
    # number_of_unpaired_qpb_correlators_files=${#list_of_unpaired_qpb_correlators_files[@]}
    number_of_unpaired_qpb_correlators_files=${#list_of_unpaired_qpb_correlators_files[@]}
    if [[ ${number_of_unpaired_qpb_log_files} -gt 0 || \
                    ${number_of_unpaired_qpb_correlators_files} -gt 0 ]]; then
        # 
        sum_of_unpaired_qpb_files=$((number_of_unpaired_qpb_log_files + number_of_unpaired_qpb_correlators_files))
        input_request_message="There are a total of "
        input_request_message+="${sum_of_unpaired_qpb_files} unpaired qpb data "
        input_request_message+="files in this 'invert' data files set "
        input_request_message+="directory. "
        log "WARNING" "$input_request_message"
        input_request_message+="Would you like to like to delete all the "
        input_request_message+="unpaired qpb data files? ([yY]/nN) "
        while true; do
            read -p "++ $input_request_message" user_response
            # Convert the response to lowercase for easier comparison
            user_response=$(echo "$user_response" | tr '[:upper:]' '[:lower:]')
            # Handle the response, treating "Enter" as "yes"
            if [[ -z "$user_response" || "$user_response" == "y" || \
                                            "$user_response" == "yes" ]]; then
                # Delete unpaired qpb log and corresponding error files
                for unpaired_qpb_log_file in \
                                    "${list_of_unpaired_qpb_log_files[@]}"; do
                    rm $unpaired_qpb_log_file
                    rm -f "${unpaired_qpb_log_file%.txt}.err"
                done
                # Delete unpaired qpb correlators and corresponding error files
                for unpaired_qpb_correlators_file in \
                            "${list_of_unpaired_qpb_correlators_files[@]}"; do
                    rm $unpaired_qpb_correlators_file
                    rm -f "${unpaired_qpb_correlators_file%.dat}.err"
                done

                log_message="All unpaired qpb data files were deleted!"
                log "INFO" "$log_message"
                echo -e "$log_message"
                break
            elif [[ "$user_response" == "n" || "$user_response" == "no" ]]; then
                error_message="Unpaired qpb data files cannot be present in an "
                error_message+="'invert' data file set directory. "
                error_message+="It is recommended that they are "
                error_message+="all removed from the directory."
                termination_output "$error_message"
                exit 1
            else
                error_message="Invalid response. Please answer with 'y', "
                error_message+="'Y', 'yes', 'n', 'N', or 'no'."
                echo $error_message
            fi
        done
    fi
fi

# SUCCESSFUL COMPLETION OUTPUT

# Construct the final message
final_message="'${data_files_set_directory_name}' data files set "
final_message+="validation completed!"
# Print the final message
echo "!! $final_message"

log "INFO" "${final_message}"
echo # Empty line

echo -e $SCRIPT_TERMINATION_MESSAGE >> "$SCRIPT_LOG_FILE_PATH"

unset SCRIPT_TERMINATION_MESSAGE
unset SCRIPT_LOG_FILE_PATH
