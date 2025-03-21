#!/bin/bash

# Test the validate_raw_data_files_set.sh script

# TODO: Add the option of removing the log files after the tests are done

count=0

# Test case: No arguments
count=$((count + 1))
echo "Test case $count: No arguments"
../checks/validate_raw_data_files_set.sh
echo "Exit status: $?"
echo

# Test case: Incorrect data files set directory
count=$((count + 1))
echo "Test case $count: Incorrect data files set directory"
../checks/validate_raw_data_files_set.sh \
    --data_files_set_directory "data_files_set"\
echo "Exit status: $?"
echo

# Test case: Empty data files set directory
count=$((count + 1))
echo "Test case $count: Empty data files set directory"
../checks/validate_raw_data_files_set.sh \
    --data_files_set_directory \
    "./mock_data/validate_raw_data_files_set/empty_data_files_set_directory" \
    --script_log_filename "empty_data_files_set_directory.log"
echo "Exit status: $?"
echo

# Test case: No qpb log files in the data files set directory
count=$((count + 1))
echo "Test case $count: No qpb log files in the data files set directory"
../checks/validate_raw_data_files_set.sh \
    --data_files_set_directory \
    "./mock_data/validate_raw_data_files_set/no_qpb_log_files" \
    --script_log_filename "no_qpb_log_files.log"
echo "Exit status: $?"
echo

# Test case: Valid Chebyshev invert data files set
count=$((count + 1))
echo "Test case $count: Valid Chebyshev invert data files set"
../checks/validate_raw_data_files_set.sh \
    --data_files_set_directory \
    "./mock_data/validate_raw_data_files_set/Chebyshev_invert" \
    --script_log_filename "Chebyshev_invert.log"
echo "Exit status: $?"
echo
