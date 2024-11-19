#!/bin/bash

WORKING_DIRECTORY=".."
SOURCE_SCRIPTS_DIRECTORY="${WORKING_DIRECTORY}/src"
RAW_FILES_DIRECTORY="${WORKING_DIRECTORY}/data_files/raw"
PROCESSED_FILES_DIRECTORY="${WORKING_DIRECTORY}/data_files/processed"

WORKING_PROJECT="sign_squared_violation/KL_several_vectors_varying_configs_and_n"
python "${SOURCE_SCRIPTS_DIRECTORY}/log_files_analysis.py" \
    -qpb_log_dir "${RAW_FILES_DIRECTORY}/${WORKING_PROJECT}" \
    -out_dir "${PROCESSED_FILES_DIRECTORY}/${WORKING_PROJECT}"
