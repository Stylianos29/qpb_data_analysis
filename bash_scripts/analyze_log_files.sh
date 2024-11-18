#!/bin/bash

WORKING_DIRECTORY=".."

python $WORKING_DIRECTORY"/src/log_files_analysis.py" \
    -log_dir $WORKING_DIRECTORY"/data_files/raw/sign_squared_violation/KL_several_vectors_varying_configs_and_n" \
    -out_dir $WORKING_DIRECTORY"/data_files/processed"