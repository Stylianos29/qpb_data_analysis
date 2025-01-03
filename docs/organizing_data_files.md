# Organizing Data Files for `qpb_data_analysis`

This guide provides detailed suggestions for organizing the `data_files/raw` and
`data_files/processed` directories to ensure consistency, ease of access, and
clarity.

---

## Directory Structure

We recommend structuring the data files under the following hierarchy:
```
data_files/
├── raw/
│ ├── <qpb_main_program>/
│ │ ├── <experiment_name>/
│ │ │ ├── raw_file1.txt
│ │ │ ├── raw_file2.txt
│ │ │ └── ...
├── processed/
│ ├── <qpb_main_program>/
│ │ ├── <experiment_name>/
│ │ │ ├── processed_file.csv
│ │ │ ├── processed_file.h5
│ │ │ ├── processed_file.log
│ │ │ ├── processed_file.md
│ │ │ └── ...
```

### Explanation of Components

- `<qpb_main_program>`: The name of the main `qpb` program that produced the
  data files. Examples:
  - `sign_squared_violation`
  - `invert`

- `<experiment_name>`: A unique name or identifier for the specific analysis, or
  experiment. Examples:
  - `KL_several_vectors_varying_configs_and_n`
  - `Chebyshev_several_N_and_m_varying_EpsCG`

---

## Guidelines for `data_files/raw`

1. **Purpose**: Store raw data files exactly as they are generated by the `qpb`
   program. Do not modify these files.
   
2. **Recommended File Types**:
   - `.txt` files: Logs of qpb main program's execution.
   - `.dat` files: Raw data files outputted specifically by a `mesons` qpb main
     program.

3. **Example**:
```
data_files/raw/sign_squared_violation/KL_several_vectors_varying_configs_and_n/
├── KL_Brillouin_mu1p0_rho1p0_cSW0_EpsCG1e-16_config0000200_n2.txt
└── KL_Brillouin_mu1p0_rho1p0_cSW0_EpsCG1e-16_config0001600_n9.txt
```

4. **Best Practices**:
- Avoid storing processed or manually modified files in this directory.
- Use descriptive filenames whenever possible to make file identification
  easier.

---

## Guidelines for `data_files/processed`

1. **Purpose**: Store files generated after processing raw data using the
`qpb_data_analysis` scripts.

2. **Recommended File Types**:
- `.csv`: Data tables suitable for analysis in spreadsheet software or further
  scripting.
- `.h5`: HDF5 files for efficient storage of large datasets.
- `.log`: Log files of various analysis performed using the `.csv` and `.h5`
  files.
- `.md`: Markdown files for further note-taking.

3. **Example**:
```
data_files/processed/sign_squared_violation/KL_several_vectors_varying_configs_and_n
├── KL_several_vectors_varying_configs_and_n.csv
├── KL_several_vectors_varying_configs_and_n.h5
├── KL_several_vectors_varying_configs_and_n.log
└── KL_several_vectors_varying_configs_and_n.md
```

4. **Best Practices**:
- Keep files grouped by `<qpb_main_program>` and `<experiment_name>` for
  consistency as in the 'raw' directory.
- Ensure file names describe their content, e.g., `summary.csv`.
- Avoid storing temporary or intermediary files unless absolutely necessary.
- For plots it might be better to be stored with the `output/plots` directory.

---

## General Notes

1. **Cross-referencing**:
- Always link raw data files with their corresponding processed files using
  directory and file naming conventions.

2. **Backup and Archival**:
- Consider backing up raw data files to avoid accidental loss.
<!-- TODO: Refer to a specific script for compressing and backing up. -->
- Archive older processed files to save space and maintain clarity.

3. **Collaboration**:
- When sharing the repository, ensure all contributors adhere to these
  organizational guidelines for consistency.

4. **Versioning**:
- If reanalyzing data under different conditions, use separate subdirectories
  for each version of processed files.

---

## Troubleshooting

1. **Misplaced Files**: If you find files in incorrect directories, move them
promptly to maintain organization.

2. **File Name Conflicts**: Use unique names for files to avoid overwriting.
Include timestamps or unique identifiers when needed, for example:
`results_2024_11_20.csv`.

3. **Missing Files**: Always check that all required raw files exist before
starting analysis. Consider creating a script to validate file presence.
<!-- TODO: I need a script that performs checks on raw data before analysis. -->

---

For further assistance, feel free to contact the project maintainers or refer to
the individual README files in the `data_files/raw` and `data_files/processed`
directories.
