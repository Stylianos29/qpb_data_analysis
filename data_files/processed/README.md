# Processed Data Files Directory

This directory is intended to store processed data files generated after
analyzing the raw data files stored in `data_files/raw` directory. Follow these
guidelines to keep your files organized:

## Organization

- Place files under directories structured as:
  <qpb_main_program>/<experiment_name>/

  For example:
  - invert/Chebyshev_several_N_and_m_varying_EpsCG/
  - sign_squared_violation/KL_several_vectors_varying_configs_and_n/

## File Types

- The processed data files typically include:
- `.csv` files
- `.h5` files (HDF5 format)
- `.log` files (text files)
- `.md` files (markdown format)

## Additional Notes

For more detailed instructions and conventions, refer to the [Organizing Data
Files documentation](../../docs/organizing_data_files.md).
