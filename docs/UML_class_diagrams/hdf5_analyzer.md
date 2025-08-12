```mermaid
    classDiagram
        class _HDF5Inspector {
            - _file: h5py.File
            - file_path: str
            - list_of_dataframe_column_names: list
            - list_of_tunable_parameter_names_from_dataframe: list
            - list_of_output_quantity_names_from_dataframe: list
            - unique_value_columns_dictionary: dict
            - multivalued_columns_count_dictionary: dict
            - list_of_single_valued_column_names: list
            - list_of_multivalued_column_names: list
            - list_of_single_valued_tunable_parameter_names: list
            - list_of_multivalued_tunable_parameter_names: list
            - list_of_single_valued_output_quantity_names: list
            - list_of_multivalued_output_quantity_names: list
            - _groups_by_level: defaultdict
            - _parameters_by_group: dict
            - _datasets_by_group: defaultdict
            - _dataset_paths: defaultdict
            - _gvar_dataset_pairs: dict
            + __init__(hdf5_file_path: str)
            + column_unique_values(column_name: str): list
            - _initialize_storage()
            - _analyze_structure()
            - _collect_groups()
            - _extract_parameters()
            - _collect_datasets()
            - _identify_gvar_pairs(dataset_names: Set)
            - _is_single_valued_dataset(values_list: List): bool
            - _categorize_columns()
            + __del__()
        }

        class _HDF5DataManager {
            - _active_groups: Set
            - _restriction_stack: list
            - _virtual_datasets: dict
            - _data_cache: dict
            - _all_deepest_groups: Set
            + __init__(hdf5_file_path: str)
            + active_groups: Set
            + reduced_multivalued_tunable_parameter_names_list: list
            + restrict_data(condition: str, filter_func: Callable)
            + restore_all_groups()
            + __enter__()
            + __exit__()
            + dataset_values(dataset_name: str, return_gvar: bool, group_path: str): Union[ndarray, List]
            + transform_dataset(source_dataset: str, transform_func: Callable, new_name: str)
            + to_dataframe(datasets: List, include_parameters: bool, flatten_arrays: bool, add_time_column: bool): DataFrame
            + group_by_multivalued_tunable_parameters(filter_out_parameters_list: List, verbose: bool): Dict
            - _all_parameters_for_group(group_path: str): Dict
            - _standard_dataset_values(dataset_name: str, group_path: str): Union[ndarray, List]
            - _gvar_dataset_values(mean_name: str, error_name: str, group_path: str): Union[ndarray, List]
            - _virtual_dataset_values(dataset_name: str, group_path: str): Union[ndarray, List]
        }

        class HDF5Analyzer {
            - _original_file_path: str
            + __init__(hdf5_file_path: Union[str, Path])
            + generate_uniqueness_report(max_width: int, separate_by_type: bool): str
            + unique_values(parameter_name: str, print_output: bool): List
            + create_dataset_dataframe(dataset_name: str, add_time_column: bool, time_offset: int, filter_func: Callable, include_group_path: bool, flatten_arrays: bool): DataFrame
            + create_merged_value_error_dataframe(base_name: str, add_time_column: bool, time_offset: int, filter_func: Callable, include_group_path: bool): DataFrame
            + save_transformed_data(output_path: Union[str, Path], include_virtual: bool, compression: str, compression_opts: int)
            + close()
            + __repr__(): str
            - _format_uniqueness_entry(name: str, value: Any): str
            - _format_single_value(value: Any): str
        }

        _HDF5Inspector <|-- _HDF5DataManager
        _HDF5DataManager <|-- HDF5Analyzer