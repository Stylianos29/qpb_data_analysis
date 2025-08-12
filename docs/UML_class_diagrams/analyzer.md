## UML Class Diagram for analyzer module

```mermaid
    classDiagram 
        class _DataFrameInspector { 
        - dataframe: pd.DataFrame 
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
        + __init__(dataframe: pd.DataFrame) 
        + column_unique_values(column_name: str): list }
        
        class DataFrameAnalyzer { 
        - original_dataframe: pd.DataFrame 
        - _filter_out_parameters_list: list 
        + __init__(dataframe: pd.DataFrame) 
        + __enter__() 
        + __exit__() 
        + restrict_dataframe(condition, filter_func) 
        + add_derived_column(new_column_name, derivation_function, expression) 
        + group_by_multivalued_tunable_parameters(filter_out_parameters_list, verbose) 
        + restore_original_dataframe() 
        + reduced_multivalued_tunable_parameter_names_list: list }
        _DataFrameInspector <|-- DataFrameAnalyzer