import decimal

def generate_config_labels(bare_mass_value, total_configs=10):
    """
    Generate a list of configuration labels based on the given bare mass value.

    Parameters:
        bare_mass_value (float or str): The bare mass value used in the computation.

    Returns:
        list of str: A list of 7-character configuration labels as strings.
    """
    bare_mass_value = decimal.Decimal(str(bare_mass_value))  # Ensure precision
    config_labels_list = []

    for k in range(total_configs):
        config_label = int((100 * k + 2200 * bare_mass_value) + 592)
        config_label %= 592
        config_label = f"00{config_label:03d}00"  # Two leading and two trailing zeros
        config_labels_list.append(config_label)

    return config_labels_list

# n = 7
# bare_mass = 0.3

# # Include counting the iterations for later use
# for (
#     combination_of_values,
#     dataframe_group,
# ) in processed_parameter_values_dataframe.groupby(
#     tunable_multivalued_parameter_names_list, observed=True
# ):
#     # Store specific tunable multivalued parameter names and values in a
#     # dedicated metadata dictionary for later use
#     if not isinstance(combination_of_values, tuple):
#         combination_of_values = [combination_of_values]
#     metadata_dictionary = dict(
#         zip(tunable_multivalued_parameter_names_list, combination_of_values)
#     )

#     print(metadata_dictionary)

#     list_of_used_configuration_labels = dataframe_group[
#         (dataframe_group["Bare_mass"] == bare_mass)
#         & (dataframe_group["KL_diagonal_order"] == n)
#     ]["Configuration_label"].values
#     list_of_target_configuration_labels = generate_config_labels(bare_mass, 10)
#     print(
#         "("
#         + " ".join(
#             f"{item}"
#             for item in list_of_target_configuration_labels
#             if item not in list_of_used_configuration_labels
#         )
#         + ")"
#     )