import decimal
import pandas as pd


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


def find_missing_configs(series, total_configs=10):
    """
    Custom aggregation function that finds missing configuration labels.

    This function:
    1. Gets the bare_mass value from the current group
    2. Generates the expected configuration labels using generate_config_labels()
    3. Finds which configuration labels are missing from the existing data

    Args:
        series: pandas Series of Configuration_label values for a specific
                (Bare_mass, Number_of_Chebyshev_terms) pair

    Returns:
        str: A formatted string showing the missing configuration labels
    """
    # Get the existing configuration labels
    existing_configs = set(series.unique())

    # Get the bare_mass value from the current group
    # This works because we're grouping by Bare_mass, so all rows have the same value
    bare_mass_value = series.index.get_level_values("Bare_mass")[0]

    # Generate expected configuration labels
    expected_configs = set(
        generate_config_labels(bare_mass_value, total_configs=total_configs)
    )

    # Find missing configuration labels
    missing_configs = expected_configs - existing_configs

    # Format the result
    if not missing_configs:
        return "Complete - all configs present"
    else:
        missing_list = sorted(missing_configs)
        return f"Missing {len(missing_list)}: {', '.join(missing_list)}"


# Alternative version that returns just the list of missing configs
def get_missing_configs_list(series):
    """
    Simpler version that just returns the comma-separated list of missing configs.
    """
    existing_configs = set(series.unique())
    bare_mass_value = series.index.get_level_values("Bare_mass")[0]
    expected_configs = set(generate_config_labels(bare_mass_value))
    missing_configs = expected_configs - existing_configs

    if not missing_configs:
        return ""
    else:
        return ", ".join(sorted(missing_configs))


# Since the aggregation function doesn't have direct access to the group keys,
# we need a slightly different approach. Let's create a factory function:


def create_missing_configs_aggregator(table_generator_instance, total_configs=10):
    """
    Factory function that creates a custom aggregator with access to the full dataframe.

    Args:
        table_generator_instance: The TableGenerator instance with the dataframe

    Returns:
        function: A custom aggregation function that can find missing configs
    """
    df = table_generator_instance.dataframe

    def find_missing_configs_with_context(series):
        # Get the first row to extract the Bare_mass value
        # All rows in this series should have the same Bare_mass value due to grouping
        first_idx = series.index[0]
        bare_mass_value = df.loc[first_idx, "Bare_mass"]

        # Get existing configuration labels
        existing_configs = set(series.unique())

        # Generate expected configuration labels
        expected_configs = set(
            generate_config_labels(bare_mass_value, total_configs=total_configs)
        )

        # Find missing configuration labels
        missing_configs = expected_configs - existing_configs

        # Format the result
        if not missing_configs:
            return "Complete"
        else:
            missing_list = sorted(missing_configs)
            if len(missing_list) <= 5:
                return f"Missing {len(missing_list)}: {', '.join(missing_list)}"
            else:
                return f"Missing {len(missing_list)}: {', '.join(missing_list[:3])}..."

    return find_missing_configs_with_context
