"""
Core processing functions for computational cost extrapolation.

Provides shared processing logic for both PCAC and pion mass-based cost
extrapolation methods. Implements two-stage analysis:
  1. Mass to bare mass conversion via fitting
  2. Cost extrapolation using shifted power law

Key capabilities:
  - Load and validate mass and cost data
  - Group data by lattice parameters
  - Fit mass vs bare mass with uncertainty propagation
  - Invert fit to derive reference bare mass
  - Average costs across configurations
  - Extrapolate costs to reference bare mass
  - Export comprehensive results to CSV
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import gvar as gv
import lsqfit

from library import load_csv, DataFrameAnalyzer

from src.analysis.cost_extrapolation._cost_extrapolation_shared_config import (
    get_grouping_parameters,
    get_grouping_excluded_parameters,
    get_filtering_config,
    get_validation_config,
    get_fit_quality_config,
    get_physical_validation_config,
    get_cost_fit_config,
    get_csv_output_config,
    get_output_column_mapping,
    get_error_handling_config,
)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================


def process_cost_extrapolation_analysis(
    cost_csv_path: str,
    mass_csv_path: str,
    output_csv_path: str,
    analysis_type: str,
    column_mapping: Dict[str, str],
    required_columns: List[str],
    logger,
) -> str:
    """
    Main entry point for cost extrapolation analysis.

    Workflow:
      1. Load mass data (PCAC or pion plateau estimates)
      2. Group mass data and fit mass vs bare mass
      3. Invert fit to get reference bare mass per group
      4. Load cost data
      5. Average cost across configurations per group
      6. Fit cost vs bare mass (shifted power law)
      7. Extrapolate cost at reference bare mass
      8. Export results to CSV

    Args:
        cost_csv_path: Path to computational cost CSV mass_csv_path:
        Path to mass plateau estimates CSV output_csv_path: Path for
        output results CSV analysis_type: "pcac" or "pion"
        column_mapping: Column name mapping for mass data
        required_columns: Required columns in mass data logger: Logger
        instance

    Returns:
        Path to output CSV file
    """
    logger.info(f"=== {analysis_type.upper()} Cost Extrapolation Analysis ===")

    # STAGE 1: Mass to Bare Mass Conversion
    logger.info("Stage 1: Converting mass to bare mass")
    derived_bare_mass_df = convert_mass_to_bare_mass(
        mass_csv_path,
        analysis_type,
        column_mapping,
        required_columns,
        logger,
    )

    if derived_bare_mass_df.empty:
        raise ValueError("Mass to bare mass conversion produced no results")

    logger.info(f"Derived bare masses for {len(derived_bare_mass_df)} parameter groups")

    # STAGE 2: Cost Extrapolation
    logger.info("Stage 2: Extrapolating computational costs")
    cost_results_df = extrapolate_computational_costs(
        cost_csv_path,
        derived_bare_mass_df,
        analysis_type,
        logger,
    )

    if cost_results_df.empty:
        raise ValueError("Cost extrapolation produced no results")

    logger.info(f"Cost extrapolation completed for {len(cost_results_df)} groups")

    # Export results
    logger.info(f"Exporting results to {output_csv_path}")
    export_cost_extrapolation_results(cost_results_df, output_csv_path, logger)

    return output_csv_path


# =============================================================================
# STAGE 1: MASS TO BARE MASS CONVERSION
# =============================================================================


def convert_mass_to_bare_mass(
    mass_csv_path: str,
    analysis_type: str,
    column_mapping: Dict[str, str],
    required_columns: List[str],
    logger,
) -> pd.DataFrame:
    """
    Convert reference mass to bare mass via linear fit inversion.

    Process:
      1. Load mass plateau data
      2. Group by lattice parameters
      3. For each group: fit mass^n vs bare mass (linear)
      4. Invert fit: bare_mass = (mass_ref^n - b) / a
      5. Return DataFrame with group keys and derived bare masses

    Args:
        mass_csv_path: Path to mass data CSV analysis_type: "pcac" or
        "pion" column_mapping: Column name mapping required_columns:
        Required columns logger: Logger instance

    Returns:
        DataFrame with group parameters and derived bare masses (value,
        error)
    """
    # Load mass data
    logger.info(f"Loading {analysis_type.upper()} mass data from {mass_csv_path}")
    try:
        mass_df = load_csv(
            mass_csv_path,
            validate_required_columns=set(required_columns),
            apply_categorical=True,
        )
        logger.info(f"Loaded {len(mass_df)} rows of mass data")
        logger.info(f"Columns: {list(mass_df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load mass data: {e}")
        raise

    # Get mass power (1 for PCAC, 2 for pion)
    if analysis_type == "pcac":
        mass_power = 1
        reference_mass = _get_reference_mass_pcac()
    elif analysis_type == "pion":
        mass_power = _get_pion_mass_power()
        reference_mass = _get_reference_mass_pion()
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    logger.info(f"Mass power: {mass_power}, Reference mass: {reference_mass}")

    # Group data
    analyzer = DataFrameAnalyzer(mass_df)
    logger.info(f"DataFrameAnalyzer created")
    logger.info(
        f"Multivalued parameters: {analyzer.list_of_multivalued_tunable_parameter_names}"
    )
    logger.info(
        f"Single-valued parameters: {analyzer.list_of_single_valued_tunable_parameter_names}"
    )

    excluded_params = get_grouping_excluded_parameters()

    # Filter exclusions to only existing parameters
    available_params = analyzer.list_of_multivalued_tunable_parameter_names
    filtered_exclusions = [p for p in excluded_params if p in available_params]

    logger.info(f"Will exclude from grouping: {filtered_exclusions}")
    logger.info(f"Grouping by remaining multivalued parameters...")

    try:
        grouped_data = analyzer.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=filtered_exclusions
        )
        # Convert generator to list to count groups
        grouped_list = list(grouped_data)
        logger.info(f"Created {len(grouped_list)} parameter groups")

        if len(grouped_list) == 0:
            logger.error(
                "No groups created! Check if data has appropriate multivalued parameters."
            )
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Grouping failed: {e}")
        raise

    # Process each group
    results = []
    validation_config = get_validation_config()
    min_points = validation_config["min_data_points_for_mass_fitting"]

    logger.info(
        f"Processing {len(grouped_list)} groups (min points required: {min_points})"
    )

    for group_keys, group_df in grouped_list:
        logger.info(f"=" * 60)
        logger.info(f"Processing mass group: {group_keys}")
        logger.info(f"Group has {len(group_df)} data points")

        if len(group_df) < min_points:
            logger.warning(
                f"  ✗ Insufficient data: {len(group_df)} < {min_points}, skipping"
            )
            continue

        try:
            # Perform fit
            logger.info(f"  Attempting fit for group...")
            fit_result = fit_mass_vs_bare_mass(
                group_df,
                column_mapping,
                mass_power,
                logger,
            )

            if fit_result is None:
                logger.warning("  ✗ Fit failed for this group")
                continue

            logger.info(
                f"  ✓ Fit successful: R²={fit_result['r_squared']:.3f}, χ²/dof={fit_result['chi2_reduced']:.3f}"
            )

            # Invert fit to get bare mass
            logger.info(f"  Inverting fit to get bare mass...")
            derived_bare_mass = invert_mass_fit(
                fit_result,
                reference_mass,
                mass_power,
                logger,
            )

            if derived_bare_mass is None:
                logger.warning("  ✗ Failed to invert fit for this group")
                continue

            logger.info(f"  ✓ Derived bare mass: {derived_bare_mass}")

            # Build result dictionary
            result = _build_mass_conversion_result(
                group_keys,
                analyzer,
                derived_bare_mass,
                fit_result,
            )

            results.append(result)
            logger.info(f"  ✓ Group successfully processed")

        except Exception as e:
            logger.error(f"  ✗ Error processing group {group_keys}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    logger.info(f"=" * 60)
    logger.info(
        f"Mass to bare mass conversion complete: {len(results)} successful groups"
    )

    if not results:
        logger.error("No successful mass to bare mass conversions")
        return pd.DataFrame()

    return pd.DataFrame(results)


def fit_mass_vs_bare_mass(
    group_df: pd.DataFrame,
    column_mapping: Dict[str, str],
    mass_power: int,
    logger,
) -> Optional[Dict[str, Any]]:
    """
    Fit mass^n vs bare mass using linear function.

    Fit: mass^n = a * bare_mass + b

    Args:
        group_df: Group data column_mapping: Column names mass_power:
        Power for mass (1 or 2) logger: Logger instance

    Returns:
        Dictionary with fit results or None if fit fails
    """
    try:
        # Extract data
        bare_mass_col = column_mapping["bare_mass"]
        mass_mean_col = column_mapping["mass_mean"]
        mass_error_col = column_mapping["mass_error"]

        logger.info(
            f"    Looking for columns: {bare_mass_col}, {mass_mean_col}, {mass_error_col}"
        )

        # Check columns exist
        missing = []
        if bare_mass_col not in group_df.columns:
            missing.append(bare_mass_col)
        if mass_mean_col not in group_df.columns:
            missing.append(mass_mean_col)
        if mass_error_col not in group_df.columns:
            missing.append(mass_error_col)

        if missing:
            logger.error(f"    Missing columns in data: {missing}")
            logger.error(f"    Available columns: {list(group_df.columns)}")
            return None

        x_data = group_df[bare_mass_col].to_numpy()
        y_mean = group_df[mass_mean_col].to_numpy()
        y_error = group_df[mass_error_col].to_numpy()

        logger.info(f"    Extracted {len(x_data)} data points")
        logger.info(f"    Bare mass range: [{x_data.min():.6f}, {x_data.max():.6f}]")
        logger.info(f"    Mass range: [{y_mean.min():.6f}, {y_mean.max():.6f}]")

        # Apply mass power transformation
        y_data = gv.gvar(y_mean, y_error)
        y_transformed = y_data**mass_power

        logger.info(f"    Applied mass power {mass_power}")

        # Linear fit with named parameters
        logger.info(f"    Performing linear fit...")

        def linear_function(x, p):
            return p["slope"] * x + p["intercept"]

        prior = {
            "slope": gv.gvar(0, 10),
            "intercept": gv.gvar(0, 1),
        }

        fit = lsqfit.nonlinear_fit(
            data=(x_data, y_transformed),
            fcn=linear_function,
            prior=prior,
        )

        logger.info(
            f"    Fit converged: slope={fit.p['slope']}, intercept={fit.p['intercept']}"
        )

        # Calculate quality metrics
        quality = _calculate_fit_quality(fit, x_data, y_transformed)

        logger.info(
            f"    Fit quality: R²={quality['r_squared']:.3f}, χ²/dof={quality['chi2_reduced']:.3f}, Q={quality['q_value']:.3f}"
        )

        return {
            "slope": fit.p["slope"],
            "intercept": fit.p["intercept"],
            "r_squared": quality["r_squared"],
            "chi2_reduced": quality["chi2_reduced"],
            "q_value": quality["q_value"],
            "n_points": len(x_data),
        }

    except Exception as e:
        logger.error(f"    Linear fit failed with error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def invert_mass_fit(
    fit_result: Dict[str, Any],
    reference_mass: float,
    mass_power: int,
    logger,
) -> Optional[gv.GVar]:
    """
    Invert mass fit to get bare mass at reference mass value.

    From fit: mass^n = a * bare_mass + b Solve for: bare_mass =
    (mass_ref^n - b) / a

    Args:
        fit_result: Fit parameters reference_mass: Target mass value
        mass_power: Power used in fit logger: Logger instance

    Returns:
        GVar with derived bare mass (value ± error) or None
    """
    try:
        slope = fit_result["slope"]
        intercept = fit_result["intercept"]

        # Check for zero slope
        if abs(gv.mean(slope)) < 1e-10:
            logger.warning("Slope too close to zero, cannot invert")
            return None

        # Calculate: bare_mass = (reference_mass^power - intercept) /
        # slope
        mass_transformed = reference_mass**mass_power
        bare_mass = (mass_transformed - intercept) / slope

        return bare_mass

    except Exception as e:
        logger.warning(f"Fit inversion failed: {e}")
        return None


def _build_mass_conversion_result(
    group_keys: Tuple,
    analyzer: DataFrameAnalyzer,
    derived_bare_mass: gv.GVar,
    fit_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build result dictionary for mass conversion."""
    # Get grouping parameter names
    grouping_params = analyzer.reduced_multivalued_tunable_parameter_names_list

    # Create result with group parameters
    result = dict(zip(grouping_params, group_keys))

    # Add single-valued parameters
    result.update(analyzer.unique_value_columns_dictionary)

    # Add derived bare mass
    result["derived_bare_mass_mean"] = float(gv.mean(derived_bare_mass))
    result["derived_bare_mass_error"] = float(gv.sdev(derived_bare_mass))

    # Add mass fit parameters
    result["mass_fit_slope_mean"] = float(gv.mean(fit_result["slope"]))
    result["mass_fit_slope_error"] = float(gv.sdev(fit_result["slope"]))
    result["mass_fit_intercept_mean"] = float(gv.mean(fit_result["intercept"]))
    result["mass_fit_intercept_error"] = float(gv.sdev(fit_result["intercept"]))

    # Add mass fit quality
    result["mass_fit_r_squared"] = fit_result["r_squared"]
    result["mass_fit_chi2_reduced"] = fit_result["chi2_reduced"]
    result["mass_fit_q_value"] = fit_result["q_value"]
    result["n_bare_mass_points"] = fit_result["n_points"]

    return result


# =============================================================================
# STAGE 2: COST EXTRAPOLATION
# =============================================================================


def extrapolate_computational_costs(
    cost_csv_path: str,
    derived_bare_mass_df: pd.DataFrame,
    analysis_type: str,
    logger,
) -> pd.DataFrame:
    """
    Extrapolate computational costs using derived bare masses.

    Process:
      1. Load cost data
      2. Average costs across configurations per group
      3. Group cost data by parameter groups (collect all bare mass
         points per group)
      4. For each mass group: find matching cost group, fit shifted
         power law
      5. Extrapolate cost at derived bare mass for that group
      6. Combine with mass conversion results

    Args:
        cost_csv_path: Path to cost data CSV derived_bare_mass_df:
        DataFrame with derived bare masses per group analysis_type:
        "pcac" or "pion" logger: Logger instance

    Returns:
        DataFrame with complete results (mass + cost)
    """
    # Load and average cost data
    logger.info(f"Loading cost data from {cost_csv_path}")
    averaged_cost_df = load_and_average_cost_data(cost_csv_path, logger)

    if averaged_cost_df.empty:
        logger.error("Cost averaging produced no results")
        return pd.DataFrame()

    logger.info(
        f"Averaged costs: {len(averaged_cost_df)} (bare_mass, group) combinations"
    )

    # Group cost data by parameter groups (excluding Bare_mass)
    cost_groups_dict = _group_cost_data_by_parameters(averaged_cost_df, logger)

    logger.info(f"Cost data grouped into {len(cost_groups_dict)} parameter groups")

    # Match groups and extrapolate
    results = []
    validation_config = get_validation_config()
    min_points = validation_config["min_data_points_for_cost_fitting"]

    logger.info(
        f"Processing {len(derived_bare_mass_df)} mass groups for cost extrapolation"
    )

    for _, mass_row in derived_bare_mass_df.iterrows():
        logger.info(f"=" * 60)

        # Get group identifier for this mass row
        group_key = _build_group_key(mass_row)
        logger.info(f"Processing cost extrapolation for group: {group_key}")

        # Find matching cost group
        cost_group_df = cost_groups_dict.get(group_key)

        if cost_group_df is None:
            logger.warning(f"  ✗ No matching cost data for this group")
            continue

        logger.info(f"  Found {len(cost_group_df)} cost data points for this group")

        if len(cost_group_df) < min_points:
            logger.warning(
                f"  ✗ Insufficient cost points: {len(cost_group_df)} < {min_points}"
            )
            continue

        # Extract derived bare mass
        derived_bare_mass = gv.gvar(
            mass_row["derived_bare_mass_mean"],
            mass_row["derived_bare_mass_error"],
        )

        logger.info(f"  Derived bare mass for extrapolation: {derived_bare_mass}")

        # Fit and extrapolate cost
        extrapolation_result = fit_and_extrapolate_cost(
            cost_group_df,
            derived_bare_mass,
            logger,
        )

        if extrapolation_result is not None:
            # Combine mass and cost results
            combined_result = {**mass_row.to_dict(), **extrapolation_result}
            results.append(combined_result)
            logger.info(f"  ✓ Cost extrapolation successful")
        else:
            logger.warning(f"  ✗ Cost extrapolation failed")

    logger.info(f"=" * 60)
    logger.info(f"Cost extrapolation complete: {len(results)} successful groups")

    if not results:
        logger.error("No successful cost extrapolations")
        return pd.DataFrame()

    return pd.DataFrame(results)


def load_and_average_cost_data(cost_csv_path: str, logger) -> pd.DataFrame:
    """
    Load cost data and average across configurations.

    Returns DataFrame with averaged costs per parameter group.
    """
    # Load cost data
    cost_df = load_csv(cost_csv_path, apply_categorical=True)

    # Check required columns
    required_cost_cols = [
        "Bare_mass",
        "Configuration_label",
        "Average_core_hours_per_spinor",
    ]
    missing = [col for col in required_cost_cols if col not in cost_df.columns]
    if missing:
        raise ValueError(f"Missing required cost columns: {missing}")

    # Create analyzer
    analyzer = DataFrameAnalyzer(cost_df)

    # Exclude configuration label and bare mass from grouping
    excluded = get_grouping_excluded_parameters()
    available_params = analyzer.list_of_multivalued_tunable_parameter_names
    filtered_exclusions = [p for p in excluded if p in available_params]

    logger.info(f"Averaging costs (excluding: {filtered_exclusions})")

    # Group and average
    grouped = analyzer.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=filtered_exclusions
    )

    # Calculate averages
    results = []
    validation_config = get_validation_config()
    min_configs = validation_config.get("min_configurations_for_averaging", 2)

    for group_keys, group_df in grouped:
        # Further group by bare mass within this parameter group
        for bare_mass, mass_group in group_df.groupby("Bare_mass"):
            n_configs = len(mass_group)

            if n_configs < min_configs:
                continue

            # Calculate average cost
            costs = mass_group["Average_core_hours_per_spinor"]
            mean_cost = costs.mean()
            error_cost = costs.sem() if n_configs > 1 else 0.0

            # Build result row
            grouping_params = analyzer.reduced_multivalued_tunable_parameter_names_list
            result = dict(zip(grouping_params, group_keys))
            result.update(analyzer.unique_value_columns_dictionary)
            result["Bare_mass"] = bare_mass
            result["cost_mean"] = mean_cost
            result["cost_error"] = error_cost if not pd.isna(error_cost) else 0.0
            result["n_configurations"] = n_configs

            results.append(result)

    return pd.DataFrame(results)


def fit_and_extrapolate_cost(
    cost_group_df: pd.DataFrame,
    derived_bare_mass: gv.GVar,
    logger,
) -> Optional[Dict[str, Any]]:
    """
    Fit shifted power law to cost data and extrapolate to derived bare
    mass.

    Fit: cost = a / (bare_mass - b) + c

    Args:
        cost_group_df: Cost data for this group (multiple bare masses)
        derived_bare_mass: Target bare mass for extrapolation (with
        uncertainty) logger: Logger instance

    Returns:
        Dictionary with extrapolation results or None if fit fails
    """
    try:
        # Extract data
        x_data = cost_group_df["Bare_mass"].to_numpy()
        y_mean = cost_group_df["cost_mean"].to_numpy()
        y_error = cost_group_df["cost_error"].to_numpy()

        logger.info(f"    Cost fitting data:")
        logger.info(f"      Bare mass range: [{x_data.min():.6f}, {x_data.max():.6f}]")
        logger.info(
            f"      Cost range: [{y_mean.min():.2f}, {y_mean.max():.2f}] core-hours"
        )

        # Create gvar data
        y_data = gv.gvar(y_mean, y_error)

        # Define shifted power law function
        def shifted_power_law(x, p):
            """cost = a / (x - b) + c"""
            return p["a"] / (x - p["b"]) + p["c"]

        # Estimate initial parameters a: amplitude (positive, larger for
        # higher costs) b: shift (should be less than min bare mass) c:
        # offset (approximate minimum cost)
        min_bare_mass = float(x_data.min())
        max_cost = float(y_mean.max())
        min_cost = float(y_mean.min())

        prior = {
            "a": gv.gvar(max_cost * 0.1, max_cost),  # Positive amplitude
            "b": gv.gvar(min_bare_mass - 0.02, 0.05),  # Shift less than min bare mass
            "c": gv.gvar(min_cost, max_cost),  # Offset around min cost
        }

        logger.info(
            f"    Prior estimates: a≈{prior['a']}, b≈{prior['b']}, c≈{prior['c']}"
        )
        logger.info(f"    Fitting shifted power law: cost = a/(bare_mass - b) + c")

        # Perform fit
        fit = lsqfit.nonlinear_fit(
            data=(x_data, y_data),
            fcn=shifted_power_law,
            prior=prior,
        )

        logger.info(f"    Fit converged:")
        logger.info(f"      a = {fit.p['a']}")
        logger.info(f"      b = {fit.p['b']}")
        logger.info(f"      c = {fit.p['c']}")

        # Calculate fit quality
        quality = _calculate_fit_quality(fit, x_data, y_data)

        logger.info(
            f"    Fit quality: R²={quality['r_squared']:.3f}, χ²/dof={quality['chi2_reduced']:.3f}, Q={quality['q_value']:.3f}"
        )

        # Validate fit quality
        fit_quality_config = get_fit_quality_config()
        if quality["r_squared"] < fit_quality_config["min_r_squared"]:
            logger.warning(
                f"    ⚠ Low R²: {quality['r_squared']:.3f} < {fit_quality_config['min_r_squared']}"
            )
        if quality["q_value"] < fit_quality_config["min_q_value"]:
            logger.warning(
                f"    ⚠ Low Q-value: {quality['q_value']:.3f} < {fit_quality_config['min_q_value']}"
            )

        # Extrapolate to derived bare mass
        logger.info(f"    Extrapolating to bare mass: {derived_bare_mass}")

        # Check if derived bare mass is within reasonable range
        bare_mass_value = float(gv.mean(derived_bare_mass))
        shift_value = float(gv.mean(fit.p["b"]))

        if abs(bare_mass_value - shift_value) < 1e-6:
            logger.error(
                f"    ✗ Bare mass too close to singularity: {bare_mass_value} ≈ {shift_value}"
            )
            return None

        if bare_mass_value < shift_value:
            logger.warning(
                f"    ⚠ Bare mass {bare_mass_value} < shift {shift_value} (negative denominator)"
            )

        # Evaluate fit at derived bare mass
        extrapolated_cost = shifted_power_law(derived_bare_mass, fit.p)

        logger.info(f"    Extrapolated cost: {extrapolated_cost} core-hours")

        # Build result dictionary
        result = {
            "extrapolated_cost_mean": float(gv.mean(extrapolated_cost)),
            "extrapolated_cost_error": float(gv.sdev(extrapolated_cost)),
            "cost_fit_param_a_mean": float(gv.mean(fit.p["a"])),
            "cost_fit_param_a_error": float(gv.sdev(fit.p["a"])),
            "cost_fit_param_b_mean": float(gv.mean(fit.p["b"])),
            "cost_fit_param_b_error": float(gv.sdev(fit.p["b"])),
            "cost_fit_param_c_mean": float(gv.mean(fit.p["c"])),
            "cost_fit_param_c_error": float(gv.sdev(fit.p["c"])),
            "cost_fit_r_squared": quality["r_squared"],
            "cost_fit_chi2_reduced": quality["chi2_reduced"],
            "cost_fit_q_value": quality["q_value"],
            "n_configurations": int(cost_group_df["n_configurations"].sum()),
        }

        return result

    except Exception as e:
        logger.error(f"    ✗ Cost fitting failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def _group_cost_data_by_parameters(
    averaged_cost_df: pd.DataFrame,
    logger,
) -> Dict[Tuple, pd.DataFrame]:
    """
    Group cost data by parameter groups (excluding Bare_mass).

    Returns dictionary mapping group_key -> DataFrame with all bare mass
    points for that group.
    """
    # Get grouping parameters
    grouping_params = get_grouping_parameters()

    # Find which grouping parameters exist in the dataframe
    available_grouping = [p for p in grouping_params if p in averaged_cost_df.columns]

    if not available_grouping:
        logger.warning(
            "No grouping parameters found in cost data, treating as single group"
        )
        return {(): averaged_cost_df}

    logger.info(f"Grouping cost data by: {available_grouping}")

    # Group the data
    grouped_dict = {}
    for group_keys, group_df in averaged_cost_df.groupby(available_grouping):
        # Ensure group_keys is a tuple
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        grouped_dict[group_keys] = group_df

    return grouped_dict


def _build_group_key(row: pd.Series) -> Tuple:
    """
    Build a hashable group key from a row's grouping parameters.
    """
    grouping_params = get_grouping_parameters()

    key_values = []
    for param in grouping_params:
        if param in row.index:
            value = row[param]
            # Convert numpy types to Python types for hashing
            if hasattr(value, "item"):
                value = value.item()
            key_values.append(value)

    return tuple(key_values)


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_cost_extrapolation_results(
    results_df: pd.DataFrame,
    output_csv_path: str,
    logger,
):
    """Export cost extrapolation results to CSV."""
    csv_config = get_csv_output_config()

    # Reorder columns (physics parameters first)
    results_df = _reorder_columns_for_export(results_df)

    # Export
    results_df.to_csv(
        output_csv_path,
        index=csv_config["index"],
        float_format=f"%.{csv_config['float_precision']}f",
    )

    logger.info(f"Exported {len(results_df)} results to {output_csv_path}")


def _reorder_columns_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to put physics parameters first."""
    priority_cols = [
        "beta",
        "volume_label",
        "kappa_critical",
        "Kernel_operator_type",
        "Overlap_operator_method",
    ]

    existing_priority = [col for col in priority_cols if col in df.columns]
    remaining = [col for col in df.columns if col not in existing_priority]

    return df[existing_priority + remaining]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _calculate_fit_quality(fit, x_data, y_data) -> Dict[str, float]:
    """Calculate fit quality metrics."""
    # Calculate predicted values using the fit function
    y_pred = fit.fcn(x_data, fit.p)

    # R-squared
    y_mean_vals = gv.mean(y_data)
    ss_tot = np.sum((y_mean_vals - np.mean(y_mean_vals)) ** 2)
    ss_res = np.sum((y_mean_vals - gv.mean(y_pred)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Chi-squared from fit result
    chi2_reduced = fit.chi2 / fit.dof if fit.dof > 0 else np.inf

    return {
        "r_squared": float(r_squared),
        "chi2_reduced": float(chi2_reduced),
        "q_value": float(fit.Q),
    }


def _get_reference_mass_pcac() -> float:
    """Get reference PCAC mass from config."""
    from src.analysis.cost_extrapolation._pcac_cost_extrapolation_config import (
        get_reference_pcac_mass,
    )

    return get_reference_pcac_mass()


def _get_reference_mass_pion() -> float:
    """Get reference pion mass from config."""
    from src.analysis.cost_extrapolation._pion_cost_extrapolation_config import (
        get_reference_pion_mass,
    )

    return get_reference_pion_mass()


def _get_pion_mass_power() -> int:
    """Get pion mass power from config."""
    from src.analysis.cost_extrapolation._pion_cost_extrapolation_config import (
        get_pion_mass_power,
    )

    return get_pion_mass_power()
