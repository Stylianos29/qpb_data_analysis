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
from library.constants.data_types import (
    PARAMETERS_WITH_EXPONENTIAL_FORMAT,
    PARAMETERS_OF_INTEGER_VALUE,
)


from src.analysis.cost_extrapolation._cost_extrapolation_shared_config import (
    get_grouping_parameters,
    get_grouping_excluded_parameters,
    get_filtering_config,
    get_validation_config,
    get_fit_quality_config,
    get_physical_validation_config,
    get_cost_data_columns,
    get_cost_fit_config,
    get_csv_output_config,
    get_output_column_mapping,
    get_error_handling_config,
)


# =============================================================================
# FITTING RANGE FILTERING
# =============================================================================


def filter_data_by_fit_range(
    df: pd.DataFrame,
    bare_mass_col: str,
    range_min: Optional[float],
    range_max: Optional[float],
) -> Tuple[pd.DataFrame, Tuple[Optional[float], Optional[float]]]:
    """
    Filter dataframe to fitting range and return actual discrete bounds
    used.

    Args:
        df: Input dataframe bare_mass_col: Name of bare mass column
        range_min: Minimum bound (None = use data minimum) range_max:
        Maximum bound (None = use data maximum)

    Returns:
        Tuple of (filtered_df, (actual_min, actual_max)) where
        actual_min/max are the discrete data values included in the
        fitting range
    """
    filtered_df = df.copy()

    # Apply user-specified constraints
    if range_min is not None:
        filtered_df = filtered_df[filtered_df[bare_mass_col] >= range_min]
    if range_max is not None:
        filtered_df = filtered_df[filtered_df[bare_mass_col] <= range_max]

    # Extract actual discrete bounds from filtered data
    if len(filtered_df) > 0:
        actual_min = float(filtered_df[bare_mass_col].min())
        actual_max = float(filtered_df[bare_mass_col].max())
    else:
        # Will be caught by validation later
        actual_min, actual_max = None, None

    return filtered_df, (actual_min, actual_max)


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

    # Get fit range configuration
    if analysis_type == "pcac":
        from src.analysis.cost_extrapolation._pcac_cost_extrapolation_config import (
            get_fit_range_config,
        )
    else:  # pion
        from src.analysis.cost_extrapolation._pion_cost_extrapolation_config import (
            get_fit_range_config,
        )

    fit_range_config = get_fit_range_config()
    logger.info(f"Fit range configuration: {fit_range_config}")

    # STAGE 1: Mass to Bare Mass Conversion
    logger.info("Stage 1: Converting mass to bare mass")
    derived_bare_mass_df = convert_mass_to_bare_mass(
        mass_csv_path,
        analysis_type,
        column_mapping,
        fit_range_config,  # Pass fit_range_config
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
        fit_range_config,  # Pass fit_range_config
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
    fit_range_config: Dict[str, Dict[str, Optional[float]]],
    logger,
) -> pd.DataFrame:
    """
    Convert reference mass to bare mass via linear fit inversion.

    Process:
      1. Load mass plateau data
      2. Group by lattice parameters
      3. Apply fitting range filter
      4. For each group: fit mass^n vs bare mass (linear)
      5. Invert fit: bare_mass = (mass_ref^n - b) / a
      6. Store results with fit range information

    Args:
        mass_csv_path: Path to mass plateau estimates CSV analysis_type:
        "pcac" or "pion" column_mapping: Column name mappings
        fit_range_config: Fitting range configuration logger: Logger
        instance

    Returns:
        DataFrame with derived bare mass and fit info per group
    """
    logger.info(f"Loading mass data from {mass_csv_path}")

    # Load data
    mass_df = load_csv(mass_csv_path)

    if mass_df.empty:
        logger.error("Mass data is empty")
        return pd.DataFrame()

    logger.info(f"Loaded {len(mass_df)} mass data points")

    # Validate required columns
    for col in column_mapping.values():
        if col not in mass_df.columns:
            raise ValueError(f"Required column '{col}' not found in mass CSV")

    # Get reference mass and power based on analysis type
    if analysis_type == "pcac":
        reference_mass = _get_reference_mass_pcac()
        mass_power = 1
    else:  # pion
        reference_mass = _get_reference_mass_pion()
        mass_power = _get_pion_mass_power()

    logger.info(
        f"Reference mass: {reference_mass}, power: {mass_power} "
        f"(fitting mass^{mass_power} vs bare mass)"
    )

    # Group data by parameters
    try:
        # Use DataFrameAnalyzer to detect grouping parameters
        analyzer = DataFrameAnalyzer(mass_df)

        # Get all multivalued parameters
        all_grouping_params = analyzer.reduced_multivalued_tunable_parameter_names_list

        # Manually exclude unwanted parameters
        excluded = get_grouping_excluded_parameters()
        grouping_params = [p for p in all_grouping_params if p not in excluded]

        # Group by the filtered parameters
        grouped_list = list(
            mass_df.groupby(grouping_params, observed=True, dropna=False)
        )

        logger.info(f"All multivalued params: {all_grouping_params}")
        logger.info(f"Excluded params: {excluded}")
        logger.info(f"Data grouped by: {grouping_params}")
        logger.info(f"Found {len(grouped_list)} parameter groups")

        if len(grouped_list) == 0:
            logger.error(
                "No groups found. Check if data has appropriate multivalued parameters."
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

        # Apply fitting range filter for mass fit
        mass_range = fit_range_config["mass_fit"]
        filtered_df, (mass_fit_min, mass_fit_max) = filter_data_by_fit_range(
            group_df,
            column_mapping["bare_mass"],
            mass_range["bare_mass_min"],
            mass_range["bare_mass_max"],
        )

        if len(filtered_df) < min_points:
            logger.warning(
                f"  ✗ Insufficient data after range filtering: "
                f"{len(filtered_df)} < {min_points}, skipping"
            )
            continue

        logger.info(
            f"  Mass fit range: [{mass_fit_min:.6f}, {mass_fit_max:.6f}] "
            f"({len(filtered_df)} points)"
        )

        try:
            # Perform fit on filtered data
            logger.info(f"  Attempting fit for group...")
            fit_result = fit_mass_vs_bare_mass(
                filtered_df,
                column_mapping,
                mass_power,
                logger,
            )

            if fit_result is None:
                logger.warning("  ✗ Fit failed for this group")
                continue

            logger.info(
                f"  ✓ Fit successful: R²={fit_result['r_squared']:.3f}, "
                f"χ²/dof={fit_result['chi2_reduced']:.3f}"
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
                grouping_params,  # Pass the filtered grouping_params
                analyzer,
                derived_bare_mass,
                fit_result,
            )

            # Add mass fit range to result
            result["mass_fit_range_min"] = mass_fit_min
            result["mass_fit_range_max"] = mass_fit_max

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
        - group_df: Group data
        - column_mapping: Column names
        - mass_power: Power for mass (1 or 2)
        - logger: Logger instance

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
        - fit_result: Fit parameters
        - reference_mass: Target mass value
        - mass_power: Power used in fit
        - logger: Logger instance

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
    grouping_params: List[str],
    analyzer: DataFrameAnalyzer,
    derived_bare_mass: gv.GVar,
    fit_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build result dictionary for mass conversion."""
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
    fit_range_config: Dict[str, Dict[str, Optional[float]]],
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
        - cost_csv_path: Path to cost data CSV
        - derived_bare_mass_df: DataFrame with derived bare masses per
          group
        - analysis_type: "pcac" or "pion"
        - logger: Logger instance

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
    cost_groups_dict, grouping_param_names = _group_cost_data_by_parameters(
        averaged_cost_df, logger
    )

    logger.info(f"Cost data grouped into {len(cost_groups_dict)} parameter groups")
    logger.info(f"Grouping parameters used: {grouping_param_names}")

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
        group_key = _build_group_key(mass_row, grouping_param_names, logger)
        if group_key is None:
            logger.warning(f"  Skipping row due to missing grouping parameters")
            continue
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
            fit_range_config["cost_fit"],  # Pass cost fit range
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
    cost_df = load_csv(cost_csv_path)

    # Check required columns
    required_cost_cols = get_cost_data_columns()
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
    fit_range_config: Dict[str, Optional[float]],
    logger,
) -> Optional[Dict[str, Any]]:
    """
    Fit cost vs bare mass and extrapolate to derived bare mass.

    Fit: cost = a / (bare_mass - b) + c

    Args:
        cost_group_df: Cost data for this group (multiple bare masses)
        derived_bare_mass: Target bare mass for extrapolation (with
        uncertainty) fit_range_config: Fitting range configuration for
        cost fit logger: Logger instance

    Returns:
        Dictionary with extrapolation results or None if fit fails
    """
    try:
        # Apply fitting range filter for cost fit
        filtered_df, (cost_fit_min, cost_fit_max) = filter_data_by_fit_range(
            cost_group_df,
            "Bare_mass",
            fit_range_config["bare_mass_min"],
            fit_range_config["bare_mass_max"],
        )

        if len(filtered_df) == 0:
            logger.warning("  No data points in specified cost fit range")
            return None

        logger.info(
            f"  Cost fit range: [{cost_fit_min:.6f}, {cost_fit_max:.6f}] "
            f"({len(filtered_df)} points)"
        )

        # Extract data from filtered DataFrame
        x_data = filtered_df["Bare_mass"].to_numpy()
        y_mean = filtered_df["cost_mean"].to_numpy()
        y_error = filtered_df["cost_error"].to_numpy()

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

        # Perform fit
        fit = lsqfit.nonlinear_fit(
            data=(x_data, y_data),
            fcn=shifted_power_law,
            prior=prior,
        )

        logger.info(
            f"    Fit converged: a={fit.p['a']}, b={fit.p['b']}, c={fit.p['c']}"
        )

        # Calculate fit quality
        quality = _calculate_fit_quality(fit, x_data, y_data)

        logger.info(
            f"    Fit quality: R²={quality['r_squared']:.3f}, "
            f"χ²/dof={quality['chi2_reduced']:.3f}, Q={quality['q_value']:.3f}"
        )

        # Check for physical validity of shift parameter
        bare_mass_value = float(gv.mean(derived_bare_mass))
        shift_value = float(gv.mean(fit.p["b"]))

        if bare_mass_value <= shift_value:
            logger.warning(
                f"    ⚠ Bare mass {bare_mass_value} <= shift {shift_value}, "
                f"extrapolation may be unreliable"
            )

        # Extrapolate to derived bare mass
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
            "cost_fit_range_min": cost_fit_min,
            "cost_fit_range_max": cost_fit_max,
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
) -> Tuple[Dict[Tuple, pd.DataFrame], List[str]]:
    """
    Group cost data by parameter groups (excluding Bare_mass).

    Uses DataFrameAnalyzer to intelligently discover multivalued
    parameters and group the data accordingly.

    Returns:
        Tuple of (grouped_dict, grouping_param_names) where: -
        grouped_dict: Dictionary mapping group_key -> DataFrame -
        grouping_param_names: List of parameter names used for grouping
    """
    # Create analyzer
    analyzer = DataFrameAnalyzer(averaged_cost_df)

    # Get exclusion list (parameters to average over, not group by)
    excluded_params = get_grouping_excluded_parameters()

    # Filter to only parameters that exist and are multivalued
    available_params = analyzer.list_of_multivalued_tunable_parameter_names
    filtered_exclusions = [p for p in excluded_params if p in available_params]

    logger.info(f"Cost data multivalued parameters: {available_params}")
    logger.info(f"Excluding from grouping: {filtered_exclusions}")

    # Group using DataFrameAnalyzer
    grouped = analyzer.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=filtered_exclusions
    )

    # Get the actual grouping parameter names used
    grouping_param_names = analyzer.reduced_multivalued_tunable_parameter_names_list

    logger.info(f"Grouping cost data by: {grouping_param_names}")

    # Convert generator to dictionary
    grouped_dict = {}
    for group_keys, group_df in grouped:
        # Ensure group_keys is a tuple
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        grouped_dict[group_keys] = group_df

    return grouped_dict, grouping_param_names


def _build_group_key(
    mass_row: pd.Series,
    grouping_param_names: List[str],
    logger,
) -> Tuple | None:
    """
    Build group key tuple from mass results row.

    Args:
        mass_row: Row from derived_bare_mass_df
        grouping_param_names: List of grouping parameter names in correct order
        logger: Logger instance

    Returns:
        Tuple of parameter values in the same order as grouping_param_names
    """
    key_values = []

    for param in grouping_param_names:
        if param in mass_row.index:
            key_values.append(mass_row[param])
        else:
            logger.warning(
                f"Parameter '{param}' not found in mass row, skipping this group"
            )
            return None

    return tuple(key_values)


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_cost_extrapolation_results(
    results_df: pd.DataFrame,
    output_csv_path: str,
    logger,
):
    """
    Export cost extrapolation results to CSV with proper formatting.

    Applies exponential notation to small parameters and standard
    notation to physics results.
    """
    csv_config = get_csv_output_config()
    precision = csv_config["float_precision"]

    # Create a copy to avoid modifying original
    export_df = results_df.copy()

    # Format columns based on their type
    for col in export_df.columns:
        if col in export_df.select_dtypes(include=["float64", "float32"]).columns:
            if col in PARAMETERS_WITH_EXPONENTIAL_FORMAT:
                # Exponential format for small parameters
                export_df[col] = export_df[col].apply(
                    lambda x: f"{x:.{precision}e}" if pd.notna(x) else x
                )
            else:
                # Standard float format for physics results
                export_df[col] = export_df[col].apply(
                    lambda x: f"{x:.{precision}f}" if pd.notna(x) else x
                )
        elif col in PARAMETERS_OF_INTEGER_VALUE:
            # Ensure integer columns stay as integers
            export_df[col] = export_df[col].astype("Int64")  # Nullable integer type

    # Reorder columns (physics parameters first)
    export_df = _reorder_columns_for_export(export_df)

    # Export without float_format (we've already formatted)
    export_df.to_csv(
        output_csv_path,
        index=csv_config["index"],
    )

    logger.info(f"Exported {len(export_df)} results to {output_csv_path}")


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
