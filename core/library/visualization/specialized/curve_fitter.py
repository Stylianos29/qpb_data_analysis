"""
Curve fitting functionality for visualization components.

This module provides the CurveFitter class for fitting mathematical
functions to data points and displaying the results on plots.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.colors import rgb2hex
from scipy.optimize import curve_fit
import gvar
import lsqfit


class CurveFitter:
    """
    Handles curve fitting operations for plot data.

    This class provides functionality for:
    - Fitting various mathematical functions to data
    - Supporting both scipy and gvar-based fitting
    - Displaying fit curves on plots
    - Formatting and displaying fit parameters
    - Handling fit failures gracefully

    Supports fit functions:
    - Linear: y = a*x + b
    - Exponential: y = a*exp(b*x) or y = a*exp(-b*x) + c
    - Power law: y = a*x^b
    - Shifted power law: y = a/(x-b) + c
    """

    def __init__(self):
        """Initialize the curve fitter."""
        # Available fit functions for scipy
        self.scipy_functions = {
            "linear": self._linear_func,
            "exponential": self._exponential_func,
            "power_law": self._power_law_func,
            "shifted_power_law": self._shifted_power_law_func,
        }

        # Available fit functions for gvar/lsqfit
        self.gvar_functions = {
            "linear": self._linear_gvar_func,
            "exponential": self._exponential_gvar_func,
            "power_law": self._power_law_gvar_func,
            "shifted_power_law": self._shifted_power_law_gvar_func,
        }

        # Initial parameter guesses
        self.initial_params = {
            "linear": [1.0, 0.0],
            "exponential": [1.0, 1.0, 0.0],
            "power_law": [1.0, 1.0],
            "shifted_power_law": [1.0, 0.0, 0.0],
        }

        # Fit label positions
        self.label_positions = {
            "top left": ((0.05, 0.95), ("left", "top")),
            "top right": ((0.95, 0.95), ("right", "top")),
            "bottom left": ((0.05, 0.05), ("left", "bottom")),
            "bottom right": ((0.95, 0.05), ("right", "bottom")),
            "center": ((0.5, 0.5), ("center", "center")),
        }

    def apply_fit(
        self,
        ax: Axes,
        x_data: np.ndarray,
        y_data: np.ndarray,
        fit_function: str,
        show_parameters: bool = True,
        curve_style: Optional[Dict[str, Any]] = None,
        parameter_format: str = ".2e",
        label_location: str = "top left",
        index_range: Optional[slice] = None,
        curve_range: Optional[Tuple[float, float]] = None,
        num_points: int = 200,
        series_color: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply curve fitting to data and display results.

        Parameters:
        -----------
        - ax : matplotlib.axes.Axes
            The axes to add the fit curve to
        - x_data : np.ndarray
            X-axis data for fitting
        - y_data : np.ndarray
            Y-axis data for fitting (can contain tuples for
            uncertainties)
        - fit_function : str
            Name of the function to fit ('linear', 'exponential',
            'power_law', 'shifted_power_law')
        - show_parameters : bool, optional
            Whether to display fit parameters on the plot
        - curve_style : dict, optional
            Style parameters for the fit curve
        - parameter_format : str, optional
            Format string for displaying parameters
        - label_location : str, optional
            Location for parameter labels on the plot
        - index_range : slice, optional
            Range of data indices to use for fitting
        - curve_range : tuple, optional
            (min, max) range for displaying the fit curve
        - num_points : int, optional
            Number of points to use for the fit curve
        - series_color : str, optional
            Color to use for the fit curve (auto-detected if None)

        Returns:
        --------
        dict or None
            Dictionary containing fit results and parameters, or None if
            fit failed
        """
        try:
            # Apply index range if specified
            if index_range is not None:
                x_data = x_data[index_range]
                y_data = y_data[index_range]

            # Check if we have uncertainty data (tuples)
            has_uncertainties = self._has_uncertainties(y_data)

            if has_uncertainties:
                return self._apply_gvar_fit(
                    ax,
                    x_data,
                    y_data,
                    fit_function,
                    show_parameters,
                    curve_style,
                    parameter_format,
                    label_location,
                    curve_range,
                    num_points,
                    series_color,
                )
            else:
                return self._apply_scipy_fit(
                    ax,
                    x_data,
                    y_data,
                    fit_function,
                    show_parameters,
                    curve_style,
                    parameter_format,
                    label_location,
                    curve_range,
                    num_points,
                    series_color,
                )

        except Exception as e:
            warnings.warn(f"Fit failed: {e}")
            return None

    def _apply_scipy_fit(
        self,
        ax: Axes,
        x_data: np.ndarray,
        y_data: np.ndarray,
        fit_function: str,
        show_parameters: bool,
        curve_style: Optional[Dict[str, Any]],
        parameter_format: str,
        label_location: str,
        curve_range: Optional[Tuple[float, float]],
        num_points: int,
        series_color: Optional[str],
    ) -> Dict[str, Any]:
        """Apply scipy-based curve fitting."""
        if fit_function not in self.scipy_functions:
            raise ValueError(f"Unsupported fit function: '{fit_function}'")

        # Get the fitting function
        func = self.scipy_functions[fit_function]

        # Remove NaN values
        valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        x_fit = x_data[valid_mask]
        y_fit = y_data[valid_mask]

        # Special validation for power law
        if fit_function == "power_law" and np.any(x_fit <= 0):
            raise ValueError("Power-law fit requires strictly positive x values.")

        # Perform the fit
        initial_guess = self.initial_params[fit_function]
        fit_params, covariance = curve_fit(func, x_fit, y_fit, p0=initial_guess)

        # Generate fit curve
        if curve_range is not None:
            x_curve = np.linspace(curve_range[0], curve_range[1], num_points)
        else:
            x_curve = np.linspace(np.min(x_fit), np.max(x_fit), num_points)

        y_curve = func(x_curve, *fit_params)

        # Plot the fit curve
        style = self._get_curve_style(curve_style, series_color, ax)
        ax.plot(x_curve, y_curve, **style)

        # Display parameters if requested
        if show_parameters:
            self._display_scipy_parameters(
                ax, fit_params, fit_function, parameter_format, label_location
            )

        # Calculate fit quality metrics
        y_pred = func(x_fit, *fit_params)
        r_squared = self._calculate_r_squared(y_fit, y_pred)

        return {
            "function": fit_function,
            "parameters": fit_params,
            "covariance": covariance,
            "r_squared": r_squared,
            "x_curve": x_curve,
            "y_curve": y_curve,
            "method": "scipy",
        }

    def _apply_gvar_fit(
        self,
        ax: Axes,
        x_data: np.ndarray,
        y_data: np.ndarray,
        fit_function: str,
        show_parameters: bool,
        curve_style: Optional[Dict[str, Any]],
        parameter_format: str,
        label_location: str,
        curve_range: Optional[Tuple[float, float]],
        num_points: int,
        series_color: Optional[str],
    ) -> Dict[str, Any]:
        """Apply gvar-based curve fitting with uncertainties."""
        if fit_function not in self.gvar_functions:
            raise ValueError(f"Unsupported fit function for gvar: '{fit_function}'")

        # Extract values and uncertainties
        y_values = np.array([t[0] for t in y_data])
        y_errors = np.array([t[1] for t in y_data])

        # Create gvar array
        y_gv = gvar.gvar(y_values, y_errors)
        x_data = np.asarray(x_data, dtype=float)

        # Get the fitting function
        func = self.gvar_functions[fit_function]

        # Perform the fit
        initial_guess = self.initial_params[fit_function]
        fit_result = lsqfit.nonlinear_fit(
            data=(x_data, y_gv),
            fcn=func,
            p0=initial_guess,
            debug=False,
        )

        # Generate fit curve
        if curve_range is not None:
            x_curve = np.linspace(curve_range[0], curve_range[1], num_points)
        else:
            x_curve = np.linspace(np.min(x_data), np.max(x_data), num_points)

        fit_params = gvar.mean(fit_result.p)
        y_curve = func(x_curve, fit_params)

        # Plot the fit curve
        style = self._get_curve_style(curve_style, series_color, ax)
        ax.plot(x_curve, gvar.mean(y_curve), **style)

        # Display parameters if requested
        if show_parameters:
            self._display_gvar_parameters(
                ax, fit_result.p, fit_function, parameter_format, label_location
            )

        return {
            "function": fit_function,
            "parameters": fit_result.p,
            "fit_result": fit_result,
            "x_curve": x_curve,
            "y_curve": y_curve,
            "method": "gvar",
        }

    def _has_uncertainties(self, y_data: np.ndarray) -> bool:
        """Check if y_data contains uncertainty tuples."""
        if len(y_data) == 0:
            return False

        # Check if first element is a tuple with 2 elements
        return isinstance(y_data[0], tuple) and len(y_data[0]) == 2

    def _get_curve_style(
        self,
        curve_style: Optional[Dict[str, Any]],
        series_color: Optional[str],
        ax: Axes,
    ) -> Dict[str, Any]:
        """Get the style dictionary for the fit curve."""
        if curve_style is not None:
            return curve_style

        # Auto-detect color from the plot
        if series_color is None:
            series_color = self._detect_series_color(ax)

        if series_color:
            rgba = to_rgba(series_color)
            lighter_rgba = (*rgba[:3], 0.5)  # Make transparent
            return {"color": lighter_rgba, "linestyle": "--"}
        else:
            return {"color": "gray", "linestyle": "--", "alpha": 0.7}

    def _detect_series_color(self, ax: Axes) -> Optional[str]:
        """Detect the color of the most recent data series."""
        try:
            # Try to get color from the last line
            if ax.lines:
                color = ax.lines[-1].get_color()
                return rgb2hex(color) if color is not None else None

            # Try to get color from the last scatter plot
            if hasattr(ax, "collections") and ax.collections:
                collection = ax.collections[-1]
                color = collection.get_facecolor()

                if len(color) > 0:
                    first_color = color[0]
                    # Ensure first_color is a tuple/list of 3 or 4 floats
                    if isinstance(first_color, (list, tuple, np.ndarray)) and len(
                        first_color
                    ) in (3, 4):
                        return rgb2hex(first_color)
                    # If it's a single float (invalid), skip

            return None

        except Exception:
            return None

    def _display_scipy_parameters(
        self,
        ax: Axes,
        parameters: np.ndarray,
        fit_function: str,
        parameter_format: str,
        label_location: str,
    ) -> None:
        """Display fit parameters on the plot for scipy fits."""
        position, alignment = self.label_positions.get(
            label_location, ((0.05, 0.95), ("left", "top"))
        )

        # Format parameters based on function type
        if fit_function == "exponential" and len(parameters) == 3:
            # For exponential: y = a*exp(-b*x) + c
            a_fmt = format(parameters[0], parameter_format)
            b_fmt = format(parameters[1], parameter_format)
            c_fmt = format(parameters[2], parameter_format)
            param_text = f"a={a_fmt}, b={b_fmt}, c={c_fmt}"
        else:
            # For other functions: generic a, b, c, ... labeling
            param_names = ["a", "b", "c", "d"][: len(parameters)]
            param_text = ", ".join(
                f"{name}={val:{parameter_format}}"
                for name, val in zip(param_names, parameters)
            )

        ax.text(
            *position,
            param_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment=alignment[1],
            horizontalalignment=alignment[0],
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    def _display_gvar_parameters(
        self,
        ax: Axes,
        parameters: List[Any],
        fit_function: str,
        parameter_format: str,
        label_location: str,
    ) -> None:
        """Display fit parameters on the plot for gvar fits."""
        position, alignment = self.label_positions.get(
            label_location, ((0.05, 0.95), ("left", "top"))
        )

        # Format parameters with uncertainties
        param_means = gvar.mean(parameters)

        if fit_function == "exponential" and len(param_means) == 3:
            a_fmt = format(param_means[0], parameter_format)
            b_fmt = format(param_means[1], parameter_format)
            c_fmt = format(param_means[2], parameter_format)
            param_text = f"a={a_fmt}, b={b_fmt}, c={c_fmt}"
        else:
            param_names = ["a", "b", "c", "d"][: len(param_means)]
            param_text = ", ".join(
                f"{name}={val:{parameter_format}}"
                for name, val in zip(param_names, param_means)
            )

        ax.text(
            *position,
            param_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment=alignment[1],
            horizontalalignment=alignment[0],
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    def _calculate_r_squared(
        self, y_actual: np.ndarray, y_predicted: np.ndarray
    ) -> float:
        """Calculate R-squared value for fit quality."""
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

        if ss_tot == 0:
            return 1.0  # Perfect fit case

        return 1 - (ss_res / ss_tot)

    # Scipy fit functions
    def _linear_func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Linear function: y = a*x + b"""
        return a * x + b

    def _exponential_func(
        self, x: np.ndarray, a: float, b: float, c: float = 0
    ) -> np.ndarray:
        """Exponential function: y = a*exp(-b*x) + c"""
        return a * np.exp(-b * x) + c

    def _power_law_func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Power law function: y = a*x^b"""
        return a * x**b

    def _shifted_power_law_func(
        self, x: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """Shifted power law function: y = a/(x-b) + c"""
        return a / (x - b) + c

    # Gvar fit functions
    def _linear_gvar_func(self, x: np.ndarray, p: List[Any]) -> np.ndarray:
        """Linear function for gvar: y = p[0]*x + p[1]"""
        return p[0] * x + p[1]

    def _exponential_gvar_func(self, x: np.ndarray, p: List[Any]) -> np.ndarray:
        """Exponential function for gvar: y = p[0]*exp(-p[1]*x) +
        p[2]"""
        return p[0] * np.exp(-p[1] * x) + p[2]

    def _power_law_gvar_func(self, x: np.ndarray, p: List[Any]) -> np.ndarray:
        """Power law function for gvar: y = p[0]*x^p[1]"""
        return p[0] * x ** p[1]

    def _shifted_power_law_gvar_func(self, x: np.ndarray, p: List[Any]) -> np.ndarray:
        """Shifted power law function for gvar: y = p[0]/(x-p[1]) +
        p[2]"""
        return p[0] / (x - p[1]) + p[2]

    def get_supported_functions(self) -> List[str]:
        """Get list of supported fit functions."""
        return list(self.scipy_functions.keys())

    def add_custom_function(
        self,
        name: str,
        scipy_func: Callable,
        gvar_func: Callable,
        initial_params: List[float],
    ) -> None:
        """
        Add a custom fitting function.

        Parameters:
        -----------
        - name : str
            Name of the custom function
        - scipy_func : callable
            Function for scipy fitting: func(x, *params)
        - gvar_func : callable
            Function for gvar fitting: func(x, params_list)
        - initial_params : list
            Initial parameter guesses
        """
        self.scipy_functions[name] = scipy_func
        self.gvar_functions[name] = gvar_func
        self.initial_params[name] = initial_params

    def format_fit_results(
        self, fit_results: Dict[str, Any], include_uncertainties: bool = True
    ) -> str:
        """
        Format fit results as a human-readable string.

        Parameters:
        -----------
        - fit_results : dict
            Results from apply_fit method
        - include_uncertainties : bool
            Whether to include uncertainty information

        Returns:
        --------
        str
            Formatted string with fit results
        """
        if fit_results is None:
            return "Fit failed"

        lines = [f"Fit function: {fit_results['function']}"]

        if fit_results["method"] == "scipy":
            params = fit_results["parameters"]
            param_names = ["a", "b", "c", "d"][: len(params)]

            for name, value in zip(param_names, params):
                lines.append(f"{name} = {value:.6f}")

            if "r_squared" in fit_results:
                lines.append(f"R² = {fit_results['r_squared']:.6f}")

        elif fit_results["method"] == "gvar" and include_uncertainties:
            params = fit_results["parameters"]
            param_names = ["a", "b", "c", "d"][: len(params)]

            for name, value in zip(param_names, params):
                lines.append(f"{name} = {value}")

        return "\n".join(lines)
