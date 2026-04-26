"""
Unit tests for the ``per_figure_overrides`` feature of
``DataPlotter.plot()``.

This test module focuses on the per-figure override callback added to
``DataPlotter.plot()``. The callback is invoked once per outer-grouping
figure with that figure's metadata dict and may return a dict of kwarg
overrides (or signal a skip) for that figure only.

The tests are organized around five behavioral contracts:

1. Backward compatibility - omitting ``per_figure_overrides`` produces
   exactly the same outputs as before the feature was introduced.
2. Callable invocation - the callback is called once per figure with the
   right metadata.
3. Override layering - returned overrides win over plot()-level kwargs;
   non-overridden kwargs retain their plot()-level values; ``None`` and
   ``{}`` mean "no overrides for this figure".
4. Skip semantics - ``{"skip_figure": True}`` (or ``{SKIP_FIGURE:
   True}``) suppresses that figure's output without affecting the
   others.
5. Validation - non-dict returns, structural-key overrides, and
   exceptions inside the callback all raise informative errors.

The tests use a small in-memory DataFrame and run the real
``DataPlotter`` end-to-end (with the ``Agg`` matplotlib backend) against
``tmp_path``, rather than mocking the loop body. The point of the
feature is its integration with the rest of ``plot()``, so testing the
integration is what gives us confidence.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing
import matplotlib.pyplot as plt

from library.visualization.plotters.data_plotter import (
    DataPlotter,
    SKIP_FIGURE,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def two_kernel_dataframe():
    """
    DataFrame with a two-figure outer grouping and a three-curve inner
    grouping per figure.

    Columns:
        - Kernel_operator_type: outer grouping (2 values: Wilson,
          Brillouin); produces 2 figures.
        - KL_diagonal_order: inner grouping_variable (3 values: 1, 2,
          3); produces 3 curves per figure.
        - Bare_mass: x-axis variable (numeric).
        - Plateau_PCAC_mass: y-axis variable (numeric output).

    Total: 2 figures x 3 curves x 4 points-per-curve = 24 rows.
    """
    rows = []
    for kernel in ["Wilson", "Brillouin"]:
        for order in [1, 2, 3]:
            for bare_mass in [0.01, 0.02, 0.03, 0.04]:
                rows.append(
                    {
                        "Kernel_operator_type": kernel,
                        "KL_diagonal_order": order,
                        "Bare_mass": bare_mass,
                        "Plateau_PCAC_mass": bare_mass * (1 + 0.1 * order)
                        + (0.0 if kernel == "Wilson" else 0.005),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def plotter(two_kernel_dataframe, tmp_path):
    """A DataPlotter ready to plot Plateau_PCAC_mass vs Bare_mass."""
    p = DataPlotter(two_kernel_dataframe, str(tmp_path))
    p.set_plot_variables("Bare_mass", "Plateau_PCAC_mass")
    return p


def _saved_files(tmp_path):
    """Return a sorted list of file paths under tmp_path matching the
    typical plot outputs."""
    return sorted(p for p in tmp_path.rglob("*") if p.is_file())


# =============================================================================
# Contract 1 - Backward compatibility (no per_figure_overrides)
# =============================================================================


class TestBackwardCompatibility:
    """Verify that omitting per_figure_overrides preserves prior
    behavior: the loop still runs, both figures are produced, both end
    up in _last_plot_figures, and the saved files exist on disk.
    """

    def test_baseline_plot_without_overrides_produces_two_figures(
        self, plotter, tmp_path
    ):
        """No per_figure_overrides: expect one figure per kernel."""
        result = plotter.plot(
            grouping_variable="KL_diagonal_order",
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        # Method chaining works
        assert result is plotter

        # Two figures registered in _last_plot_figures
        assert len(plotter._last_plot_figures) == 2
        assert len(plotter._last_plot_paths) == 2

        # Two files saved
        files = _saved_files(tmp_path)
        assert len(files) == 2
        for f in files:
            assert f.suffix == ".pdf"
            assert f.stat().st_size > 0  # non-empty output

    def test_explicit_none_callable_is_equivalent_to_omitting_it(
        self, plotter, tmp_path
    ):
        """per_figure_overrides=None must behave exactly like the
        parameter not being passed."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=None,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        assert len(plotter._last_plot_figures) == 2
        assert len(_saved_files(tmp_path)) == 2


# =============================================================================
# Contract 2 - Callable invocation
# =============================================================================


class TestCallableInvocation:
    """Verify that the callback is called once per figure with the
    correct metadata dict."""

    def test_callback_called_once_per_figure(self, plotter, tmp_path):
        """Two outer-grouping figures should produce two callback
        invocations."""
        invocations = []

        def callback(meta):
            invocations.append(dict(meta))  # snapshot, not reference
            return {}

        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=callback,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        assert len(invocations) == 2

    def test_metadata_contains_outer_grouping_axis(self, plotter, tmp_path):
        """For each figure, the metadata dict must include the
        outer-grouping axis (Kernel_operator_type)."""
        seen_kernels = []

        def callback(meta):
            seen_kernels.append(meta.get("Kernel_operator_type"))
            return {}

        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=callback,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        # Both kernels should have been observed exactly once
        assert sorted(seen_kernels) == ["Brillouin", "Wilson"]

    def test_metadata_passed_is_a_dict(self, plotter, tmp_path):
        """The argument to the callback must be a dict (not e.g. a
        tuple or a NamedTuple), to match the documented contract."""
        observed_types = []

        def callback(meta):
            observed_types.append(type(meta).__name__)
            return {}

        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=callback,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        assert all(t == "dict" for t in observed_types)
        assert len(observed_types) == 2


# =============================================================================
# Contract 3 - Override layering
# =============================================================================


class TestOverrideLayering:
    """Verify how returned override dicts interact with plot()-level
    kwargs."""

    def test_returning_empty_dict_changes_nothing(self, plotter, tmp_path):
        """A callback that always returns {} must produce the same
        output as no callback at all (number of figures, files saved)."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=lambda meta: {},
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        assert len(plotter._last_plot_figures) == 2
        assert len(_saved_files(tmp_path)) == 2

    def test_returning_none_changes_nothing(self, plotter, tmp_path):
        """A callback that returns None for some figures behaves the
        same as returning {} for those figures."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=lambda meta: None,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        assert len(plotter._last_plot_figures) == 2
        assert len(_saved_files(tmp_path)) == 2

    def test_overridden_kwarg_takes_effect(self, plotter, tmp_path):
        """An overridden xlim should be visible on the saved figure's
        axes object."""

        # Force one kernel's xlim to a distinctive value; leave the
        # other untouched so we can assert the difference.
        def callback(meta):
            if meta.get("Kernel_operator_type") == "Brillouin":
                return {"xlim": (10.0, 20.0)}
            return {}

        plotter.plot(
            grouping_variable="KL_diagonal_order",
            xlim=(0.0, 1.0),  # plot()-level default
            per_figure_overrides=callback,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        # _last_plot_figures is keyed by group_keys; we don't know the
        # exact keys, so introspect each figure to find the right one.
        kernel_to_xlim = {}
        for group_keys, (fig, ax, group_df) in plotter._last_plot_figures.items():
            kernel_value = group_df["Kernel_operator_type"].iloc[0]
            kernel_to_xlim[kernel_value] = ax.get_xlim()

        assert kernel_to_xlim["Brillouin"] == (10.0, 20.0)
        assert kernel_to_xlim["Wilson"] == (0.0, 1.0)

    def test_non_overridden_kwarg_retains_plot_level_value(self, plotter, tmp_path):
        """If the callback returns {'xlim': ...}, the ylim should still
        come from the plot()-level call."""

        def callback(meta):
            return {"xlim": (10.0, 20.0)}  # no ylim in the override

        plotter.plot(
            grouping_variable="KL_diagonal_order",
            xlim=(0.0, 1.0),
            ylim=(-0.5, 0.5),  # plot()-level
            per_figure_overrides=callback,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        for _, (fig, ax, _) in plotter._last_plot_figures.items():
            assert ax.get_xlim() == (10.0, 20.0)
            assert ax.get_ylim() == (-0.5, 0.5)


# =============================================================================
# Contract 4 - Skip semantics
# =============================================================================


class TestSkipFigure:
    """Verify that a figure can be skipped entirely via the override
    return value."""

    def test_skip_one_figure_via_string_literal(self, plotter, tmp_path):
        """skip_figure=True suppresses that figure but not the others."""

        def callback(meta):
            if meta.get("Kernel_operator_type") == "Brillouin":
                return {"skip_figure": True}
            return {}

        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=callback,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        # Only Wilson should have made it to _last_plot_figures
        assert len(plotter._last_plot_figures) == 1
        assert len(plotter._last_plot_paths) == 1
        kernels_kept = [
            group_df["Kernel_operator_type"].iloc[0]
            for _, _, group_df in plotter._last_plot_figures.values()
        ]
        assert kernels_kept == ["Wilson"]

        # Only one file on disk
        assert len(_saved_files(tmp_path)) == 1

    def test_skip_via_constant_alias(self, plotter, tmp_path):
        """SKIP_FIGURE constant should work identically to the literal
        string 'skip_figure'."""

        def callback(meta):
            if meta.get("Kernel_operator_type") == "Wilson":
                return {SKIP_FIGURE: True}
            return {}

        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=callback,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        assert len(plotter._last_plot_figures) == 1
        assert len(_saved_files(tmp_path)) == 1

    def test_skip_false_does_not_skip(self, plotter, tmp_path):
        """{'skip_figure': False} must be treated as 'do not skip',
        not as a structural error or no-op-with-warning."""
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=lambda meta: {"skip_figure": False},
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        assert len(plotter._last_plot_figures) == 2
        assert len(_saved_files(tmp_path)) == 2

    def test_skip_all_figures_yields_empty_output(self, plotter, tmp_path):
        """If every figure is skipped, no files are saved and
        _last_plot_figures stays empty - but plot() returns normally."""
        result = plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=lambda meta: {SKIP_FIGURE: True},
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        assert result is plotter  # method chaining still works
        assert len(plotter._last_plot_figures) == 0
        assert len(plotter._last_plot_paths) == 0
        assert len(_saved_files(tmp_path)) == 0

    def test_skip_signal_does_not_leak_into_kwargs(self, plotter, tmp_path):
        """A skip flag should be removed from the merged kwargs even
        in branches where it's False. The non-skipped figure must not
        receive 'skip_figure' as a stray kwarg."""

        # If the skip flag leaked into base_plot_kwargs, the second
        # iteration would either KeyError (if skip_figure isn't a
        # rebind target) or silently apply it. Both are bugs.
        def callback(meta):
            if meta.get("Kernel_operator_type") == "Brillouin":
                return {SKIP_FIGURE: True}
            return {SKIP_FIGURE: False, "xlim": (0.0, 1.0)}

        # If this raises or silently misbehaves, the test fails.
        plotter.plot(
            grouping_variable="KL_diagonal_order",
            per_figure_overrides=callback,
            include_legend=False,
            include_plot_title=False,
            file_format="pdf",
            verbose=False,
        )

        # Wilson must have been kept with its xlim override applied
        assert len(plotter._last_plot_figures) == 1
        for _, (fig, ax, _) in plotter._last_plot_figures.items():
            assert ax.get_xlim() == (0.0, 1.0)


# =============================================================================
# Contract 5 - Validation
# =============================================================================


class TestValidation:
    """Verify that misuse of per_figure_overrides surfaces as
    informative errors at the right point."""

    def test_non_dict_return_raises_type_error(self, plotter, tmp_path):
        """Returning a list (or any non-dict, non-None) from the
        callback must raise TypeError."""
        with pytest.raises(TypeError, match="must return a dict or None"):
            plotter.plot(
                grouping_variable="KL_diagonal_order",
                per_figure_overrides=lambda meta: ["not", "a", "dict"],
                include_legend=False,
                include_plot_title=False,
                file_format="pdf",
                verbose=False,
            )

    def test_string_return_raises_type_error(self, plotter, tmp_path):
        """A string is technically a Mapping-adjacent type, but it is
        not a dict and must also be rejected."""
        with pytest.raises(TypeError, match="must return a dict or None"):
            plotter.plot(
                grouping_variable="KL_diagonal_order",
                per_figure_overrides=lambda meta: "skip_figure",
                include_legend=False,
                include_plot_title=False,
                file_format="pdf",
                verbose=False,
            )

    @pytest.mark.parametrize(
        "forbidden_key",
        [
            "grouping_variable",
            "excluded_from_grouping_list",
            "styling_variable",
            "target_ax",
            "is_inset",
            "verbose",
            "per_figure_overrides",
        ],
    )
    def test_structural_keys_are_rejected(self, plotter, tmp_path, forbidden_key):
        """Each structurally-forbidden key must trigger ValueError when
        present in the override dict."""
        with pytest.raises(ValueError, match="structural"):
            plotter.plot(
                grouping_variable="KL_diagonal_order",
                per_figure_overrides=lambda meta: {forbidden_key: "anything"},
                include_legend=False,
                include_plot_title=False,
                file_format="pdf",
                verbose=False,
            )

    def test_structural_error_message_includes_metadata(self, plotter, tmp_path):
        """The error must include the offending figure's metadata so
        the user can locate which figure tripped it."""
        with pytest.raises(ValueError) as excinfo:
            plotter.plot(
                grouping_variable="KL_diagonal_order",
                per_figure_overrides=lambda meta: {"grouping_variable": "Bare_mass"},
                include_legend=False,
                include_plot_title=False,
                file_format="pdf",
                verbose=False,
            )

        msg = str(excinfo.value)
        assert "grouping_variable" in msg
        # One kernel value must appear in the error message because
        # metadata is included.
        assert "Wilson" in msg or "Brillouin" in msg

    def test_callback_exception_is_wrapped_with_metadata(self, plotter, tmp_path):
        """If the user's callback raises, the wrapper re-raises a
        RuntimeError that names the original exception type and includes
        the figure's metadata for debugging."""

        def boom(meta):
            raise RuntimeError("user code blew up")

        with pytest.raises(RuntimeError) as excinfo:
            plotter.plot(
                grouping_variable="KL_diagonal_order",
                per_figure_overrides=boom,
                include_legend=False,
                include_plot_title=False,
                file_format="pdf",
                verbose=False,
            )

        msg = str(excinfo.value)
        # The wrapper mentions the original exception type
        assert "RuntimeError" in msg
        # ... and the user's exception message
        assert "user code blew up" in msg
        # ... and includes some metadata for debugging
        assert "Wilson" in msg or "Brillouin" in msg

    def test_non_callback_value_error_has_actionable_message(self, plotter, tmp_path):
        """ValueError from the structural check should mention the
        figure's metadata, the offending key, and a hint about why."""
        with pytest.raises(ValueError) as excinfo:
            plotter.plot(
                grouping_variable="KL_diagonal_order",
                per_figure_overrides=lambda meta: {"styling_variable": "X"},
                include_legend=False,
                include_plot_title=False,
                file_format="pdf",
                verbose=False,
            )
        msg = str(excinfo.value)
        assert "styling_variable" in msg
        # 'structural' or 'plot() call site' should appear so the
        # user knows where to put the offending kwarg instead.
        assert "structural" in msg or "plot()" in msg


# =============================================================================
# Resolver unit tests (in isolation, no plotting)
# =============================================================================


class TestResolverInIsolation:
    """Pure unit tests of _resolve_per_figure_overrides. These don't
    plot anything; they exercise the resolver method directly to keep
    the validation logic well-covered without paying the cost of full
    plot() runs."""

    @pytest.fixture
    def base_kwargs(self):
        return {"xlim": (0, 1), "color_index_shift": 0, "marker_size": 8}

    @pytest.fixture
    def metadata(self):
        return {"Kernel_operator_type": "Wilson", "Bare_mass": 0.04}

    def test_none_callback_returns_base_kwargs_unchanged(
        self, plotter, base_kwargs, metadata
    ):
        result = plotter._resolve_per_figure_overrides(None, metadata, base_kwargs)
        assert result is base_kwargs  # no copy needed for the no-op path

    def test_callback_returning_none_returns_base_kwargs(
        self, plotter, base_kwargs, metadata
    ):
        result = plotter._resolve_per_figure_overrides(
            lambda m: None, metadata, base_kwargs
        )
        assert result == base_kwargs

    def test_callback_returning_empty_dict_returns_base_equivalent(
        self, plotter, base_kwargs, metadata
    ):
        result = plotter._resolve_per_figure_overrides(
            lambda m: {}, metadata, base_kwargs
        )
        assert result == base_kwargs
        # And must not have mutated base_kwargs
        assert base_kwargs == {
            "xlim": (0, 1),
            "color_index_shift": 0,
            "marker_size": 8,
        }

    def test_overrides_layer_on_top_of_base(self, plotter, base_kwargs, metadata):
        result = plotter._resolve_per_figure_overrides(
            lambda m: {"xlim": (5, 10), "marker_size": 99},
            metadata,
            base_kwargs,
        )
        assert result["xlim"] == (5, 10)
        assert result["marker_size"] == 99
        # Untouched key preserved
        assert result["color_index_shift"] == 0

    def test_skip_signal_returns_none(self, plotter, base_kwargs, metadata):
        result = plotter._resolve_per_figure_overrides(
            lambda m: {SKIP_FIGURE: True}, metadata, base_kwargs
        )
        assert result is None

    def test_skip_signal_takes_precedence_over_other_keys(
        self, plotter, base_kwargs, metadata
    ):
        """If skip_figure=True is set, the rest of the override dict
        is irrelevant - the figure is skipped."""
        result = plotter._resolve_per_figure_overrides(
            lambda m: {SKIP_FIGURE: True, "xlim": (5, 10)},
            metadata,
            base_kwargs,
        )
        assert result is None

    def test_skip_false_does_not_appear_in_result(self, plotter, base_kwargs, metadata):
        """A skip_figure=False entry must be stripped from the merged
        result - it should not leak into downstream kwargs."""
        result = plotter._resolve_per_figure_overrides(
            lambda m: {SKIP_FIGURE: False, "xlim": (5, 10)},
            metadata,
            base_kwargs,
        )
        assert result is not None
        assert SKIP_FIGURE not in result
        assert result["xlim"] == (5, 10)
