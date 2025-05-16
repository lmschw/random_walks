"""
copied and adapted from https://github.com/garland-culbreth/pymdea/
"""

import logging
from pathlib import Path
from typing import Literal, Self

import numpy as np
import polars as pl
import stochastic.processes.continuous
import stochastic.processes.noise
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from scipy import stats
from scipy.optimize import curve_fit

from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from random_walk_types.levy_walk import levy_walk_simulation, levy_walk, levy_walk_2
from random_walk_types.brownian_motion import brownian_motion_2d_without_sigma
from random_walk_types.correlated_random_walk import correlated_random_walk_2d
import data_loader

def _power_log(x: float, a: float, b: float) -> float:
    """Log power law for curve fit."""
    return (a * np.log(x)) + b

class DeaEngine:
    """Run diffusion entropy analysis."""

    def __init__(self: Self, data) -> Self:
        """Run diffusion entropy analysis."""
        self.data = data
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            TextColumn("elapsed"),
            TimeElapsedColumn(),
        )

    def _apply_stripes(self: Self) -> Self:
        """Round `data` to `stripes` evenly spaced intervals."""
        if np.min(self.data) <= 0:
            self.data = self.data + np.abs(np.min(self.data))
        elif np.min(self.data) > 0:
            self.data = self.data - np.abs(np.min(self.data))
        data_width = np.abs(self.data.max() - self.data.min())
        stripe_size = data_width / self.number_of_stripes
        self.series = self.data / stripe_size
        return self

    def _get_events(self: Self) -> Self:
        """Record an event (1) when `series` changes value."""
        no_upper_crossing = self.series[1:] < np.floor(self.series[:-1]) + 1
        no_lower_crossing = self.series[1:] > np.ceil(self.series[:-1]) - 1
        events = np.where(no_lower_crossing & no_upper_crossing, 0, 1)
        self.events = np.append([0], events)  # Event impossible at index 0
        return self

    def _make_trajectory(self: Self) -> Self:
        """Construct diffusion trajectory from events."""
        self.trajectory = np.cumsum(self.events)
        return self

    def _calculate_entropy(self: Self, window_stop: float = 0.25) -> Self:
        """Calculate the Shannon Entropy of the diffusion trajectory.

        Parameters
        ----------
        window_stop : float, optional, default: 0.25
            Proportion of total data length at which to cap the maximum
            window length. Large window lengths rarely produce
            informative entropy.

        """
        entropies = []
        window_lengths = np.unique(
            np.logspace(
                start=0,
                stop=np.log10(window_stop * len(self.trajectory)),
                num=1000,
                dtype=np.int32,
            ),
        )
        with self.progress as p:
            for window_length in p.track(window_lengths):
                window_starts = np.arange(0, len(self.trajectory) - window_length, 1)
                window_ends = np.arange(window_length, len(self.trajectory), 1)
                displacements = (
                    self.trajectory[window_ends] - self.trajectory[window_starts]
                )
                counts, bin_edge = np.histogram(displacements, bins="doane")
                counts = np.array(counts[counts != 0])
                binsize = bin_edge[1] - bin_edge[0]
                distribution = counts / np.sum(counts)
                entropies.append(
                    -np.sum(distribution * np.log(distribution)) + np.log(binsize),
                )
        self.entropies = np.asarray(entropies)
        self.window_lengths = window_lengths
        return self

    def _calculate_scaling(self: Self) -> Self:
        """Calculate scaling."""
        start_index = np.floor(self.fit_start * len(self.window_lengths)).astype(int)
        stop_index = np.floor(self.fit_stop * len(self.window_lengths)).astype(int)
        s_slice = self.entropies[start_index:stop_index]
        length_slice = self.window_lengths[start_index:stop_index]
        if self.fit_method == "ls":
            logging.warning(
                """Least-squares linear fits can introduce systematic error when
                applied to log-scale data. Prefer the more robust 'theilsen' or
                'siegel' methods.""",
            )
            coefficients = curve_fit(
                f=_power_log,
                xdata=length_slice,
                ydata=s_slice,
            )[0]  # 0 is coeffs, 1 is uncertainty, uncertainty not yet used
        if self.fit_method == "theilsen":
            coefficients = stats.theilslopes(s_slice, np.log(length_slice))
        if self.fit_method == "siegel":
            coefficients = stats.siegelslopes(s_slice, np.log(length_slice))
        self.length_slice = length_slice
        self.fit_coefficients = coefficients
        self.delta = coefficients[0]
        return self

    def _calculate_mu(self: Self) -> Self:
        """Calculate powerlaw index for inter-event time distribution.

        - mu1 is the index calculated by the rule `1 + delta`.
        - mu2 is the index calculated by the rule `1 + (1 / delta)`.

        Returns
        -------
        Self @ Engine
            Object containing the results and inputs of the diffusion
            entropy analysis.

        Notes
        -----
        mu is calculated by both rules. later both are plotted
        against the line relating delta and mu, to hopefully
        let users graphically determine the correct mu.

        """
        self.mu1 = 1 + self.delta
        self.mu2 = 1 + (1 / self.delta)
        return self

    def print_result(self: Self) -> str:
        """Print out result of analysis."""
        self.result = Table(title="Result", box=box.SIMPLE)
        self.result.add_column("δ")
        self.result.add_column("μ (rule 1)")
        self.result.add_column("μ (rule 2)")
        self.result.add_row(f"{self.delta:.5f}", f"{self.mu1:.5f}", f"{self.mu2:.5f}")
        console = Console()
        console.print(self.result)
        return self

    def analyze_with_stripes(
        self: Self,
        fit_start: int,
        fit_stop: int,
        fit_method: Literal["siegel", "theilsen", "ls"] = "siegel",
        n_stripes: int = 20,
    ) -> Self:
        """Run a modified diffusion entropy analysis.

        Parameters
        ----------
        fit_start : float
            Fraction of maximum window length at which to start linear fit.
        fit_stop : float
            Fraction of maximum window length at which to stop linear fit.
        fit_method : str {"siegel", "theilsen", "ls"}, optional
            Linear fit method to use. By default "siegel"
        n_stripes : int, optional, default: 20
            Number of stripes to apply to input time-series during
            analysis.

        Returns
        -------
        Self @ Engine
            Object containing the results and inputs of the diffusion
            entropy analysis.

        Raises
        ------
        ValueError
            If n_stripes < 2. At least two stripes must be applied for
            DEA to provide a meaningful result.

        Notes
        -----
        Prefer the siegel or theilsen methods. Least squares linear
        fits can introduce bias when done over log-scale data, see
        Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
        distributions in empirical data. SIAM review, 51(4), pp.661-703.
        https://doi.org/10.1137/070710111.
        https://arxiv.org/pdf/0706.1062.pdf.

        """
        if n_stripes < 2:  # noqa: PLR2004
            msg = "n_stripes must be greater than 1"
            raise ValueError(msg)
        self.number_of_stripes = n_stripes
        self.fit_start = fit_start
        self.fit_stop = fit_stop
        self.fit_method = fit_method
        self._apply_stripes()
        self._get_events()
        self._make_trajectory()
        self._calculate_entropy()
        self._calculate_scaling()
        self._calculate_mu()
        self.print_result()

    def analyze_without_stripes(
        self: Self,
        fit_start: int,
        fit_stop: int,
        fit_method: Literal["siegel", "theilsen", "ls"] = "siegel",
    ) -> Self:
        """Run a regular diffusion entropy analysis.

        Parameters
        ----------
        fit_start : float
            Fraction of maximum window length at which to start linear fit.
        fit_stop : float
            Fraction of maximum window length at which to stop linear fit.
        fit_method : str {"siegel", "theilsen", "ls"}, optional
            Linear fit method to use. By default "siegel"

        Returns
        -------
        Self @ Engine
            Object containing the results and inputs of the diffusion
            entropy analysis.

        Notes
        -----
        Prefer the siegel or theilsen methods. Least squares linear
        fits can introduce bias when done over log-scale data, see
        Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
        distributions in empirical data. SIAM review, 51(4), pp.661-703.
        https://doi.org/10.1137/070710111.
        https://arxiv.org/pdf/0706.1062.pdf.

        """
        self.trajectory = self.data
        self.fit_start = fit_start
        self.fit_stop = fit_stop
        self.fit_method = fit_method
        self._calculate_entropy()
        self._calculate_scaling()
        self._calculate_mu()
        self.print_result()

"""Plotting functions."""


class DeaPlotter:
    """Plot DEA results."""

    def __init__(
        self: Self,
        model: DeaEngine,
        theme: None | str = None,
    ) -> Self:
        """Plot DEA results.

        Parameters
        ----------
        model : Self@DeaEngine
            Object containing the results of a DEA analysis to be plotted.
        theme : None | str, optional, default: None
            Must be either None or a string corresponding to a
            matplotlib.pyplot style.

        """
        if theme is not None:
            plt.style.use(style=theme)
        self.window_lengths = model.window_lengths
        self.entropies = model.entropies
        self.delta = model.fit_coefficients[0]
        self.y_intercept = model.fit_coefficients[1]
        self.mu1 = model.mu1
        self.mu2 = model.mu2

    def s_vs_l(self: Self, fig_width: int = 4, fig_height: int = 3) -> None:
        """Plot the slope of entropy vs window length.

        Parameters
        ----------
        fig_width : int, optional, default: 4
            Width, in inches, of the figure.
        fig_height : int, optional, default: 3
            Height, in inches, of the figure.

        """
        x_line = np.linspace(start=1, stop=np.max(self.window_lengths), num=3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")
        ax.plot(
            self.window_lengths,
            self.entropies,
            linestyle="none",
            marker="o",
            markersize=3,
            fillstyle="none",
        )
        ax.plot(
            x_line,
            self.delta * np.log(x_line) + self.y_intercept,
            color="k",
            label=f"$\\delta = {np.round(self.delta, 3)}$",
        )
        ax.set_xscale("log")
        ax.set_xlabel("$\\ln(L)$")
        ax.set_ylabel("$S(L)$")
        ax.legend(loc=0)
        sns.despine(trim=True)
        self.fig_s_vs_l = fig


    def mu_candidates(self: Self, fig_width: int = 4, fig_height: int = 3) -> None:
        """Plot the possible values of mu.

        Parameters
        ----------
        fig_width : int, optional, default: 4
            Width, in inches, of the figure.
        fig_height : int, optional, default: 3
            Height, in inches, of the figure.

        """
        x1 = np.linspace(1, 2, 100)
        x2 = np.linspace(2, 3, 100)
        x3 = np.linspace(3, 4, 100)
        y1 = x1 - 1
        y2 = 1 / (x2 - 1)
        y3 = np.full(100, 0.5)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")
        ax.plot(x1, y1, color="k")
        ax.plot(x2, y2, color="k")
        ax.plot(x3, y3, color="k")
        ax.plot(
            self.mu1,
            self.delta,
            marker="o",
            fillstyle="none",
            markersize=5,
            markeredgewidth=2,
            linestyle="none",
            label=f"$\\mu$ = {np.round(self.mu1, 2)}",
        )
        ax.plot(
            self.mu2,
            self.delta,
            marker="o",
            fillstyle="none",
            markersize=5,
            markeredgewidth=2,
            linestyle="none",
            label=f"$\\mu$ = {np.round(self.mu2, 2)}",
        )
        ax.set_xticks(ticks=np.linspace(1, 4, 7))
        ax.set_yticks(ticks=np.linspace(0, 1, 5))
        ax.set_xlabel("$\\mu$")
        ax.set_ylabel("$\\delta$")
        ax.legend(loc=0)
        ax.grid(visible=True)
        sns.despine(left=True, bottom=True)
        self.fig_mu_candidates = fig


for n in [1, 25, 49, 100]:
    for type in ['brown', 'levy', 'correlated']:
        for i in range(1,2):
            filename = f"{type}_free_{n}_run{i}"
            print(filename)
            trajectory = data_loader.load_data(f"c:/Users/lschw/dev/mas-random-walk/mas_random_walk/results/2D/2025-03-20_11-21-14/{filename}.pickle")

            dea_engine = DeaEngine(trajectory)
            dea_engine.analyze_with_stripes(fit_start=0.1, fit_stop=0.9, n_stripes=60)

            dea_plot = DeaPlotter(dea_engine)
            dea_plot.s_vs_l()
            #dea_plot.mu_candidates()
            #plt.show()

