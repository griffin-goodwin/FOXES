import os
import re
import yaml
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from matplotlib import rcParams


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def setup_barlow_font():
    """Setup Barlow font for matplotlib plots"""
    try:
        # Try to find Barlow font with more specific search
        barlow_fonts = []
        for font in fm.fontManager.ttflist:
            if 'barlow' in font.name.lower() or 'barlow' in font.fname.lower():
                barlow_fonts.append(font.name)

        if barlow_fonts:
            rcParams['font.family'] = 'Barlow'
            print(f"Using Barlow font: {barlow_fonts[0]}")
        else:
            # Try alternative approach - directly specify font file
            barlow_path = '/usr/share/fonts/truetype/barlow/Barlow-Regular.ttf'
            barlow_path2 = os.path.expanduser('~/Library/Fonts/Barlow-Regular.otf')
            if os.path.exists(barlow_path):
                # Add the font file directly to matplotlib
                fm.fontManager.addfont(barlow_path)
                rcParams['font.family'] = 'Barlow'
                print(f"Using Barlow font from: {barlow_path}")
            elif os.path.exists(barlow_path2):
                fm.fontManager.addfont(barlow_path2)
                rcParams['font.family'] = 'Barlow'
                print(f"Using Barlow font from: {barlow_path2}")
            else:
                # Fallback to sans-serif
                rcParams['font.family'] = 'sans-serif'
                print("Barlow font not found, using default sans-serif")
    except Exception as e:
        print(f"Font setup error: {e}, using default font")


class FOXESEvaluator:
    """
    Solar flare evaluation system for FOXES model predictions.

    This class provides functionality for evaluating FOXES solar flare predictions
    against ground truth data. It includes quantitative metrics calculation and
    regression analysis visualization.

    Key Features:
        - Performance metrics calculation (MSE, RMSE, MAE, R², Pearson correlation)
        - Flare class-specific analysis (Below-B, B, C, M, X classes)
        - Gaussian NLL and predictive interval calibration
        - Optional per-timestamp patch flux and uncertainty maps
    """

    FLARE_CLASSES = {
        'Below-B': (0, 1e-7),
        'B': (1e-7, 1e-6),
        'C': (1e-6, 1e-5),
        'M': (1e-5, 1e-4),
        'X': (1e-4, np.inf),
    }

    GAUSSIAN_UNCERTAINTY_COLUMNS = {
        'prediction_normalized',
        'groundtruth_normalized',
        'variance_normalized',
    }
    AIA_WAVELENGTHS = (94, 131, 171, 193, 211, 304, 335)

    def __init__(self,
                 csv_path,
                 output_dir="./foxes_evaluation",
                 plot_background='black',
                 evaluate_uncertainty=True,
                 inference_patch_dir=None,
                 max_patch_maps=None,
                 aia_dir=None,
                 aia_wavelength=193):
        """
        Initialize the FOXES evaluation system.

        Args:
            csv_path (str): Path to model prediction results CSV
            output_dir (str): Base output directory for results
            plot_background (str): Regression plot background theme ('black' or 'white')
            inference_patch_dir (str, optional): Parent folder containing
                flux/ and patch_uncertainty/ NPY maps from inference.
            max_patch_maps (int, optional): Maximum number of timestamp figures.
            aia_dir (str, optional): Folder of matching seven-channel AIA NPYs.
            aia_wavelength (int): AIA channel to show beside the patch maps.
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.plot_background = (plot_background or 'black').lower()
        self.evaluate_uncertainty = _as_bool(evaluate_uncertainty)
        self.inference_patch_dir = (
            Path(inference_patch_dir).expanduser()
            if inference_patch_dir else None
        )
        if max_patch_maps is not None and (
            isinstance(max_patch_maps, bool)
            or not isinstance(max_patch_maps, int)
            or max_patch_maps <= 0
        ):
            raise ValueError("max_patch_maps must be a positive integer or null")
        self.max_patch_maps = max_patch_maps
        self.aia_dir = Path(aia_dir).expanduser() if aia_dir else None
        self.aia_wavelength = int(aia_wavelength)
        if self.aia_wavelength not in self.AIA_WAVELENGTHS:
            raise ValueError(
                f"aia_wavelength must be one of {self.AIA_WAVELENGTHS}"
            )

        # Create output directory structure
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.plots_dir = os.path.join(output_dir, "plots")

        for dir_path in [self.metrics_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize data holders
        self.df = None
        self.y_true = None
        self.y_pred = None
        self.has_uncertainty = False

    def load_data(self):
        """
        Load and prepare prediction data.

        Returns:
            None
        """
        self.df = pd.read_csv(self.csv_path)
        required = {'groundtruth', 'predictions'}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(
                f"Prediction CSV is missing required columns: {sorted(missing)}"
            )

        valid = (
            np.isfinite(self.df['groundtruth'])
            & np.isfinite(self.df['predictions'])
            & (self.df['groundtruth'] > 0)
            & (self.df['predictions'] > 0)
        )
        if not np.all(valid):
            dropped = int((~valid).sum())
            print(
                f"Warning: dropping {dropped} rows with missing, non-finite, "
                "or non-positive prediction/ground truth values"
            )
            self.df = self.df.loc[valid].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(
                "No evaluable rows remain. Prediction-only CSVs cannot be "
                "scored without ground truth."
            )

        self.y_true = self.df['groundtruth'].values
        self.y_pred = self.df['predictions'].values
        gaussian_present = 'variance_normalized' in self.df.columns
        if gaussian_present and not self.GAUSSIAN_UNCERTAINTY_COLUMNS.issubset(
            self.df.columns
        ):
            raise ValueError(
                "Prediction CSV has incomplete Gaussian uncertainty outputs; "
                f"missing {sorted(self.GAUSSIAN_UNCERTAINTY_COLUMNS.difference(self.df.columns))}"
            )
        self.has_uncertainty = self.GAUSSIAN_UNCERTAINTY_COLUMNS.issubset(
            self.df.columns
        )
        print(f"Loaded model data with {len(self.df)} records")
        if self.has_uncertainty:
            print("Detected Gaussian uncertainty columns")

    def calculate_metrics(self):
        """
        Calculate and save performance metrics.

        Computes standard regression metrics (MSE, RMSE, MAE, R², Pearson correlation)
        in log-space, plus class-specific metrics for different flare classes
        (Below-B, B, C, M, X).

        Returns:
            pandas.DataFrame: DataFrame containing all calculated metrics
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("No prediction data available. Load data first.")

        main_metrics = {
            'Model': 'FOXES',
            'MSE': mean_squared_error(np.log10(self.y_true), np.log10(self.y_pred)),
            'RMSE': np.sqrt(mean_squared_error(np.log10(self.y_true), np.log10(self.y_pred))),
            'MAE': mean_absolute_error(np.log10(self.y_true), np.log10(self.y_pred)),
            'R2': r2_score(np.log10(self.y_true), np.log10(self.y_pred)),
            'Pearson_Corr': np.corrcoef(np.log10(self.y_true), np.log10(self.y_pred))[0, 1],
            'Sample_Count': len(self.y_true),
        }

        # Calculate metrics for each flare class
        flare_class_metrics = []

        for class_name, (lower_bound, upper_bound) in self.FLARE_CLASSES.items():
            # Create mask for current flare class
            if upper_bound == np.inf:
                mask = self.y_true >= lower_bound
            else:
                mask = (self.y_true >= lower_bound) & (self.y_true < upper_bound)

            # Skip if no samples in this class
            if not np.any(mask):
                print(f"Warning: No samples found for flare class {class_name}")
                continue

            # Get true and predicted values for this class
            y_true_class = self.y_true[mask]
            y_pred_class = self.y_pred[mask]

            # Calculate metrics for this flare class
            class_metrics = {
                'Model': f'FOXES_{class_name}',
                'MSE': mean_squared_error(np.log10(y_true_class), np.log10(y_pred_class)),
                'RMSE': np.sqrt(mean_squared_error(np.log10(y_true_class), np.log10(y_pred_class))),
                'MAE': mean_absolute_error(np.log10(y_true_class), np.log10(y_pred_class)),
                'R2': (
                    r2_score(np.log10(y_true_class), np.log10(y_pred_class))
                    if len(y_true_class) >= 2 else np.nan
                ),
                'Sample_Count': len(y_true_class),
                'Pearson_Corr': (
                    np.corrcoef(
                        np.log10(y_true_class), np.log10(y_pred_class)
                    )[0, 1]
                    if len(y_true_class) >= 2 else np.nan
                ),
            }

            flare_class_metrics.append(class_metrics)

        metrics_list = [main_metrics] + flare_class_metrics

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_list)
        metrics_path = os.path.join(self.metrics_dir, "performance_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Generate regression plot
        self._plot_regression()

        return metrics_df

    def _class_masks(self):
        """Return overall and ground-truth flare-class masks."""
        masks = {'Overall': np.ones(len(self.y_true), dtype=bool)}
        for class_name, (lower_bound, upper_bound) in self.FLARE_CLASSES.items():
            if np.isinf(upper_bound):
                masks[class_name] = self.y_true >= lower_bound
            else:
                masks[class_name] = (
                    (self.y_true >= lower_bound)
                    & (self.y_true < upper_bound)
                )
        return masks

    def calculate_uncertainty_metrics(self):
        """Evaluate Gaussian uncertainty in normalized log space.

        This reports both calibration (coverage and standardized residuals)
        and sharpness (typical predicted sigma). A model can obtain high
        coverage merely by widening its intervals, so both are needed.
        """
        if not self.has_uncertainty:
            raise ValueError(
                "The prediction CSV does not contain complete Gaussian "
                "uncertainty columns."
            )

        mean = self.df['prediction_normalized'].to_numpy(dtype=float)
        target = self.df['groundtruth_normalized'].to_numpy(dtype=float)
        variance = self.df['variance_normalized'].to_numpy(dtype=float)
        valid = (
            np.isfinite(mean) & np.isfinite(target) & np.isfinite(variance)
            & (variance > 0)
        )
        if not np.all(valid):
            print(
                f"Warning: excluding {int((~valid).sum())} rows from "
                "uncertainty metrics because their Gaussian outputs are invalid"
            )
        if not np.any(valid):
            raise ValueError("No finite positive uncertainty predictions to evaluate")

        sigma = np.sqrt(np.maximum(variance, np.finfo(float).tiny))
        residual_z = (target - mean) / sigma
        sigma_dex = (
            self.df['sigma_dex'].to_numpy(dtype=float)
            if 'sigma_dex' in self.df.columns
            else np.full(len(self.df), np.nan)
        )

        rows = []
        for group_name, group_mask in self._class_masks().items():
            mask = group_mask & valid
            if not np.any(mask):
                print(
                    f"Warning: No valid uncertainty samples for {group_name}"
                )
                continue

            group_variance = variance[mask]
            group_error = target[mask] - mean[mask]
            group_z = residual_z[mask]
            nll = 0.5 * (
                np.log(2.0 * np.pi * group_variance)
                + np.square(group_error) / group_variance
            )
            row = {
                'Group': group_name,
                'Sample_Count': int(mask.sum()),
                'Gaussian_NLL_Normalized': float(np.mean(nll)),
                'Coverage_68': float(np.mean(np.abs(group_z) <= 1.0)),
                'Coverage_95': float(np.mean(np.abs(group_z) <= 1.96)),
                'Mean_Sigma_Normalized': float(np.mean(sigma[mask])),
                'Mean_Sigma_Dex': (
                    float(np.nanmean(sigma_dex[mask]))
                    if np.any(np.isfinite(sigma_dex[mask])) else np.nan
                ),
                'Mean_Absolute_Z': float(np.mean(np.abs(group_z))),
                'Z_Mean': float(np.mean(group_z)),
                'Z_Std': (
                    float(np.std(group_z, ddof=1))
                    if mask.sum() >= 2 else np.nan
                ),
            }
            if {'lower_68', 'upper_68'}.issubset(self.df.columns):
                row['Mean_68_Interval_Width_Raw'] = float(np.mean(
                    self.df.loc[mask, 'upper_68'].to_numpy(dtype=float)
                    - self.df.loc[mask, 'lower_68'].to_numpy(dtype=float)
                ))
            if {'lower_95', 'upper_95'}.issubset(self.df.columns):
                row['Mean_95_Interval_Width_Raw'] = float(np.mean(
                    self.df.loc[mask, 'upper_95'].to_numpy(dtype=float)
                    - self.df.loc[mask, 'lower_95'].to_numpy(dtype=float)
                ))
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        metrics_path = os.path.join(
            self.metrics_dir, 'uncertainty_metrics.csv'
        )
        metrics_df.to_csv(metrics_path, index=False)
        self._plot_uncertainty_calibration(residual_z, valid)
        self._plot_prediction_intervals()
        print(f"Saved uncertainty metrics to {metrics_path}")
        return metrics_df

    def _plot_prediction_intervals(self):
        """Show a small, class-balanced sample of prediction intervals."""
        interval_levels = [
            (label, lower, upper, color)
            for label, lower, upper, color in (
                ('68%', 'lower_68', 'upper_68', '#43BEEA'),
                ('95%', 'lower_95', 'upper_95', '#B899F3'),
            )
            if {lower, upper}.issubset(self.df.columns)
        ]
        if not interval_levels:
            return

        interval_columns = [column for _, lower, upper, _ in interval_levels
                            for column in (lower, upper)]
        intervals = self.df[interval_columns].to_numpy(dtype=float)
        valid = np.isfinite(intervals).all(axis=1) & (intervals > 0).all(axis=1)
        for _, lower, upper, _ in interval_levels:
            valid &= (
                (self.df[lower].to_numpy(dtype=float) <= self.y_pred)
                & (self.y_pred <= self.df[upper].to_numpy(dtype=float))
            )

        rng = np.random.default_rng(42)
        selected = []
        for class_name, class_mask in self._class_masks().items():
            if class_name == 'Overall':
                continue
            candidates = np.flatnonzero(valid & class_mask)
            if candidates.size:
                selected.extend(rng.choice(
                    candidates, size=min(12, candidates.size), replace=False
                ))
        if not selected:
            print("No valid raw prediction intervals to plot")
            return
        selected = np.asarray(selected, dtype=int)

        theme = 'white' if self.plot_background in ('white', 'light') else 'black'
        figure_color = '#FFFFFF' if theme == 'white' else '#000000'
        axes_color = '#FFFFFF' if theme == 'white' else '#151522'
        text_color = '#111111' if theme == 'white' else '#FFFFFF'
        grid_color = '#BBBBBB' if theme == 'white' else '#55556B'

        fig, axes = plt.subplots(
            1, len(interval_levels), figsize=(6 * len(interval_levels), 5.5),
            sharex=True, sharey=True, squeeze=False, facecolor=figure_color,
        )
        axes = axes.ravel()
        truth = self.y_true[selected]
        prediction = self.y_pred[selected]
        lower_values = [self.df[lower].to_numpy(dtype=float)[selected]
                        for _, lower, _, _ in interval_levels]
        upper_values = [self.df[upper].to_numpy(dtype=float)[selected]
                        for _, _, upper, _ in interval_levels]
        x_limits = (truth.min() * 0.8, truth.max() * 1.2)
        y_limits = (
            min(lower.min() for lower in lower_values) * 0.8,
            max(upper.max() for upper in upper_values) * 1.2,
        )
        reference_limits = (min(x_limits[0], y_limits[0]),
                            max(x_limits[1], y_limits[1]))

        for ax, (label, _, _, color), lower, upper in zip(
                axes, interval_levels, lower_values, upper_values):
            ax.set_facecolor(axes_color)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.plot(reference_limits, reference_limits, '--',
                    color=grid_color, linewidth=1.2, label='Perfect prediction')
            ax.vlines(truth, lower, upper, color=color, linewidth=1.5,
                      alpha=0.75, label=f'{label} interval')
            ax.scatter(truth, prediction, color='#F4A340', s=22,
                       edgecolors=axes_color, linewidths=0.4, zorder=3,
                       label='Prediction')
            ax.set(xlim=x_limits, ylim=y_limits,
                   xlabel=r'Ground Truth Flux (W/m$^{2}$)',
                   title=f'{label} predictive interval')
            ax.grid(alpha=0.25, color=grid_color, which='major')
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)
        axes[0].set_ylabel(r'Predicted Flux (W/m$^{2}$)', color=text_color)
        axes[0].legend(fontsize=9)
        fig.suptitle('Predictions with uncertainty intervals', color=text_color)
        fig.text(
            0.5, 0.02,
            f'{len(selected)} class-balanced examples (up to 12 per class); '
            'coverage metrics use all valid predictions.',
            ha='center', color=text_color, fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        plot_path = os.path.join(self.plots_dir, 'prediction_intervals.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight',
                    facecolor=figure_color)
        plt.close(fig)
        print(f"Saved prediction interval plot to {plot_path}")

    def _plot_uncertainty_calibration(self, residual_z, valid):
        """Plot nominal-vs-empirical coverage and standardized residuals."""
        setup_barlow_font()
        nominal = np.linspace(0.05, 0.99, 20)
        z_limits = np.array([
            NormalDist().inv_cdf((1.0 + probability) / 2.0)
            for probability in nominal
        ])

        theme = 'white' if self.plot_background in ('white', 'light') else 'black'
        figure_color = '#FFFFFF' if theme == 'white' else '#000000'
        axes_color = '#FFFFFF' if theme == 'white' else '#151522'
        text_color = '#111111' if theme == 'white' else '#FFFFFF'

        fig, (coverage_ax, residual_ax) = plt.subplots(
            1, 2, figsize=(12, 5), facecolor=figure_color
        )
        colors = {
            'Overall': '#E24A33', 'Below-B': '#8C8C8C',
            'B': '#59A14F', 'C': '#348ABD', 'M': '#988ED5',
            'X': '#FBC15E',
        }
        for ax in (coverage_ax, residual_ax):
            ax.set_facecolor(axes_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)
            ax.grid(alpha=0.25)

        coverage_ax.plot(
            [0, 1], [0, 1], '--', color=text_color, linewidth=1,
            label='Ideal calibration',
        )
        for group_name, group_mask in self._class_masks().items():
            mask = group_mask & valid
            if not np.any(mask):
                continue
            empirical = [
                np.mean(np.abs(residual_z[mask]) <= z_limit)
                for z_limit in z_limits
            ]
            coverage_ax.plot(
                nominal, empirical, marker='o', markersize=3,
                color=colors[group_name], label=group_name,
            )
        coverage_ax.set(
            xlabel='Nominal central interval coverage',
            ylabel='Empirical coverage', xlim=(0, 1), ylim=(0, 1),
            title='Predictive interval calibration',
        )
        coverage_ax.legend(fontsize=9)

        overall_z = residual_z[valid]
        residual_ax.hist(
            overall_z, bins=np.linspace(-5, 5, 51), density=True,
            alpha=0.75, color='#348ABD', label='FOXES residuals',
        )
        normal_x = np.linspace(-5, 5, 300)
        residual_ax.plot(
            normal_x,
            np.exp(-0.5 * normal_x ** 2) / np.sqrt(2.0 * np.pi),
            color='#E24A33', linewidth=2, label='Standard normal',
        )
        residual_ax.set(
            xlabel='Standardized residual (target - mean) / sigma',
            ylabel='Density', title='Standardized residual distribution',
        )
        residual_ax.legend(fontsize=9)

        for ax in (coverage_ax, residual_ax):
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)

        fig.tight_layout()
        plot_path = os.path.join(
            self.plots_dir, 'uncertainty_calibration.png'
        )
        fig.savefig(
            plot_path, dpi=300, bbox_inches='tight', facecolor=figure_color
        )
        plt.close(fig)
        print(f"Saved uncertainty calibration plot to {plot_path}")

    def plot_patch_maps(self):
        """Plot timestamp-matched AIA, flux, and variance-contribution maps."""
        if self.inference_patch_dir is None:
            return 0
        if self.df is None:
            raise ValueError("Load prediction data before plotting patch maps")
        if 'timestamp' not in self.df.columns:
            raise ValueError("Prediction CSV needs a timestamp column for patch maps")
        if not self.inference_patch_dir.is_dir():
            raise FileNotFoundError(self.inference_patch_dir)
        if self.aia_dir is not None and not self.aia_dir.is_dir():
            raise FileNotFoundError(self.aia_dir)

        available = []
        for row in self.df.itertuples(index=False):
            stamp = str(row.timestamp)
            flux_path = self.inference_patch_dir / 'flux' / f'{stamp}.npy'
            uncertainty_path = (
                self.inference_patch_dir / 'patch_uncertainty' / f'{stamp}.npy'
            )
            if flux_path.is_file() or uncertainty_path.is_file():
                available.append((row, flux_path, uncertainty_path))
        if not available:
            print("No patch maps match timestamps in the prediction CSV")
            return 0

        # Read the small NPY maps once. A single range across every matching
        # timestamp and both map types keeps equal values the same color.
        maps = {}
        minimum = np.inf
        maximum = 0.0
        for _, flux_path, uncertainty_path in available:
            for path in (flux_path, uncertainty_path):
                if not path.is_file() or path in maps:
                    continue
                patch_map = np.load(path, allow_pickle=False)
                if patch_map.ndim != 2 or not np.isfinite(patch_map).all():
                    raise ValueError(f"Invalid 2D patch map: {path}")
                maps[path] = patch_map
                positive = patch_map[patch_map > 0]
                if positive.size:
                    minimum = min(minimum, float(np.min(positive)))
                    maximum = max(maximum, float(np.max(positive)))
        if not np.isfinite(minimum):
            raise ValueError("Patch maps contain no positive values for log plotting")
        scale_min, scale_max = minimum, maximum
        if scale_min == scale_max:
            scale_min /= 10.0
            scale_max *= 10.0
        print(
            f"Patch-map color scale: {scale_min:.1e} to {scale_max:.1e} "
            "W/m²"
        )
        if self.max_patch_maps is not None and len(available) > self.max_patch_maps:
            indices = np.linspace(
                0, len(available) - 1, self.max_patch_maps, dtype=int
            )
            available = [available[index] for index in indices]

        output_dir = Path(self.plots_dir) / 'patch_maps'
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_barlow_font()
        figure_color = (
            '#FFFFFF' if self.plot_background in ('white', 'light')
            else '#000000'
        )
        text_color = '#111111' if figure_color == '#FFFFFF' else '#FFFFFF'

        for row, flux_path, uncertainty_path in available:
            stamp = str(row.timestamp)
            panels = []
            image_shape = None
            if self.aia_dir is not None:
                aia_path = self.aia_dir / f'{stamp}.npy'
                if aia_path.is_file():
                    image = np.load(aia_path, allow_pickle=False)
                    if image.ndim == 1 and image.size == 7 * 512 * 512:
                        image = image.reshape(7, 512, 512)
                    if image.ndim != 3 or image.shape[0] != 7:
                        raise ValueError(
                            f"Expected seven AIA channels in {aia_path}; "
                            f"got {image.shape}"
                        )
                    channel = image[self.AIA_WAVELENGTHS.index(
                        self.aia_wavelength
                    )]
                    image_shape = channel.shape
                    panels.append((
                        channel, f'AIA {self.aia_wavelength} Å',
                        'gray', 'Processed AIA intensity', False,
                    ))
                else:
                    print(f"Warning: no AIA image for {stamp}")
            for path, title, cmap, label in (
                (flux_path, 'Patch flux', 'inferno',
                 r'Flux contribution (W m$^{-2}$)'),
                (uncertainty_path, 'Patch uncertainty', 'inferno',
                 r'Square-root variance contribution (W m$^{-2}$)'),
            ):
                if path not in maps:
                    continue
                panels.append((maps[path], title, cmap, label, True))
            if not panels:
                continue
            if image_shape is None:
                image_shape = panels[0][0].shape
            extent = (0, image_shape[1], 0, image_shape[0])
            fig, axes = plt.subplots(
                1, len(panels), figsize=(5.2 * len(panels), 5.4),
                squeeze=False, constrained_layout=True,
                facecolor=figure_color,
            )
            for ax, (array, title, cmap, label, is_patch) in zip(
                axes.ravel(), panels
            ):
                if not is_patch:
                    finite = array[np.isfinite(array)]
                    if finite.size == 0:
                        raise ValueError(f"No finite AIA values for {stamp}")
                    low, high = np.percentile(finite, [1, 99.5])
                    if low == high:
                        high = low + 1
                    artist = ax.imshow(
                        array, origin='lower', extent=extent,
                        cmap=cmap, vmin=low, vmax=high,
                    )
                else:
                    # A single physical-flux scale makes different timestamps
                    # and the two map types directly comparable. Nonpositive
                    # values cannot be shown on a logarithmic scale.
                    artist = ax.imshow(
                        np.ma.masked_less_equal(array, 0),
                        origin='lower', extent=extent, cmap=cmap,
                        norm=LogNorm(
                            vmin=scale_min,
                            vmax=scale_max,
                            clip=True,
                        ),
                    )
                ax.set(
                    title=title, xlabel='Image x (pixels)',
                    ylabel='Image y (pixels)',
                )
                ax.tick_params(colors=text_color)
                ax.xaxis.label.set_color(text_color)
                ax.yaxis.label.set_color(text_color)
                ax.title.set_color(text_color)
                colorbar = fig.colorbar(artist, ax=ax, shrink=0.8)
                colorbar.set_label(label, color=text_color)
                colorbar.ax.tick_params(colors=text_color)
            fig.suptitle(
                f'{stamp}  |  prediction {row.predictions:.3e} '
                r'W m$^{-2}$  |  '
                f'truth {row.groundtruth:.3e} '
                r'W m$^{-2}$',
                color=text_color,
            )
            filename = re.sub(r'[^A-Za-z0-9._-]', '_', stamp) + '.png'
            fig.savefig(
                output_dir / filename, dpi=160,
                facecolor=figure_color,
            )
            plt.close(fig)
        print(f"Saved {len(available)} patch-map figures to {output_dir}")
        return len(available)

    def _plot_regression(self):
        """
        Generate regression plot with MAE contours.

        Creates a comprehensive visualization showing:
        - 2D histogram of predicted vs. actual values
        - Perfect prediction line (1:1 relationship)
        - MAE contour bands showing prediction uncertainty
        - Flare class boundaries (A, B, C, M, X)
        - Logarithmic scaling for both axes
        - Professional styling with Barlow font and custom color scheme
        """
        setup_barlow_font()
        flare_classes = {
            'A1.0': 1e-8,
            'B1.0': 1e-7,
            'C1.0': 1e-6,
            'M1.0': 1e-5,
            'X1.0': 1e-4,
            'X10.0': 1e-3,
            'X100.0': 1e-2,
        }

        theme = 'white' if self.plot_background in ('white', 'light') else 'black'
        axis_facecolor = '#FFFFFF' if theme == 'white' else '#FFFFFF'
        text_color = '#111111' if theme == 'white' else '#FFFFFF'
        legend_facecolor = '#FFFFFF' if theme == 'white' else '#1E1E2F'
        grid_color = '#CCCCCC' if theme == 'white' else '#3A3A5A'
        minor_grid_color = '#E6E6E6' if theme == 'white' else '#1F1F35'
        legend_edge_color = 'black' if theme == 'white' else '#3A3A5A'
        colorbar_facecolor = axis_facecolor
        figure_facecolor = '#FFFFFF' if theme == 'white' else '#000000'

        def add_flare_class_axes(ax, tick_color):
            """Helper function to add flare class secondary axes"""
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()

            # Create secondary axis for flare classes (top)
            ax_top = ax.twiny()
            ax_top.set_xlim(x_limits)
            ax_top.set_xscale('log')
            # Make secondary axis background transparent
            ax_top.patch.set_alpha(0.0)

            # Create secondary axis for flare classes (right)
            ax_right = ax.twinx()
            ax_right.set_ylim(y_limits)
            ax_right.set_yscale('log')
            # Make secondary axis background transparent
            ax_right.patch.set_alpha(0.0)

            # Each secondary axis gets only the flare thresholds it displays.
            x_ticks = [(flux, name) for name, flux in flare_classes.items()
                       if min(x_limits) <= flux <= max(x_limits)]
            y_ticks = [(flux, name) for name, flux in flare_classes.items()
                       if min(y_limits) <= flux <= max(y_limits)]

            if x_ticks:
                ax_top.set_xticks([flux for flux, _ in x_ticks])
                ax_top.set_xticklabels([name for _, name in x_ticks], fontsize=12,
                                      color=tick_color, fontfamily='Barlow')
                ax_top.tick_params(colors=tick_color)

                ax_top.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs='auto', numticks=100))
                ax_top.tick_params(which='minor', colors=tick_color)

            if y_ticks:
                ax_right.set_yticks([flux for flux, _ in y_ticks])
                ax_right.set_yticklabels([name for _, name in y_ticks], fontsize=12,
                                         color=tick_color, fontfamily='Barlow')
                ax_right.tick_params(colors=tick_color)

                ax_right.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs='auto', numticks=100))
                ax_right.tick_params(which='minor', colors=tick_color)

        def draw_mae_contours(plot_ax):
            """Draw class MAE bands over the observed ground-truth range."""
            y_true = self.y_true
            y_pred = self.y_pred

            # Define flare classes
            flare_classes_mae = {
                #'A': (1e-8, 1e-7, "#FFAAA5"),
                'B': (1e-7, 1e-6, "#FFAAA5"),
                'C': (1e-6, 1e-5, "#FFAAA5"),
                'M': (1e-5, 1e-4, "#FFAAA5"),
                'X': (1e-4, np.inf, "#FFAAA5")
            }

            for class_name, (min_flux, max_flux, color) in flare_classes_mae.items():
                # Filter data points within this flare class range
                mask = (y_true >= min_flux) & (y_true < max_flux)
                if not np.any(mask):
                    continue

                true_subset = y_true[mask]
                pred_subset = y_pred[mask]

                # Calculate MAE in log space
                log_true = np.log10(true_subset)
                log_pred = np.log10(pred_subset)
                log_mae = mean_absolute_error(log_true, log_pred)

                # Extend slightly beyond the samples, without filling empty
                # decades or crossing into a neighboring flare class.
                buffer_factor = 1.1
                x_start = max(min_flux, true_subset.min() / buffer_factor)
                x_stop = min(max_flux, true_subset.max() * buffer_factor)
                x_class = np.geomspace(x_start, x_stop, 100)

                # Upper and lower MAE bounds
                upper_bound = x_class * 10 ** log_mae
                lower_bound = x_class * 10 ** -log_mae

                # Plot MAE contours on the 1-to-1 plot
                if class_name == 'X':
                    plot_ax.fill_between(x_class, lower_bound, upper_bound,
                                         alpha=0.75,
                                         label=f'MAE', color=color)
                else:
                    plot_ax.fill_between(x_class, lower_bound, upper_bound,
                                         alpha=0.75, color=color)

        log_bins = np.logspace(np.log10(min(min(self.y_true), min(self.y_pred))),
                               np.log10(max(max(self.y_true), max(self.y_pred))), 100)

        shared_norm = LogNorm(vmin=1, vmax=1000)

        # Create figure with transparent background but solid plot area
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
        # Set figure background according to theme
        fig.patch.set_facecolor(figure_facecolor)
        fig.patch.set_alpha(1.0)

        # 1-to-1 plot with MAE contours
        min_val = min(min(self.y_true), min(self.y_pred)) * 0.9
        max_val = max(max(self.y_true), max(self.y_pred)) * 1.1

        # Perfect prediction line
        ax1.plot([min_val, max_val], [min_val, max_val],
                 label='Perfect Prediction', color='#A00503', linestyle='-', linewidth=1, zorder=5)

        # 2D histogram
        h1 = ax1.hist2d(self.y_true, self.y_pred, bins=[log_bins, log_bins],
                        cmap="bone", norm=shared_norm, alpha=1)

        # Draw MAE contours on main plot
        draw_mae_contours(ax1)

        # Set plot area background to dark blue-purple that complements fire colormap
        ax1.set_facecolor(axis_facecolor)
        ax1.patch.set_alpha(1.0)

        # Set labels and styling
        ax1.set_xlabel(r'Ground Truth Flux (W/m$^{2}$)', fontsize=14, color=text_color, fontfamily='Barlow')
        ax1.set_ylabel(r'Predicted Flux (W/m$^{2}$)', fontsize=14, color=text_color, fontfamily='Barlow')
        ax1.tick_params(labelsize=12, colors=text_color)

        # Set tick labels to Barlow font
        for label in ax1.get_xticklabels():
            label.set_fontfamily('Barlow')
            label.set_color(text_color)
        for label in ax1.get_yticklabels():
            label.set_fontfamily('Barlow')
            label.set_color(text_color)

        # Style the legend
        legend = ax1.legend(loc='upper left',
                            prop={'family': 'Barlow', 'size': 12})
        legend.get_frame().set_facecolor(legend_facecolor)
        legend.get_frame().set_edgecolor(legend_edge_color)
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color(text_color)
            text.set_fontsize(12)
            text.set_fontfamily('Barlow')

        # Grid styling
        ax1.set_axisbelow(True)
        ax1.grid(True, alpha=0.6, color=grid_color, linestyle='-', linewidth=0.5)
        ax1.tick_params()
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # Add minor ticks for main plot
        ax1.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs='auto', numticks=100))
        ax1.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs='auto', numticks=100))
        ax1.tick_params(which='minor', colors=text_color)
        ax1.grid(True, which='minor', alpha=0.15, linewidth=0.25, linestyle='--', color=minor_grid_color)

        # Add flare class axes to main plot
        add_flare_class_axes(ax1, text_color)

        # Colorbar styling
        cbar = fig.colorbar(h1[3], ax=ax1, orientation='vertical', pad=.1)
        cbar.ax.yaxis.set_tick_params(labelsize=12, colors=text_color)
        cbar.set_label("Count", fontsize=14, color=text_color, fontfamily='Barlow')
        cbar.ax.tick_params(colors=text_color)
        # make cbar small ticks white
        cbar.ax.yaxis.set_tick_params(colors=text_color)
        cbar.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs='auto', numticks=100))
        cbar.ax.tick_params(which='minor', colors=text_color)
        # Make colorbar background match the plot area
        cbar.ax.set_facecolor(colorbar_facecolor)
        cbar.ax.patch.set_alpha(1.0)

        # Set colorbar tick labels to Barlow font
        for label in cbar.ax.get_yticklabels():
            label.set_fontfamily('Barlow')
            label.set_color(text_color)

        # Set spines to match text color
        for spine in ax1.spines.values():
            spine.set_color(text_color)

        # Save with transparent background - now only the figure background will be transparent
        plot_path = os.path.join(self.plots_dir, "regression_plot.png")
        plt.savefig(plot_path, dpi=500, bbox_inches='tight',
                    facecolor=figure_facecolor)
        plt.close()
        print(f"Saved regression plot to {plot_path}")

    def run_full_evaluation(self):
        """
        Run complete evaluation pipeline.

        Executes the full evaluation workflow including:
        1. Data loading
        2. Quantitative metrics calculation and saving
        3. Regression plot generation

        Returns:
            pandas.DataFrame: Performance metrics dataframe
        """
        print("=== FOXES Solar Flare Evaluation ===")
        print(f"Output will be saved to: {self.output_dir}")

        # Load all data
        print("\nLoading data...")
        self.load_data()

        # Quantitative evaluation
        print("\nCalculating performance metrics...")
        metrics_df = self.calculate_metrics()

        print("\n=== Performance Metrics ===")
        print(metrics_df.to_string(index=False))

        if self.evaluate_uncertainty and self.has_uncertainty:
            print("\nCalculating uncertainty calibration metrics...")
            uncertainty_df = self.calculate_uncertainty_metrics()
            print("\n=== Uncertainty Metrics ===")
            print(uncertainty_df.to_string(index=False))
        elif self.evaluate_uncertainty:
            print(
                "\nNo Gaussian uncertainty columns found; skipping uncertainty "
                "evaluation."
            )

        if self.inference_patch_dir is not None:
            print("\nPlotting inference patch maps...")
            self.plot_patch_maps()

        print("\nEvaluation complete!")
        return metrics_df


def load_evaluation_config(config_path):
    """
    Load evaluation configuration from YAML file.

    Reads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Loaded and processed configuration dictionary
    """
    with open(config_path, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    return config_data


def main():
    """
    Main function to run evaluation with config file.

    Parses command line arguments, loads configuration, and executes the
    complete evaluation pipeline using the FOXESEvaluator class.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run FOXES solar flare evaluation')
    parser.add_argument('--config', type=str, default='evaluation_config.yaml',
                        help='Path to evaluation config YAML file')
    args = parser.parse_args()

    # Load configuration
    config = load_evaluation_config(args.config)

    # Extract parameters from config
    model_predictions = config['model_predictions']
    evaluation = config['evaluation']
    plotting_config = config.get('plotting', {})

    print(f"Loaded evaluation config from: {args.config}")
    print(f"Model CSV: {model_predictions['main_model_csv']}")
    print(f"Output directory: {evaluation['output_dir']}")

    # Initialize evaluator
    evaluator = FOXESEvaluator(
        csv_path=model_predictions['main_model_csv'],
        output_dir=evaluation['output_dir'],
        plot_background=plotting_config.get('regression_background', 'black'),
        evaluate_uncertainty=_as_bool(
            evaluation.get('evaluate_uncertainty', True)
        ),
        inference_patch_dir=plotting_config.get('inference_patch_dir'),
        max_patch_maps=plotting_config.get('max_patch_maps'),
        aia_dir=plotting_config.get('aia_dir'),
        aia_wavelength=plotting_config.get('aia_wavelength', 193),
    )

    # Run complete evaluation
    print("Starting evaluation...")
    evaluator.run_full_evaluation()
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
