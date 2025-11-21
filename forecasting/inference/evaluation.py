import os
import glob
import re
import sys
import yaml
from multiprocessing import Pool
from datetime import timedelta, datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from scipy.ndimage import zoom
from matplotlib.colors import AsinhNorm, LogNorm
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import sunpy.visualization.colormaps as cm
import matplotlib.font_manager as fm
from matplotlib import rcParams
import colormaps as cmaps


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
            if os.path.exists(barlow_path):
                # Add the font file directly to matplotlib
                fm.fontManager.addfont(barlow_path)
                rcParams['font.family'] = 'Barlow'
                print(f"Using Barlow font from: {barlow_path}")
            else:
                # Fallback to sans-serif
                rcParams['font.family'] = 'sans-serif'
                print("Barlow font not found, using default sans-serif")
    except Exception as e:
        print(f"Font setup error: {e}, using default font")


class SolarFlareEvaluator:
    """
    Comprehensive solar flare evaluation system with baseline model comparison capabilities.
    
    This class provides functionality for evaluating solar flare prediction models by comparing
    their performance against ground truth data and baseline models. It includes quantitative
    metrics calculation, regression analysis visualization, and attention-based movie generation.
    
    Key Features:
        - Performance metrics calculation (MSE, RMSE, MAE, R², Pearson correlation)
        - Flare class-specific analysis (A, B, C, M, X classes)
        - Baseline model comparison
        - Uncertainty quantification and visualization
        - Attention map visualization with AIA imagery
        - Multi-process frame generation for efficient movie creation
    """
    
    def __init__(self,
                 csv_path=None,
                 aia_dir=None,
                 weight_path=None,
                 baseline_csv_path=None,
                 output_dir="./solar_flare_evaluation",
                 sxr_cutoff=None,
                 plot_background='black'):
        """
        Initialize the solar flare evaluation system with baseline comparison.

        Args:
            csv_path (str): Path to main model prediction results CSV (optional)
            aia_dir (str): Path to AIA image data directory
            weight_path (str): Path to main model attention weights directory (optional)
            baseline_csv_path (str): Path to baseline model prediction results CSV
            output_dir (str): Base output directory for results
            sxr_cutoff (float): Minimum SXR value threshold for ground truth filtering (optional)
            plot_background (str): Regression plot background theme ('black' or 'white')
        """
        # Set paths
        self.csv_path = csv_path
        self.aia_dir = aia_dir
        self.weight_path = weight_path
        self.baseline_csv_path = baseline_csv_path
        self.output_dir = output_dir
        self.sxr_cutoff = sxr_cutoff
        self.plot_background = (plot_background or 'black').lower()
        
        # Determine if we're in baseline-only mode
        self.baseline_only_mode = (csv_path is None or not os.path.exists(csv_path)) and baseline_csv_path is not None

        # Create output directory structure
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.frames_dir = os.path.join(output_dir, "movie_frames")
        self.comparison_dir = os.path.join(output_dir, "baseline_comparison")

        for dir_path in [self.metrics_dir, self.plots_dir, self.frames_dir, self.comparison_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize data holders
        self.df = None
        self.baseline_df = None
        self.sxr_df = None
        self.y_true = None
        self.y_pred = None
        self.y_baseline = None

    def load_data(self):
        """
        Load and prepare all required data including baseline.
        
        This method handles data loading for both regular comparison mode and baseline-only mode.
        It applies outlier filtering and SXR cutoff filtering as specified during initialization.
        Ground truth uncertainty is automatically calculated as 20% of the ground truth values.
        
        Returns:
            None
        """
        if self.baseline_only_mode:
            # In baseline-only mode, use baseline data as the main data
            if self.baseline_csv_path and os.path.exists(self.baseline_csv_path):
                self.df = pd.read_csv(self.baseline_csv_path)
                outlier_threshold = 1 # 0.01
                self.mask = self.df['predictions'] < outlier_threshold
                
                # Apply SXR cutoff filter if specified
                if self.sxr_cutoff is not None:
                    sxr_mask = self.df['groundtruth'] >= self.sxr_cutoff
                    self.mask = self.mask & sxr_mask
                    print(f"Applied SXR cutoff filter: ground truth >= {self.sxr_cutoff}")
                
                self.df = self.df[self.mask]
                self.y_true = self.df['groundtruth'].values
                #add 20% uncertainty to ground truth
                self.y_true_uncertainty = 0.2 * self.y_true
                self.y_pred = self.df['predictions'].values
                if 'uncertainty' in self.df.columns and self.df['uncertainty'] is not None:
                    self.y_uncertainty = self.df['uncertainty'].values
                else:
                    self.y_uncertainty = None
                print(f"Loaded baseline model data with {len(self.df)} records (baseline-only mode)")
                
                # Set baseline data to None since we're using it as main data
                self.baseline_df = None
                self.y_baseline = None
                self.y_baseline_uncertainty = None
            else:
                raise ValueError("Baseline CSV path is required in baseline-only mode")
        else:
            # Original behavior for main model + baseline comparison
            # Load main model prediction data
            if self.csv_path and os.path.exists(self.csv_path):
                self.df = pd.read_csv(self.csv_path)
                outlier_threshold = 999999 # 0.01
                self.mask = self.df['predictions'] < outlier_threshold
                
                # Apply SXR cutoff filter if specified
                if self.sxr_cutoff is not None:
                    sxr_mask = self.df['groundtruth'] >= self.sxr_cutoff
                    self.mask = self.mask & sxr_mask
                    print(f"Applied SXR cutoff filter: ground truth >= {self.sxr_cutoff}")
                
                self.df = self.df[self.mask]
                self.y_true = self.df['groundtruth'].values
                #add 20% uncertainty to ground truth
                self.y_true_uncertainty = 0.2 * self.y_true
                self.y_pred = self.df['predictions'].values
                if 'uncertainty' in self.df.columns and self.df['uncertainty'] is not None:
                    self.y_uncertainty = self.df['uncertainty'].values
                else:
                    self.y_uncertainty = None
                print(f"Loaded main model data with {len(self.df)} records")

            # Load baseline model prediction data
            if self.baseline_csv_path and os.path.exists(self.baseline_csv_path):
                self.baseline_df = pd.read_csv(self.baseline_csv_path)
                self.baseline_df = self.baseline_df[self.mask]
                self.y_baseline = self.baseline_df['predictions'].values
                if 'uncertainty' in self.baseline_df.columns and self.baseline_df['uncertainty'] is not None:
                    self.y_baseline_uncertainty = self.baseline_df['uncertainty'].values
                else:
                    self.y_baseline_uncertainty = None
                print(f"Loaded baseline model data with {len(self.baseline_df)} records")


    def calculate_metrics(self):
        """
        Calculate and save performance metrics for both models.
        
        Computes standard regression metrics (MSE, RMSE, MAE, R², Pearson correlation) 
        in log-space for both the main model and baseline model. Additionally calculates
        class-specific metrics for different flare classes (Quiet, C, M, X).
        
        Returns:
            pandas.DataFrame: DataFrame containing all calculated metrics
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("No prediction data available. Load data first.")

        # Calculate metrics for main model (or baseline in baseline-only mode)
        model_name = 'Baseline' if self.baseline_only_mode else 'ViT'
        main_metrics = {
            'Model': model_name,
            'MSE': mean_squared_error(np.log10(self.y_true), np.log10(self.y_pred)),
            'RMSE': np.sqrt(mean_squared_error(np.log10(self.y_true), np.log10(self.y_pred))),
            'MAE': mean_absolute_error(np.log10(self.y_true), np.log10(self.y_pred)),
            'R2': r2_score(np.log10(self.y_true), np.log10(self.y_pred)),
            'Pearson_Corr': np.corrcoef(np.log10(self.y_true), np.log10(self.y_pred))[0, 1],
        }

        # Calculate metrics for each flare class
        flare_classes = {
            'Quiet': (0, 1e-6),  # Below 1e-6
            'C': (1e-6, 1e-5),  # 1e-6 to 1e-5
            'M': (1e-5, 1e-4),  # 1e-5 to 1e-4
            'X': (1e-4, np.inf)  # Above 1e-4
        }

        flare_class_metrics = []

        for class_name, (lower_bound, upper_bound) in flare_classes.items():
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
            class_model_name = f'{model_name}_{class_name}'
            class_metrics = {
                'Model': class_model_name,
                'MSE': mean_squared_error(np.log10(y_true_class), np.log10(y_pred_class)),
                'RMSE': np.sqrt(mean_squared_error(np.log10(y_true_class), np.log10(y_pred_class))),
                'MAE': mean_absolute_error(np.log10(y_true_class), np.log10(y_pred_class)),
                'R2': r2_score(np.log10(y_true_class), np.log10(y_pred_class)),
                'Sample_Count': len(y_true_class),
                'Pearson_Corr': np.corrcoef(np.log10(y_true_class), np.log10(y_pred_class))[0, 1],
            }

            flare_class_metrics.append(class_metrics)

            # If baseline exists and we're not in baseline-only mode, calculate baseline metrics for this class too
            if self.y_baseline is not None and not self.baseline_only_mode:
                y_baseline_class = self.y_baseline[mask]

                baseline_class_metrics = {
                    'Model': f'Baseline_{class_name}',
                    'MSE': mean_squared_error(np.log10(y_true_class), np.log10(y_baseline_class)),
                    'RMSE': np.sqrt(mean_squared_error(np.log10(y_true_class), np.log10(y_baseline_class))),
                    'MAE': mean_absolute_error(np.log10(y_true_class), np.log10(y_baseline_class)),
                    'R2': r2_score(np.log10(y_true_class), np.log10(y_baseline_class)),
                    'Sample_Count': len(y_true_class),
                    'Pearson_Corr': np.corrcoef(np.log10(y_true_class), np.log10(y_baseline_class))[0, 1],
                }

                flare_class_metrics.append(baseline_class_metrics)

        metrics_list = [main_metrics] + flare_class_metrics

        # Calculate metrics for baseline model if available and not in baseline-only mode
        if self.y_baseline is not None and not self.baseline_only_mode:
            baseline_metrics = {
                'Model': 'Baseline',
                'MSE': mean_squared_error(np.log10(self.y_true), np.log10(self.y_baseline)),
                'RMSE': np.sqrt(mean_squared_error(np.log10(self.y_true), np.log10(self.y_baseline))),
                'MAE': mean_absolute_error(np.log10(self.y_true), np.log10(self.y_baseline)),
                'R2': r2_score(np.log10(self.y_true), np.log10(self.y_baseline)),
                'Pearson_Corr': np.corrcoef(np.log10(self.y_true), np.log10(self.y_baseline))[0, 1]
            }
            print(baseline_metrics)
            metrics_list.append(baseline_metrics)

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_list)
        metrics_path = os.path.join(self.metrics_dir, "performance_comparison.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Generate comparison plots
        self._plot_regression_comparison()

        return metrics_df

    def _calculate_tss(self, y_true, y_pred, threshold=None):
        """
        Calculate True Skill Statistic (TSS) for binary classification performance.
        
        TSS is calculated as Sensitivity + Specificity - 1, providing a measure of
        classification skill that accounts for both true positives and true negatives.
        
        Args:
            y_true (array-like): Ground truth values
            y_pred (array-like): Predicted values
            threshold (float, optional): Classification threshold. Defaults to median of y_true.
        
        Returns:
            float: True Skill Statistic value
        """
        if threshold is None:
            threshold = np.median(y_true)

        y_true_bin = (y_true > threshold).astype(int)
        y_pred_bin = (y_pred > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return sensitivity + specificity - 1

    def _plot_regression_comparison(self):
        """
        Generate regression comparison plot with MAE contours and residuals plot.
        
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
            'A1.0': (1e-8, 1e-7),
            'B1.0': (1e-7, 1e-6),
            'C1.0': (1e-6, 1e-5),
            'M1.0': (1e-5, 1e-4),
            'X1.0': (1e-4, 1e-3)
        }

        theme = 'white' if self.plot_background in ('white', 'light') else 'black'
        axis_facecolor = '#FFFFFF' if theme == 'white' else '#FFFFFF'
        text_color = '#111111' if theme == 'white' else '#FFFFFF'
        legend_facecolor = '#F5F5F5' if theme == 'white' else '#1E1E2F'
        grid_color = '#CCCCCC' if theme == 'white' else '#3A3A5A'
        minor_grid_color = '#E6E6E6' if theme == 'white' else '#1F1F35'
        legend_edge_color = '#BABABA' if theme == 'white' else '#3A3A5A'
        colorbar_facecolor = axis_facecolor
        figure_facecolor = '#FFFFFF' if theme == 'white' else '#000000'

        def add_flare_class_axes(ax, min_val, max_val, tick_color):
            """Helper function to add flare class secondary axes"""
            # Create secondary axis for flare classes (top)
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            ax_top.set_xscale('log')
            # Make secondary axis background transparent
            ax_top.patch.set_alpha(0.0)

            # Create secondary axis for flare classes (right)
            ax_right = ax.twinx()
            ax_right.set_ylim(ax.get_ylim())
            ax_right.set_yscale('log')
            # Make secondary axis background transparent
            ax_right.patch.set_alpha(0.0)

            # Set flare class tick positions and labels
            flare_positions = []
            flare_labels = []
            for class_name, (min_flux, max_flux) in flare_classes.items():
                if min_flux >= min_val and min_flux <= max_val:
                    flare_positions.append(min_flux)
                    flare_labels.append(f'{class_name}')
                if max_flux >= min_val and max_flux <= max_val and max_flux != min_flux:
                    flare_positions.append(max_flux)
                    flare_labels.append(f'{class_name}')

            if flare_positions:
                ax_top.set_xticks(flare_positions)
                ax_top.set_xticklabels(flare_labels, fontsize=12, color=tick_color, fontfamily='Barlow')
                ax_top.tick_params(colors=tick_color)

                ax_top.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs='auto', numticks=100))
                ax_top.tick_params(which='minor', colors=tick_color)

                ax_right.set_yticks(flare_positions)
                ax_right.set_yticklabels(flare_labels, fontsize=12, color=tick_color, fontfamily='Barlow')
                ax_right.tick_params(colors=tick_color)

                ax_right.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs='auto', numticks=100))
                ax_right.tick_params(which='minor', colors=tick_color)

        def draw_mae_contours(plot_ax, min_val, max_val):
            """Draw MAE contours on the 1-to-1 plot"""
            import numpy as np

            y_true = self.y_true
            y_pred = self.y_pred

            # Define flare classes
            flare_classes_mae = {
                'A': (1e-8, 1e-7, "#FFAAA5"),
                'B': (1e-7, 1e-6,  "#FFAAA5"),
                'C': (1e-6, 1e-5, "#FFAAA5"),
                'M': (1e-5, 1e-4, "#FFAAA5"),
                'X': (1e-4, 1e-2, "#FFAAA5")
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

                # Create smooth curve within this class range
                x_class = np.logspace(np.log10(min_flux), np.log10(max_flux), 100)

                # Upper and lower MAE bounds
                upper_bound = x_class * np.exp(log_mae)
                lower_bound = x_class * np.exp(-log_mae)

                # Plot MAE contours on the 1-to-1 plot
                if class_name == 'X':
                    plot_ax.fill_between(x_class, lower_bound, upper_bound,
                                        alpha=0.75,
                                        label=f'MAE',color=color)
                else:
                    plot_ax.fill_between(x_class, lower_bound, upper_bound,
                                        alpha=0.75,color=color)

        min_val = min(min(self.y_true), min(self.y_pred))
        max_val = max(max(self.y_true), max(self.y_pred))
        log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)

        shared_norm = LogNorm(vmin=1, vmax=None)

        # Create figure with transparent background but solid plot area
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
        # Set figure background according to theme
        fig.patch.set_facecolor(figure_facecolor)
        fig.patch.set_alpha(1.0)

        # Main model plot (1-to-1 with MAE contours)
        min_val = min(min(self.y_true), min(self.y_pred))
        max_val = max(max(self.y_true), max(self.y_pred))

        # Perfect prediction line
        ax1.plot([min_val, max_val], [min_val, max_val],
                label='Perfect Prediction', color='#A00503', linestyle='-', linewidth=1, zorder=5)

        # 2D histogram
        h1 = ax1.hist2d(self.y_true, self.y_pred, bins=[log_bins, log_bins],
                        cmap="bone", norm=shared_norm, alpha=1)

        # Draw MAE contours on main plot
        draw_mae_contours(ax1, min_val, max_val)

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
        
        title = 'Baseline Model Performance with MAE Overlay' if self.baseline_only_mode else 'FOXES Model Performance with MAE Overlay'
        #ax1.set_title(title, fontsize=16, color='white', pad=20, fontfamily='Barlow')
        
        # Style the legend
        legend = ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                            prop={'family': 'Barlow', 'size': 12})
        legend.get_frame().set_facecolor(legend_facecolor)
        legend.get_frame().set_edgecolor(legend_edge_color)
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color(text_color)
            text.set_fontsize(12)
            text.set_fontfamily('Barlow')
        
        # Grid styling
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
        add_flare_class_axes(ax1, min_val, max_val, text_color)

        # Colorbar styling
        cbar = fig.colorbar(h1[3], ax=ax1, orientation='vertical', pad=.1)
        cbar.ax.yaxis.set_tick_params(labelsize=12, colors=text_color)
        cbar.set_label("Count", fontsize=14, color=text_color, fontfamily='Barlow')
        cbar.ax.tick_params(colors=text_color)
        #make cbar small ticks white
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
        plot_path = os.path.join(self.comparison_dir, "regression_comparison.png")
        plt.savefig(plot_path, dpi=500, bbox_inches='tight',
                    facecolor=figure_facecolor)
        plt.close()
        print(f"Saved regression comparison plot to {plot_path}")


    @staticmethod
    def init_worker(csv_data, baseline_csv_data):
        """
        Initialize each worker process with CSV data.
        
        This static method is used as the initializer function for multiprocessing.Pool.
        It loads the CSV data into global variables accessible by worker processes.
        
        Args:
            csv_data (pandas.DataFrame): Main model CSV data
            baseline_csv_data (pandas.DataFrame): Baseline model CSV data
        """
        global csv_data_global, baseline_csv_data_global
        csv_data_global = csv_data
        baseline_csv_data_global = baseline_csv_data
        print(f"Worker {os.getpid()}: CSV data loaded")

    def load_csv_data(self):
        """
        Load and prepare CSV data for workers.
        
        Prepares CSV data for multiprocessing by converting timestamps to datetime 
        objects and adding ground truth uncertainty. Handles both regular and 
        baseline-only modes.
        
        Returns:
            tuple: (csv_data, baseline_csv_data) - Main and baseline CSV dataframes
        """
        if self.baseline_only_mode:
            # In baseline-only mode, use baseline data as main data
            csv_data = pd.read_csv(self.baseline_csv_path)
            if 'timestamp' in csv_data.columns:
                csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'])
            csv_data['groundtruth_uncertainty'] = 0.2 * csv_data['groundtruth']  # Add 20% uncertainty to ground truth
            baseline_data = None
        else:
            # Load main model CSV
            csv_data = pd.read_csv(self.csv_path)
            if 'timestamp' in csv_data.columns:
                csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'])
            csv_data['groundtruth_uncertainty'] = 0.2 * csv_data['groundtruth']  # Add 20% uncertainty to ground truth
            # Load baseline CSV
            try:
                if self.baseline_csv_path:
                    baseline_data = pd.read_csv(self.baseline_csv_path)
                    if 'timestamp' in baseline_data.columns:
                        baseline_data['timestamp'] = pd.to_datetime(baseline_data['timestamp'])
                else:
                    baseline_data = None
            except:
                baseline_data = None
        return csv_data, baseline_data

    def load_aia_image(self, timestamp):
        """
        Load AIA image for given timestamp.
        
        Searches for AIA image files matching the timestamp pattern in the specified directory.
        
        Args:
            timestamp (str): Timestamp in format YYYY-MM-DDTHH:MM:SS
            
        Returns:
            numpy.ndarray or None: AIA image data if found, None otherwise
        """
        pattern = f"{self.aia_dir}/*{timestamp}*"
        files = glob.glob(pattern)
        if files:
            return np.load(files[0])
        return None

    def load_attention_map(self, timestamp):
        """
        Load attention map for given timestamp.
        
        Loads attention weights from text files and resizes them to match AIA image dimensions.
        
        Args:
            timestamp (str): Timestamp in format YYYY-MM-DDTHH:MM:SS
            
        Returns:
            numpy.ndarray or None: Resized attention map or None if loading failed
        """
        if not self.weight_path or not os.path.exists(self.weight_path):
            print(f"Weight path not available: {self.weight_path}")
            return None
            
        filepath = os.path.join(self.weight_path, f"{timestamp}")
        try:
            attention = np.loadtxt(filepath, delimiter=",")
            target_shape = [512, 512]
            zoom_factors = (target_shape[0] / attention.shape[0],
                            target_shape[1] / attention.shape[1])
            return zoom(attention, zoom_factors, order=1)
        except Exception as e:
            print(f"Could not load attention map for {timestamp}: {e}")
            return None

    def get_sxr_data_for_timestamp(self, timestamp, window_hours=16):
        """
        Get SXR data around the given timestamp from CSV files.
        
        Retrieves both current timestamp data and surrounding temporal window data 
        for comprehensive visualization. Merges main model and baseline predictions.
        
        Args:
            timestamp (str): Target timestamp for data retrieval
            window_hours (int): Temporal window size in hours (default: 16)
            
        Returns:
            tuple: (window_data, current_data, target_time) - 
                   Window dataframe, current timestamp data, and datetime object
        """
        try:
            # Access global CSV data loaded in worker
            global csv_data_global, baseline_csv_data_global

            target_time = pd.to_datetime(timestamp)

            # Find matching row in main model CSV
            main_row = csv_data_global[csv_data_global['timestamp'] == target_time]
            baseline_row = baseline_csv_data_global[baseline_csv_data_global[
                                                        'timestamp'] == target_time] if baseline_csv_data_global is not None else pd.DataFrame()

            if main_row.empty:
                print(f"No main model data found for timestamp {timestamp}")
                return None, None, None

            # Extract baseline predictions
            if baseline_row.empty:
                print(f"No baseline data found for timestamp {timestamp}")
                baseline_pred = None
                baseline_uncertainty = None
            else:
                baseline_pred = baseline_row.iloc[0]['predictions']
                if 'uncertainty' in baseline_row.columns:
                    baseline_uncertainty = baseline_row.iloc[0]['uncertainty']
                else:
                    baseline_uncertainty = None

            # Extract main model uncertainty
            if 'uncertainty' in main_row.columns:
                main_uncertainty = main_row.iloc[0]['uncertainty']
            else:
                main_uncertainty = None

            # Create current data dictionary
            current_data = {
                'groundtruth': main_row.iloc[0]['groundtruth'],
                'groundtruth_uncertainty': main_row.iloc[0]['groundtruth_uncertainty'],
                'predictions': main_row.iloc[0]['predictions'],
                'uncertainty': main_uncertainty,
                'baseline_predictions': baseline_pred,
                'baseline_uncertainty': baseline_uncertainty,
                'timestamp': target_time
            }

            # Create window data (get surrounding timestamps within window_hours)
            time_window_start = target_time - pd.Timedelta(hours=window_hours / 2)
            time_window_end = target_time + pd.Timedelta(hours=window_hours / 2)

            # Filter data within window
            main_window = csv_data_global[
                (csv_data_global['timestamp'] >= time_window_start) &
                (csv_data_global['timestamp'] <= time_window_end)
                ].copy()

            # Merge the windows for plotting
            window_data = main_window[['timestamp', 'groundtruth', 'groundtruth_uncertainty' , 'predictions']].copy()

            # Add main model uncertainty if available
            if 'uncertainty' in main_window.columns:
                window_data['uncertainty'] = main_window['uncertainty']

            if baseline_csv_data_global is not None:
                baseline_window = baseline_csv_data_global[
                    (baseline_csv_data_global['timestamp'] >= time_window_start) &
                    (baseline_csv_data_global['timestamp'] <= time_window_end)
                    ].copy()

                if not baseline_window.empty:
                    # Merge baseline predictions
                    baseline_cols = ['timestamp', 'predictions']
                    if 'uncertainty' in baseline_window.columns:
                        baseline_cols.append('uncertainty')

                    baseline_pred_col = baseline_window[baseline_cols].rename(
                        columns={'predictions': 'baseline_predictions', 'uncertainty': 'baseline_uncertainty'})
                    window_data = window_data.merge(baseline_pred_col, on='timestamp', how='left')
                else:
                    window_data['baseline_predictions'] = None
                    window_data['baseline_uncertainty'] = None
            else:
                window_data['baseline_predictions'] = None
                window_data['baseline_uncertainty'] = None

            return window_data, current_data, target_time

        except Exception as e:
            print(f"Could not get SXR data for timestamp {timestamp}: {e}")
            return None, None, None

    def generate_frame_worker(self, timestamp):
        """
        Worker function to generate a single frame.
        
        This method is designed for multiprocessing and creates a comprehensive 
        visualization frame showing:
        - AIA 131Å image with attention overlay
        - SXR time series with ground truth, model predictions, and uncertainties
        - Current timestamp marker
        - Quantitative performance metrics
        - Professional styling with consistent color scheme
        
        Args:
            timestamp (str): Timestamp for frame generation
            
        Returns:
            str or None: Path to generated frame file or None if generation failed
        """
        try:
            print(f"Worker {os.getpid()}: Processing {timestamp}")

            # Load data
            aia_data = self.load_aia_image(timestamp)
            attention_data = self.load_attention_map(timestamp)

            if aia_data is None:
                print(f"Worker {os.getpid()}: Skipping {timestamp} (missing AIA data)")
                return None

            # Get SXR data from CSV
            sxr_window, sxr_current, target_time = self.get_sxr_data_for_timestamp(timestamp)

            # Generate frame
            save_path = os.path.join(self.frames_dir, f"{timestamp}.png")

            # Setup Barlow font
            setup_barlow_font()

            # Create figure with transparent background
            fig = plt.figure(figsize=(10, 5))
            fig.patch.set_alpha(0.0)  # Transparent background
            gs_left = fig.add_gridspec(1, 1, left=0.0, right=0.35, width_ratios=[1], hspace=0, wspace=0.1)

            # Right gridspec for SXR plot (column 3) with more padding
            gs_right = fig.add_gridspec(2, 1, left=0.45, right=1, hspace=0.1)

            wavs = ['94', '131', '171', '193', '211', '304']
            att_max = np.percentile(attention_data, 100)
            att_min = np.percentile(attention_data, 0)
            att_norm = AsinhNorm(vmin=att_min, vmax=att_max, clip=False)
            max_attention_idx = np.unravel_index(np.argmax(attention_data), attention_data.shape)

            row = 0
            col = 0
            ax = fig.add_subplot(gs_left[row, col])

            aia_img = aia_data[1]

            ax.imshow(aia_img, cmap=cm.cmlist['sdoaia131'], origin='lower')
            ax.imshow(attention_data, cmap='hot', origin='lower', alpha=0.5,norm=att_norm)

            ax.set_title(f'AIA {wavs[1]} Å', fontsize=12, fontfamily='Barlow', color='white')
            ax.axis('off')


            # Plot SXR data with uncertainty bands
            sxr_ax = fig.add_subplot(gs_right[:, 0])
            
            # Set SXR plot background to have light background inside plot area
            sxr_ax.set_facecolor('#FFEEE6')  # Light background for SXR plot area
            sxr_ax.patch.set_alpha(1.0)      # Make axes patch opaque

            if sxr_window is not None and not sxr_window.empty:
                # Plot ground truth (no uncertainty)
                sxr_ax.plot(sxr_window['timestamp'], sxr_window['groundtruth'],
                            label='Ground Truth', linewidth=2.5, alpha=1, markersize=5, color="#F78E69")

                gt = sxr_window['groundtruth'].values
                uncertainties = sxr_window['groundtruth_uncertainty'].values

                # Calculate uncertainty bounds
                lower_bound = gt - uncertainties
                upper_bound = gt + uncertainties
                
                # Ensure bounds are positive for log scale
                lower_bound = np.maximum(lower_bound, 1e-12)

                # Plot model predictions with uncertainty bands
                model_label = 'Baseline Model' if self.baseline_only_mode else 'FOXES Model'
                model_color = "#94ECBE" if self.baseline_only_mode else "#C0B9DD"
                sxr_ax.plot(sxr_window['timestamp'], sxr_window['predictions'],
                                                  label=model_label, linewidth=2.5, alpha=1, markersize=5,
                                                  color=model_color)

                # Plot baseline predictions if available and not in baseline-only mode
                if not self.baseline_only_mode and 'baseline_predictions' in sxr_window.columns and sxr_window[
                    'baseline_predictions'].notna().any():
                    baseline_line = sxr_ax.plot(sxr_window['timestamp'], sxr_window['baseline_predictions'],
                                                label='Baseline Model', linewidth=1.5, alpha=1, markersize=5,
                                                color="#94ECBE")

                # Mark current time
                if sxr_current is not None:
                    sxr_ax.axvline(target_time, color='black', linestyle='--',
                                   linewidth=2, alpha=0.4, label='Current Time')

                    # Create info text with all available values
                    model_name = 'Baseline' if self.baseline_only_mode else 'FOXES'
                    info_lines = ["Current Values:",
                                  f"Ground Truth: {sxr_current['groundtruth']:.2e}",
                                  f"{model_name}: {sxr_current['predictions']:.2e}"]

                    # Add model uncertainty if available
                    if sxr_current['uncertainty'] is not None:
                        uncertainty_label = f"{model_name} σ" if self.baseline_only_mode else "ViT σ"
                        info_lines.append(f"{uncertainty_label}: {sxr_current['uncertainty']:.2e}")

                    # Add baseline prediction if available and not in baseline-only mode
                    if not self.baseline_only_mode and sxr_current['baseline_predictions'] is not None:
                        info_lines.append(f"Base: {sxr_current['baseline_predictions']:.2e}")

                        # Add baseline uncertainty if available
                        if sxr_current['baseline_uncertainty'] is not None:
                            info_lines.append(f"Base σ: {sxr_current['baseline_uncertainty']:.2e}")

                    info_text = "\n".join(info_lines)
                    sxr_ax.text(0.02, 0.98, info_text, transform=sxr_ax.transAxes,
                                fontsize=8, verticalalignment='top', fontfamily='Barlow',
                                bbox=dict(boxstyle='round', alpha=0.9, facecolor='#FFEEE6'))

                sxr_ax.set_xlim([pd.to_datetime(timestamp) - pd.Timedelta(hours=4),pd.to_datetime(timestamp) + pd.Timedelta(hours=4)])
                sxr_ax.set_ylim([5e-7, 5e-4])  # Set y-limits for SXR data
                sxr_ax.set_ylabel(r'SXR Flux (W/m$^2$)', fontsize=12, fontfamily='Barlow', color='white')
                sxr_ax.set_xlabel('Time', fontsize=12, fontfamily='Barlow', color='white')
                title = 'Baseline Prediction vs. Ground Truth Comparison' if self.baseline_only_mode else 'FOXES Prediction vs. Ground Truth Comparison'
                sxr_ax.set_title(title, fontsize=12, fontfamily='Barlow', color='white')
                
                # Style the legend to match regression plot
                legend1 = sxr_ax.legend(fontsize=8, loc='upper right', prop={'family': 'Barlow', 'size': 8})
                legend1.get_frame().set_facecolor('#FFEEE6')
                legend1.get_frame().set_alpha(0.9)
                for text in legend1.get_texts():
                    text.set_color('black')
                    text.set_fontfamily('Barlow')
                
                sxr_ax.grid(True, alpha=0.3, color='black')
                sxr_ax.tick_params(axis='x', rotation=15, labelsize=12, colors='white', 
                                )
                sxr_ax.tick_params(axis='y', labelsize=12, colors='white',
                                )
                
                # Set tick labels to Barlow font and white color
                for label in sxr_ax.get_xticklabels():
                    label.set_fontfamily('Barlow')
                    label.set_color('white')
                for label in sxr_ax.get_yticklabels():
                    label.set_fontfamily('Barlow')
                    label.set_color('white')
                
                # Set graph border (spines) to white
                for spine in sxr_ax.spines.values():
                    spine.set_color('white')
                try:
                    sxr_ax.set_yscale('log')
                except:
                    pass  # Skip log scale if data doesn't support it
            else:
                sxr_ax.text(0.5, 0.5, 'No SXR Data\nAvailable',
                            transform=sxr_ax.transAxes, fontsize=12, fontfamily='Barlow',
                            horizontalalignment='center', verticalalignment='center')
                sxr_ax.set_title('SXR Data Comparison with Uncertainties', fontsize=12, fontfamily='Barlow')
            plt.savefig(save_path, dpi=500, facecolor='none',bbox_inches='tight')
            plt.close()

            print(f"Worker {os.getpid()}: Completed {timestamp}")
            return save_path

        except Exception as e:
            print(f"Worker {os.getpid()}: Error processing {timestamp}: {e}")
            plt.close('all')  # Clean up any open figures
            return None

    def create_attention_movie(self, timestamps, auto_cleanup=True):
        """
        Generate attention visualization movie with baseline comparison and uncertainties.
        
        Creates a comprehensive movie showing attention maps overlaid on AIA images
        alongside SXR time series comparisons. Uses multiprocessing for efficient 
        parallel frame generation and compiles frames into a video file.
        
        Args:
            timestamps (list): List of timestamps for frame generation
            auto_cleanup (bool): Whether to automatically delete individual frame files after movie creation
            
        Returns:
            None
        """
        print(f"Generated {len(timestamps)} timestamps to process")

        # Load CSV data once
        print("Loading CSV data...")
        csv_data, baseline_csv_data = self.load_csv_data()

        # Determine number of processes
        num_processes = min(os.cpu_count(), len(timestamps))  # Don't use more processes than timestamps
        num_processes = max(1, num_processes - 1)  # Leave one CPU free
        print(f"Using {num_processes} processes")

        # Process frames in parallel
        import time
        start_time = time.time()

        with Pool(processes=num_processes, initializer=self.init_worker,
                  initargs=(csv_data, baseline_csv_data)) as pool:
            # Use map to process all timestamps
            results = pool.map(self.generate_frame_worker, timestamps)

        # Filter out failed frames
        frame_paths = [path for path in results if path is not None]

        processing_time = time.time() - start_time
        print(f"Generated {len(frame_paths)} frames in {processing_time:.2f} seconds")
        if len(frame_paths) > 0:
            print(f"Average: {processing_time / len(frame_paths):.2f} seconds per frame")

        if not frame_paths:
            print("No frames were successfully generated. Cannot create movie.")
            return

        # Compile into video
        print("Creating movie...")
        video_start = time.time()

        # Sort frame paths by timestamp to ensure correct order
        frame_paths.sort(key=lambda x: os.path.basename(x))

        movie_path = os.path.join(self.output_dir, f"AIA_{timestamps[0].split('T')[0]}.mp4")
        with imageio.get_writer(movie_path, fps=30, codec='libx264', format='ffmpeg') as writer:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    image = imageio.imread(frame_path)
                    writer.append_data(image)

        video_time = time.time() - video_start
        total_time = time.time() - start_time

        print(f"Video creation took {video_time:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"✅ Movie saved to: {movie_path}")

        #Optional: Clean up frame files
        if auto_cleanup:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    #os.remove(frame_path)
                    print("Frame files not deleted")
        else:
            cleanup = input("Delete individual frame files? (y/n): ").lower().strip()
            if cleanup == 'y':
                for frame_path in frame_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                print("Frame files deleted")

    def run_full_evaluation(self, timestamps=None):
        """
        Run complete evaluation pipeline with baseline comparison and uncertainties.
        
        Executes the full evaluation workflow including:
        1. Data loading and preprocessing
        2. Quantitative metrics calculation and saving
        3. Regression comparison plot generation
        4. Attention movie creation (if timestamps provided)
        
        Args:
            timestamps (list, optional): List of timestamps for movie generation
            
        Returns:
            pandas.DataFrame: Performance metrics dataframe
        """
        print("=== Solar Flare Evaluation with Baseline Comparison and Uncertainties ===")
        print(f"Output will be saved to: {self.output_dir}")

        # Load all data
        print("\nLoading data...")
        self.load_data()

        # Quantitative evaluation
        print("\nCalculating performance metrics...")
        metrics_df = self.calculate_metrics()

        print("\n=== Performance Metrics Comparison ===")
        print(metrics_df.to_string(index=False))

        # Visual evaluation if timestamps provided
        if timestamps:
            print("\nGenerating attention visualizations with uncertainties...")
            self.create_attention_movie(timestamps)

        print("\nEvaluation complete!")
        return metrics_df




def resolve_config_variables(config_dict):
    """
    Recursively resolve ${variable} references within the config.
    
    This function processes configuration dictionaries to substitute variable 
    references of the form ${variable_name} with their actual values defined 
    elsewhere in the configuration.
    
    Args:
        config_dict (dict): Configuration dictionary with potential variable references
        
    Returns:
        dict: Configuration dictionary with resolved variable substitutions
    """
    variables = {}
    for key, value in config_dict.items():
        if isinstance(value, str) and not value.startswith('${'):
            variables[key] = value

    def substitute_value(value, variables):
        if isinstance(value, str):
            pattern = r'\$\{([^}]+)\}'
            for match in re.finditer(pattern, value):
                var_name = match.group(1)
                if var_name in variables:
                    value = value.replace(f'${{{var_name}}}', variables[var_name])
        return value

    def recursive_substitute(obj, variables):
        if isinstance(obj, dict):
            return {k: recursive_substitute(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_substitute(item, variables) for item in obj]
        else:
            return substitute_value(obj, variables)

    return recursive_substitute(config_dict, variables)


def load_evaluation_config(config_path):
    """
    Load evaluation configuration from YAML file.
    
    Reads a YAML configuration file and applies variable substitution to 
    resolve any ${variable} references within the configuration.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Loaded and processed configuration dictionary
    """
    with open(config_path, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)
    
    # Resolve variable substitutions
    config_data = resolve_config_variables(config_data)
    return config_data


def generate_timestamps(start_time_str, end_time_str, interval_minutes):
    """
    Generate list of timestamps for evaluation.
    
    Creates a sequence of timestamps within the specified time range at 
    regular intervals for evaluation purposes.
    
    Args:
        start_time_str (str): Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
        end_time_str (str): End time in ISO format (YYYY-MM-DDTHH:MM:SS)
        interval_minutes (int): Time interval between timestamps in minutes
        
    Returns:
        list: List of timestamp strings in format YYYY-MM-DDTHH:MM:SS
    """
    start_time = datetime.fromisoformat(start_time_str)
    end_time = datetime.fromisoformat(end_time_str)
    interval = timedelta(minutes=interval_minutes)
    
    timestamps = []
    current_time = start_time
    while current_time <= end_time:
        timestamps.append(current_time.strftime("%Y-%m-%dT%H:%M:%S"))
        current_time += interval
    
    return timestamps


def main():
    """
    Main function to run evaluation with config file.
    
    Parses command line arguments, loads configuration, generates timestamps, 
    and executes the complete evaluation pipeline using the SolarFlareEvaluator class.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run solar flare evaluation')
    parser.add_argument('-config', type=str, default='evaluation_config.yaml', 
                       help='Path to evaluation config YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_evaluation_config(args.config)
    
    # Extract parameters from config
    model_predictions = config['model_predictions']
    data = config['data']
    evaluation = config['evaluation']
    time_range = config['time_range']
    plotting_config = config.get('plotting', {})
    
    # Generate timestamps
    timestamps = generate_timestamps(
        time_range['start_time'],
        time_range['end_time'],
        time_range['interval_minutes']
    )
    
    print(f"Loaded evaluation config from: {args.config}")
    print(f"Main model CSV: {model_predictions['main_model_csv']}")
    print(f"Baseline CSV: {model_predictions['baseline_csv']}")
    print(f"AIA directory: {data['aia_dir']}")
    print(f"Output directory: {evaluation['output_dir']}")
    print(f"Time range: {time_range['start_time']} to {time_range['end_time']}")
    print(f"Number of timestamps: {len(timestamps)}")
    if evaluation.get('sxr_cutoff') is not None:
        print(f"SXR cutoff filter: ground truth >= {evaluation['sxr_cutoff']}")
    else:
        print("SXR cutoff filter: disabled")
    
    # Check if we're in baseline-only mode
    main_csv = model_predictions['main_model_csv']
    if main_csv is None or main_csv == 'null' or not os.path.exists(main_csv):
        print("Running in baseline-only mode")
    else:
        print("Running in comparison mode (main model + baseline)")
    
    # Initialize evaluator
    evaluator = SolarFlareEvaluator(
        csv_path=main_csv if main_csv != 'null' else None,
        baseline_csv_path=model_predictions['baseline_csv'],
        aia_dir=data['aia_dir'],
        weight_path=data['weight_path'],
        output_dir=evaluation['output_dir'],
        sxr_cutoff=evaluation.get('sxr_cutoff'),
        plot_background=plotting_config.get('regression_background', 'black')
    )
    
    # Run complete evaluation
    print("Starting evaluation...")
    evaluator.run_full_evaluation(timestamps=timestamps)
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
