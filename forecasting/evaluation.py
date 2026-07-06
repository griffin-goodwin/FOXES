import os
import re
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from matplotlib import rcParams


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
        - Flare class-specific analysis (Quiet, C, M, X classes)
    """

    def __init__(self,
                 csv_path,
                 output_dir="./foxes_evaluation",
                 plot_background='black'):
        """
        Initialize the FOXES evaluation system.

        Args:
            csv_path (str): Path to model prediction results CSV
            output_dir (str): Base output directory for results
            plot_background (str): Regression plot background theme ('black' or 'white')
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.plot_background = (plot_background or 'black').lower()

        # Create output directory structure
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.plots_dir = os.path.join(output_dir, "plots")

        for dir_path in [self.metrics_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize data holders
        self.df = None
        self.y_true = None
        self.y_pred = None

    def load_data(self):
        """
        Load and prepare prediction data.

        Returns:
            None
        """
        self.df = pd.read_csv(self.csv_path)
        self.y_true = self.df['groundtruth'].values
        self.y_pred = self.df['predictions'].values
        print(f"Loaded model data with {len(self.df)} records")

    def calculate_metrics(self):
        """
        Calculate and save performance metrics.

        Computes standard regression metrics (MSE, RMSE, MAE, R², Pearson correlation)
        in log-space, plus class-specific metrics for different flare classes
        (Quiet, C, M, X).

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
            class_metrics = {
                'Model': f'FOXES_{class_name}',
                'MSE': mean_squared_error(np.log10(y_true_class), np.log10(y_pred_class)),
                'RMSE': np.sqrt(mean_squared_error(np.log10(y_true_class), np.log10(y_pred_class))),
                'MAE': mean_absolute_error(np.log10(y_true_class), np.log10(y_pred_class)),
                'R2': r2_score(np.log10(y_true_class), np.log10(y_pred_class)),
                'Sample_Count': len(y_true_class),
                'Pearson_Corr': np.corrcoef(np.log10(y_true_class), np.log10(y_pred_class))[0, 1],
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
            'A1.0': (1e-8, 1e-7),
            'B1.0': (1e-7, 1e-6),
            'C1.0': (1e-6, 1e-5),
            'M1.0': (1e-5, 1e-4),
            'X1.0': (1e-4, 1e-3)
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
            y_true = self.y_true
            y_pred = self.y_pred

            # Define flare classes
            flare_classes_mae = {
                'A': (1e-8, 1e-7, "#FFAAA5"),
                'B': (1e-7, 1e-6, "#FFAAA5"),
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
        add_flare_class_axes(ax1, min_val, max_val, text_color)

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


def main():
    """
    Main function to run evaluation with config file.

    Parses command line arguments, loads configuration, and executes the
    complete evaluation pipeline using the FOXESEvaluator class.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run FOXES solar flare evaluation')
    parser.add_argument('-config', type=str, default='evaluation_config.yaml',
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
        plot_background=plotting_config.get('regression_background', 'black')
    )

    # Run complete evaluation
    print("Starting evaluation...")
    evaluator.run_full_evaluation()
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
