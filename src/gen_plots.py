import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

FONT_SCALE = 1.10  # ← master knob
_BASE_FONT = 14

SECONDS_TO_HOUR = 3600
SECONDS_TO_DAY = 86_400

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=FONT_SCALE)

plt.rcParams.update({
      'font.serif': ['Times New Roman', 'Times'],
      'mathtext.fontset': 'stix',
      'axes.linewidth': 1.0,
      'axes.edgecolor': 'grey',    
      'axes.labelsize': _BASE_FONT,
      'axes.titlesize': _BASE_FONT,
      'xtick.major.size': 2,
      'ytick.major.size': 2,
      'xtick.labelsize': _BASE_FONT,
      'ytick.labelsize': _BASE_FONT,
      'legend.fontsize': _BASE_FONT,
      'legend.title_fontsize': _BASE_FONT,
      'figure.dpi': 300,
})

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
FIGURES_DIR = ROOT_DIR / 'figs' 

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    'merge_time', 'closure_time', 'description_length', 'code_churn', 'num_files', 
    'num_commits', 'comments', 'num_inline_comments', 'changes_requested'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Heatmap for PR metrics effect sizes.")
    parser.add_argument(
        '-d', '--dataset', 
        type=int, 
        choices=[1, 2, 3, 4], 
        required=True,
        help="Select the dataset matching the generated data."
    )
    parser.add_argument(
        '-f', '--figure', 
        type=int, 
        required=True,
        help="Select the figure number to show in the title."
    )
    parser.add_argument(
        '-met', '--metrics', 
        nargs='*', 
        required=False,
        help="Select the metrics to plot."
    )
    # --- NEW PARAMETER ---
    parser.add_argument(
        '--remove-empty', 
        action='store_true',
        help="If set, remove effect size magnitudes (rows) that have zeros across all metrics."
    )

    return parser.parse_args()

def get_effect_category(val):
    if pd.isna(val):
        return None
    abs_val = abs(val)
    direction = "(-)" if val > 0 else "(+)"
    if abs_val < 0.147:
        return "Negligible"
    elif abs_val < 0.33:
        return f"Small {direction}"
    elif abs_val < 0.474:
        return f"Medium {direction}"
    else:
        return f"Large {direction}"

def prepare_heatmap_data(df, metrics):
    y_order = [
        'Large (+)', 'Medium (+)', 'Small (+)', 
        'Negligible', 
        'Small (-)', 'Medium (-)', 'Large (-)'
    ]
    
    heatmap_matrix = pd.DataFrame(0, index=y_order, columns=metrics)
    for metric in metrics:
        delta_col = f'{metric}_delta'
        p_col = f'{metric}_p_value'
        
        if delta_col not in df.columns or p_col not in df.columns:
            continue
            
        significant_df = df[df[p_col] < 0.05]
        categories = significant_df[delta_col].apply(get_effect_category)
        counts = categories.value_counts()
        
        for cat, count in counts.items():
            if cat in heatmap_matrix.index:
                heatmap_matrix.loc[cat, metric] = count
                
    return heatmap_matrix

def plot_effect_size_heatmap(data_matrix, figure):
    # Sort the x-axis (columns) alphabetically
    data_matrix = data_matrix.sort_index(axis=1)
    
    # If you also want to sort the y-axis (Effect Size Magnitude)
    # data_matrix = data_matrix.sort_index(axis=0)

    fig_size = (8, 6)
    if figure in [1, 2]:
        fig_size = (5, 5)
    elif figure in [3, 4]:
        fig_size = (3, 4)

    plt.figure(figsize=fig_size)
    
    ax = sns.heatmap(
        data_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=.2,
        annot_kws={"size": 16}
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Metric")
    plt.ylabel("Effect Size Magnitude")
    plt.tight_layout()
    
    output_filename = f'effect_size_heatmap_mode_{figure}.pdf'
    output_path = FIGURES_DIR / output_filename
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"Heatmap saved to {output_path}")

def main():
    args = parse_args()
    dataset_file = args.dataset

    if dataset_file in [1, 3]:
        dataset_file = 1
    elif dataset_file in [2, 4]:
        dataset_file = 2

    input_filename = f'repo_agent_metrics_comparison_mode_{args.dataset}.csv'
    input_path = DATA_DIR / input_filename
    
    try:
        df = pd.read_csv(input_path)
        # Determine prefixes based on actual mode file usage
        prefix_base = 'before'
        prefix_target = 'after' if dataset_file == 1 else 'after_last'
        
        df = df[
            (df[f'pr_count_{prefix_base}'] >= 5) &
            (df[f'pr_count_{prefix_target}'] >= 5)
        ]
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
        return

    metrics = args.metrics
    if metrics:
        metrics = list(set(METRICS).intersection(metrics))
    else:
        metrics = METRICS

    heatmap_matrix = prepare_heatmap_data(df, metrics)

    # --- APPLY FILTERING LOGIC ---
    if args.remove_empty:
        # Keep rows where the sum of counts across all columns is greater than 0
        heatmap_matrix = heatmap_matrix[heatmap_matrix.sum(axis=1) > 0]
        print("Removed empty effect size categories from Y-axis.")
    
    if heatmap_matrix.empty or heatmap_matrix.sum().sum() == 0:
        print("Warning: No significant results to display.")
    else:
        plot_effect_size_heatmap(heatmap_matrix, args.figure)

if __name__ == "__main__":
    main()