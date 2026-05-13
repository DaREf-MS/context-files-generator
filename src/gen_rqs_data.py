import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from helper import mannUandCliffdelta
from logging_config import get_logger

logger = get_logger()

ROOT_DIR = Path(__file__).resolve().parent.parent

# Configuration
DATA_DIR = ROOT_DIR / 'data'
PR_FEAT_FILE = DATA_DIR / 'prs_features.csv'
# Updated to use context_files.csv
CONTEXT_FILES_FILE = DATA_DIR / 'context_files.csv'
OUTPUT_FILE = DATA_DIR / 'repo_agent_metrics_comparison.csv'
MARKDOWN_METRICS_FILE = DATA_DIR / 'context_file_metrics.csv'

# Metrics to analyze statistically
METRICS_TO_TEST = [
    'merge_time', 'closure_time', 'description_length', 'code_churn', 'num_files', 
    'num_commits', 'comments', 'num_inline_comments', 'changes_requested' # <-- Added changes_requested
]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate PR metrics based on context file creation dates.")
    parser.add_argument(
        '-m', '--mode', 
        type=int, 
        choices=[1, 2, 3, 4], 
        required=True,
        help="""
        Mode 1: Before First File vs. After First File.
        Mode 2: Before First File vs. After Last File.
        Mode 3: (Merged PRs) Before First vs. After First.
        Mode 4: (Merged PRs) Before First vs. After Last.
        """
    )
    
    parser.add_argument(
        '-met', '--metrics', 
        nargs='*', 
        required=False,
        help="Select the metrics to plot."
    )

    return parser.parse_args()

def calculate_metrics(df):
    if df.empty:
        # Added an extra 0.0 to account for the new changes_requested metric
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
    
    all_pr_count = len(df)
    merged_count = df[df['state'].astype(str).str.lower() == 'merged'].shape[0]
    merge_rate = merged_count / all_pr_count
    
    # Calculate Medians
    merge_time = df['merge_time'].median()
    closure_time = df['closure_time'].median()
    num_files = df['num_files'].median()
    code_churn = df['code_churn'].median()
    num_commits = df['num_commits'].median()
    comments = df['comments'].median()
    # num_issues = df['num_issues'].median()
    num_issues = None
    description_length = df['description_length'].median()
    num_inline_comments = df['num_inline_comments'].median()
    
    # Calculate median for changes_requested, with a fallback if the column is missing
    changes_requested = df['changes_requested'].median() if 'changes_requested' in df.columns else 0.0
    
    return (merge_rate, merge_time, closure_time, num_files, code_churn, 
            num_commits, comments, num_issues, description_length, 
            num_inline_comments, changes_requested, all_pr_count)

def calc_merge_rate_improvement_pct(merge_rate_baseline, merge_rate_target, on_zero="inf"):
    try:
        b = float(merge_rate_baseline)
        t = float(merge_rate_target)
    except (TypeError, ValueError):
        raise ValueError("merge_rate_baseline and merge_rate_target must be numeric")

    if b == 0.0:
        if t == 0.0: return 0.0
        return float("inf") if on_zero == "inf" else (t - b)
    
    improvement_pct = ((t - b) / b) * 100.0
    return round(improvement_pct, 2) if np.isfinite(improvement_pct) else improvement_pct

def calculate_stats(baseline_df, target_df, metric_list):
    stats_results = {}
    for metric in metric_list:
        p, d, eff_size = None, None, None
        if not baseline_df.empty and not target_df.empty:
            # Added a check to ensure the metric actually exists in the columns before testing
            if metric in baseline_df.columns and metric in target_df.columns:
                b_data = baseline_df[metric].dropna()
                a_data = target_df[metric].dropna()
                if len(b_data) > 0 and len(a_data) > 0:
                    try:
                        _, p, d, eff_size = mannUandCliffdelta(b_data, a_data)
                    except Exception: pass
        stats_results[f'{metric}_p_value'] = p
        stats_results[f'{metric}_delta'] = d
        stats_results[f'{metric}_eff_size'] = eff_size
    return stats_results

def parse_github_url(url):
    # Pattern to capture owner/repo
    pattern = r"github\.com[:/]([^/]+/[^/\s?#.]+?)(?:\.git)?(?:/|$|\?|#)"
    match = re.search(pattern, url)
    if match:
        repo = match.group(1)
        return repo
    return None

def main():
    args = parse_args()

    try:    
        pr_df = pd.read_csv(PR_FEAT_FILE)
        # Load the new context files CSV
        context_df = pd.read_csv(CONTEXT_FILES_FILE)
        # print(f"{pr_df['repo']}")
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Preprocessing
    pr_df['created_at'] = pd.to_datetime(pr_df['created_at'])
    pr_df['closed_at'] = pd.to_datetime(pr_df['closed_at'])
    # Using 'created_at' from context_files.csv
    
    context_df = context_df[context_df['is_reference']==False]
    context_df['created_at'] = pd.to_datetime(context_df['created_at'])
    
    pr_df = pr_df[pr_df['repo'].isin(context_df['repo_name'].unique())]

    results = []
    
    if args.mode in [1, 3]:
        prefix_base, prefix_target = 'before', 'after'
        print(f"Running Mode {args.mode}: Before First vs After First")
    elif args.mode in [2, 4]:
        prefix_base, prefix_target = 'before', 'after_last'
        print(f"Running Mode {args.mode}: Before First vs After Last")

    repos = pr_df['repo'].unique()

    for repo in repos:
        repo_prs = pr_df[pr_df['repo'] == repo]

        # Filter context files using 'repo_name'
        repo_context = context_df[context_df['repo_name'] == repo]
        
        if args.mode in [3, 4]:
            repo_prs = repo_prs[repo_prs['state'] == 'merged']

        if repo_context.empty or repo_prs.empty:
            continue

        # Determine timestamps based on context file creation
        repo_context = repo_context.sort_values('created_at')
        num_context_files = len(repo_context)
        
        first_date = repo_context.iloc[0]['created_at']
        last_date = repo_context.iloc[-1]['created_at']
        first_commit_context_ai_created = repo_context.iloc[0]['first_commit_ai_created']

        df_base = pd.DataFrame()
        df_target = pd.DataFrame()
        valid_case = False

        if args.mode in [1, 3]:
            df_base = repo_prs[repo_prs['closed_at'] < first_date]
            df_target = repo_prs[repo_prs['closed_at'] >= first_date]
            valid_case = True
        elif args.mode in [2, 4]:
            df_base = repo_prs[repo_prs['closed_at'] < first_date]
            df_target = repo_prs[repo_prs['closed_at'] >= last_date]
            valid_case = True

        if valid_case:
            res_base = calculate_metrics(df_base)
            res_target = calculate_metrics(df_target)

            row = {
                'repo': repo,
                'num_context_files': num_context_files,
                'mode': args.mode,
                'first_commit_context_ai_created': first_commit_context_ai_created, 
                'merge_impro_pct': calc_merge_rate_improvement_pct(res_base[0], res_target[0]),
                # Baseline
                f'merge_rate_{prefix_base}': res_base[0],
                f'merge_time_{prefix_base}': res_base[1],
                f'closure_time_{prefix_base}': res_base[2],
                f'num_files_{prefix_base}': res_base[3],
                f'code_churn_{prefix_base}': res_base[4],
                f'num_commits_{prefix_base}': res_base[5],
                f'comments_{prefix_base}': res_base[6],
                # f'num_issues_{prefix_base}': res_base[7],
                f'description_length_{prefix_base}': res_base[8],
                f'num_inline_comments_{prefix_base}': res_base[9],
                f'changes_requested_{prefix_base}': res_base[10], # <-- Mapped Baseline
                f'pr_count_{prefix_base}': res_base[11],          # <-- Shifted to 11

                # Target
                f'merge_rate_{prefix_target}': res_target[0],
                f'merge_time_{prefix_target}': res_target[1],
                f'closure_time_{prefix_target}': res_target[2],
                f'num_files_{prefix_target}': res_target[3],
                f'code_churn_{prefix_target}': res_target[4],
                f'num_commits_{prefix_target}': res_target[5],
                f'comments_{prefix_target}': res_target[6],
                # f'num_issues_{prefix_target}': res_target[7],
                f'description_length_{prefix_target}': res_target[8],
                f'num_inline_comments_{prefix_target}': res_target[9],
                f'changes_requested_{prefix_target}': res_target[10], # <-- Mapped Target
                f'pr_count_{prefix_target}': res_target[11],          # <-- Shifted to 11
                'first_date': first_date,
                'last_date': last_date
            }

            stats = calculate_stats(df_base, df_target, METRICS_TO_TEST)
            row.update(stats)
            results.append(row)

    if not results:
        print("No results found.")
        return

    results_df = pd.DataFrame(results)
    non_stats_cols = [c for c in results_df.columns if 'p_value' not in c and 'eff_size' not in c]
    results_df[non_stats_cols] = results_df[non_stats_cols].fillna(0)
    
    # Merge with Markdown Metrics
    if MARKDOWN_METRICS_FILE.exists():
        markdown_metrics_df = pd.read_csv(MARKDOWN_METRICS_FILE)
        # results_df['repo_alias'] = results_df['repo'].map(lambda x: x.replace('/', '_'))
        print(f"READ {MARKDOWN_METRICS_FILE}")
        results_df = pd.merge(
            results_df, markdown_metrics_df, on='repo', how='left'
        )

    output_filename = str(OUTPUT_FILE).replace('.csv', f'_mode_{args.mode}.csv')
    results_df.to_csv(output_filename, index=False)
    print(f"Analysis complete. Results saved to {output_filename}")

if __name__ == "__main__":
    main()