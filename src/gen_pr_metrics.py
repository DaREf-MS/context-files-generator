import pandas as pd
import re
import numpy as np
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_CSV = DATA_DIR / "prs_features.csv"
MARKDOWN_METRICS_FILE = DATA_DIR / 'context_file_metrics.csv'

def count_inline_comments_in_diff(file_patches):
    """
    Analyzes code patches to count comment-like strings in added lines.
    """
    if not isinstance(file_patches, list) or len(file_patches) == 0:
        return 0

    count = 0
    for patch in file_patches:
        if not isinstance(patch, str):
            continue
        for line in patch.splitlines():
            line = line.strip()
            # Only count ADDED lines (real code changes)
            if line.startswith("+") and not line.startswith("+++"):
                code = line[1:].strip()
                if (
                    re.search(r"#", code) or          # Python / Bash
                    re.search(r"//", code) or         # C / Java / JS
                    re.search(r"/\*", code) or        # Block comment start
                    re.search(r"^\*", code)           # Block comment continuation (starts with *)
                ):
                    count += 1
    return count

def process_prs_from_csv():
    print("Loading CSV files...")
    
    # 1. Load Dataframes
    pr_df = pd.read_csv(DATA_DIR / 'pull_request_combined.csv')
    repo_df = pd.read_csv(DATA_DIR / 'repository_combined.csv')
    pr_comments_df = pd.read_csv(DATA_DIR / 'pr_comments_combined.csv')
    pr_commits_df = pd.read_csv(DATA_DIR / "pr_commits_combined.csv")
    pr_commit_details_df = pd.read_csv(DATA_DIR / "pr_commit_details_combined.csv")
    pr_reviews_df = pd.read_csv(DATA_DIR / "pr_reviews_combined.csv")
    
    repo_df = repo_df.rename(columns={'id': 'repo_id'})

    # 2. Base Filtering
    # Set state to 'merged' if merged_at exists, otherwise keep existing state
    # We filter out 'open' PRs as per your original logic
    base_df = pr_df.copy()
    base_df['state'] = np.where(base_df['merged_at'].notna(), 'merged', base_df['state'])
    base_df = base_df[base_df['state'] != 'open']
    base_df = base_df.rename(columns={'id': 'pr_id'})

    print("Aggregating metrics...")

    # 3. Aggregations
    
    # A. Commits Count
    commit_counts = pr_commits_df.groupby('pr_id')['sha'].count().reset_index(name='num_commits')

    # B. File Count and Code Churn
    commit_details_agg = pr_commit_details_df.groupby('pr_id').agg(
        num_files=('filename', 'nunique'),
        total_additions=('additions', 'sum'),
        total_deletions=('deletions', 'sum')
    ).reset_index()
    commit_details_agg['code_churn'] = commit_details_agg['total_additions'] + commit_details_agg['total_deletions']

    # C. General Comments
    comment_counts = pr_comments_df.groupby('pr_id')['id'].count().reset_index(name='comments')

    # D. Inline Comments (Parsing patches from details)
    pr_commit_details_df['patch'] = pr_commit_details_df['patch'].fillna('')
    inline_comment_counts_df = pr_commit_details_df.groupby('pr_id')['patch'].apply(list).reset_index()
    inline_comment_counts_df['num_inline_comments'] = inline_comment_counts_df['patch'].map(count_inline_comments_in_diff)
    
    # E. Requested reviews
    changes_requested_counts_df = pr_reviews_df[pr_reviews_df['state'].str.lower() == 'changes_requested'].groupby('pr_id')['id'].count().reset_index(name='changes_requested')

    # 4. Merging
    final_df = base_df.merge(commit_counts, on='pr_id', how='left')
    final_df = final_df.merge(commit_details_agg[['pr_id', 'num_files', 'code_churn']], on='pr_id', how='left')
    final_df = final_df.merge(comment_counts, on='pr_id', how='left')
    final_df = final_df.merge(inline_comment_counts_df[['pr_id', 'num_inline_comments']], on='pr_id', how='left')
    final_df = final_df.merge(changes_requested_counts_df[['pr_id', 'changes_requested']], on='pr_id', how='left')

    # 5. Feature Engineering (Dates & Times)
    for col in ['created_at', 'closed_at', 'merged_at']:
        final_df[col] = pd.to_datetime(final_df[col], errors='coerce')

    # Duration in hours
    final_df['closure_time'] = (final_df['closed_at'] - final_df['created_at']).dt.total_seconds() / 3600
    final_df['merge_time'] = (final_df['merged_at'] - final_df['created_at']).dt.total_seconds() / 3600
    
    # Description Length
    final_df['description_length'] = final_df['body'].str.len().fillna(0).astype(int)

    # Repository extraction (assuming standard GitHub URL format)
    # if 'repo_url' in final_df.columns:
    #     final_df['repo'] = final_df['repo_url'].str.replace('https://api.github.com/repos/', '', regex=False)
    
    final_df = final_df.merge(
        repo_df[['repo_id', 'full_name']],
        on='repo_id',
        how='left'
    ).rename(columns={'full_name': 'repo'})

    # 6. Cleaning and Final Selection
    metrics_to_fill = ['num_files', 'code_churn', 'num_commits', 'comments', 'num_inline_comments', 'changes_requested']
    print(f"{final_df.columns=}")
    final_df[metrics_to_fill] = final_df[metrics_to_fill].fillna(0).astype(int)
    
    # Rename 'number' to 'pr_number' to match your required output
    final_df.rename(columns={'number': 'pr_number'}, inplace=True)

    # Handle 'user' (extracting login if it's a dict-string, otherwise using column)
    # If the CSV already parsed 'user' to a string, this just ensures it exists
    if 'user' not in final_df.columns and 'user_login' in final_df.columns:
        final_df['user'] = final_df['user_login']

    target_columns = [
        'repo', 'agent', 'pr_number', 'title', 'state', 'user',
        'created_at', 'closed_at', 'merge_time', 'closure_time', 'num_files', 'code_churn',
        'num_commits', 'comments', 'description_length', 'num_inline_comments', 'changes_requested'
    ]

    # Filter only available columns to avoid errors
    available_cols = [c for c in target_columns if c in final_df.columns]
    result_df = final_df[available_cols]
    
    # Read MARKDOWN_METRICS_FILE
    markdown_metrics_df = pd.read_csv(MARKDOWN_METRICS_FILE)
    
    # Merge dfs
    result_df = pd.merge(
        result_df,
        markdown_metrics_df,
        on='repo',
        how='left'
    )

    print(f"Saving {len(result_df)} records to {OUTPUT_CSV}...")
    result_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    process_prs_from_csv()