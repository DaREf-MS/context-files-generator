import math
import argparse
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import random
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from logging_config import get_logger

logger = get_logger(__name__)

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
FIGS_PATH = ROOT_DIR / "figs"
DATA_DIR = ROOT_DIR / "data"

METRICS_FILE = "repo_agent_metrics_comparison_mode_1.csv"
TOPICS_JSONL_FILE = DATA_DIR / "assignment_corrected_sample.jsonl"
CONFIGURED_REPOS_FILE = DATA_DIR / "repos_with_deps_count.csv"
COMMITS_METADATA_FILE = DATA_DIR / "context_files_commits_metadata.csv"
PRS_FEATURES_FILE = DATA_DIR / "prs_features.csv"
CONTEXT_FILES_SUBSET_FILE = DATA_DIR / "context_files_subset.csv"
ALL_COMMIT_FILES_FILE = DATA_DIR / "commit_files_results.csv"
REFINEMENT_FILE = DATA_DIR / "refinement.md"
SEPARATOR = '&SEP&'

COUNT_PR_THRESHOLD = 20

FIGS_PATH.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """
You are an expert annotator. Your task is to analyze and reason what changes has been made to an Agent Context File (ACF), such as `AGENTS.md`, and assign the most appropriate label that best reflects the nature of the change. ACF contains instructions and guidelines for AI agents when working with code repositories.

## Instructions

To complete this task, you should follow these steps:

1. Analyze the particular changes to any of the topic labels.
2. Briefly explain what is the main in a given topic. Base your explanation on the commit message and the ACF diff provided.
3. Assign the label that best represents the change to the ACF.

## Typical example

<TOPICS>
<TOPIC>Build, Test & Development Commands</TOPIC>
<TOPIC>Testing & Quality Verification</TOPIC>
<TOPIC>Code Style, Conventions</TOPIC>
<TOPIC>Naming, Git, Commit & Branch Workflow</TOPIC>
<TOPIC>Documentation Maintenance, Navigation & Instruction Hierarchy</TOPIC>
<TOPIC>Change Scope, Repository Policy & Worktree Execution</TOPIC>
</TOPICS>

<COMMIT_MESSAGE>
Windows reproducible build & ccache hit rate improvement (#4419)
## Motivation

Fixes #4519, parent issue #4195

## Technical Details

* Reproducible builds on Windows via `/Brepro` for MSVC toolchain
Ensure amd-llvm is reproducible & COFF header contains a reproducible
hash instead of current timestamp.
* `compiler_check=content` for Windows
Switch `compiler_check` from `mtime` to `content`. Since the LLVM
toolchain on Windows is statically linked, hashing the compiler binary
content is sufficient and more reliable for detecting actual compiler
changes across CI runs.
* Support PCH
Add `pch_defines` and `time_micros` to ccache sloppiness. amd-llvm on
Windows builds uses precompiled headers by default.
* Misc changes to ccache config
  Max size bump to 10G, rename `secondary_storage` to `remote_storage`

## Test Plan

Validate cache hitrate on Windows following subsequent buil.ds
</COMMIT_MESSAGE>

<ACF_DIFF>
@@ -276,7 +276,8 @@ Processing services are FastAPI applications that implement the AMI ML API contr
 **Health Checks:**
 - Cached status with 3 retries and exponential backoff (0s, 2s, 4s)
 - Celery Beat task runs periodic checks (`ami.ml.tasks.check_processing_services_online`)
-- Status stored in `ProcessingService.last_checked_live` boolean field
+- Status stored in `ProcessingService.last_seen_live` boolean field
+- Async/pull-mode services update status via `mark_seen()` when they register pipelines
 - UI shows red/green indicator based on cached status
 
 Location: `processing_services/` directory contains example implementations
</ACF_DIFF>

Output:

**Analysis**: The commit message and the ACF diff indicate that the changes are focused on build on Windows platform, specifically improving reproducibility and cache hit rates. The changes are related to build, test, and development commands.
**Labeling**: Multi-platform build coverage: guiding the agent on how to run the build on an additional platform (windows). This lead to the following recommendation: "make sure you consider how to run the build on all the platforms that you support".

## Input data

<TOPICS>
{topics}
</TOPICS>

<COMMIT_MESSAGE>
{commit_message}
</COMMIT_MESSAGE>

<ACF_DIFF>
{file_patch}
</ACF_DIFF>

Your response:
"""

# --- Custom Formatting Helpers ---


def format_km(x, pos=None):
    """Formats numbers with 'k' (thousands) and 'm' (millions)."""
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}m".replace('.0m', 'm')
    if x >= 1_000:
        return f"{x/1_000:.1f}k".replace('.0k', 'k')
    return f"{int(x)}" if x % 1 == 0 else f"{x:.1f}"


def format_hour(x, pos=None):
    """Formats numbers by appending 'h' for hours."""
    return f"{int(x)}h" if x % 1 == 0 else f"{x:.1f}h"


def format_pct(x, pos=None):
    """Formats numbers by appending '%'."""
    return f"{int(x)}%" if x % 1 == 0 else f"{x:.1f}%"


def get_suffix(one_version=False, ai_filter="all"):
    """Generates a file suffix based on runtime arguments."""
    s = ""
    if one_version:
        s += "_one_version"
    if ai_filter != "all":
        s += f"_{ai_filter}"
    return s


def load_and_filter_metrics():
    """Loads metrics and applies the PR threshold filter (>= 20 after)."""
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)

        # Apply PR Thresholds.
        if 'prs_before' in df_metrics.columns and 'prs_after' in df_metrics.columns:
            df_metrics = df_metrics[(
                df_metrics['prs_after'] >= COUNT_PR_THRESHOLD)]
        elif 'totalPullRequests' in df_metrics.columns:
            df_metrics = df_metrics[df_metrics['totalPullRequests'] >= (
                COUNT_PR_THRESHOLD * 2)]

        return df_metrics
    except FileNotFoundError as e:
        print(f"Error loading metrics: {e}")
        return pd.DataFrame()


def filter_by_ai(df, ai_filter):
    """Filters the dataset to only include AI-generated or Human-generated files."""
    if ai_filter == "all":
        return df

    try:
        df_context = pd.read_csv(CONTEXT_FILES_SUBSET_FILE)

        # Create a clean file path for merging (removes the &SEP& timestamp if present)
        df['file_path_merge'] = df['context_file'].apply(
            lambda x: x.split(SEPARATOR)[0] if SEPARATOR in x else x
        )

        # Merge on repo and the cleaned file path
        df_merged = pd.merge(
            df,
            df_context[['repo_name', 'context_file',
                        'first_commit_ai_created']],
            left_on=['repo', 'file_path_merge'],
            right_on=['repo_name', 'context_file'],
            how='inner',
            suffixes=('', '_csv')
        )

        # Determine AI creation safely handling boolean or string 'True'/'False'
        is_ai = df_merged['first_commit_ai_created'].astype(
            str).str.lower().isin(['true', '1', 'yes', 't', 'y'])

        if ai_filter == "ai":
            filtered_df = df_merged[is_ai].copy()
        elif ai_filter == "human":
            filtered_df = df_merged[~is_ai].copy()

        # Clean up the helper columns used for merging to retain original dataframe structure
        filtered_df = filtered_df.drop(columns=[
                                       'file_path_merge', 'repo_name', 'context_file_csv', 'first_commit_ai_created'])

        print(
            f"Applied AI Filter ({ai_filter}): {len(df)} -> {len(filtered_df)} versions.")
        return filtered_df

    except FileNotFoundError as e:
        print(
            f"Warning: {CONTEXT_FILES_SUBSET_FILE} not found. Skipping AI filtering. ({e})")
        return df


def read_refinement_topics():
    topics = []
    with open(REFINEMENT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                group_pattern = r"\]\s(.*?)\s\(Count"
                match = re.search(group_pattern, line)
                if match:
                    topic = match.group(1)
                    topics.append(topic)
    return topics


def parse_jsonl_file():
    """Reads the JSONL file and returns all versions across all files."""
    data = []
    refinement_topics = read_refinement_topics()
    with open(TOPICS_JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_id = row.get("id", "")
            raw_responses = row.get("responses", "")
            timestamp = row.get("timestamp", None)

            if SEPARATOR in raw_id:
                repo_part, file_part = raw_id.split(SEPARATOR, 1)
                repo_name = repo_part.replace("§", "/")
                context_file = file_part.replace("§", "/")
                timestamp = context_file.split(SEPARATOR)[1].replace(
                    ".md", "").replace('_', ' ')
            else:
                repo_name = raw_id.replace("§", "/")
                context_file = "N/A"

            topics = re.findall(r"\[\d+\]\s*(.*?):", raw_responses)

            cleaned_topics = []
            for t in topics:
                t = str(t).strip()

                # 1. Reformat and clean unicode characters
                t = unicodedata.normalize('NFKC', t)

                # Swap the missing Unicode hyphen for a standard dash
                t = t.replace('\u2010', '-')

                # Un-escape double-escaped unicode characters
                try:
                    t = t.encode('raw_unicode_escape').decode('unicode_escape')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass

                # 2. Filter out mismatched or empty topics
                if (not t) or (t not in refinement_topics):
                # if not t or '*' in t or '"' in t:
                    continue

                cleaned_topics.append(t)

            data.append({
                "repo": repo_name,
                "context_file": context_file,
                "topic_labels": cleaned_topics,
                "timestamp": timestamp,
            })

    df = pd.DataFrame(data)
    if not df.empty:
        # Enforce UTC so we can compare accurately with PR timestamps later
        df["timestamp_parsed"] = pd.to_datetime(
            df["timestamp"], format="mixed", errors="coerce", utc=True)
    return df


def filter_one_version(df):
    """Filters dataset to only the first occurrence of each file."""
    if df.empty:
        return df
    df_sorted = df.sort_values(
        by=["repo", "context_file", "timestamp_parsed"], ascending=True)
    return df_sorted.drop_duplicates(subset=["repo", "context_file"], keep="first")


def deduplicate_topics_per_project(df):
    """Groups by project and flattens to ensure one topic occurrence per project."""
    if df.empty:
        return df
    return df.groupby("repo", as_index=False).agg({
        "context_file": lambda x: ", ".join(x.astype(str).unique()),
        "topic_labels": lambda x: list(set(topic for sublist in x for topic in sublist)),
        "timestamp": "first"
    })


def keep_last_topics_per_project(df: pd.DataFrame):
    """Groups by project and flattens to ensure one topic occurrence per project."""
    if df.empty:
        return df
    return df.sort_values(["repo", "context_file", "timestamp"], ascending=[0, 0, 0]).drop_duplicates(subset=["repo", "context_file"], keep="last")


# -----------------------------------------------------------
# 1. Global Topic Distribution (Barplot)
# -----------------------------------------------------------
def plot_global_topic_distribution(df_topics, one_version=False, ai_filter="all"):
    df_metrics = load_and_filter_metrics()
    if df_metrics.empty:
        return

    df = pd.merge(df_metrics, df_topics, on="repo", how="inner")
    total_repos = df["repo"].nunique()

    exploded_df = df.explode("topic_labels").copy()
    exploded_df["topic_labels"] = exploded_df["topic_labels"].astype(
        str).str.strip()
    exploded_df = exploded_df[
        exploded_df["topic_labels"].notna() &
        (exploded_df["topic_labels"] != "") &
        (exploded_df["topic_labels"] != "nan")
    ]

    topic_counts = exploded_df.groupby("topic_labels")[
        "repo"].nunique().reset_index(name="Unique_Repo_Count")
    topic_counts["Percentage"] = (
        topic_counts["Unique_Repo_Count"] / total_repos) * 100
    topic_counts = topic_counts.sort_values(by="Percentage", ascending=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 10))

    ax = sns.barplot(
        data=topic_counts, y="topic_labels", x="Percentage",
        color="#3498db", edgecolor="black", linewidth=0.5
    )

    for container in ax.containers:
        labels = [f"{w:.1f}%" if (w := bar.get_width())
                  > 0 else "" for bar in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=10)

    plt.xlabel("Percentage of repositories", fontsize=12)
    plt.ylabel("TopicGPT labels", fontsize=12)

    max_pct = topic_counts["Percentage"].max(
    ) if not topic_counts.empty else 100
    plt.xlim(0, max_pct + 10)
    plt.tight_layout()

    suffix = get_suffix(one_version, ai_filter)
    out_pdf = FIGS_PATH / f"global_topic_distribution{suffix}.pdf"
    out_img = FIGS_PATH / f"global_topic_distribution{suffix}.png"

    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Global topic distribution saved to: {out_img}")
    plt.close()


# -----------------------------------------------------------
# 2. Global Topic Co-occurrence (Heatmap)
# -----------------------------------------------------------
def plot_global_topic_cooccurrence(df_topics, one_version=False, ai_filter="all"):
    df_metrics = load_and_filter_metrics()
    if df_metrics.empty:
        return

    df = pd.merge(df_metrics, df_topics, on="repo", how="inner")
    valid_topics = df['topic_labels'].apply(
        lambda x: x if isinstance(x, list) else [])
    master_topics = sorted(list(set(valid_topics.explode().dropna())))

    if not master_topics:
        return

    pair_counts = Counter()
    for topics in df['topic_labels']:
        if isinstance(topics, list) and len(topics) > 1:
            topics_sorted = sorted(topics)
            for pair in combinations(topics_sorted, 2):
                pair_counts[pair] += 1

    matrix = pd.DataFrame(0, index=master_topics, columns=master_topics)
    for (t1, t2), count in pair_counts.items():
        matrix.at[t1, t2] = count
        matrix.at[t2, t1] = count

    active_topics = matrix.sum(axis=1)[matrix.sum(axis=1) > 0].index.tolist()
    matrix = matrix.loc[active_topics, active_topics]

    if matrix.empty:
        return

    sns.set_theme(style="white")
    fig_size = max(12, len(active_topics) * 0.6)
    plt.figure(figsize=(fig_size, fig_size * 0.8))

    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(
        matrix, mask=mask, cmap="Purples", annot=True, fmt="d", linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Co-occurrence Frequency"}
    )

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    suffix = get_suffix(one_version, ai_filter)
    out_img = FIGS_PATH / f"global_topic_cooccurrence{suffix}.png"
    out_pdf = FIGS_PATH / f"global_topic_cooccurrence{suffix}.pdf"
    plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Global co-occurrence heatmap saved to: {out_img}")
    plt.close()


# -----------------------------------------------------------
# 3. Global Topic Churn / Evolution
# -----------------------------------------------------------
def plot_global_topic_churn(df_all_versions, ai_filter="all"):
    df_metrics = load_and_filter_metrics()
    if df_metrics.empty:
        return

    df_versions = df_all_versions.copy()
    if 'file_path' not in df_versions.columns:
        df_versions[['file_path', 'file_timestamp']] = df_versions['context_file'].str.split(
            '&SEP&', n=1, expand=True)

    df_sorted = df_versions.sort_values(
        by=["repo", "file_path"])
    valid_repos = set(df_metrics['repo'].unique())
    df_sorted = df_sorted[df_sorted['repo'].isin(valid_repos)]

    records = []

    # NEW: List to hold our CSV records
    csv_records = []

    for (repo, file_path), group in df_sorted.groupby(['repo', 'file_path']):
        is_first_version = True
        prev_topics = set()

        # Check if the current context file has more than 1 version
        has_multiple_versions = len(group) > 1

        # NEW: Dictionary to store topics per version order
        version_dict = {}
        topic_stats_dict = {}

        # ---> REMOVED version_counter = 1

        for _, row in group.iterrows():
            curr_topics = set(row['topic_labels']) if isinstance(
                row['topic_labels'], list) else set()

            # ---> CHANGED: Add the current topics to our version dictionary using the file_timestamp
            timestamp_key = str(row['file_timestamp'])
            version_dict[timestamp_key] = list(curr_topics)

            if is_first_version:
                is_first_version = False

                # If only 1 version exists, skip comparison and don't record transitions
                if not has_multiple_versions:
                    prev_topics = curr_topics.copy()
                    continue
                # If multiple versions exist, we skip the `continue`
                # and compare against `prev_topics` (which starts as an empty set).

            added = curr_topics.difference(prev_topics)
            removed = prev_topics.difference(curr_topics)
            unchanged = curr_topics.intersection(prev_topics)

            # NEW: Helper function to initialize missing topics in the stats dictionary
            def init_topic_stats(t):
                if t not in topic_stats_dict:
                    topic_stats_dict[t] = {'added': 0,
                                           'removed': 0, 'unchanged': 0}

            for t in added:
                records.append(
                    {'repo': repo, 'file_path': file_path, 'topic': t, 'status': 'Added'})
                init_topic_stats(t)
                topic_stats_dict[t]['added'] += 1

            for t in removed:
                records.append(
                    {'repo': repo, 'file_path': file_path, 'topic': t, 'status': 'Removed'})
                init_topic_stats(t)
                topic_stats_dict[t]['removed'] += 1

            for t in unchanged:
                records.append(
                    {'repo': repo, 'file_path': file_path, 'topic': t, 'status': 'Unchanged'})
                init_topic_stats(t)
                topic_stats_dict[t]['unchanged'] += 1

            prev_topics = curr_topics.copy()

        # NEW: Append the aggregated file history to the csv_records list
        csv_records.append({
            'repo': repo,
            'context file': file_path,
            'dictionary': json.dumps(version_dict),
            'topic stats': json.dumps(topic_stats_dict)
        })

    # NEW: Export the dictionary records to a CSV file
    suffix = get_suffix(False, ai_filter)
    df_csv_export = pd.DataFrame(csv_records)
    csv_file_path = FIGS_PATH / f"topic_versions_per_file{suffix}.csv"
    df_csv_export.to_csv(csv_file_path, index=False)
    print(f"Topic version dictionaries saved to CSV: {csv_file_path}")

    df_transitions = pd.DataFrame(records)
    if df_transitions.empty:
        return

    # =========================================================================
    # GLOBAL CHART & STATS LOGIC
    # =========================================================================
    agg_df = df_transitions.groupby(
        ['topic', 'status']).size().reset_index(name='count')
    pivot_df = agg_df.pivot_table(
        index='topic', columns='status', values='count', fill_value=0).reset_index()

    for col in ['Added', 'Unchanged', 'Removed']:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    pivot_df['total'] = pivot_df['Added'] + \
        pivot_df['Unchanged'] + pivot_df['Removed']
    pivot_df = pivot_df.sort_values(
        by='total', ascending=True).set_index('topic')

    pivot_df['% Added'] = (pivot_df['Added'] / pivot_df['total']) * 100
    pivot_df['% Unchanged'] = (pivot_df['Unchanged'] / pivot_df['total']) * 100
    pivot_df['% Removed'] = (pivot_df['Removed'] / pivot_df['total']) * 100

    stats_file_path = FIGS_PATH / \
        f"global_topic_churn_percentages_stats{suffix}.txt"

    with open(stats_file_path, "w", encoding="utf-8") as f:
        f.write("Global Topic Churn Percentage Statistics & Breakdown\n")
        f.write("====================================================\n\n")

        f.write("PART 1: SUMMARY STATISTICS (MIN, MAX, MEDIAN)\n")
        f.write("---------------------------------------------\n")
        for col in ['% Added', '% Unchanged', '% Removed']:
            max_val = pivot_df[col].max()
            min_val = pivot_df[col].min()
            median_val = pivot_df[col].median()

            max_topics = pivot_df[pivot_df[col] == max_val].index.tolist()
            min_topics = pivot_df[pivot_df[col] == min_val].index.tolist()

            diff = (pivot_df[col] - median_val).abs()
            median_topics = pivot_df[diff == diff.min()].index.tolist()

            f.write(f"--- {col} ---\n")
            f.write(
                f"Max    : {max_val:.2f}% | Topic(s): {', '.join(max_topics)}\n")
            f.write(
                f"Min    : {min_val:.2f}% | Topic(s): {', '.join(min_topics)}\n")
            f.write(
                f"Median : {median_val:.2f}% | Closest Topic(s): {', '.join(median_topics)}\n\n")

        f.write("\nPART 2: COMPLETE PER-TOPIC BREAKDOWN (Sorted by Total Actions)\n")
        f.write("--------------------------------------------------------------\n")
        for topic_name, row in pivot_df.sort_values(by='total', ascending=False).iterrows():
            f.write(f"Topic: {topic_name}\n")
            f.write(f"  Total Actions : {int(row['total'])}\n")
            f.write(f"  % Added       : {row['% Added']:.2f}%\n")
            f.write(f"  % Unchanged   : {row['% Unchanged']:.2f}%\n")
            f.write(f"  % Removed     : {row['% Removed']:.2f}%\n")
            f.write("-" * 40 + "\n")

    print(
        f"Comprehensive topic churn statistics and breakdown saved to: {stats_file_path}")

    # Plot Global Bar Chart
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, max(8, len(pivot_df) * 0.4)))

    churn_attrs = ['Added', 'Unchanged']
    ax = pivot_df[churn_attrs].plot(
        kind='barh', stacked=True, color=['#3498db', '#bdc3c7', '#e67e22'],
        edgecolor='black', linewidth=0.5, ax=plt.gca()
    )

    max_x = pivot_df[churn_attrs].sum(axis=1).max()
    space_threshold = max_x * 0.04

    for row_idx, (topic_name, row) in enumerate(pivot_df.iterrows()):
        total_val = row[churn_attrs].sum()
        if total_val == 0:
            continue

        small_vals = []
        for col_idx, col_name in enumerate(churn_attrs):
            val = row[col_name]
            if val == 0:
                continue

            start_x = sum(row[churn_attrs][:col_idx])
            if val >= space_threshold:
                text_color = 'black' if col_name == 'Unchanged' else 'white'
                # Reduced font size for better box fitting (fontsize=11)
                ax.text(start_x + val / 2, row_idx, str(int(val)),
                        va='center', ha='center', fontsize=11, weight='bold', color=text_color)
            else:
                small_vals.append(f"{col_name[0]}: {int(val)}")

        if small_vals:
            # Scaled overflow text to match (fontsize=10)
            ax.text(total_val + (max_x * 0.015), row_idx, " (" + ", ".join(small_vals) + ")",
                    va='center', ha='left', fontsize=9, color='#333333', style='italic', weight='bold')

    # Main label and axis font sizing kept large for article readability
    plt.xlabel("Occurrences over file versions", fontsize=22, labelpad=14, fontweight='bold')
    plt.ylabel("TopicGPT labels", fontsize=22, labelpad=14, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(title="Status", title_fontsize='18', fontsize='16',
               loc="lower right", bbox_to_anchor=(1.0, 0.0))
    plt.tight_layout()

    out_img = FIGS_PATH / f"global_topic_churn_over_versions{suffix}.png"
    out_pdf = FIGS_PATH / f"global_topic_churn_over_versions{suffix}.pdf"
    plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Global topic churn plot saved to: {out_img}")
    plt.close()


# -----------------------------------------------------------
# 4. Global Topic Attribute Barplots (Top-30% vs Bottom-30%)
# -----------------------------------------------------------
def plot_global_topic_attributes(df_topics, one_version=False, ai_filter="all"):
    df_metrics = load_and_filter_metrics()
    if df_metrics.empty:
        return

    try:
        df_repos = pd.read_csv(CONFIGURED_REPOS_FILE)
    except FileNotFoundError as e:
        print(f"Error loading repos file: {e}")
        return

    df = pd.merge(df_metrics, df_topics, on="repo", how="inner")
    df = pd.merge(df, df_repos, left_on="repo",
                  right_on="repo_name", how="inner")

    attributes = ["contributors", "stargazers", "commits",
                  "totalPullRequests", "codeLines", "dependency_count"]
    sns.set_theme(style="whitegrid")

    for attr in attributes:
        if attr not in df.columns:
            continue

        df_proj = df[['repo', attr]].drop_duplicates().dropna(subset=[attr])
        df_proj[attr] = pd.to_numeric(df_proj[attr], errors='coerce')
        df_proj = df_proj.dropna(subset=[attr])

        if df_proj.empty:
            continue

        q30 = df_proj[attr].quantile(0.3)
        q70 = df_proj[attr].quantile(0.7)

        top_30_repos = df_proj[df_proj[attr] >= q70]['repo'].unique()
        bottom_30_repos = df_proj[df_proj[attr] <= q30]['repo'].unique()

        n_top = len(top_30_repos)
        n_bottom = len(bottom_30_repos)

        if n_top == 0 or n_bottom == 0:
            continue

        df_exploded = df.explode("topic_labels").copy()
        df_exploded["topic_labels"] = df_exploded["topic_labels"].astype(
            str).str.strip()
        df_exploded = df_exploded[
            df_exploded["topic_labels"].notna() &
            (df_exploded["topic_labels"] != "") &
            (df_exploded["topic_labels"] != "nan")
        ].drop_duplicates(subset=["repo", "topic_labels"])

        top_df = df_exploded[df_exploded['repo'].isin(top_30_repos)]
        bottom_df = df_exploded[df_exploded['repo'].isin(bottom_30_repos)]

        # 1. Get raw counts for the Chi-square contingency table
        top_counts = top_df.groupby("topic_labels")['repo'].nunique()
        bottom_counts = bottom_df.groupby("topic_labels")['repo'].nunique()

        plot_data = pd.DataFrame({
            'Top 30% Count': top_counts,
            'Bottom 30% Count': bottom_counts
        }).fillna(0).reset_index()

# 2. Perform Chi-square test for each topic
        topics_list = []
        p_values_list = []
        
        for _, row in plot_data.iterrows():
            topic = row['topic_labels']
            t_c = int(row['Top 30% Count'])
            b_c = int(row['Bottom 30% Count'])

            contingency_table = [
                [t_c, n_top - t_c],
                [b_c, n_bottom - b_c]
            ]

            try:
                chi2, p, dof, ex = chi2_contingency(contingency_table)
            except ValueError:
                p = 1.0
                
            topics_list.append(topic)
            p_values_list.append(p)

        # Apply Holm Correction
        reject, pvals_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='holm')
        significant_topics = set(t for t, r in zip(topics_list, reject) if r)

        # 3. Calculate percentages for plotting
        plot_data['Top 30%'] = (plot_data['Top 30% Count'] / n_top) * 100
        plot_data['Bottom 30%'] = (plot_data['Bottom 30% Count'] / n_bottom) * 100
        plot_data = plot_data.sort_values(by="Top 30%", ascending=False)

        plot_data_melted = plot_data.melt(
            id_vars="topic_labels",
            value_vars=['Top 30%', 'Bottom 30%'],
            var_name="Group",
            value_name="Percentage"
        )

        plt.figure(figsize=(14, 10))
        
        ax = sns.barplot(
            data=plot_data_melted, x="Percentage", y="topic_labels", hue="Group",
            palette=["#3498db", "#e74c3c"], edgecolor="black", linewidth=0.5
        )

        # 4. HORIZONTAL ASTERISKS (Updated)
        x_max = ax.get_xlim()[1]

        for i, topic in enumerate(plot_data['topic_labels']):
            if topic in significant_topics:
                top_val = plot_data.iloc[i]['Top 30%']
                bot_val = plot_data.iloc[i]['Bottom 30%']
                max_val = max(top_val, bot_val)

                # Scaled asterisk font size
                ax.text(max_val, i, '*', ha='left', va='center',
                        fontsize=24, fontweight='bold', color='black')

        ax.set_xlim(0, x_max + (x_max * 0.1))

        # INCREASED FONT SIZES FOR PUBLICATION READABILITY:
        plt.xlabel("% of projects", fontsize=22, labelpad=14, fontweight='bold')
        plt.ylabel("TopicGPT labels", fontsize=22, labelpad=14, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)  # Topic labels
        
        # Legend sizing
        plt.legend(title=f"{attr.capitalize()} Percentiles", title_fontsize='18', fontsize='16',
                   loc='lower right', borderaxespad=1.5)
        
        plt.tight_layout()

        suffix = get_suffix(one_version, ai_filter)
        out_img = FIGS_PATH / \
            f"global_topic_{attr.lower()}_barplot{suffix}.png"
        out_pdf = FIGS_PATH / \
            f"global_topic_{attr.lower()}_barplot{suffix}.pdf"
        plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Global attribute barplot ({attr}) saved to: {out_img}")
        plt.close()


# -----------------------------------------------------------
# 5. PR Metrics Barplots (Merge Rate + PR Granular Features)
# -----------------------------------------------------------
def plot_topic_pr_metrics_boxplots(df_topics, df_all_versions, one_version=False, ai_filter="all"):
    df_metrics = load_and_filter_metrics()
    if df_metrics.empty:
        return

    try:
        df_prs = pd.read_csv(PRS_FEATURES_FILE)
    except FileNotFoundError as e:
        print(f"Error loading PR features file: {e}")
        return

    sns.set_theme(style="whitegrid")
    suffix = get_suffix(one_version, ai_filter)

    def plot_percentile_barplot(df_source, metric_col, title_prefix, is_pr_metric=False):
        df_source = df_source.copy()
        df_source[metric_col] = pd.to_numeric(df_source[metric_col], errors='coerce')
        
        df_exploded = df_topics.explode("topic_labels").copy()
        df_exploded["topic_labels"] = df_exploded["topic_labels"].astype(str).str.strip()
        df_exploded = df_exploded[
            df_exploded["topic_labels"].notna() &
            (df_exploded["topic_labels"] != "") &
            (df_exploded["topic_labels"] != "nan")
        ].drop_duplicates(subset=["repo", "topic_labels"])

        if is_pr_metric:
            df_valid = df_source.dropna(subset=['repo', metric_col])
            
            if df_valid.empty:
                return

            q30 = df_valid[metric_col].quantile(0.3)
            q70 = df_valid[metric_col].quantile(0.7)

            top_subset = df_valid[df_valid[metric_col] >= q70]
            bottom_subset = df_valid[df_valid[metric_col] <= q30]

            n_top = len(top_subset)
            n_bottom = len(bottom_subset)

            if n_top == 0 or n_bottom == 0:
                return

            top_merged = pd.merge(top_subset, df_exploded, on='repo', how='inner')
            bottom_merged = pd.merge(bottom_subset, df_exploded, on='repo', how='inner')

            top_counts = top_merged.groupby("topic_labels").size()
            bottom_counts = bottom_merged.groupby("topic_labels").size()
            
            x_label = "% of PRs"

        else:
            df_metric_proj = df_source[['repo', metric_col]].drop_duplicates().dropna()

            if df_metric_proj.empty:
                return

            q30 = df_metric_proj[metric_col].quantile(0.3)
            q70 = df_metric_proj[metric_col].quantile(0.7)

            top_30_repos = df_metric_proj[df_metric_proj[metric_col] >= q70]['repo'].unique()
            bottom_30_repos = df_metric_proj[df_metric_proj[metric_col] <= q30]['repo'].unique()

            n_top = len(top_30_repos)
            n_bottom = len(bottom_30_repos)

            if n_top == 0 or n_bottom == 0:
                return

            top_df = df_exploded[df_exploded['repo'].isin(top_30_repos)]
            bottom_df = df_exploded[df_exploded['repo'].isin(bottom_30_repos)]

            top_counts = top_df.groupby("topic_labels")['repo'].nunique()
            bottom_counts = bottom_df.groupby("topic_labels")['repo'].nunique()
            
            x_label = "% of projects"

        plot_data = pd.DataFrame({
            'Top 30% Count': top_counts,
            'Bottom 30% Count': bottom_counts
        }).fillna(0).reset_index()

        # Collect p-values
        topics_list = []
        p_values_list = []
        
        for _, row in plot_data.iterrows():
            topic = row['topic_labels']
            t_c = int(row['Top 30% Count'])
            b_c = int(row['Bottom 30% Count'])

            contingency_table = [
                [t_c, n_top - t_c],
                [b_c, n_bottom - b_c]
            ]

            try:
                chi2, p, dof, ex = chi2_contingency(contingency_table)
            except ValueError:
                p = 1.0
                
            topics_list.append(topic)
            p_values_list.append(p)

        # Apply Holm Correction
        reject, pvals_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='holm')
        significant_topics = set(t for t, r in zip(topics_list, reject) if r)

        plot_data['Top 30%'] = (plot_data['Top 30% Count'] / n_top) * 100
        plot_data['Bottom 30%'] = (plot_data['Bottom 30% Count'] / n_bottom) * 100

        plot_data = plot_data.sort_values(by="Top 30%", ascending=False)

        plot_data_melted = plot_data.melt(
            id_vars="topic_labels", 
            value_vars=['Top 30%', 'Bottom 30%'],
            var_name="Group", 
            value_name="Percentage"
        )

        plt.figure(figsize=(14, 10))
        
        ax = sns.barplot(
            data=plot_data_melted, x="Percentage", y="topic_labels", hue="Group",
            palette=["#2ecc71", "#e74c3c"], edgecolor="black", linewidth=0.5
        )

        # Asterisk styling (Updated to use Holm results)
        x_max = ax.get_xlim()[1]

        for i, topic in enumerate(plot_data['topic_labels']):
            if topic in significant_topics:
                top_val = plot_data.iloc[i]['Top 30%']
                bot_val = plot_data.iloc[i]['Bottom 30%']
                max_val = max(top_val, bot_val)

                ax.text(max_val, i, '*', ha='left', va='center',
                        fontsize=24, fontweight='bold', color='black')

        ax.set_xlim(0, x_max + (x_max * 0.1))

        # INCREASED FONT SIZES:
        plt.xlabel(x_label, fontsize=22, labelpad=14, fontweight='bold')
        plt.ylabel("TopicGPT labels", fontsize=22, labelpad=14, fontweight='bold') 
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)  # Topic tick labels on y-axis
        
        # Legend font sizing
        plt.legend(
            title=f"{title_prefix} Percentiles", 
            title_fontsize='18', 
            fontsize='16',
            loc='lower right', 
            borderaxespad=1.5
        )
        
        plt.tight_layout()

        out_img = FIGS_PATH / f"topic_{metric_col.lower()}_percentile_barplot{suffix}.png"
        out_pdf = FIGS_PATH / f"topic_{metric_col.lower()}_percentile_barplot{suffix}.pdf"
        plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Barplot for {metric_col} saved to: {out_img}")
        plt.close()

    # --- 5A. Project-Level Merge Rate ---
    df_proj = pd.merge(df_metrics, df_topics, on="repo", how="inner")
    merge_col = next((col for col in ['merge_impro_pct'] if col in df_proj.columns), None)

    if merge_col:
        plot_percentile_barplot(
            df_proj, merge_col, "Project Merge Rate", is_pr_metric=False)

    # --- 5B. PR-Level Granular Metrics ---
    project_first_ctx_dates = df_all_versions.groupby('repo')['timestamp_parsed'].min().reset_index(name='first_ctx_time')
    df_prs['created_at'] = pd.to_datetime(df_prs['created_at'], errors='coerce', utc=True)
    df_prs = pd.merge(df_prs, project_first_ctx_dates, on="repo", how="inner")

    initial_count = len(df_prs)
    df_prs_filtered = df_prs[df_prs['created_at'] > df_prs['first_ctx_time']].copy()
    filtered_count = len(df_prs_filtered)

    print(f"Applied PR Date Filter: {initial_count} total PRs -> {filtered_count} PRs created strictly after their project's first context file.")

    pr_metrics_to_plot = ['changes_requested', 'closure_time',
                          'merge_time', 'comments', 'code_churn', 'description_length']

    for attr in pr_metrics_to_plot:
        if attr in df_prs_filtered.columns:
            plot_percentile_barplot(
                df_prs_filtered, attr, f"{attr.replace('_', ' ').capitalize()}", is_pr_metric=True)


# -----------------------------------------------------------
# 6. Global Association Rules (FP-Growth Algorithm)
# -----------------------------------------------------------
def run_global_association_rules(df_topics, one_version=False, ai_filter="all"):
    df_metrics = load_and_filter_metrics()
    if df_metrics.empty:
        return

    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import fpgrowth, association_rules
    except ImportError:
        print("Error: The 'mlxtend' library is not installed. Please run 'pip install mlxtend'.")
        return

    df = pd.merge(df_metrics, df_topics, on="repo", how="inner")

    transactions = df['topic_labels'].apply(
        lambda x: list(set(x)) if isinstance(x, list) else []
    ).tolist()

    transactions = [t for t in transactions if len(t) > 0]

    if not transactions:
        print("No valid transactions found for association rules.")
        return

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(
        df_onehot, min_support=0.04, use_colnames=True)

    if frequent_itemsets.empty:
        print("No frequent itemsets found at the current support threshold.")
        return

    rules = association_rules(frequent_itemsets, metric="confidence",
                              min_threshold=0.03, num_itemsets=len(transactions))

    if rules.empty:
        print("No rules met the confidence threshold.")
        return

    rules = rules[(rules['antecedents'].apply(len) == 1) &
                  (rules['consequents'].apply(len) == 1)].copy()

    rules['topic_a'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['topic_b'] = rules['consequents'].apply(lambda x: list(x)[0])

    df_final_rules = rules[['topic_a', 'topic_b',
                            'support', 'confidence', 'lift']]
    df_final_rules = df_final_rules.sort_values(
        by=['confidence', 'lift'], ascending=[False, False])

    suffix = get_suffix(one_version, ai_filter)
    out_path = DATA_DIR / \
        f"global_topic_association_rules_fpgrowth{suffix}.csv"
    df_final_rules.to_csv(out_path, index=False)
    print(
        f"Global Association rules (FP-Growth) successfully saved to: {out_path}")

    # --- Generate LaTeX Table for Top 20 ---
    top_20 = df_final_rules.head(20).copy()

    def escape_tex(text):
        if not isinstance(text, str):
            text = str(text)
        for char in ['&', '%', '$', '#', '_', '{', '}']:
            text = text.replace(char, '\\' + char)
        return text

    latex_str = "\\begin{table*}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Top 20 Association Rules globally sorted by Confidence}\n"
    latex_str += "\\label{tab:global_assoc_rules}\n"
    latex_str += "\\begin{tabular}{llcc}\n"
    latex_str += "\\toprule\n"
    latex_str += "\\textbf{Antecedent (Topic A)} & \\textbf{Consequent (Topic B)} & \\textbf{Support} & \\textbf{Confidence} \\\\\n"
    latex_str += "\\midrule\n"

    for _, row in top_20.iterrows():
        topic_a = escape_tex(row['topic_a'])
        topic_b = escape_tex(row['topic_b'])
        support = f"{row['support']:.3f}"
        confidence = f"{row['confidence']:.3f}"

        latex_str += f"{topic_a} & {topic_b} & {support} & {confidence} \\\\\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"

    out_path_tex = DATA_DIR / \
        f"global_topic_association_rules_fpgrowth_top20{suffix}.tex"
    with open(out_path_tex, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"Top 20 Association Rules LaTeX table saved to: {out_path_tex}")


def plot_repo_file_stats(df_all_versions, ai_filter="all"):
    """
    Plots boxplots showing:
    1. The number of unique context files per repository.
    2. The distribution of the median number of file versions per repository.
    Also calculates specific project inclusion stats and saves them to a text file.
    """
    # Filter metrics to ensure we only process valid repos
    try:
        df_metrics = load_and_filter_metrics()
        valid_repos = set(df_metrics['repo'].unique())
    except NameError:
        valid_repos = set(df_all_versions['repo'].unique())

    df_versions = df_all_versions.copy()

    if 'file_path' not in df_versions.columns and 'context_file' in df_versions.columns:
        df_versions[['file_path', 'file_timestamp']] = df_versions['context_file'].str.split(
            '&SEP&', n=1, expand=True)

    df_versions = df_versions[df_versions['repo'].isin(valid_repos)]

    # =========================================================================
    # METRIC CALCULATIONS
    # =========================================================================
    files_per_repo = df_versions.groupby(
        'repo')['file_path'].nunique().reset_index(name='num_context_files')

    versions_per_file = df_versions.groupby(
        ['repo', 'file_path']).size().reset_index(name='version_count')
    median_versions_per_repo = versions_per_file.groupby(
        'repo')['version_count'].median().reset_index(name='median_versions')

    # --- NEW: Compute Requested Statistics ---
    total_projects = len(files_per_repo['repo'].unique())

    # 1. Projects with at least two context files
    repos_multiple_files = files_per_repo[files_per_repo['num_context_files'] >= 2]
    count_multiple_files = len(repos_multiple_files)
    pct_multiple_files = (count_multiple_files /
                          total_projects * 100) if total_projects > 0 else 0

    # 2. Projects with at least one context file with more than one version
    files_multiple_versions = versions_per_file[versions_per_file['version_count'] > 1]
    repos_multiple_versions = files_multiple_versions['repo'].unique()
    count_multiple_versions = len(repos_multiple_versions)
    pct_multiple_versions = (count_multiple_versions /
                             total_projects * 100) if total_projects > 0 else 0

    # =========================================================================
    # PLOTTING
    # =========================================================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Number of context files ---
    sns.boxplot(
        y='num_context_files', data=files_per_repo, ax=axes[0],
        color='#3498db', width=0.4, fliersize=0
    )
    sns.stripplot(
        y='num_context_files', data=files_per_repo, ax=axes[0],
        color='#2c3e50', alpha=0.6, jitter=True, size=5
    )
    axes[0].set_title('Number of Context Files per Repository',
                      fontsize=14, weight='bold')
    axes[0].set_ylabel('Number of Unique Files', fontsize=12)
    axes[0].set_xlabel('Repositories', fontsize=12)
    axes[0].set_xticks([])

    # --- Plot 2: Median versions per file ---
    sns.boxplot(
        y='median_versions', data=median_versions_per_repo, ax=axes[1],
        color='#e67e22', width=0.4, fliersize=0
    )
    sns.stripplot(
        y='median_versions', data=median_versions_per_repo, ax=axes[1],
        color='#2c3e50', alpha=0.6, jitter=True, size=5
    )
    axes[1].set_title('Median File Versions per Repository',
                      fontsize=14, weight='bold')
    axes[1].set_ylabel('Median Number of Versions', fontsize=12)
    axes[1].set_xlabel('Repositories', fontsize=12)
    axes[1].set_xticks([])

    plt.tight_layout()

    # =========================================================================
    # SAVING EXPORTS (Plots & Text Stats)
    # =========================================================================
    try:
        suffix = get_suffix(False, ai_filter)

        # Save Text Statistics
        stats_file_path = FIGS_PATH / f"repo_file_stats_summary{suffix}.txt"
        with open(stats_file_path, "w", encoding="utf-8") as f:
            f.write("Repository File Statistics Summary\n")
            f.write("==================================\n\n")
            f.write(f"Total valid projects analyzed: {total_projects}\n\n")
            f.write(f"Projects with >= 2 context files:\n")
            f.write(f"  Count: {count_multiple_files}\n")
            f.write(f"  Percentage: {pct_multiple_files:.2f}%\n\n")
            f.write(f"Projects containing >= 1 context file with > 1 version:\n")
            f.write(f"  Count: {count_multiple_versions}\n")
            f.write(f"  Percentage: {pct_multiple_versions:.2f}%\n")
        print(f"Statistics summary saved to: {stats_file_path}")

        # Save Plots
        out_img = FIGS_PATH / f"repo_file_stats_boxplots{suffix}.png"
        out_pdf = FIGS_PATH / f"repo_file_stats_boxplots{suffix}.pdf"
        plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Repository file stats plots saved to: {out_img}")

    except NameError:
        # Fallback if standard paths/suffix helpers are missing
        plt.show()
        print("\n--- Repository File Statistics Summary ---")
        print(f"Total valid projects: {total_projects}")
        print(
            f"Projects with >= 2 context files: {count_multiple_files} ({pct_multiple_files:.2f}%)")
        print(
            f"Projects with >= 1 file having > 1 version: {count_multiple_versions} ({pct_multiple_versions:.2f}%)")
        print("\n(Note: Variables get_suffix or FIGS_PATH not found, skipped saving files to disk).")

    plt.close()


def enrich_with_ai_tools(df_all_versions, subset_csv_path):
    """
    Merges df_all_versions with the context_files_subset.csv to append
    the 'ai_commit_tools' column based on repository and context file.
    """
    # 1. Load the subset CSV
    try:
        df_subset = pd.read_csv(subset_csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the subset file at {subset_csv_path}")
        return df_all_versions

    # Ensure the required columns exist in the subset
    if not {'repo_name', 'context_file', 'ai_commit_tools'}.issubset(df_subset.columns):
        print("Error: The subset CSV is missing one or more required columns.")
        return df_all_versions

    # 2. Keep only the columns we need to join and append
    df_subset_slim = df_subset[['repo_name',
                                'context_file', 'ai_commit_tools']]

    # 3. Perform the merge
    # We use a 'left' join to keep all original versions, adding the tools where available.
    df_merged = df_all_versions.merge(
        df_subset_slim,
        left_on=['repo', 'context_file'],
        right_on=['repo_name', 'context_file'],
        how='left'
    )

    # 4. Clean up the redundant 'repo_name' column left over from the right side of the join
    df_merged = df_merged.drop(columns=['repo_name'])

    # Optional: Fill missing ai_commit_tools with an empty string or 'Unknown'
    df_merged['ai_commit_tools'] = df_merged['ai_commit_tools'].fillna(
        "Unknown")

    print(
        f"Successfully joined ai_commit_tools. Dataframe shape: {df_merged.shape}")

    return df_merged


def plot_agent_distribution(df: pd.DataFrame):
    """
    Reads the context files subset CSV, extracts agents from the 'ai_commit_tools' column,
    and plots the distribution (number of files) per agent.
    """
    # Check if the required column exists
    if 'ai_commit_tools' not in df.columns:
        print("Error: Column 'ai_commit_tools' not found in the dataset.")
        return

    # 2. Clean and Process the Data
    # Drop rows where 'ai_commit_tools' is NaN
    df_agents = df.dropna(subset=['ai_commit_tools']).copy()

    # Split the comma-separated strings into lists, handling any extra whitespace
    df_agents['agent_list'] = df_agents['ai_commit_tools'].astype(
        str).str.split(',')
    print(df_agents['agent_list'])
    df_agents['agent_list'] = df_agents['agent_list'].apply(
        lambda x: [agent.strip() for agent in x if agent.strip()]
    )

    # Explode the dataframe: this transforms each element of a list-like to a row,
    # duplicating the index values. (e.g., Row 1 with [AgentA, AgentB] becomes two rows)
    df_exploded = df_agents.explode('agent_list')

    # Count the number of files per agent
    agent_counts = df_exploded['agent_list'].value_counts().reset_index()
    agent_counts.columns = ['Agent', 'File Count']

    # 3. Plotting the Distribution
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, max(6, len(agent_counts) * 0.4)))

    ax = sns.barplot(
        data=agent_counts,
        x='File Count',
        y='Agent',
        palette='viridis',
        edgecolor='black'
    )

    # Add numeric labels to the end of each bar for readability
    max_count = agent_counts['File Count'].max()
    for i, count in enumerate(agent_counts['File Count']):
        ax.text(count + (max_count * 0.01), i, str(int(count)),
                va='center', fontsize=10, weight='bold', color='#333')

    plt.title('Distribution of Context Files per AI Agent',
              fontsize=14, weight='bold')
    plt.xlabel('Number of Context Files', fontsize=12)
    plt.ylabel('AI Agent (from ai_commit_tools)', fontsize=12)
    plt.tight_layout()

    # 4. Save and Output
    try:
        # Assuming FIGS_PATH is defined in your broader script
        out_img = FIGS_PATH / "agent_distribution.png"
        out_pdf = FIGS_PATH / "agent_distribution.pdf"
        plt.savefig(out_img, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Agent distribution plot saved to: {out_img}")
    except NameError:
        plt.show()
        print("Plot rendered. (Variable FIGS_PATH not found, skipped saving to disk).")

    plt.close()

    # Optionally print or return the dataframe for inspection
    print("\n--- Agent File Counts ---")
    print(agent_counts.to_string(index=False))
    return agent_counts


def generate_topic_explanations_latex(seed=20):
    """
    Extracts topics and their explanations from the JSONL file,
    counts unique topics per repository using parse_jsonl_file(),
    and saves them to a LaTeX longtable document grouped by themes.
    """
    # 1. Use the built-in function to get the clean DataFrame
    df = parse_jsonl_file()
    
    if df.empty:
        print("No data found from parse_jsonl_file.")
        return

    # Total unique repositories for percentage calculation
    total_repos = df["repo"].nunique()

    # Calculate frequency of unique topics per repository
    df_exploded = df.explode("topic_labels")
    topic_counts = df_exploded.groupby("topic_labels")["repo"].nunique().to_dict()

    # 2. Extract explanation candidates directly from the file
    topic_explanations = defaultdict(list)
    try:
        with open(TOPICS_JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_responses = row.get("responses", "")
                
                # Regex to capture the topic name and its explanation
                matches = re.finditer(
                    r"\[\d+\]\s*(.*?):\s*(.*?)(?=(?:\[\d+\]|$))", raw_responses, re.DOTALL)
                
                for match in matches:
                    topic_name = str(match.group(1)).strip()
                    explanation = str(match.group(2)).strip()
                    
                    # Apply identical unicode cleaning rules to match df['topic_labels']
                    topic_name = unicodedata.normalize('NFKC', topic_name)
                    topic_name = topic_name.replace('\u2010', '-')
                    try:
                        topic_name = topic_name.encode('raw_unicode_escape').decode('unicode_escape')
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                        
                    clean_explanation = " ".join(explanation.split())
                    topic_explanations[topic_name].append(clean_explanation)
    except FileNotFoundError as e:
        print(f"Error loading JSONL for explanations: {e}")
        return

    # 3. Define the Sub-theme to Theme mapping
    THEME_MAPPING = {
        "Build, Test & Development Commands": "Development",
        "Dependency, Toolchain & Version Management": "Development",
        "Configuration & Environment Management": "Development",
        "Generated-Artifact & Cross-Source Synchronization": "Development",
        "Release & Version Publishing": "Development",
        "Testing & Quality Verification": "Testing",
        "Documentation Maintenance, Navigation & Instruction Hierarchy": "Documentation",
        "Documentation Authoring Standards": "Documentation",
        "Code Style, Conventions & Naming": "Conventions",
        "Communication Standards & Language": "Conventions",
        "System Architecture & Module Boundaries": "Architecture",
        "API Design, Schema & Backward Compatibility": "Architecture",
        "Plugin/Extension Architecture & Integrations": "Architecture",
        "Multilayer Network/Graph Domain Semantics & API Selection": "Architecture",
        "Git, Commit & Branch Workflow": "Workflow",
        "Code Search, Discovery & External Research": "Workflow",
        "Change Scope, Repository Policy & Worktree Execution": "Workflow",
        "Pull Request, Issue & Code Review Process": "Workflow",
        "Security, Secrets & Safe Operations": "Security",
        "Database, Migrations & Schema Management": "Data",
        "Reproducibility, Provenance & Source-of-Truth Consistency": "Data",
        "Error Handling & Runtime Robustness": "Observability",
        "Debugging, Troubleshooting & Diagnostics": "Observability",
        "Agent Communication, Clarification & Multi-Agent Coordination": "Agents",
        "Session/Task Tracking & Agent Working Files": "Agents",
        "Logging & Observability": "Observability",
        "Performance, Scaling & Resource Limits": "Operations & Performance",
        "Deployment, Infrastructure & Service Operations": "Operations & Performance",
        "Cross-Platform, Terminal & Feature-Gating Constraints": "Platform",
        "Internationalization & Localization": "Platform",
        "Accessibility": "Platform"
    }

    random.seed(seed)
    
    # Helper to escape characters for LaTeX rendering
    def escape_latex(text):
        replacements = [
            ('&', r'\&'), ('%', r'\%'), ('#', r'\#'),
            ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
            ('<', r'$<$'), ('>', r'$>$'), ('→', r'$\rightarrow$'),
            ('“', '``'), ('”', "''") 
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    # 4. Organize items into themes
    theme_groups = defaultdict(list)
    for topic, count in topic_counts.items():
        if topic not in THEME_MAPPING:
            continue
        
        theme = THEME_MAPPING[topic]
        
        if topic in topic_explanations and topic_explanations[topic]:
            explanation = random.choice(topic_explanations[topic])
        else:
            explanation = "No explanation available."
            
        theme_groups[theme].append({
            'topic': topic,
            'count': count,
            'explanation': explanation
        })

    # Sort within themes by sub-theme frequency
    for theme in theme_groups:
        theme_groups[theme].sort(key=lambda x: x['count'], reverse=True)
        
    # Sort themes by their most frequent sub-theme
    sorted_themes = sorted(theme_groups.keys(), key=lambda t: theme_groups[t][0]['count'], reverse=True)

    # 5. Generate and write the LaTeX document
    out_file = FIGS_PATH / "topic_explanations_table.tex"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("{\\footnotesize\n")
        f.write("\\begin{longtable}{|p{0.14\\linewidth}|p{0.18\\linewidth}|p{0.5\\linewidth}|r|}\n")
        
        # Headers
        f.write("\\hline\n")
        f.write(f"\\textbf{{Theme}} & \\textbf{{Sub-theme}} & \\textbf{{Description}} & \\textbf{{\\%}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\endfirsthead\n\n")
        
        f.write("\\hline\n")
        f.write(f"\\textbf{{Theme}} & \\textbf{{Sub-theme}} & \\textbf{{Description}} & \\textbf{{\\%}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\endhead\n\n")
        
        f.write("\\hline\n")
        f.write("\\endfoot\n\n")
        
        f.write("\\hline\n")
        f.write("\\endlastfoot\n\n")

        # Body rows
        for theme in sorted_themes:
            items = theme_groups[theme]
            n = len(items)
            for i, item in enumerate(items):
                theme_cell = f"\\multirow{{{n}}}{{=}}{{\\centering {theme}}}" if i == 0 else ""
                sub_escaped = escape_latex(item['topic'])
                desc_escaped = escape_latex(item['explanation'])
                
                pct_val = (item['count'] / total_repos) * 100
                pct_str = f"{pct_val:.1f}\\%"
                
                row = f"{theme_cell} & {sub_escaped} & {desc_escaped} & {pct_str} \\\\\n"
                f.write(row)
                
                if i < n - 1:
                    f.write("\\hhline{~||-|-|-|}\n")
            f.write("\\hline\n")

        f.write("\\end{longtable}\n")
        f.write("}\n")

    print(f"Topic explanations LaTeX table saved to: {out_file}")


def export_topics_to_excel(df_versions, filename="generated_topics_export.xlsx"):
    """
    Exports the parsed topics DataFrame to an Excel file with the specified columns:
    Repo, context_file, timestamp, and topic_labels.
    """
    if df_versions.empty:
        print("No data available to export to Excel.")
        return

    # Create a copy to avoid modifying the original dataframe
    df_export = df_versions.copy()

    # Convert the list of topics into a comma-separated string for Excel readability
    df_export['topic_labels'] = df_export['topic_labels'].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    

    df_export['context_file'] = df_export['context_file'].map(
        lambda x: x.split(SEPARATOR)[0])
    df_export['github_link'] = 'https://www.github.com/' + \
        df_export['repo'] + '/blob/main/' + df_export['context_file']

    # Rename 'repo' to 'Repo' to strictly match the requested column names
    df_export = df_export.rename(columns={'repo': 'Repo'})

    # Select only the specific columns requested
    cols_to_export = ['github_link', 'timestamp', 'topic_labels']
    df_final = df_export[cols_to_export]

    # Save the Excel file in the DATA_DIR directory
    out_path = DATA_DIR / filename

    try:
        # Requires 'openpyxl' to be installed
        df_final.to_excel(out_path, index=False, engine='openpyxl')
        print(f"Data successfully exported to Excel: {out_path}")
    except ModuleNotFoundError:
        print("Error: The 'openpyxl' library is required to export to Excel.")
        print("Please install it by running: pip install openpyxl")


def build_system_prompt(row):
    commit_message = row.get('commit_message')
    commit_message = commit_message if pd.notna(commit_message) else "N/A"
    file_patch = row.get('file_patch')
    file_patch = file_patch if pd.notna(file_patch) else "N/A"
    topics = [f"<TOPIC>{t.strip()}<TOPIC>"for t in row.get('topic_labels').split('&SEP&')]
    prompt = SYSTEM_PROMPT.format(
        commit_message=commit_message, file_patch=file_patch, topics="\n".join(topics))

    return prompt


def sample_statistically_significant_acf_versions_to_csv(df_all_versions, random_seed=98):
    """
    Selects a random sample of ACF versions (excluding the very first version, 
    and from ACFs with min 2 versions total) using a 95% confidence level and 
    a 5% margin of error. Employs stratified sampling to ensure maximum project 
    (repo) representation across all versions. 
    Only includes commits that ONLY modified context files.
    """
    print("\n--- Sampling ACF Versions (95% Confidence, 5% Margin of Error) ---")

    df_versions = df_all_versions.copy()
    df_versions = df_versions.rename(columns={"timestamp": "git_committer_date"})

    # Pass the exact format to stop Pandas from guessing timezone offsets
    df_versions['git_committer_date'] = pd.to_datetime(
        df_versions['git_committer_date'],
        format="%Y-%m-%d %H-%M-%S",
        errors="coerce"
    )

    # Ensure file_path is isolated from the timestamp separator
    if 'file_path' not in df_versions.columns:
        df_versions['context_file'] = df_versions['context_file'].apply(
            lambda x: x.split(SEPARATOR)[0] if SEPARATOR in str(x) else x
        )

    # -------------------------------------------------------------------------
    # NEW: Move metadata loading and filtering BEFORE sampling to protect stats
    # -------------------------------------------------------------------------
    try:
        # 1. Load context commits
        df_commits = pd.read_csv(COMMITS_METADATA_FILE)
        df_commits = df_commits.rename(columns={"repo_name": "repo"})

        if 'git_committer_date' in df_commits.columns:
            parsed_dates = pd.to_datetime(df_commits['git_committer_date'], utc=True, errors='coerce')
            df_commits['git_committer_date'] = parsed_dates.dt.tz_localize(None)

        df_commits = df_commits[['repo', 'context_file', 'commit_hash', 'git_committer_date', 'commit_message', 'file_patch']]
        
        # 2. Load all commit files
        all_commit_files_df = pd.read_csv(ALL_COMMIT_FILES_FILE)

        # 3. Filter for commits that ONLY contain context files
        if 'commit_hash' in all_commit_files_df.columns:
            # Count total files per commit
            total_files_counts = all_commit_files_df.groupby('commit_hash').size().reset_index(name='total_files')
            # Count context files per commit
            context_files_counts = df_commits.groupby('commit_hash').size().reset_index(name='context_files')
            
            # Merge counts to compare
            commit_comparison = pd.merge(total_files_counts, context_files_counts, on='commit_hash', how='inner')
            
            # Identify valid commits where total files modified exactly matches context files modified
            valid_commit_hashes = commit_comparison[
                commit_comparison['total_files'] == commit_comparison['context_files']
            ]['commit_hash']
            
            # Filter df_commits to only keep these pure context-file commits
            df_commits = df_commits[df_commits['commit_hash'].isin(valid_commit_hashes)]
        
        # 4. Inner merge to ensure our population ONLY consists of these valid commits
        df_versions = df_versions.merge(
            df_commits,
            on=['repo', 'context_file', 'git_committer_date'],
            how='inner'
        )

    except FileNotFoundError as e:
        print(f"Warning: Metadata file missing ({e}). Commits will default to 'main' and filtering is skipped.")
        df_versions['commit_hash'] = pd.NA
        df_versions['commit_message'] = "None"
        df_versions['file_patch'] = "None"

    # -------------------------------------------------------------------------

    # 1. Group by repo and context_file to count total versions per ACF
    acf_counts = df_versions.groupby(
        ['repo', 'context_file']).size().reset_index(name='version_count')

    # 2. Filter for ACFs that have at least two versions
    multi_version_acfs = acf_counts[acf_counts['version_count'] >= 2]

    # Create our population by grabbing ALL versions belonging to these multi-version ACFs
    valid_versions = df_versions.merge(
        multi_version_acfs,
        on=['repo', 'context_file'],
        how='inner'
    )

    # -------------------------------------------------------------------------
    # Exclude the first version chronologically for each ACF
    # -------------------------------------------------------------------------
    # Sort by date so cumcount() reliably flags the oldest version as index 0
    valid_versions = valid_versions.sort_values(
        by=['repo', 'context_file', 'git_committer_date'], 
        ascending=[True, True, True]
    )
    
    # cumcount() yields 0 for the first/oldest version, 1 for the second, etc.
    valid_versions['version_chronological_index'] = valid_versions.groupby(
        ['repo', 'context_file']
    ).cumcount()
    
    # Keep only versions where index > 0 (excludes the first version)
    valid_versions = valid_versions[valid_versions['version_chronological_index'] > 0]
    # -------------------------------------------------------------------------

    # The population N is now the total number of *eligible* versions (minus the first versions)
    N = len(valid_versions)

    if N == 0:
        print("No versions found belonging to ACFs after excluding the first versions and applying file constraints.")
        return

    # 3. Calculate Statistical Sample Size
    Z = 1.96  # Z-score for 95% confidence
    p = 0.5   # Proportion representing maximum variability
    E = 0.05  # 5% Margin of Error

    # Cochran's formula & Finite population correction
    n_0 = (Z**2 * p * (1 - p)) / (E**2)
    target_sample_size = math.ceil(n_0 / (1 + (n_0 - 1) / N))
    target_sample_size = min(target_sample_size, N)

    print(f"Population size (N): {N} valid ACF versions")
    print(f"Calculated target sample size (n): {target_sample_size} versions")

    # 4. Perform the Random Sample (STRATIFIED TO ENSURE PROJECT COVERAGE)
    unique_repos = valid_versions['repo'].unique()

    if target_sample_size >= len(unique_repos):
        # Guarantee at least 1 version from EVERY project
        sample_base = valid_versions.groupby(
            'repo').sample(n=1, random_state=random_seed)

        # Fill the rest of the sample size randomly from the remaining versions
        remaining_versions = valid_versions.drop(sample_base.index)
        remaining_sample_size = target_sample_size - len(sample_base)

        if remaining_sample_size > 0:
            sample_remainder = remaining_versions.sample(
                n=remaining_sample_size, random_state=random_seed)
            sampled_versions = pd.concat([sample_base, sample_remainder])
        else:
            sampled_versions = sample_base
    else:
        # If target size is smaller than repo count, sample repos first, then grab 1 version per sampled repo
        sampled_repos = pd.Series(unique_repos).sample(
            n=target_sample_size, random_state=random_seed)
        sampled_versions = valid_versions[valid_versions['repo'].isin(
            sampled_repos)].groupby('repo').sample(n=1, random_state=random_seed)

    print(
        f"Selected {len(sampled_versions)} versions distributed across {sampled_versions['repo'].nunique()} projects.")

    # 5. Extract metadata (Metadata is already merged from Step 1, so we just clean up)
    selected_metadata = sampled_versions.copy()

    # Format lists of topic labels
    if 'topic_labels' in selected_metadata.columns:
        selected_metadata['topic_labels'] = selected_metadata['topic_labels'].apply(
            lambda x: "&SEP&".join(x) if isinstance(x, list) else str(x)
        )

    # Construct GitHub URLs
    def build_github_link(row):
        found_commit = row.get('commit_hash')
        branch_or_commit = found_commit if pd.notna(found_commit) and isinstance(
            found_commit, str) and found_commit.strip() else 'main'
        return f"https://github.com/{row['repo']}/commit/{branch_or_commit}"

    selected_metadata['github_link'] = selected_metadata.apply(build_github_link, axis=1)
    selected_metadata['description'] = "None"
    selected_metadata['label'] = "None"

    if 'git_committer_date' not in selected_metadata.columns:
        selected_metadata['git_committer_date'] = pd.NaT

    # Keep columns that actually exist to avoid KeyError
    cols_to_export = [
        'repo', 'context_file', 'version_count', 'git_committer_date', 'github_link',
        'topic_labels', 'commit_message', 'file_patch', 'description', 'label'
    ]
    cols_to_export = [col for col in cols_to_export if col in selected_metadata.columns]

    final_df = selected_metadata[cols_to_export]

    # 6. Sort the final dataset for readability (Chose descending for output, but pool filtering used ascending)
    final_df = final_df.sort_values(
        by=['repo', 'context_file', 'git_committer_date'], ascending=[True, True, False])

    final_df['system_prompt'] = final_df.apply(build_system_prompt, axis=1)

    # Remove file_patch to avoid clutter in the Excel output
    final_df = final_df.rename(columns={'file_patch': 'diff'})

    # 7. Save to Excel
    out_path = DATA_DIR / "statistically_representative_acf_sample.xlsx"
    final_df.to_excel(out_path, index=False)
    print(f"Saved statistically representative ACF version metadata to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Global Topic Distributions and Statistics.")
    parser.add_argument("--one-version", action="store_true",
                        help="Consider only the first version of the context file.")
    parser.add_argument("--ai-filter", type=str, choices=["all", "ai", "human"], default="all",
                        help="Filter context files: 'all' (default), 'ai' (AI-created), 'human' (not AI-created)")
    args = parser.parse_args()

    print("Fetching and parsing raw data...")
    df_all_versions = parse_jsonl_file()
    logger.info(f"df_all_versions['topic_labels'] sample: {df_all_versions['topic_labels'].head()}")

    print("Exporting topics to Excel...")
    export_topics_to_excel(
        df_all_versions, filename="all_generated_topics.xlsx")

    sample_statistically_significant_acf_versions_to_csv(df_all_versions)

    print(f"Applying AI filter: {args.ai_filter}...")
    df_all_versions = filter_by_ai(df_all_versions, args.ai_filter)

    print("Generating Global Topic Churn...")
    plot_global_topic_churn(df_all_versions, ai_filter=args.ai_filter)

    # print("Generating Global Repo ACF stats...")
    # plot_repo_file_stats(df_all_versions, ai_filter=args.ai_filter)

    # df_all_versions_enriched = enrich_with_ai_tools(df_all_versions, CONTEXT_FILES_SUBSET_FILE)

    # plot_agent_distribution(df_all_versions_enriched)

    if args.one_version:
        df_static = filter_one_version(df_all_versions)
    else:
        df_sorted = df_all_versions.sort_values(
            by=["repo", "context_file", "timestamp_parsed"])
        df_static = df_sorted.drop_duplicates(
            subset=["repo", "context_file"], keep="last")

    print("Deduplicating topics per project...")
    df_static_dedup = deduplicate_topics_per_project(df_static)

    # df_last_topic = keep_last_topics_per_project(df_sorted)

    # print("Generating Global Topic Distribution Plot...")
    # plot_global_topic_distribution(
    #     df_last_topic, one_version=args.one_version, ai_filter=args.ai_filter)

    # print("Generating Topic Explanations Dictionary...")
    # generate_topic_explanations_latex()

    # print("Generating Global Topic Co-occurrence Matrix...")
    # plot_global_topic_cooccurrence(
    #     df_static_dedup, one_version=args.one_version, ai_filter=args.ai_filter)

    # # print("Generating Global Project Attribute Boxplots...")
    # plot_global_topic_attributes(
    #     df_static_dedup, one_version=args.one_version, ai_filter=args.ai_filter)

    # print("Generating PR-level Metrics Boxplots...")
    plot_topic_pr_metrics_boxplots(
        df_static_dedup, df_all_versions, one_version=args.one_version, ai_filter=args.ai_filter)

    # print("Generating Global Association Rules via FP-Growth...")
    # run_global_association_rules(
    #     df_static_dedup, one_version=args.one_version, ai_filter=args.ai_filter)

    print("All global analyses completed successfully!")
