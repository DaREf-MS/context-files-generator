import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from helper import mannUandCliffdelta

from itertools import combinations
from collections import Counter

from helper import mannUandCliffdelta

# -----------------------------------------------------------
# Statistical Testing Between the Two Heatmaps
# -----------------------------------------------------------
from scipy.stats import wilcoxon, ks_2samp, spearmanr

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
FIGS_PATH = ROOT_DIR / "figs"
DATA_DIR = ROOT_DIR / "data"

METRICS_FILE = "repo_agent_metrics_comparison_mode_1.csv"
TOPICS_JSONL_FILE = DATA_DIR / "assignment_corrected.jsonl"
CONFIGURED_REPOS_FILE = DATA_DIR / "repos_with_deps_count.csv"
COMMITS_METADATA_FILE = DATA_DIR / "context_files_commits_metadata.csv"

COUNT_PR_THRESHOLD = 5
MERGE_RATE_THRESHOLD = 20

FIGS_PATH.mkdir(parents=True, exist_ok=True)


def get_significance_marker(data1, data2):
    """
    Computes the Mann-Whitney U test and returns a significance marker
    based on the p-value and the magnitude of the effect size.
    Effect size used is the rank-biserial correlation magnitude: r = |U1 - U2| / (n1 * n2)
    Thresholds: Negligible (<0.1), Small (>=0.1), Medium (>=0.3), Large (>=0.5)
    """
    if len(data1) < 2 or len(data2) < 2:
        return None

    try:
        _, p, _, size = mannUandCliffdelta(data1, data2)
    except ValueError:
        return None  # Fails if all values are identical across both arrays

    if p < 0.05:
        if size == 'large':
            return "****"
        elif size == 'medium':
            return "***"
        elif size == 'small':
            return "**"
        elif size == 'negligible':
            return "*"

    return None


def parse_jsonl_file():
    """Reads the JSONL file and returns all versions across all files."""
    data = []

    with open(TOPICS_JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            raw_id = row.get("id", "")
            raw_responses = row.get("responses", "")
            timestamp = row.get("timestamp", None)

            if "&SEP&" in raw_id:
                repo_part, file_part = raw_id.split("&SEP&", 1)
                repo_name = repo_part.replace("§", "/")
                context_file = file_part.replace("§", "/")
            else:
                repo_name = raw_id.replace("§", "/")
                context_file = "N/A"

            topics = re.findall(r"\[\d+\]\s*(.*?):", raw_responses)

            cleaned_topics = []
            for t in topics:
                t = str(t).strip()
                if not t:
                    continue
                cleaned_topics.append(t)

            data.append({
                "repo": repo_name,
                "context_file": context_file,
                "topic_labels": cleaned_topics,
                "timestamp": timestamp,
            })

    df = pd.DataFrame(data)

    # Parse timestamp safely
    if not df.empty:
        df["timestamp_parsed"] = pd.to_datetime(
            df["timestamp"], errors="coerce")

    return df


def filter_one_version(df):
    """Filters dataset to only the first occurrence of each file."""
    if df.empty:
        return df

    df_sorted = df.sort_values(
        by=["repo", "context_file", "timestamp_parsed"], ascending=True)
    df_filtered = df_sorted.drop_duplicates(
        subset=["repo", "context_file"], keep="first")
    return df_filtered


def deduplicate_topics_per_project(df):
    """Groups by project and flattens to ensure one topic occurrence per project."""
    if df.empty:
        return df

    df_grouped = df.groupby("repo", as_index=False).agg({
        "context_file": lambda x: ", ".join(x.astype(str).unique()),
        "topic_labels": lambda x: list(set(topic for sublist in x for topic in sublist)),
        "timestamp": "first"
    })
    return df_grouped


def plot_topic_evolution(df_all_versions: pd.DataFrame):
    """Calculates and plots the Added/Removed/Unchanged evolution of topics across versions."""
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 1. Split context_file using '&SEP&' into file_path and file_timestamp
    df_all_versions[['file_path', 'file_timestamp']
                    ] = df_all_versions['context_file'].str.split('&SEP&', n=1, expand=True)

    # 2. Sort sequentially to track changes across file versions.
    df_sorted = df_all_versions.sort_values(
        by=["repo", "file_path", "timestamp_parsed"])

    records = []

    # 3. Compare state of topics version-by-version
    for (repo, _), group in df_sorted.groupby(['repo', 'file_path']):

        is_first_version = True
        prev_topics = set()

        for _, row in group.iterrows():
            curr_topics = set(row['topic_labels']) if isinstance(
                row['topic_labels'], list) else set()

            # Skip comparison for the very first version, just set it as the baseline
            if is_first_version:
                prev_topics = curr_topics.copy()
                is_first_version = False
                continue

            added = curr_topics.difference(prev_topics)
            removed = prev_topics.difference(curr_topics)
            unchanged = curr_topics.intersection(prev_topics)

            for t in added:
                records.append({'repo': repo, 'topic': t, 'status': 'Added'})
            for t in removed:
                records.append({'repo': repo, 'topic': t, 'status': 'Removed'})
            for t in unchanged:
                records.append(
                    {'repo': repo, 'topic': t, 'status': 'Unchanged'})

            prev_topics = curr_topics.copy()

    df_transitions = pd.DataFrame(records)

    # Merge and categorize
    df_merged = pd.merge(df_transitions, df_metrics, on="repo", how="inner")

    increase_label = f"Merge-rate increase (≥{MERGE_RATE_THRESHOLD}%)"
    decrease_label = f"Merge-rate decrease (≤-{MERGE_RATE_THRESHOLD}%)"

    df_merged["Impact Category"] = pd.NA
    df_merged.loc[df_merged["merge_impro_pct"] >=
                  MERGE_RATE_THRESHOLD, "Impact Category"] = increase_label
    df_merged.loc[df_merged["merge_impro_pct"] <= -
                  MERGE_RATE_THRESHOLD, "Impact Category"] = decrease_label

    analysis_df = df_merged.dropna(subset=["Impact Category"]).copy()

    # Aggregate to compute volume of statuses
    agg_df = analysis_df.groupby(
        ['Impact Category', 'topic', 'status']).size().reset_index(name='count')
    pivot_df = agg_df.pivot_table(
        index=['Impact Category', 'topic'], columns='status', values='count', fill_value=0).reset_index()

    for col in ['Added', 'Unchanged', 'Removed']:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    pivot_df['total'] = pivot_df['Added'] + \
        pivot_df['Unchanged'] + pivot_df['Removed']

    # Total cross-category sorting map for consistent Y-axis
    topic_totals = pivot_df.groupby(
        'topic')['total'].sum().sort_values(ascending=True)

    # -----------------------------------------------------------
    # Plot Configuration
    # -----------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 12), sharey=True)

    # Distinct colors avoiding collision with general Increase/Decrease Greens/Reds
    status_colors = ['#3498db', '#bdc3c7', '#e67e22']  # Blue, Gray, Orange

    for i, category in enumerate([increase_label, decrease_label]):
        ax = axes[i]

        cat_df = pivot_df[pivot_df['Impact Category']
                          == category].set_index('topic')
        cat_df = cat_df.reindex(topic_totals.index).fillna(0)

        if not cat_df.empty:
            cat_df[['Added', 'Unchanged', 'Removed']].plot(
                kind='barh',
                stacked=True,
                ax=ax,
                color=status_colors,
                edgecolor='black',
                linewidth=0.5
            )

            # --- DATA LABELING LOGIC ---
            # Evaluate bounds to know when text fits vs needs to sit on the right
            max_x = cat_df[['Added', 'Unchanged', 'Removed']].sum(axis=1).max()
            # 5% of max width acts as the text collision threshold
            space_threshold = max_x * 0.04
            has_right_labels = False

            for row_idx, (topic_name, row) in enumerate(cat_df.iterrows()):
                total_val = row[['Added', 'Unchanged', 'Removed']].sum()
                if total_val == 0:
                    continue

                small_vals = []

                for col_idx, col_name in enumerate(['Added', 'Unchanged', 'Removed']):
                    val = row[col_name]
                    if val == 0:
                        continue

                    # Calculate segment's start X
                    start_x = sum(
                        row[['Added', 'Unchanged', 'Removed']][:col_idx])

                    if val >= space_threshold:
                        # Sufficient Space: Plot centered inside the segment
                        # Set text colors based on background contrast
                        text_color = 'black' if col_name == 'Unchanged' else 'white'
                        ax.text(start_x + val / 2, row_idx, str(int(val)),
                                va='center', ha='center', fontsize=10, weight='bold', color=text_color)
                    else:
                        # Insufficient Space: Prepare it for the right-side box
                        small_vals.append(
                            f"{col_name[0]}: {int(val)}")  # Ex: "A: 1"

                # If there were small segment sizes, plot them right of the bar total
                if small_vals:
                    has_right_labels = True
                    # Position slightly right of the total width
                    ax.text(total_val + (max_x * 0.015), row_idx, " (" + ", ".join(small_vals) + ")",
                            va='center', ha='left', fontsize=9, color='#333333', style='italic', weight='bold')

            # Expand X bounds dynamically to stop right-side annotations from getting clipped
            if has_right_labels:
                ax.set_xlim(right=max_x * 1.20)
            # ---------------------------

        ax.set_title(f"{category}", fontsize=15, weight='bold', pad=15)
        ax.set_xlabel("Occurrences Over File Versions", fontsize=12)
        if i == 0:
            ax.set_ylabel("TopicGPT Labels", fontsize=12)
        else:
            ax.set_ylabel("")

        ax.legend(title="Topic Evolution", loc="lower right")

    plt.suptitle("Topic Modification Dynamics (Added / Unchanged / Removed) Across Context File Versions",
                 fontsize=18, weight='bold', y=1.02)
    plt.tight_layout()

    output_file_pdf = str(FIGS_PATH / "topic_evolution_over_versions.pdf")
    output_file_img = str(FIGS_PATH / "topic_evolution_over_versions.png")

    plt.savefig(output_file_pdf, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.savefig(output_file_img, bbox_inches="tight", pad_inches=0.01, dpi=300)
    print(f"Evolution plot saved to: {output_file_img}")
    plt.close()


def plot_latest_topic_cooccurrence(df_all_versions: pd.DataFrame):
    """Calculates and plots topic co-occurrence matrices, filtering out topics with 0 co-occurrences."""
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 1. Split context_file using '&SEP&' into file_path and file_timestamp
    df_all_versions[['file_path', 'file_timestamp']
                    ] = df_all_versions['context_file'].str.split('&SEP&', n=1, expand=True)

    # 2. Sort by time and isolate the final version of each file
    df_sorted = df_all_versions.sort_values(
        by=["repo", "file_path", "timestamp_parsed"])
    df_latest = df_sorted.drop_duplicates(
        subset=["repo", "file_path"], keep="last").copy()

    print(
        f"Data reduced: {len(df_all_versions)} total versions -> {len(df_latest)} final file versions.")

    # Extract the definitive unique list of ALL topics in the latest versions
    valid_topics = df_latest['topic_labels'].apply(
        lambda x: x if isinstance(x, list) else [])
    master_topics = sorted(list(set(valid_topics.explode().dropna())))

    if not master_topics:
        print("No topics found in the latest file versions.")
        return

    # 3. Merge to get impact categories
    df_merged = pd.merge(df_latest, df_metrics, on="repo", how="inner")

    increase_label = f"Merge-rate increase (≥{MERGE_RATE_THRESHOLD}%)"
    decrease_label = f"Merge-rate decrease (≤-{MERGE_RATE_THRESHOLD}%)"

    df_merged["Impact Category"] = pd.NA
    df_merged.loc[df_merged["merge_impro_pct"] >=
                  MERGE_RATE_THRESHOLD, "Impact Category"] = increase_label
    df_merged.loc[df_merged["merge_impro_pct"] <= -
                  MERGE_RATE_THRESHOLD, "Impact Category"] = decrease_label

    analysis_df = df_merged.dropna(subset=["Impact Category"]).copy()

    # Helper function to count co-occurrences
    def build_cooccurrence_counts(df_subset):
        pair_counts = Counter()
        for topics in df_subset['topic_labels']:
            if isinstance(topics, list) and len(topics) > 1:
                topics_sorted = sorted(topics)
                for pair in combinations(topics_sorted, 2):
                    pair_counts[pair] += 1
        return pair_counts

    # Extract pairs for both categories
    inc_df = analysis_df[analysis_df['Impact Category'] == increase_label]
    dec_df = analysis_df[analysis_df['Impact Category'] == decrease_label]

    inc_counts = build_cooccurrence_counts(inc_df)
    dec_counts = build_cooccurrence_counts(dec_df)

    # Initialize empty dataframes (Matrix: Topic x Topic) using the MASTER list
    inc_matrix = pd.DataFrame(0, index=master_topics, columns=master_topics)
    dec_matrix = pd.DataFrame(0, index=master_topics, columns=master_topics)

    # Populate matrices with actual counts
    for (t1, t2), count in inc_counts.items():
        inc_matrix.at[t1, t2] = count
        inc_matrix.at[t2, t1] = count  # Symmetrical

    for (t1, t2), count in dec_counts.items():
        dec_matrix.at[t1, t2] = count
        dec_matrix.at[t2, t1] = count  # Symmetrical

    # --- FILTERING LOGIC ---
    # Sum the rows across both matrices to find topics that never co-occur with anything
    total_cooccurrences = inc_matrix.sum(axis=1) + dec_matrix.sum(axis=1)

    # Keep only topics that have a sum > 0
    active_topics = total_cooccurrences[total_cooccurrences > 0].index.tolist()

    if not active_topics:
        print("No co-occurring topics found to plot.")
        return

    print(
        f"Filtered axes: {len(master_topics)} total topics -> {len(active_topics)} topics with ≥1 co-occurrence.")

    # Extract the upper triangular part of the matrices (excluding diagonal)
    # to avoid duplicating symmetric pairs and self-co-occurrences
    upper_tri_indices = np.triu_indices_from(inc_matrix.values, k=1)
    inc_vals = inc_matrix.values[upper_tri_indices]
    dec_vals = dec_matrix.values[upper_tri_indices]

    # 1. Wilcoxon Signed-Rank Test (paired comparison of magnitudes)
    try:
        w_stat, w_pval = wilcoxon(inc_vals, dec_vals, zero_method='zsplit')
    except ValueError:
        w_stat, w_pval = float('nan'), float('nan')

    # 2. Kolmogorov-Smirnov Test (overall distribution shape difference)
    ks_stat, ks_pval = ks_2samp(inc_vals, dec_vals)

    # 3. Spearman Rank Correlation (structural pattern similarity)
    corr, corr_pval = spearmanr(inc_vals, dec_vals)

    # Write results to text file
    stats_out_path = DATA_DIR / "heatmap_stats_comparison.txt"
    with open(stats_out_path, "w", encoding="utf-8") as f:
        f.write("Statistical Comparison Between Increase and Decrease Heatmaps\n")
        f.write("="*65 + "\n\n")
        f.write(
            f"Number of distinct topic pairs analyzed: {len(inc_vals)}\n\n")

        f.write("1. Wilcoxon Signed-Rank Test (Global paired shift in magnitude)\n")
        f.write(
            "   Tests if one heatmap consistently has higher co-occurrences than the other.\n")
        f.write(f"   Statistic: {w_stat:.4f}\n")
        f.write(f"   P-value:   {w_pval:.4e}\n\n")

        f.write("2. Kolmogorov-Smirnov 2-Sample Test (Distribution shape difference)\n")
        f.write(
            "   Tests if the distribution of count values differs between conditions.\n")
        f.write(f"   Statistic: {ks_stat:.4f}\n")
        f.write(f"   P-value:   {ks_pval:.4e}\n\n")

        f.write("3. Spearman Rank Correlation (Pattern similarity)\n")
        f.write(
            "   Tests if the structure of topic clusters (high vs low values) is maintained.\n")
        f.write(f"   Correlation: {corr:.4f}\n")
        f.write(f"   P-value:     {corr_pval:.4e}\n")

    print(f"Heatmap statistical comparison saved to: {stats_out_path}")

    # Slice the matrices to remove the 0-occurrence rows and columns
    inc_matrix = inc_matrix.loc[active_topics, active_topics]
    dec_matrix = dec_matrix.loc[active_topics, active_topics]

    # Global max to share the color map scale (vmin, vmax) across both subplots
    global_max = max(inc_matrix.max().max(), dec_matrix.max().max())
    if global_max == 0:
        global_max = 1

    # -----------------------------------------------------------
    # Plot Configuration
    # -----------------------------------------------------------
    sns.set_theme(style="white")

    # --- WIDENED FIGURE SIZE ---
    # Increased base width from 20 to 24, and multiplier from 0.6 to 0.85
    # This guarantees enough horizontal space for labels to not overlap
    fig_width = max(24, len(active_topics) * 0.85)
    fig_height = max(10, len(active_topics) * 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(
        fig_width, fig_height), sharey=True)

    # Create a mask for the upper triangle (and diagonal) to keep it clean
    mask = np.triu(np.ones_like(inc_matrix, dtype=bool))

    # Color palettes
    cmap_inc = "Blues"
    cmap_dec = "Reds"

    for i, (matrix, category, cmap) in enumerate(zip(
        [inc_matrix, dec_matrix],
        [increase_label, decrease_label],
        [cmap_inc, cmap_dec]
    )):
        ax = axes[i]

        sns.heatmap(
            matrix,
            mask=mask,
            cmap=cmap,
            vmax=global_max,
            vmin=0,
            annot=True,
            fmt="d",
            linewidths=.5,
            cbar_kws={"shrink": .8, "label": "Co-occurrence Frequency"},
            ax=ax
        )

        ax.set_title(f"{category}", fontsize=16, weight='bold', pad=15)
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Adjust tick parameters for better visibility
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

        # Align x-labels properly to anchor them exactly at the tick mark
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    plt.suptitle("Topic Co-occurrence Matrix (Latest File Versions Only)",
                 fontsize=20, weight='bold', y=1.02)

    plt.tight_layout()

    output_file_pdf = str(FIGS_PATH / "latest_topic_cooccurrence_heatmap.pdf")
    output_file_img = str(FIGS_PATH / "latest_topic_cooccurrence_heatmap.png")

    plt.savefig(output_file_pdf, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.savefig(output_file_img, bbox_inches="tight", pad_inches=0.1, dpi=300)

    print(f"Latest version co-occurrence heatmap saved to: {output_file_img}")
    plt.close()


def plot_topic_distribution(df_topics, one_version=False):
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    df = pd.merge(df_metrics, df_topics, on="repo", how="inner")

    increase_label = f"Merge-rate increase (≥{MERGE_RATE_THRESHOLD}%)"
    decrease_label = f"Merge-rate decrease (≤-{MERGE_RATE_THRESHOLD}%)"

    df["Impact Category"] = pd.NA
    df.loc[df["merge_impro_pct"] >= MERGE_RATE_THRESHOLD,
           "Impact Category"] = increase_label
    df.loc[df["merge_impro_pct"] <= -MERGE_RATE_THRESHOLD,
           "Impact Category"] = decrease_label

    analysis_df = df.dropna(subset=["Impact Category"]).copy()

    exploded_df = analysis_df.explode("topic_labels").copy()
    exploded_df["topic_labels"] = exploded_df["topic_labels"].astype(
        str).str.strip()

    exploded_df = exploded_df[
        exploded_df["topic_labels"].notna()
        & (exploded_df["topic_labels"] != "")
        & (exploded_df["topic_labels"] != "nan")
    ].copy()

    exploded_df = exploded_df.drop_duplicates(
        subset=["repo", "Impact Category", "topic_labels"])
    category_totals = analysis_df.groupby("Impact Category")[
        "repo"].nunique().to_dict()

    topic_counts = (
        exploded_df.groupby(["topic_labels", "Impact Category"])["repo"]
        .nunique()
        .reset_index(name="Unique_Repo_Count")
    )

    topic_counts["Percentage"] = topic_counts.apply(
        lambda row: (row["Unique_Repo_Count"] / category_totals[row["Impact Category"]]) * 100, axis=1
    )

    topic_order = topic_counts.groupby("topic_labels")[
        "Percentage"].max().sort_values(ascending=False).index

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 11))

    ax = sns.barplot(
        data=topic_counts,
        y="topic_labels",
        x="Percentage",
        hue="Impact Category",
        order=topic_order,
        palette={increase_label: "#2ecc71", decrease_label: "#e74c3c"},
        edgecolor="black",
        linewidth=0.5,
    )

    for container in ax.containers:
        labels = []
        for bar in container:
            width = bar.get_width()
            labels.append(f"{width:.1f}%" if width > 0 else "")
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)

    plt.title("Topic Prevalence by Merge-rate Impact Category\n(Percentages are relative to each category)",
              fontsize=15, weight="bold", pad=20)
    plt.xlabel("Percentage of Repositories (%)", fontsize=12)
    plt.ylabel("TopicGPT Labels", fontsize=12)

    max_pct = topic_counts["Percentage"].max(
    ) if not topic_counts.empty else 100
    plt.xlim(0, max_pct + 10)

    plt.legend(title="Impact Category", loc="lower right")
    plt.tight_layout()

    output_file_pdf = str(FIGS_PATH / "context_file_topics_per_merge_rate.pdf")
    output_file_img = str(FIGS_PATH / "context_file_topics_per_merge_rate.png")

    if one_version:
        output_file_pdf = output_file_pdf.replace(".pdf", "_one_version.pdf")
        output_file_img = output_file_img.replace(".png", "_one_version.png")

    plt.savefig(output_file_pdf, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.savefig(output_file_img, bbox_inches="tight", pad_inches=0.01, dpi=300)
    print(f"Plot saved to: {output_file_img}")
    plt.close()


def plot_repo_metrics_barplots(df_topics):
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)
        df_repos = pd.read_csv(CONFIGURED_REPOS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    df = pd.merge(df_metrics, df_topics, on="repo", how="inner")

    increase_label = f"Merge-rate increase (≥{MERGE_RATE_THRESHOLD}%)"
    decrease_label = f"Merge-rate decrease (≤-{MERGE_RATE_THRESHOLD}%)"

    df["Impact Category"] = pd.NA
    df.loc[df["merge_impro_pct"] >= MERGE_RATE_THRESHOLD,
           "Impact Category"] = increase_label
    df.loc[df["merge_impro_pct"] <= -MERGE_RATE_THRESHOLD,
           "Impact Category"] = decrease_label

    analysis_df = df.dropna(subset=["Impact Category"]).copy()
    analysis_df = pd.merge(analysis_df, df_repos,
                           left_on="repo", right_on="repo_name", how="inner")

    df_unique_repos = analysis_df.drop_duplicates(subset=["repo"]).copy()

    metrics_config = [
        {"column": "contributors", "label": "Number of Contributors",
            "file_suffix": "contributors"},
        {"column": "stargazers", "label": "Number of Stars", "file_suffix": "stars"},
        {"column": "size",
            "label": "Repository Size (KB)", "file_suffix": "size"}
    ]

    palette = {increase_label: "#2ecc71", decrease_label: "#e74c3c"}
    sns.set_theme(style="whitegrid")

    for metric in metrics_config:
        col = metric["column"]
        label = metric["label"]
        suffix = metric["file_suffix"]

        if col not in df_unique_repos.columns:
            continue

        plt.figure(figsize=(11, 7))
        log_scale = col in ["stargazers", "size"]

        if log_scale:
            df_unique_repos[col] = df_unique_repos[col].clip(lower=1)

        ax = sns.histplot(
            data=df_unique_repos,
            x=col,
            hue="Impact Category",
            hue_order=[increase_label, decrease_label],
            multiple="dodge",
            shrink=0.8,
            bins=10,
            palette=palette,
            log_scale=log_scale,
            edgecolor="black",
            linewidth=0.5
        )

        xlabel = f"{label} (Log Scale intervals)" if log_scale else f"{label} (Intervals)"
        plt.title(
            f"Project Distribution by {label}\nand Merge-rate Impact Category", fontsize=14, weight="bold", pad=20)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Number of Projects", fontsize=12)
        plt.legend(title="Impact Category", loc="upper right",
                   labels=[increase_label, decrease_label])
        plt.tight_layout()

        output_file_pdf = FIGS_PATH / f"context_file_projects_per_{suffix}.pdf"
        output_file_img = FIGS_PATH / f"context_file_projects_per_{suffix}.png"

        plt.savefig(output_file_pdf, bbox_inches="tight",
                    pad_inches=0.01, dpi=300)
        plt.savefig(output_file_img, bbox_inches="tight",
                    pad_inches=0.01, dpi=300)
        print(f"Bar plot saved to: {output_file_img}")
        plt.close()


def plot_topic_attribute_boxplots(df_topics, one_version=False):
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)
        df_repos = pd.read_csv(CONFIGURED_REPOS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    df = pd.merge(df_metrics, df_topics, on="repo", how="inner")

    increase_label = f"Merge-rate increase (≥{MERGE_RATE_THRESHOLD}%)"
    decrease_label = f"Merge-rate decrease (≤-{MERGE_RATE_THRESHOLD}%)"

    df["Impact Category"] = pd.NA
    df.loc[df["merge_impro_pct"] >= MERGE_RATE_THRESHOLD,
           "Impact Category"] = increase_label
    df.loc[df["merge_impro_pct"] <= -MERGE_RATE_THRESHOLD,
           "Impact Category"] = decrease_label

    analysis_df = df.dropna(subset=["Impact Category"]).copy()
    analysis_df = pd.merge(analysis_df, df_repos,
                           left_on="repo", right_on="repo_name", how="inner")

    exploded_df = analysis_df.explode("topic_labels").copy()
    exploded_df["topic_labels"] = exploded_df["topic_labels"].astype(
        str).str.strip()

    exploded_df = exploded_df[
        exploded_df["topic_labels"].notna()
        & (exploded_df["topic_labels"] != "")
        & (exploded_df["topic_labels"] != "nan")
    ].copy()

    exploded_df = exploded_df.drop_duplicates(
        subset=["repo", "Impact Category", "topic_labels"])

    attributes = ["contributors", "stargazers", "commits",
                  "totalPullRequests", "codeLines", "dependency_count"]
    palette = {increase_label: "#2ecc71", decrease_label: "#e74c3c"}
    sns.set_theme(style="whitegrid")

    for attr in attributes:
        if attr not in exploded_df.columns:
            continue

        plt.figure(figsize=(16, 9))

        exploded_df[attr] = pd.to_numeric(exploded_df[attr], errors='coerce')
        plot_df = exploded_df.dropna(subset=[attr]).copy()
        plot_df[attr] = plot_df[attr].clip(lower=1)

        ax = sns.boxplot(
            data=plot_df,
            x="topic_labels",
            y=attr,
            hue="Impact Category",
            palette=palette,
            fliersize=3,
            linewidth=1.2
        )

        plt.yscale('log')
        plt.title(f"Distribution of {attr} by Topic and Merge-rate Impact Category",
                  fontsize=15, weight="bold", pad=20)
        plt.xlabel("Topic", fontsize=12)
        plt.ylabel(f"{attr} (Log Scale)", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.legend(title="Impact Category", loc="upper right")

        # -----------------------------------------------------------
        # Statistical Annotations
        # -----------------------------------------------------------
        ymin, ymax = ax.get_ylim()
        ymin = max(ymin, 1e-5)  # Safe lower bound for log10
        log_ymin, log_ymax_ax = np.log10(ymin), np.log10(ymax)
        yrange = log_ymax_ax - log_ymin
        highest_annot_log = log_ymax_ax

        for i, tick in enumerate(ax.get_xticklabels()):
            topic_str = tick.get_text()
            d1 = plot_df[(plot_df["topic_labels"] == topic_str) & (
                plot_df["Impact Category"] == increase_label)][attr].dropna()
            d2 = plot_df[(plot_df["topic_labels"] == topic_str) & (
                plot_df["Impact Category"] == decrease_label)][attr].dropna()

            sym = get_significance_marker(d1, d2)
            if sym:
                local_max = max(d1.max() if len(d1) > 0 else ymin,
                                d2.max() if len(d2) > 0 else ymin)
                local_log = np.log10(local_max)

                # Offset bounds appropriately for log-space
                y_tick = 10**(local_log + 0.02 * yrange)
                y_bar = 10**(local_log + 0.05 * yrange)

                # Draw Bracket & Text
                x1, x2 = i - 0.2, i + 0.2
                ax.plot([x1, x1, x2, x2], [y_tick, y_bar,
                        y_bar, y_tick], lw=1.2, c='black')
                ax.text((x1 + x2) / 2, y_bar, sym, ha='center', va='bottom',
                        color='black', fontsize=12, fontweight='bold')

                # Track the highest required point to stretch the axes
                highest_annot_log = max(
                    highest_annot_log, local_log + 0.15 * yrange)

        if highest_annot_log > log_ymax_ax:
            ax.set_ylim(top=10**highest_annot_log)

        plt.tight_layout()

        suffix = attr.lower()
        output_file_pdf = str(FIGS_PATH / f"topic_{suffix}_boxplot.pdf")
        output_file_img = str(FIGS_PATH / f"topic_{suffix}_boxplot.png")

        if one_version:
            output_file_pdf = output_file_pdf.replace(
                ".pdf", "_one_version.pdf")
            output_file_img = output_file_img.replace(
                ".png", "_one_version.png")

        plt.savefig(output_file_pdf, bbox_inches="tight",
                    pad_inches=0.01, dpi=300)
        plt.savefig(output_file_img, bbox_inches="tight",
                    pad_inches=0.01, dpi=300)
        print(f"Boxplot saved to: {output_file_img}")
        plt.close()


def plot_lines_changed_comparison():
    """Plots the distribution of file additions and deletions across context files."""
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)
        df_commits = pd.read_csv(COMMITS_METADATA_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Rename repo_name to repo to match the metrics dataframe
    if "repo_name" in df_commits.columns:
        df_commits = df_commits.rename(columns={"repo_name": "repo"})

    # Merge commit metadata with the impact categories
    df = pd.merge(df_commits, df_metrics, on="repo", how="inner")

    increase_label = f"Merge-rate increase (≥{MERGE_RATE_THRESHOLD}%)"
    decrease_label = f"Merge-rate decrease (≤-{MERGE_RATE_THRESHOLD}%)"

    df["Impact Category"] = pd.NA
    df.loc[df["merge_impro_pct"] >= MERGE_RATE_THRESHOLD,
           "Impact Category"] = increase_label
    df.loc[df["merge_impro_pct"] <= -MERGE_RATE_THRESHOLD,
           "Impact Category"] = decrease_label

    analysis_df = df.dropna(subset=["Impact Category"]).copy()

    # Ensure numeric types
    analysis_df["file_additions"] = pd.to_numeric(
        analysis_df["file_additions"], errors="coerce").fillna(0)
    analysis_df["file_deletions"] = pd.to_numeric(
        analysis_df["file_deletions"], errors="coerce").fillna(0)

    # Melt dataframe for side-by-side seaborn boxplots
    melted_df = analysis_df.melt(
        id_vars=["repo", "context_file", "Impact Category"],
        value_vars=["file_additions", "file_deletions"],
        var_name="Change Type",
        value_name="Lines"
    )

    # Format labels cleanly and apply log-scale bottom limit
    melted_df["Change Type"] = melted_df["Change Type"].map({
        "file_additions": "Lines Added",
        "file_deletions": "Lines Deleted"
    })
    melted_df["Lines"] = melted_df["Lines"].clip(lower=1)

    # -----------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    palette = {increase_label: "#2ecc71", decrease_label: "#e74c3c"}

    ax = sns.boxplot(
        data=melted_df,
        x="Change Type",
        y="Lines",
        hue="Impact Category",
        palette=palette,
        fliersize=3,
        linewidth=1.2
    )

    # Log scale is vital here, as line changes can range from 1 to 10,000+
    plt.yscale('log')

    plt.title("Distribution of Lines Added and Deleted in Context Files",
              fontsize=15, weight="bold", pad=20)
    plt.xlabel("", fontsize=12)
    plt.ylabel("Number of Lines (Log Scale)", fontsize=12)
    plt.legend(title="Impact Category", loc="upper right")

    # -----------------------------------------------------------
    # Statistical Annotations
    # -----------------------------------------------------------
    ymin, ymax = ax.get_ylim()
    ymin = max(ymin, 1e-5)
    log_ymin, log_ymax_ax = np.log10(ymin), np.log10(ymax)
    yrange = log_ymax_ax - log_ymin
    highest_annot_log = log_ymax_ax

    for i, tick in enumerate(ax.get_xticklabels()):
        ctype_str = tick.get_text()
        d1 = melted_df[(melted_df["Change Type"] == ctype_str) & (
            melted_df["Impact Category"] == increase_label)]["Lines"].dropna()
        d2 = melted_df[(melted_df["Change Type"] == ctype_str) & (
            melted_df["Impact Category"] == decrease_label)]["Lines"].dropna()

        sym = get_significance_marker(d1, d2)
        if sym:
            local_max = max(d1.max() if len(d1) > 0 else ymin,
                            d2.max() if len(d2) > 0 else ymin)
            local_log = np.log10(local_max)

            y_tick = 10**(local_log + 0.02 * yrange)
            y_bar = 10**(local_log + 0.05 * yrange)

            x1, x2 = i - 0.2, i + 0.2
            ax.plot([x1, x1, x2, x2], [y_tick, y_bar,
                    y_bar, y_tick], lw=1.2, c='black')
            ax.text((x1 + x2) / 2, y_bar, sym, ha='center', va='bottom',
                    color='black', fontsize=12, fontweight='bold')

            highest_annot_log = max(
                highest_annot_log, local_log + 0.15 * yrange)

    if highest_annot_log > log_ymax_ax:
        ax.set_ylim(top=10**highest_annot_log)

    plt.tight_layout()

    # -----------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------
    output_file_pdf = str(FIGS_PATH / "lines_changed_boxplot.pdf")
    output_file_img = str(FIGS_PATH / "lines_changed_boxplot.png")

    plt.savefig(output_file_pdf, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.savefig(output_file_img, bbox_inches="tight", pad_inches=0.01, dpi=300)
    print(f"Lines changed plot saved to: {output_file_img}")
    plt.close()


def analyze_topic_count_impact_latest(df_all_versions):
    """
    Filters to the latest version of each context file, deduplicates topics per project,
    measures the correlation between the number of topics and its merge-rate,
    and plots a boxplot of topic counts per project group.
    """
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 1. Split context_file using '&SEP&' into file_path and file_timestamp (if not already done)
    df = df_all_versions.copy()
    if 'file_path' not in df.columns:
        df[['file_path', 'file_timestamp']] = df['context_file'].str.split(
            '&SEP&', n=1, expand=True)

    # 2. Sort by time and isolate the final version of each file
    df_sorted = df.sort_values(by=["repo", "file_path", "timestamp_parsed"])
    df_latest = df_sorted.drop_duplicates(
        subset=["repo", "file_path"], keep="last").copy()

    # 3. Deduplicate topics per project for these latest files
    df_latest_dedup = deduplicate_topics_per_project(df_latest)

    # Calculate number of unique topics per project
    df_latest_dedup['topic_count'] = df_latest_dedup['topic_labels'].apply(
        lambda x: len(x) if isinstance(x, list) else 0)

    # Merge deduplicated topic data with metrics
    df_merged = pd.merge(df_metrics, df_latest_dedup, on="repo", how="inner")

    # -----------------------------------------------------------
    # 1. Measure Correlation
    # -----------------------------------------------------------
    corr_df = df_merged.dropna(subset=['topic_count', 'merge_impro_pct'])
    if not corr_df.empty:
        # Using Spearman rank correlation as topic counts and merge-rates may not be normally distributed
        corr_stat, corr_pval = spearmanr(
            corr_df['topic_count'], corr_df['merge_impro_pct'])

        stats_out_path = DATA_DIR / "topic_count_correlation_latest.txt"
        with open(stats_out_path, "w", encoding="utf-8") as f:
            f.write(
                "Correlation Between Project Topic Count (Latest Versions) and Merge-Rate Improvement\n")
            f.write("="*80 + "\n\n")
            f.write(f"Spearman Correlation Coefficient: {corr_stat:.4f}\n")
            f.write(f"P-value: {corr_pval:.4e}\n")

        print(f"Topic count correlation saved to: {stats_out_path}")

    # -----------------------------------------------------------
    # 2. Plot Boxplot
    # -----------------------------------------------------------
    increase_label = f"Merge-rate increase (≥{MERGE_RATE_THRESHOLD}%)"
    decrease_label = f"Merge-rate decrease (≤-{MERGE_RATE_THRESHOLD}%)"

    df_merged["Impact Category"] = pd.NA
    df_merged.loc[df_merged["merge_impro_pct"] >=
                  MERGE_RATE_THRESHOLD, "Impact Category"] = increase_label
    df_merged.loc[df_merged["merge_impro_pct"] <= -
                  MERGE_RATE_THRESHOLD, "Impact Category"] = decrease_label

    analysis_df = df_merged.dropna(subset=["Impact Category"]).copy()

    if analysis_df.empty:
        print("No valid data points for the topic count boxplot after categorization.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 7))
    palette = {increase_label: "#2ecc71", decrease_label: "#e74c3c"}

    ax = sns.boxplot(
        data=analysis_df,
        x="Impact Category",
        y="topic_count",
        palette=palette,
        fliersize=3,
        linewidth=1.2,
        width=0.5
    )

    plt.title("Number of Unique Topics per Project by Merge-rate Impact Category\n(Latest File Versions Only)",
              fontsize=14, weight="bold", pad=20)
    plt.xlabel("", fontsize=12)
    plt.ylabel("Number of Unique Topics", fontsize=12)

    # Statistical Annotations
    d1 = analysis_df[analysis_df["Impact Category"]
                     == increase_label]["topic_count"].dropna()
    d2 = analysis_df[analysis_df["Impact Category"]
                     == decrease_label]["topic_count"].dropna()

    sym = get_significance_marker(d1, d2)
    if sym:
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin

        local_max = max(d1.max() if len(d1) > 0 else 0,
                        d2.max() if len(d2) > 0 else 0)

        y_tick = local_max + 0.05 * yrange
        y_bar = local_max + 0.10 * yrange

        # Coordinates for the two categories (0 and 1 on x-axis)
        x1, x2 = 0, 1
        ax.plot([x1, x1, x2, x2], [y_tick, y_bar,
                y_bar, y_tick], lw=1.2, c='black')
        ax.text((x1 + x2) / 2, y_bar, sym, ha='center', va='bottom',
                color='black', fontsize=12, fontweight='bold')
        ax.set_ylim(top=ymax + 0.15 * yrange)

    plt.tight_layout()

    output_file_pdf = str(
        FIGS_PATH / "topic_count_per_project_boxplot_latest.pdf")
    output_file_img = str(
        FIGS_PATH / "topic_count_per_project_boxplot_latest.png")

    plt.savefig(output_file_pdf, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.savefig(output_file_img, bbox_inches="tight", pad_inches=0.01, dpi=300)
    print(f"Topic count boxplot (latest versions) saved to: {output_file_img}")
    plt.close()


def generate_metrics_latex_table():
    """
    Generates a LaTeX table comparing project metrics for repositories with 
    a merge-rate decrease (<= -20%) and a merge-rate increase (>= 20%).
    """
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)
        df_repos = pd.read_csv(CONFIGURED_REPOS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Merge metrics and repo metadata
    df = pd.merge(df_metrics, df_repos, left_on="repo",
                  right_on="repo_name", how="inner")

    # Filter into the two threshold groups
    inc_group = df[df["merge_impro_pct"] >= MERGE_RATE_THRESHOLD]
    dec_group = df[df["merge_impro_pct"] <= -MERGE_RATE_THRESHOLD]

    # Map the human-readable names to the dataframe columns
    metrics_config = [
        ("Contributors", "contributors"),
        ("Stars", "stargazers"),
        ("Dependencies", "dependency_count"),
        ("Total PRs", "totalPullRequests"),
        ("Code Lines", "codeLines"),
        ("Commits", "commits")
    ]

    # Initialize LaTeX table string
    latex_str = "\\begin{table*}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Statistics of project metrics for Merge-rate Decrease ($\\le -20\\%$) vs. Increase ($\\ge 20\\%$)}\n"
    latex_str += "\\label{tab:repo_metrics_stats}\n"
    latex_str += "\\resizebox{\\textwidth}{!}{\n"
    latex_str += "\\begin{tabular}{l|cc|cc|cc|cc|ccc}\n"
    latex_str += "\\toprule\n"

    # Table Headers
    latex_str += "\\multirow{2}{*}{\\textbf{Metric}} & " \
                 "\\multicolumn{2}{c|}{\\textbf{Min}} & " \
                 "\\multicolumn{2}{c|}{\\textbf{Median}} & " \
                 "\\multicolumn{2}{c|}{\\textbf{Mean}} & " \
                 "\\multicolumn{2}{c|}{\\textbf{Max}} & " \
                 "\\multirow{2}{*}{\\textbf{$p$-value}} & " \
                 "\\multirow{2}{*}{\\textbf{Effect Size}} & " \
                 "\\multirow{2}{*}{\\textbf{Significance}} \\\\\n"

    # Sub-headers for <= -20% and >= 20%
    dec_hdr = "$\\le -20\\%$"
    inc_hdr = "$\\ge 20\\%$"
    latex_str += f" & {dec_hdr} & {inc_hdr} " \
        f"& {dec_hdr} & {inc_hdr} " \
        f"& {dec_hdr} & {inc_hdr} " \
        f"& {dec_hdr} & {inc_hdr} & & & \\\\\n"
    latex_str += "\\midrule\n"

    # Formatting helper for large numbers
    def fmt(val, is_mean=False):
        if pd.isna(val):
            return "N/A"
        return f"{val:,.1f}" if is_mean or (isinstance(val, float) and not val.is_integer()) else f"{int(val):,}"

    for label, col in metrics_config:
        if col not in dec_group.columns or col not in inc_group.columns:
            continue

        dec_data = pd.to_numeric(dec_group[col], errors='coerce').dropna()
        inc_data = pd.to_numeric(inc_group[col], errors='coerce').dropna()

        if len(dec_data) == 0 or len(inc_data) == 0:
            continue

        # Calculate Statistics
        d_min, i_min = dec_data.min(), inc_data.min()
        d_med, i_med = dec_data.median(), inc_data.median()
        d_mean, i_mean = dec_data.mean(), inc_data.mean()
        d_max, i_max = dec_data.max(), inc_data.max()

        # Calculate Statistics & Effect Size (Rank-Biserial correlation calculation included as fallback)
        try:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(dec_data, inc_data, alternative='two-sided')

            n1, n2 = len(dec_data), len(inc_data)
            # Rank-Biserial Correlation r = 1 - (2 * U) / (n1 * n2)
            r = 1 - (2 * stat) / (n1 * n2)
            abs_r = abs(r)

            if abs_r < 0.1:
                size_label = 'negligible'
            elif abs_r < 0.3:
                size_label = 'small'
            elif abs_r < 0.5:
                size_label = 'medium'
            else:
                size_label = 'large'

            effect_size_str = f"{abs_r:.3f} ({size_label})"
        except Exception:
            p = float('nan')
            effect_size_str = "N/A"

        # Determine Significance
        significance = "Yes" if not pd.isna(p) and p < 0.05 else "No"
        p_str = f"{p:.4f}" if not pd.isna(p) else "N/A"
        if p < 0.0001:
            p_str = "$< 0.0001$"

        # Build Row
        row = f"{label} " \
            f"& {fmt(d_min)} & {fmt(i_min)} " \
            f"& {fmt(d_med)} & {fmt(i_med)} " \
            f"& {fmt(d_mean, is_mean=True)} & {fmt(i_mean, is_mean=True)} " \
            f"& {fmt(d_max)} & {fmt(i_max)} " \
            f"& {p_str} & {effect_size_str} & {significance} \\\\\n"

        latex_str += row

    # Closing LaTeX tags
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "}\n"
    latex_str += "\\end{table}\n"

    # Export to file
    out_path = DATA_DIR / "repo_metrics_comparison_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"Metrics LaTeX table successfully saved to: {out_path}")
    return latex_str


def run_association_rules(df_all_versions, one_version=False):
    """
    Runs association rule algorithm using mlxtend on topics of the context files 
    across projects with >=20% merge-rate increase and <=-20% merge-rate decrease.
    Outputs the rules to a CSV file.
    """
    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        print("Error: The 'mlxtend' library is not installed. Please install it using 'pip install mlxtend'.")
        return

    print(
        f"Running association rule mining via mlxtend (one_version={one_version})...")

    # 1. Filter versions if required & deduplicate per project
    if one_version:
        df_to_process = filter_one_version(df_all_versions)
    else:
        df_to_process = df_all_versions.copy()

    df_proj = deduplicate_topics_per_project(df_to_process)

    # 2. Load metrics and merge
    try:
        df_metrics = pd.read_csv(DATA_DIR / METRICS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    df_merged = pd.merge(df_proj, df_metrics, on="repo", how="inner")
    print(f"df_merged...")

    # 3. Split into the two threshold groups
    inc_df = df_merged[df_merged["merge_impro_pct"] >= MERGE_RATE_THRESHOLD]
    dec_df = df_merged[df_merged["merge_impro_pct"] <= -MERGE_RATE_THRESHOLD]

    def get_rules_for_group(df_group):
        # Extract unique topics per project as a list of lists
        transactions = df_group['topic_labels'].apply(
            lambda x: list(set(x)) if isinstance(x, list) else []
        ).tolist()

        # Remove empty transactions
        transactions = [t for t in transactions if len(t) > 0]

        if not transactions:
            return pd.DataFrame()

        # One-hot encode the transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

        # Apriori algorithm to find frequent itemsets
        frequent_itemsets = apriori(
            df_onehot, min_support=0.04, use_colnames=True)
        print(f"{len(frequent_itemsets)=}...")

        if frequent_itemsets.empty:
            return pd.DataFrame()

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence",
                                  min_threshold=0.019, num_itemsets=len(transactions))
        print(f"{len(rules)=}...")

        # We only want pairs (topic_a -> topic_b), so filter for single antecedents and single consequents
        rules = rules[(rules['antecedents'].apply(len) == 1) &
                      (rules['consequents'].apply(len) == 1)].copy()

        # Unpack the frozensets into strings
        rules['topic_a'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['topic_b'] = rules['consequents'].apply(lambda x: list(x)[0])

        # Keep only the relevant columns and rename them
        return rules[['topic_a', 'topic_b', 'support', 'confidence']]

    # 4. Calculate rules for both groups
    rules_inc = get_rules_for_group(inc_df)
    print(f"rules_inc...")
    rules_dec = get_rules_for_group(dec_df)
    print(f"rules_dec...")

    # Rename metrics columns specifically for the increase group
    if not rules_inc.empty:
        rules_inc = rules_inc.rename(columns={
            'support': 'support_merg_rate_inc',
            'confidence': 'confidence_merge_rate_inc'
        })
    else:
        rules_inc = pd.DataFrame(columns=[
                                 'topic_a', 'topic_b', 'support_merg_rate_inc', 'confidence_merge_rate_inc'])

    # Rename metrics columns specifically for the decrease group
    if not rules_dec.empty:
        rules_dec = rules_dec.rename(columns={
            'support': 'support_merg_rate_decr',
            'confidence': 'confidence_merg_rate_decr'
        })
    else:
        rules_dec = pd.DataFrame(columns=[
                                 'topic_a', 'topic_b', 'support_merg_rate_decr', 'confidence_merg_rate_decr'])

    # 5. Merge the two DataFrames on the topic pairs
    df_final_rules = pd.merge(rules_inc, rules_dec, on=[
                              'topic_a', 'topic_b'], how='outer')

    # Fill NaN values with 0.0 for pairs that didn't appear in one of the groups
    df_final_rules = df_final_rules.fillna(0.0)

    # Sort by the increase confidence for readability
    df_final_rules = df_final_rules.sort_values(
        by=["confidence_merge_rate_inc", "support_merg_rate_inc"],
        ascending=[False, False]
    )

    # -----------------------------------------------------------
    # STATISTICAL TESTING ON CONFIDENCE DISTRIBUTIONS
    # -----------------------------------------------------------
    if not df_final_rules.empty:
        conf_inc_vals = df_final_rules['confidence_merge_rate_inc'].values
        conf_dec_vals = df_final_rules['confidence_merg_rate_decr'].values

        # Wilcoxon Signed-Rank Test (Paired testing across the same rules)
        try:
            w_stat, w_pval = wilcoxon(
                conf_inc_vals, conf_dec_vals, zero_method='zsplit')
        except ValueError:
            w_stat, w_pval = float('nan'), float('nan')

        # Mann-Whitney U Test (Unpaired distribution comparison)
        try:
            u, p, d, size = mannUandCliffdelta(conf_inc_vals, conf_dec_vals)
        except ValueError:
            u, p, d, size = float('nan'), float('nan'), float('nan'), str("")

        # Export Stats
        stats_file_name = "association_rules_confidence_stats.txt"
        if one_version:
            stats_file_name = stats_file_name.replace(
                ".txt", "_one_version.txt")

        stats_out_path = DATA_DIR / stats_file_name

        with open(stats_out_path, "w", encoding="utf-8") as f:
            f.write(
                "Statistical Comparison of Rule Confidence Between Increase and Decrease Groups\n")
            f.write("="*80 + "\n\n")
            f.write(f"Number of rules analyzed: {len(df_final_rules)}\n\n")

            f.write("1. Wilcoxon Signed-Rank Test (Paired comparison)\n")
            f.write(
                "   Tests if confidence for the SAME topic pair differs systematically between groups.\n")
            f.write(f"   Statistic: {w_stat:.4f}\n")
            f.write(f"   P-value:   {w_pval:.4e}\n\n")

            f.write("2. Mann-Whitney U Test (Unpaired distribution comparison)\n")
            f.write(
                "   Tests if the overall distribution of confidence values differs between groups.\n")
            f.write(f"   Statistic: {u:.4f}\n")
            f.write(f"   P-value:   {p:.4e}\n")
            f.write(f"   Delta:   {d:.4e}\n")
            f.write(f"   Size:   {size}\n")

        print(f"Confidence statistical comparison saved to: {stats_out_path}")

    # 6. Save to CSV
    file_name = "topic_association_rules_mlxtend.csv"
    if one_version:
        file_name = file_name.replace(".csv", "_one_version.csv")

    out_path = DATA_DIR / file_name
    df_final_rules.to_csv(out_path, index=False)
    print(f"Association rules (mlxtend) successfully saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter and plot topic distributions from repo metrics.")
    parser.add_argument("--one-version", action="store_true",
                        help="Consider only the first version of the context file.")
    args = parser.parse_args()

    # 1. Fetch full historical base DataFrame
    df_all_versions = parse_jsonl_file()

    # 2. Plot Version-to-Version Evolution Dynamics
    # (Requires all versions natively to observe diffs)
    plot_topic_evolution(df_all_versions)
    plot_latest_topic_cooccurrence(df_all_versions)

    # 3. Handle standard dataset transformations for standard static plots
    if args.one_version:
        df_static = filter_one_version(df_all_versions)
    else:
        df_static = df_all_versions.copy()

    # 4. Strict topic deduplication per project before standard plots
    # df_static_deduplicated = deduplicate_topics_per_project(df_static)

    # Clean up redundant timestamp parsing column and export
    # df_export = df_static_deduplicated.drop(
    #     columns=["timestamp_parsed"], errors="ignore")
    # df_export.to_csv(DATA_DIR / "context_file_topics.csv", index=None)

    # plot_lines_changed_comparison()

    # 5. Run static plotting sequence
    # plot_topic_distribution(df_static_deduplicated,
    #                         one_version=args.one_version)
    # plot_repo_metrics_barplots(df_static_deduplicated)
    # plot_topic_attribute_boxplots(
    #     df_static_deduplicated, one_version=args.one_version)

    # NEW: Run the correlation and plot the topic count per project (Using latest files)
    # analyze_topic_count_impact_latest(df_all_versions)

    # NEW: Generate the LaTeX statistical table for repo metrics
    # generate_metrics_latex_table()

    # NEW: Run robust association rule mining via mlxtend
    run_association_rules(df_all_versions, one_version=args.one_version)
