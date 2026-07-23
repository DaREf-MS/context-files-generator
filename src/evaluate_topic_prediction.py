import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import cross_validate
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

import seaborn as sns

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("Please install XGBoost: pip install xgboost")

# Suppress minor convergence warnings from linear models for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGS_DIR = ROOT_DIR / "figs"
INPUT_DATASET_FILE = DATA_DIR / "final_model_dataset.parquet"

# New paths for saving/loading results
GLOBAL_RESULTS_FILE = DATA_DIR / "cv_global_results.csv"
TOPIC_RESULTS_FILE = DATA_DIR / "cv_per_topic_results.csv"

# Set to True if you want to bypass the CSV cache and run the CV pipeline again
FORCE_RECOMPUTE = False 

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features_and_targets(df, min_samples=15):
    """
    Separates the target topics, numerical features, and the embeddings.
    Filters out rare topics that will break cross-validation.
    """
    print("Preparing features and targets...")

    # 1. Isolate the target columns
    topic_cols = [c for c in df.columns if c.startswith('topic_')]

    # Filter out extremely rare topics
    topic_counts = df[topic_cols].sum()
    valid_topics = topic_counts[topic_counts >= min_samples].index.tolist()

    dropped_count = len(topic_cols) - len(valid_topics)
    if dropped_count > 0:
        print(
            f"[!] Dropped {dropped_count} rare topics that occurred fewer than {min_samples} times.")

    topic_cols = valid_topics
    Y = df[topic_cols].values

    # 2. Isolate the explicit numerical independent variables
    exclude_cols = ['repo', 'readme', 'readme_embedding'] + \
        [c for c in df.columns if c.startswith(
            'topic_') or c.startswith('emb_dim_')]
    if 'first_acf_date' in df.columns:
        exclude_cols.append('first_acf_date')

    num_features_cols = [c for c in df.columns if c not in exclude_cols]

    # Ensure numerical data is correctly typed
    X_num = df[num_features_cols].astype(float).values

    # 3. Process the Embeddings
    print(f"Extracting 384-dimensional embeddings for {len(df)} records...")
    X_emb = np.vstack(df['readme_embedding'].values)

    # 4. Final feature matrix (currently only numerical)
    X_final = X_num
    num_indices = list(range(X_num.shape[1]))
    feature_names = num_features_cols

    print(f"Final Feature Space: {X_final.shape[1]} dimensions.")
    print(f"Target Space: {Y.shape[1]} distinct valid topics.\n")

    return X_final, Y, num_indices, topic_cols, feature_names


def build_models():
    """
    Initializes the multi-label machine learning models. 
    """
    models = {
        "Logistic Regression": MultiOutputClassifier(
            LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42)
        ),
        "Random Forest": MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=450, class_weight='balanced_subsample', max_depth=20, random_state=42, n_jobs=-1
            )
        ),
        "XGBoost": MultiOutputClassifier(
            XGBClassifier(
                max_depth=25,
                learning_rate=0.01,
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            )
        )
    }
    return models


def plot_global_feature_importance_heatmap(pipeline, topic_cols, num_features_cols):
    """
    Extracts tabular Gini importances across all topics from the multi-output RF model 
    and plots them in a single, comprehensive heatmap.
    """
    print("\n" + "="*60)
    print("GENERATING GLOBAL FEATURE IMPORTANCE HEATMAP")
    print("="*60)

    rf_multi_model = pipeline.named_steps['classifier']
    num_features_count = len(num_features_cols)

    importance_data = []
    clean_topic_names = [t.replace('topic_', '') for t in topic_cols]

    for topic_idx in range(len(topic_cols)):
        estimator = rf_multi_model.estimators_[topic_idx]
        num_importances = estimator.feature_importances_[:num_features_count]
        importance_data.append(num_importances)

    df_importance = pd.DataFrame(
        importance_data,
        index=clean_topic_names,
        columns=num_features_cols
    ).T

    df_importance['mean_importance'] = df_importance.mean(axis=1)
    df_importance = df_importance.sort_values(
        by='mean_importance', ascending=False)
    df_importance = df_importance.drop(columns=['mean_importance'])

    fig_width = max(10, len(clean_topic_names) * 0.8)
    fig_height = max(6, len(num_features_cols) * 0.5)

    plt.figure(figsize=(fig_width, fig_height))

    sns.heatmap(
        df_importance,
        cmap="Blues",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={'label': 'Gini Importance (Mean Decrease Impurity)'}
    )

    plt.title("Feature Importance Matrix Across All Configurations",
              fontsize=16, pad=20)
    plt.xlabel("TopicGPT Labels", fontsize=12, labelpad=10)
    plt.ylabel("Repository & PR Metrics", fontsize=12, labelpad=10)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    out_path = FIGS_DIR / "global_feature_importance_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved global feature importance heatmap to: {out_path}")



def plot_per_topic_performance_barplots(df_topics):
    """
    Generates horizontal grouped bar plots for Precision and Recall per topic across all models.
    """
    print("\n" + "="*60)
    print("GENERATING PER-TOPIC PERFORMANCE HORIZONTAL BAR PLOTS")
    print("="*60)

    metrics = ['Precision', 'Recall']

    for metric in metrics:
        # Filter columns corresponding to the current metric
        metric_cols = [c for c in df_topics.columns if metric in c]
        if not metric_cols:
            continue

        df_metric = df_topics[metric_cols].copy()

        # Reverse the dataframe so the first topic appears at the top of the horizontal plot
        df_metric = df_metric.iloc[::-1]

        # Clean column names for the legend
        df_metric.columns = [c.replace(f" {metric}", "")
                             for c in df_metric.columns]

        # Convert to percentages for plotting
        df_metric = df_metric * 100

        # Dynamically size the figure based on the number of topics to prevent crowding.
        fig_height = max(8, len(df_metric) * 0.5)
        fig_width = 12

        ax = df_metric.plot(kind='barh', figsize=(
            fig_width, fig_height), width=0.8, colormap='viridis')
        
        # Swap x and y labels for horizontal orientation
        plt.ylabel("Topics", fontsize=18, labelpad=12)
        plt.xlabel(f"{metric} (%)", fontsize=18, labelpad=12)

        # Apply the 25% interval range to the x-axis
        plt.yticks(fontsize=14)
        plt.xticks(np.arange(0, 101, 25), fontsize=14)

        # Apply the percentage formatter to the x-axis
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

        plt.legend(title="Models", title_fontsize='14', fontsize='12',
                   bbox_to_anchor=(1.01, 1), loc='upper left')

        # Switch the grid to the x-axis
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        out_path = FIGS_DIR / f"per_topic_{metric.lower()}_barplot.png"
        out_pdf = FIGS_DIR / f"per_topic_{metric.lower()}_barplot.pdf"
        plt.savefig(out_path, dpi=300, pad_inches=0.01, bbox_inches='tight')
        plt.savefig(out_pdf, dpi=300, pad_inches=0.01, bbox_inches='tight')
        plt.close()

        print(f"Saved horizontal {metric} barplot to: {out_path}")
        

def generate_latex_longtable(df_topics):
    """
    Generates a LaTeX longtable for model comparison across topics.
    Adjusts font size and column spacing to help fit the page width.
    Requires \\usepackage{longtable}, \\usepackage{booktabs}, and \\usepackage{multirow}.
    """
    print("\n" + "="*60)
    print("GENERATING LATEX LONGTABLE (WIDTH-ADJUSTED)")
    print("="*60)
    
    # Sort by Random Forest F1-score in descending order
    df_topics = df_topics.sort_values(by='Random Forest F1', ascending=False)
    
    out_path = FIGS_DIR / "per_topic_performance.tex"
    
    # Models in the exact order requested
    models = ["Logistic Regression", "XGBoost", "Random Forest"]
    
    latex_str = [
        # Use a group to localize font size and column separation changes
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{l ccc ccc ccc}",
        r"\caption{Per-Topic Performance Across Models} \label{tab:topic_performance} \\",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Topic}} & "
        r"\multicolumn{3}{c}{\textbf{Logistic Regression}} & "
        r"\multicolumn{3}{c}{\textbf{XGBoost}} & "
        r"\multicolumn{3}{c}{\textbf{Random Forest}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r"& \textbf{Prec} & \textbf{Rec} & \textbf{F1} "
        r"& \textbf{Prec} & \textbf{Rec} & \textbf{F1} "
        r"& \textbf{Prec} & \textbf{Rec} & \textbf{F1} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{10}{c}%",
        r"{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Topic}} & "
        r"\multicolumn{3}{c}{\textbf{Logistic Regression}} & "
        r"\multicolumn{3}{c}{\textbf{XGBoost}} & "
        r"\multicolumn{3}{c}{\textbf{Random Forest}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r"& \textbf{Prec} & \textbf{Rec} & \textbf{F1} "
        r"& \textbf{Prec} & \textbf{Rec} & \textbf{F1} "
        r"& \textbf{Prec} & \textbf{Rec} & \textbf{F1} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{10}{r}{{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot"
    ]
    
    for index, row in df_topics.iterrows():
        # Escape both underscores and ampersands for LaTeX compilation
        clean_topic = str(index).replace('_', r'\_').replace('&', r'\&')
        row_str = f"{clean_topic}"
        
        for model in models:
            # Multiply by 100 and format to 2 decimal places
            prec = f"{row.get(f'{model} Precision', 0) * 100:.2f}"
            rec = f"{row.get(f'{model} Recall', 0) * 100:.2f}"
            f1 = f"{row.get(f'{model} F1', 0) * 100:.2f}"
            
            row_str += f" & {prec} & {rec} & {f1}"
            
        row_str += r" \\"
        latex_str.append(row_str)
        
    latex_str.extend([
        r"\end{longtable}",
        r"}" # Close the font size and tabcolsep group
    ])
    
    final_latex = "\n".join(latex_str)
    
    with open(out_path, "w") as f:
        f.write(final_latex)
        
    print(f"Saved LaTeX table to: {out_path}")
    

def main():
    # --- Check for Precomputed Results ---
    if not FORCE_RECOMPUTE and GLOBAL_RESULTS_FILE.exists() and TOPIC_RESULTS_FILE.exists():
        print(f"Loading precomputed results from {DATA_DIR} to save time...")
        df_global = pd.read_csv(GLOBAL_RESULTS_FILE, index_col="Model")
        df_topics = pd.read_csv(TOPIC_RESULTS_FILE, index_col="Topic")
        
    else:
        # --- Run Full Training Pipeline ---
        print("Precomputed results not found or FORCE_RECOMPUTE is True. Starting training...")
        try:
            df = pd.read_parquet(INPUT_DATASET_FILE)
        except FileNotFoundError:
            print(f"Error: Could not find {INPUT_DATASET_FILE}. Please run the dataset builder script first.")
            return

        X, Y, num_indices, topic_cols, feature_names = prepare_features_and_targets(df)
        models = build_models()

        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, num_indices)
            ],
            remainder='passthrough'
        )

        kf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Build Scoring Dictionary
        scoring = {
            'f1_micro': 'f1_micro',
            'f1_macro': 'f1_macro',
            'precision_micro': 'precision_micro',
            'precision_macro': 'precision_macro',
            'recall_micro': 'recall_micro',
            'recall_macro': 'recall_macro'
        }

        clean_topic_names = [t.replace('topic_', '') for t in topic_cols]
        for idx, topic_name in enumerate(clean_topic_names):
            scoring[f'precision_{topic_name}'] = make_scorer(
                lambda y_true, y_pred, i=idx: precision_score(
                    y_true[:, i], y_pred[:, i], zero_division=0)
            )
            scoring[f'recall_{topic_name}'] = make_scorer(
                lambda y_true, y_pred, i=idx: recall_score(
                    y_true[:, i], y_pred[:, i], zero_division=0)
            )
            # Add the F1 scorer here
            scoring[f'f1_{topic_name}'] = make_scorer(
                lambda y_true, y_pred, i=idx: f1_score(
                    y_true[:, i], y_pred[:, i], zero_division=0)
            )

        global_results = []
        per_topic_results = {topic: {"Topic": topic} for topic in clean_topic_names}

        print("Starting 10-Fold Cross Validation...\n" + "-"*50)

        for name, model in models.items():
            print(f"Evaluating: {name}...")

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            cv_scores = cross_validate(
                pipeline, X, Y,
                cv=kf,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False
            )

            # Aggregate Global Metrics
            res = {
                "Model": name,
                "F1 (Micro)": np.mean(cv_scores['test_f1_micro']),
                "F1 (Macro)": np.mean(cv_scores['test_f1_macro']),
                "Precision (Micro)": np.mean(cv_scores['test_precision_micro']),
                "Recall (Micro)": np.mean(cv_scores['test_recall_micro']),
                "Precision (Macro)": np.mean(cv_scores['test_precision_macro']),
                "Recall (Macro)": np.mean(cv_scores['test_recall_macro'])
            }
            global_results.append(res)

            # Aggregate Per-Topic Metrics
            for topic_name in clean_topic_names:
                prec_mean = np.mean(cv_scores[f'test_precision_{topic_name}'])
                rec_mean = np.mean(cv_scores[f'test_recall_{topic_name}'])
                f1_mean = np.mean(cv_scores[f'test_f1_{topic_name}']) # Extract F1

                per_topic_results[topic_name][f"{name} Precision"] = prec_mean
                per_topic_results[topic_name][f"{name} Recall"] = rec_mean
                per_topic_results[topic_name][f"{name} F1"] = f1_mean # Save F1
                
        # Generate DataFrames
        df_global = pd.DataFrame(global_results).set_index("Model")
        df_topics = pd.DataFrame(list(per_topic_results.values())).set_index("Topic")
        
        # Save results to CSV for future runs
        print(f"\nSaving results to {DATA_DIR}...")
        df_global.to_csv(GLOBAL_RESULTS_FILE)
        df_topics.to_csv(TOPIC_RESULTS_FILE)
        print("Results successfully saved.")

        #### The contribution of each feature (Only run when training from scratch) ####
        rf_model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', rf_model)
        ])

        print("\nTraining final Random Forest model on all data for feature importance...")
        pipeline.fit(X, Y)

        # plot_global_feature_importance_heatmap(pipeline, topic_cols, [
        #     feat for feat in feature_names if not feat.startswith('emb_dim_')
        # ])

    # --- Print the Global Report (Applies whether loaded or computed) ---
    print("\n" + "="*80)
    print("1. GLOBAL 10-FOLD CROSS-VALIDATION PERFORMANCE (MICRO & MACRO)")
    print("="*80)
    df_global_formatted = (df_global * 100).round(2).astype(str) + '%'
    print(df_global_formatted.to_markdown())

    # --- Print the Per-Topic Report (Applies whether loaded or computed) ---
    print("\n" + "="*80)
    print("2. PER-TOPIC PRECISION & RECALL (AVERAGED ACROSS FOLDS)")
    print("="*80)
    df_topics_formatted = (df_topics * 100).round(2).astype(str) + '%'
    print(df_topics_formatted.to_markdown())

    # Plot Per-Topic Performance
    plot_per_topic_performance_barplots(df_topics)
    
    # Generate the LaTeX Table
    generate_latex_longtable(df_topics)

    print("\nNote: ")
    print("* 'Micro' metrics look at the aggregate predictive power across all topics combined.")
    print("* 'Macro' metrics average the performance of each individual topic equally, which is highly sensitive to class imbalance.")


if __name__ == "__main__":
    main()