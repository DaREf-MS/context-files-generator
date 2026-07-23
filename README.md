# Replication package

This is the replication package for the paper **Characterizing Topics in Agent Context Files: Prevalence, Characteristics, and Emerging Insights**

This project uses the dataset for the paper **"A Dataset of Agentic AI Coding Tool Configurations"**, accepted at the [AIWare 2026 Datasets and Benchmarks track](https://2026.aiwareconf.org/track/aiware-2026-benchmark---dataset-track).

---

## 📊 Overview

we conduct a large-scale empirical study of 8,732 ACF versions collected from 430 engineered, actively maintained, and history-rich software repositories. Our study aims to (1) systematically identify the topics mentioned in the ACF files, (2) investigate how ACFs evolve over time by inspecting whether developers add new topics with the evolution of their projects, (3) examine the association between the characteristics of projects and the presence of an ACF topic. Finally, with the goal to guide developers writing a ACF files, (4) we evaluate using the characteristics of a project the effectiveness of machine learning models in predicting ACF topics for that project. 

### Dataset Features
* **4,738** unique GitHub repositories.
* Data on **5 major agentic tools**:
  * **Claude Code**
  * **Gemini CLI**
  * **Codex CLI**
  * **Copilot CLI**
  * **Cursor**

---

## 📁 Repository Structure

| Directory | Description |
| :--- | :--- |
| `/data` | Contains the raw configuration artifacts (JSON/Text). |
| `/docs` | Contains the TopicGPT system prompts and the resulting refined topics. |
| `/src`  | Contains source code files to reporduce the experiments and results of the study. |
| `/figs` | Contains plots used in the paper. |

---

## 🛠️ Setup and Reproducibility

The `src/` directory contains the scripts required to reproduce the experiments, analyses, and figures presented in the paper. The typical workflow is outlined below.

### 1. Data Collection

Collect the raw data from the GitHub API.

| Script | Description |
|--------|-------------|
| `src/fetch_pr_metadata.py` | Retrieves pull request metadata from GitHub repositories. |
| `src/collect_context_file_snapshots.py` | Collects snapshots of Agent Context Files (ACFs) together with their associated commit metadata. |
| `src/collect_deps.py` | Collects repository dependency information from the GitHub API. |

### 2. Metric Generation

Generate repository-, ACF-, and pull request-level metrics used throughout the study.

| Script | Description |
|--------|-------------|
| `src/gen_cont_file_metr.py` | Computes metrics for Agent Context Files. |
| `src/gen_pr_metrics.py` | Computes pull request metrics. |
| `src/gen_rqs_data.py` | Generates repository metrics before and after the introduction of ACFs for answering the research questions. |
| `src/collect_pre_acf_metrics.py` | Collects repository metrics prior to the introduction of ACFs for the machine learning study. |
| `src/collect_advanced_repo_metrics.py` | Collects additional repository-level metrics used for predicting ACF topics. |

### 3. Topic Extraction

Prepare the collected ACFs for topic classification.

| Script | Description |
|--------|-------------|
| `src/acf_jsonl_exporter.py` | Exports collected Agent Context Files into JSONL format for topic classification using TopicGPT. |

### 4. Machine Learning Pipeline

Build and evaluate models for predicting ACF topics.

| Script | Description |
|--------|-------------|
| `src/build_model_dataset.py` | Builds the dataset used to train and evaluate topic prediction models. |
| `src/evaluate_topic_prediction.py` | Trains and evaluates machine learning models for ACF topic prediction. |

### 5. Analysis and Visualization

Generate the analyses, figures, and tables reported in the paper.

| Script | Description |
|--------|-------------|
| `src/gen_global_topic_analysis.py` | Produces the statistical analyses and figures used to answer the research questions. |
| `src/gen_latex_tab_from_xlsx.py` | Converts the manually annotated Excel results into publication-ready LaTeX tables. |
| `src/main.ipynb` | Explores the dataset and compute the inter-rater reliability score between labelers. |

---

## 📋 Recommended Execution Order

To fully reproduce the study, execute the scripts in the following order:

1. `fetch_pr_metadata.py`
2. `collect_context_file_snapshots.py`
3. `collect_deps.py`
4. `gen_cont_file_metr.py`
5. `gen_pr_metrics.py`
6. `collect_pre_acf_metrics.py`
7. `collect_advanced_repo_metrics.py`
8. `gen_rqs_data.py`
9. `acf_jsonl_exporter.py`
10. Run TopicGPT to classify ACF topics.
11. `build_model_dataset.py`
12. `evaluate_topic_prediction.py`
13. `gen_global_topic_analysis.py`
14. `gen_latex_tab_from_xlsx.py`