# An Automated Approach to Generating Projects-specifi Agentic Context files 

This project uses the dataset for the paper **"A Dataset of Agentic AI Coding Tool Configurations"**, accepted at the [AIWare 2026 Datasets and Benchmarks track](https://2026.aiwareconf.org/track/aiware-2026-benchmark---dataset-track).

[![Conference](https://img.shields.io/badge/AIWare-2026-blue)](https://2026.aiwareconf.org/track/aiware-2026-benchmark---dataset-track)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Overview

This dataset tracks how developers configure agentic AI coding assistants across thousands of real-world projects. It captures the repository-level artifacts that define how these agents behave, what context they access, and how they interact with the codebase.

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
| `/docs` | Contains the documentation resources for the project. |
| `/scripts` | Utility scripts for parsing and analyzing the dataset. |
| `/src` | Contains source code files to reporduce the experiments and results of the study. |
| `/tests` | Contains unit and integration test files. |

---

## 🛠 Usage

To use this project in your own research:

1. **Clone the repo:**
   ```bash
   git clone git@github.com:DaREf-MS/context-files-generator.git
   ```