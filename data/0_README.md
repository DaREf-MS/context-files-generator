# A Dataset of Agentic AI Coding Tool Configurations

This dataset contains data on repository-level configuration artifacts for five agentic AI coding tools (Claude Code, Gemini CLI, Codex CLI, Copilot CLI, Cursor) across 4,738 GitHub repositories. It extends the data originally collected for the study by Galster et al. (2026) with per-artifact git metadata (creation dates, commit counts) and AI commit authorship detection.

## Associated Papers

- Matthias Galster, Seyedmoein Mohsenimofidi, Jai Lal Lulla, Muhammad Auwal Abubakar, Christoph Treude, Sebastian Baltes. **Configuring Agentic AI Coding Tools: An Exploratory Study.** In *Proceedings of the 3rd ACM/IEEE International Conference on AI-powered Software (AIware 2026)*.
- Seyedmoein Mohsenimofidi, Matthias Galster, Christoph Treude, Sebastian Baltes. **Context Engineering for AI Agents in Open-Source Software.** In *Proceedings of the 23rd IEEE/ACM International Conference on Mining Software Repositories (MSR 2026)*. Available at: https://arxiv.org/abs/2510.21413

## Dataset Overview

The dataset is organized into a main repository file and per-artifact detail files:

| File | Rows | Description |
|------|------|-------------|
| `repos.csv` | 40,585 | Sampling frame with GitHub metadata, classification, and tool/config detection flags |
| `commits.csv` | 148,519 | AI-co-authored commits across repositories with detected configurations |
| `context_files.csv` | 9,470 | Context file artifacts (e.g., `CLAUDE.md`, `AGENTS.md`, `.cursorrules`) |
| `settings.csv` | 472 | Settings configuration files (e.g., `.claude/settings.json`) |
| `skills.csv` | 2,430 | Skill definitions (e.g., `.claude/skills/*/SKILL.md`) |
| `subagents.csv` | 884 | Subagent definitions (e.g., `.claude/agents/*.md`) |
| `commands.csv` | 1,098 | Custom command files (e.g., `.claude/commands/*.md`) |
| `hooks.csv` | 102 | Hook configurations in settings files |
| `rules.csv` | 997 | Rule files (e.g., `.cursor/rules/*.mdc`) |
| `mcp.csv` | 138 | MCP (Model Context Protocol) configurations |

## Provenance

The dataset was constructed through a three-stage pipeline (described in detail in the associated dataset paper):

1. **Sampling frame** (40,585 repos): Repositories were sourced from the GitHub Search dataset, filtered by activity, license, and language (C, C#, C++, Go, Java, JavaScript, PHP, Python, Rust, TypeScript). Archived, forked, and educational repositories were excluded.
2. **Classification** (39,426 annotated): Repository README files were classified using GPT-5.2 to determine whether each repository is an engineered software project. 36,710 were classified as engineered projects.
3. **Configuration detection** (4,738 repos): Engineered repositories were cloned and scanned for AI tool configuration artifacts. 4,738 repositories contained at least one artifact.

All three levels are represented in `repos.csv`:
- All 40,585 rows have GitHub metadata (`mainLanguage`, `stargazers`, etc.)
- 39,426 rows have classification data (`engineered_project`, `justification`)
- 4,738 rows have tool/config detection flags (`claude`, `copilot`, `context_file`, etc.)

## File Descriptions

### repos.csv

Main repository dataset. One row per repository in the sampling frame.

| Column | Description |
|--------|-------------|
| `repo_name` | GitHub repository identifier (`owner/name`) |
| `id` | GitHub repository ID |
| `default_branch` | Default branch name |
| `scanned_branch` | Branch scanned during data collection (repos with detected configurations only) |
| `scanned_at` | Timestamp of data collection scan (repos with detected configurations only) |
| `engineered_project` | Whether the repo was classified as an engineered project |
| `github_link` | Link to the repository on GitHub (points to `scanned_branch` if available, otherwise `default_branch`) |
| `justification` | GPT-5.2 classification justification text |
| `claude` | Repository has Claude Code configuration |
| `gemini` | Repository has Gemini CLI configuration |
| `codex` | Repository has Codex CLI configuration |
| `copilot` | Repository has Copilot CLI configuration |
| `cursor` | Repository has Cursor configuration |
| `claude.md` | Repository has a CLAUDE.md context file |
| `gemini.md` | Repository has a GEMINI.md context file |
| `agents.md` | Repository has an AGENTS.md context file |
| `copilot_instructions` | Repository has Copilot instructions (`.github/copilot-instructions.md` or `.github/instructions/` directory) |
| `cursorrules` | Repository has a `.cursorrules` file |
| `context_file` | Repository has at least one context file |
| `settings` | Repository has tool settings files |
| `skills` | Repository has skill definitions |
| `subagents` | Repository has subagent definitions |
| `commands` | Repository has custom command files |
| `hooks` | Repository has hook configurations |
| `rules` | Repository has rule files |
| `mcp` | Repository has MCP configurations |
| `agents_md_only` | Repository has only an AGENTS.md file (no tool-specific config) |
| `mainLanguage` | Primary programming language |
| `license` | Repository license |
| `homepage` | Repository homepage URL |
| `createdAt` | Repository creation date |
| `size` | Repository size in KB |
| `commits` | Total number of commits |
| `branches` | Number of branches |
| `releases` | Number of releases |
| `forks` | Number of forks |
| `watchers` | Number of watchers |
| `stargazers` | Number of stars |
| `contributors` | Number of contributors |
| `totalIssues` | Total number of issues |
| `openIssues` | Number of open issues |
| `totalPullRequests` | Total number of pull requests |
| `openPullRequests` | Number of open pull requests |
| `codeLines` | Total lines of code |
| `blankLines` | Total blank lines |
| `commentLines` | Total comment lines |
| `languages` | JSON object mapping language names to byte counts |
| `topics` | JSON array of GitHub topics |

### commits.csv

AI-co-authored commits detected across repositories with detected configurations. One row per commit.

| Column | Description |
|--------|-------------|
| `repo_name` | GitHub repository identifier |
| `commit_timestamp` | Commit timestamp |
| `branch` | Branch the commit was found on |
| `commit_sha` | Commit SHA |
| `github_link` | Link to the commit on GitHub |
| `commit_message` | Full commit message (newlines escaped as `\n`) |
| `ai_tool` | AI tool that co-authored the commit |

### context_files.csv

Context files providing instructions to AI coding tools. One row per context file.

| Column | Description |
|--------|-------------|
| `repo_name` | GitHub repository identifier |
| `context_file` | Relative path to the context file |
| `timestamp` | Timestamp of the repository snapshot |
| `branch` | Branch where the file was found |
| `github_link` | Link to the file on GitHub |
| `created_at` | Timestamp of the first commit introducing the file |
| `#commits` | Number of commits touching the file |
| `is_empty` | Whether the file is empty |
| `language` | Detected natural language of the file content |
| `line_count` | Number of lines in the file |
| `is_reference` | Whether the file is a reference to another file |
| `referenced_file` | Path of the referenced file (if `is_reference` is TRUE) |
| `first_commit_sha` | SHA of the first commit introducing the file |
| `last_commit_sha` | SHA of the most recent commit touching the file |
| `first_commit_ai_created` | Whether the first commit was made by an AI tool |
| `first_commit_ai_tool` | AI tool that made the first commit (if applicable) |
| `#ai_commits` | Number of AI-co-authored commits touching the file |
| `ai_commit_tools` | AI tools that co-authored commits touching the file |

### settings.csv, skills.csv, subagents.csv, commands.csv, hooks.csv, rules.csv, mcp.csv

Per-artifact detail files. Each file contains one row per configuration artifact found across repositories with detected configurations. All share a common set of columns:

| Column | Description |
|--------|-------------|
| `repo_name` | GitHub repository identifier |
| *(artifact path)* | Relative path to the artifact (column name varies by file) |
| `timestamp` | Timestamp of the repository snapshot |
| `branch` | Branch where the artifact was found |
| `github_link` | Link to the artifact on GitHub |
| `created_at` | Timestamp of the first commit introducing the artifact |
| `#commits` | Number of commits touching the artifact |
| `is_empty` | Whether the artifact file is empty |
| `first_commit_sha` | SHA of the first commit introducing the artifact |
| `last_commit_sha` | SHA of the most recent commit touching the artifact |

Some files have additional columns:

- **skills.csv**: `name` (skill name), `scripts`/`references`/`assets` (whether the skill has these subdirectories), `language` (detected natural language)
- **subagents.csv**: `name` (agent name), `memory` (whether a memory file is configured), `language`
- **commands.csv**: `name` (command name), `language`
- **rules.csv**: `name` (rule name), `language`

## Compressed Archives

Large files are distributed as 7z archives:

| Archive | Size | Contents |
|---------|------|----------|
| `repos.csv.7z` | 7.2 MB | `repos.csv` (compressed from 37 MB) |
| `commits.csv.7z` | 26.7 MB | `commits.csv` (compressed from 151 MB) |
| `repos_data.7z` | 22.9 MB | `repos/` directory containing the collected configuration artifacts for 4,738 repositories (18,167 files). Each subdirectory is named `owner§name` (with `§` replacing `/`). |

Extract with: `7z x <archive> -y`

## Usage Examples

```python
import pandas as pd
import json

# Load the main dataset
repos = pd.read_csv("repos.csv")

# Filter to repos with detected configurations
configured_repos = repos[repos["scanned_at"].notna()]
print(f"Repos with configurations: {len(configured_repos)}")

# Filter to repos using Claude Code
claude_repos = repos[repos["claude"] == True]

# Filter to engineered projects (stored as lowercase string)
engineered = repos[repos["engineered_project"] == "true"]

# Parse the languages JSON column
repos["languages_parsed"] = repos["languages"].apply(json.loads)

# Join context files with repo metadata
context_files = pd.read_csv("context_files.csv")
merged = context_files.merge(repos[["repo_name", "mainLanguage", "stargazers"]], on="repo_name")
```

## License

This dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Disclaimer:** The configuration artifact files and commit messages included in this dataset were collected from public GitHub repositories. The content of individual files and commit messages remains subject to the license terms of their respective source repositories. The CC BY 4.0 license applies to the dataset compilation and metadata, not to the reproduced third-party content.
