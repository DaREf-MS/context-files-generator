import os
import subprocess
import shutil
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from dotenv import load_dotenv
from logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

# --- Taxonomies ---
BUILD_FILES_TAXONOMY = [
    "Makefile", "build.gradle", "pom.xml", "CMakeLists.txt", 
    "setup.py", "package.json", "Cargo.toml", "mix.exs", "*.csproj"
]

IAC_FILES_TAXONOMY = [
    "*.tf", "*.tfvars", "docker-compose.yml", "Dockerfile", 
    "Vagrantfile", "Chart.yaml", "*.k8s.yaml", "ansible*.yml"
]

LINTER_CONFIG_TAXONOMY = [
    ".eslintrc*", ".prettierrc*", "tox.ini", ".flake8", 
    "pylintrc", ".stylelintrc*", "tslint.json", ".rubocop.yml"
]

DEPENDENCY_FILES_TAXONOMY = [
    "package.json", "requirements.txt", "pom.xml", "build.gradle", 
    "Cargo.toml", "go.mod", "Gemfile", "pyproject.toml", "composer.json"
]

class RepoMetricsExtractor:
    """
    OOP Class to compute repository metrics using git, grep, and find.
    """
    def __init__(self, repo: str, first_acf_date: str, clone_dir: Path):
        # Handle both "owner/repo" format and full URLs
        self.repo = repo
        self.repo_url = f"https://github.com/{repo}.git" if not repo.startswith("http") else repo
        self.first_acf_date = first_acf_date
        self.clone_dir = clone_dir
        self.repo_path = self.clone_dir / self.repo_url.split('/')[-1].replace('.git', '')
        
        self.target_commit = None # Will be resolved dynamically
        self.total_loc = 0 # Cached LoC for density calculations
        
        # Hardcoded GitHub token as requested (can also be overridden by env var)
        self.github_token = os.getenv("GITHUB_TOKEN", "TOKEN_1_JAPAN_PROJECT")

    def _run_cmd(self, cmd: str, cwd: Path = None, shell: bool = True) -> str:
        """Helper to run shell commands safely and return standard output."""
        try:
            target_cwd = cwd if cwd else self.repo_path
            result = subprocess.check_output(cmd, shell=shell, cwd=target_cwd, stderr=subprocess.DEVNULL, text=True)
            return result.strip()
        except subprocess.CalledProcessError:
            return ""
        except FileNotFoundError:
            return ""

    def clone_and_checkout(self) -> bool:
        """Clones the repository, finds the commit before first_acf_date, and checks it out."""
        auth_url = self.repo_url
        
        # Inject token into URL if available to bypass rate limits and access private repos
        if self.github_token and "https://github.com" in self.repo_url and "@" not in self.repo_url:
            auth_url = self.repo_url.replace("https://github.com/", f"https://{self.github_token}@github.com/")

        # 1. Clone or Fetch
        if not self.repo_path.exists():
            clone_cmd = f"git clone {auth_url} {self.repo_path}"
            if subprocess.run(clone_cmd, shell=True, stderr=subprocess.DEVNULL).returncode != 0:
                print(f"\nFailed to clone {self.repo_url}")
                return False
        else:
            # If repo exists, fetch the latest to ensure we have full history
            self._run_cmd("git fetch --all", cwd=self.repo_path)

        # 2. Resolve target commit based on the date
        find_commit_cmd = f"git rev-list -1 --before='{self.first_acf_date}' --all"
        resolved_commit = self._run_cmd(find_commit_cmd, cwd=self.repo_path)
        
        if not resolved_commit:
            print(f"\nNo commit found before {self.first_acf_date} for {self.repo_url}")
            return False
            
        self.target_commit = resolved_commit

        # 3. Checkout the target commit
        checkout_cmd = f"git checkout --force {self.target_commit}"
        if subprocess.run(checkout_cmd, shell=True, cwd=self.repo_path, stderr=subprocess.DEVNULL).returncode != 0:
            print(f"\nFailed to checkout commit {self.target_commit} in {self.repo_url}")
            return False
            
        return True

    # --- 1. Build Scripts ---
    def compute_build_scripts(self) -> dict:
        names = " -o ".join([f"-name '{f}'" for f in BUILD_FILES_TAXONOMY])
        find_cmd = f"find . -type f \\( {names} \\) | wc -l"
        count = int(self._run_cmd(find_cmd) or 0)
        
        # Frequency of modifications
        freq_cmd = f"git log --oneline --name-only | grep -E '{'|'.join([f.replace('*.', '') for f in BUILD_FILES_TAXONOMY])}' | wc -l"
        mod_freq = int(self._run_cmd(freq_cmd) or 0)
        
        return {"build_scripts_count": count, "build_scripts_mod_freq": mod_freq}

    # --- 2. Test Files ---
    def compute_test_files(self) -> dict:
        total_files_cmd = "git ls-files | wc -l"
        test_files_cmd = "git ls-files | grep -i 'test' | wc -l"
        
        total_files = int(self._run_cmd(total_files_cmd) or 1) # avoid div by zero
        test_files = int(self._run_cmd(test_files_cmd) or 0)
        
        freq_cmd = "git log --oneline --name-only | grep -i 'test' | wc -l"
        mod_freq = int(self._run_cmd(freq_cmd) or 0)
        
        return {
            "test_file_ratio": test_files / total_files, 
            "test_scripts_mod_freq": mod_freq
        }

    # --- 3. Markdown / RST files ---
    def compute_doc_files(self) -> dict:
        doc_files_cmd = "find . -type f \\( -name '*.md' -o -name '*.rst' \\) | wc -l"
        doc_count = int(self._run_cmd(doc_files_cmd) or 0)
        
        depth_cmd = "find . -type f \\( -name '*.md' -o -name '*.rst' \\) | awk -F'/' '{print NF-1}' | sort -nr | head -n 1"
        max_depth = int(self._run_cmd(depth_cmd) or 0)
        
        doc_changes_cmd = r"git log --oneline --name-only | grep -E '\.md$|\.rst$' | wc -l"
        all_changes_cmd = "git log --oneline --name-only | wc -l"
        
        doc_changes = int(self._run_cmd(doc_changes_cmd) or 0)
        all_changes = int(self._run_cmd(all_changes_cmd) or 1)
        
        return {
            "doc_file_count": doc_count, 
            "doc_max_nesting": max_depth,
            "doc_change_ratio": doc_changes / all_changes
        }

    # --- 4. Linter/Config Files ---
    def compute_linter_configs(self) -> dict:
        names = " -o ".join([f"-name '{f}'" for f in LINTER_CONFIG_TAXONOMY])
        find_cmd = f"find . -type f \\( {names} \\)"
        files = self._run_cmd(find_cmd).splitlines()
        
        size = 0
        for f in files:
            size += int(self._run_cmd(f"wc -c < '{f}'") or 0)
            
        return {"linter_config_count": len(files), "linter_config_total_bytes": size}

    # --- 5. Directory Hierarchy ---
    def compute_directory_depth(self) -> dict:
        depths_cmd = "find . -type d | awk -F'/' '{print NF-1}'"
        depths_raw = self._run_cmd(depths_cmd).splitlines()
        depths = [int(d) for d in depths_raw if d.isdigit()]
        
        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths)/len(depths) if depths else 0.0
        
        return {"dir_max_depth": max_depth, "dir_avg_depth": avg_depth}

    # --- 6. External Dependencies ---
    def compute_dependencies(self) -> dict:
        names = " -o ".join([f"-name '{f}'" for f in DEPENDENCY_FILES_TAXONOMY])
        find_cmd = f"find . -type f \\( {names} \\)"
        files = self._run_cmd(find_cmd).splitlines()
        
        dep_lines = 0
        for f in files:
            dep_lines += int(self._run_cmd(f"wc -l < '{f}'") or 0)
            
        total_commits = int(self._run_cmd("git rev-list --count HEAD") or 1)
        dep_mod_cmd = f"git log --oneline --name-only | grep -E '{'|'.join([f.replace('*.', '') for f in DEPENDENCY_FILES_TAXONOMY])}' | wc -l"
        dep_mods = int(self._run_cmd(dep_mod_cmd) or 0)
        
        return {
            "dependency_files_count": len(files), 
            "dependency_proxy_size": dep_lines,
            "dependency_mod_ratio": dep_mods / total_commits
        }

    # --- 7. Total LoC ---
    def compute_loc(self) -> dict:
        loc_cmd = "git ls-files | xargs wc -l | tail -n 1 | awk '{print $1}'"
        self.total_loc = int(self._run_cmd(loc_cmd) or 1) # Store for density metrics
        return {"total_loc": self.total_loc}

    # --- 8. Env Variables ---
    def compute_env_templates(self) -> dict:
        cmd = "find . -type f \\( -name '*.env*' -o -name '*config.yaml' \\) | wc -l"
        count = int(self._run_cmd(cmd) or 0)
        return {"env_templates_count": count}

    # --- 9. Git Hooks ---
    def compute_git_hooks(self) -> dict:
        hook_files_cmd = "find .git/hooks -type f 2>/dev/null | wc -l"
        husky_cmd = "find .husky -type f 2>/dev/null | wc -l"
        
        hooks = int(self._run_cmd(hook_files_cmd) or 0) + int(self._run_cmd(husky_cmd) or 0)
        return {"git_hooks_count": hooks}

    # --- 10. Blast Radius & Diversity ---
    def compute_blast_radius(self) -> dict:
        blast_cmd = "git log --numstat --pretty='%H' | awk 'NF==3 {files++} NF==1 {if (files) print files; files=0} END {if (files) print files}' | awk '{sum+=$1} END {print sum/NR}'"
        avg_blast = float(self._run_cmd(blast_cmd) or 0.0)
        
        div_cmd = "git ls-files | awk -F. '{if (NF>1) print $NF}' | sort -u | wc -l"
        diversity = int(self._run_cmd(div_cmd) or 0)
        
        return {"avg_blast_radius_files": avg_blast, "file_extension_diversity": diversity}

    # --- 11. PR Templates ---
    def compute_pr_templates(self) -> dict:
        cmd = "find .github -iname '*PULL_REQUEST_TEMPLATE*' -type f | wc -l"
        count = int(self._run_cmd(cmd) or 0)
        return {"pr_templates_count": count}

    # --- 12. Try-Catch Density ---
    def compute_exception_density(self) -> dict:
        cmd = r"grep -rIE 'try\s*\{|catch\s*\(|except ' . | wc -l"
        count = int(self._run_cmd(cmd) or 0)
        kloc = (self.total_loc / 1000) or 1
        return {"exception_density_per_kloc": count / kloc}

    # --- 13. Logging Density ---
    def compute_logging_density(self) -> dict:
        cmd = r"grep -rIE 'logger\.|console\.|log\.|logging\.' . | wc -l"
        count = int(self._run_cmd(cmd) or 0)
        kloc = (self.total_loc / 1000) or 1
        return {"logging_density_per_kloc": count / kloc}

    # --- 14. SQL Files ---
    def compute_sql_files(self) -> dict:
        cmd = "find . -type f -name '*.sql' | wc -l"
        count = int(self._run_cmd(cmd) or 0)
        return {"sql_files_count": count}

    # --- 15. Compilation Flags ---
    def compute_platform_flags(self) -> dict:
        flag_cmd = "grep -rIE '#ifdef|#ifndef|#if defined' . | wc -l"
        dir_cmd = "find . -type d \\( -name 'win32' -o -name 'posix' -o -name 'linux' -o -name 'darwin' \\) | wc -l"
        
        flags = int(self._run_cmd(flag_cmd) or 0)
        dirs = int(self._run_cmd(dir_cmd) or 0)
        kloc = (self.total_loc / 1000) or 1
        
        return {"platform_flags_density_per_kloc": flags / kloc, "platform_dirs_count": dirs}

    # --- 16. IaC Manifests ---
    def compute_iac_files(self) -> dict:
        names = " -o ".join([f"-name '{f}'" for f in IAC_FILES_TAXONOMY])
        cmd = f"find . -type f \\( {names} \\) | wc -l"
        count = int(self._run_cmd(cmd) or 0)
        return {"iac_files_count": count}

    # --- 17. Workflow Files ---
    def compute_workflows(self) -> dict:
        cmd = "find .github/workflows -type f 2>/dev/null | wc -l"
        count = int(self._run_cmd(cmd) or 0)
        return {"github_workflows_count": count}

    # --- 18. Changelog Churn ---
    def compute_changelog_churn(self) -> dict:
        cmd = "git log --oneline -- CHANGELOG.md 2>/dev/null | wc -l"
        count = int(self._run_cmd(cmd) or 0)
        return {"changelog_churn_commits": count}

    # --- 19. Translation Files ---
    def compute_translation_volume(self) -> dict:
        cmd = "find . -type f \\( -name '*.po' -o -name '*.mo' -o -name '*i18n*.json' \\) | xargs wc -c 2>/dev/null | tail -n 1 | awk '{print $1}'"
        size = int(self._run_cmd(cmd) or 0)
        return {"translation_files_bytes": size}

    def run_all_metrics(self) -> Dict[str, Any]:
        """Executes all metric computation functions and aggregates results."""
        metrics = {}
        
        # NOTE: compute_loc must run first to populate self.total_loc for density metrics
        metrics.update(self.compute_loc())
        
        metrics.update(self.compute_build_scripts())
        metrics.update(self.compute_test_files())
        metrics.update(self.compute_doc_files())
        metrics.update(self.compute_linter_configs())
        metrics.update(self.compute_directory_depth())
        metrics.update(self.compute_dependencies())
        metrics.update(self.compute_env_templates())
        metrics.update(self.compute_git_hooks())
        metrics.update(self.compute_blast_radius())
        metrics.update(self.compute_pr_templates())
        metrics.update(self.compute_exception_density())
        metrics.update(self.compute_logging_density())
        metrics.update(self.compute_sql_files())
        metrics.update(self.compute_platform_flags())
        metrics.update(self.compute_iac_files())
        metrics.update(self.compute_workflows())
        metrics.update(self.compute_changelog_churn())
        metrics.update(self.compute_translation_volume())
        
        return metrics

    def cleanup(self):
        """Removes the cloned repository to save disk space."""
        if self.repo_path.exists():
            shutil.rmtree(self.repo_path, ignore_errors=True)


# --- Main Execution Loop ---
def process_repositories(input_csv: str, output_csv: str, clone_dir: Path):
    """
    Reads target repositories, extracts metrics, tracks progress, and rate limits.
    Assumes input_csv has columns: 'repo', 'first_acf_date'
    """
    clone_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    df_targets = pd.read_csv(input_csv)
    
    # Check for existing progress to resume safely
    processed_repos = set()
    if Path(output_csv).exists():
        df_existing = pd.read_csv(output_csv)
        if 'repo' in df_existing.columns:
            processed_repos = set(df_existing['repo'].tolist())
            print(f"Resuming: Found {len(processed_repos)} already processed repositories.")

    results = []
    
    with tqdm(total=len(df_targets), desc="Mining Repositories") as pbar:
        for _, row in df_targets.iterrows():
            repo = row['repo']
            first_acf_date = row['first_acf_date']
            
            if repo in processed_repos:
                pbar.update(1)
                continue
                
            analyzer = RepoMetricsExtractor(repo, first_acf_date, clone_dir)
            
            pbar.set_postfix_str(f"Resolving {repo}...")
            if analyzer.clone_and_checkout():
                pbar.set_postfix_str("Computing metrics...")
                repo_metrics = analyzer.run_all_metrics()
                
                # Metadata for tracking
                repo_metrics['repo'] = repo
                repo_metrics['first_acf_date'] = first_acf_date
                repo_metrics['resolved_commit'] = analyzer.target_commit 
                
                results.append(repo_metrics)
                
                # Save progressively to avoid data loss
                df_results = pd.DataFrame([repo_metrics])
                df_results.to_csv(output_csv, mode='a', header=not Path(output_csv).exists(), index=False)
            
            # Uncomment the line below to clean up repos as you go (saves disk space)
            analyzer.cleanup() 
            
            # Rate limiting
            time.sleep(2) 
            pbar.update(1)

    print(f"\nCompleted! Metrics saved to {output_csv}")

if __name__ == "__main__":
    # Example execution paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    CLONE_WORKSPACE = ROOT_DIR / "workspace_clones"
    
    INPUT_DATASET = DATA_DIR / "pre_acf_model_features.csv"
    OUTPUT_METRICS = DATA_DIR / "extracted_repo_metrics.csv"
    
    # Mock creation of input file for testing if it doesn't exist
    if not INPUT_DATASET.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"repo": "pallets/flask", "first_acf_date": "2023-10-13T00:00:00Z"},
            {"repo": "psf/requests", "first_acf_date": "2023-11-01T12:00:00Z"}
        ]).to_csv(INPUT_DATASET, index=False)
        print(f"Created sample input file at {INPUT_DATASET}")
    
    process_repositories(str(INPUT_DATASET), str(OUTPUT_METRICS), CLONE_WORKSPACE)