use clap::Parser;
use quantik_core::bench::portability::build_report;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "quantik-portability-report",
    about = "Emit a normalized Quantik API portability report"
)]
struct Cli {
    #[arg(long)]
    contracts_root: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();

    // Resolve contracts_root with default
    let contracts_root = match &cli.contracts_root {
        Some(path) => path.clone(),
        None => {
            // Default to ../quantik-core-contracts relative to the repository root
            let repo_root = find_repo_root();
            repo_root.join("../quantik-core-contracts")
        }
    };

    // Verify contracts_root exists
    if !contracts_root.exists() {
        eprintln!(
            "quantik-portability-report: contracts root not found at {}; use --contracts-root to specify",
            contracts_root.display()
        );
        return std::process::ExitCode::FAILURE;
    }

    // Build the report
    let report = match build_report(&contracts_root) {
        Ok(r) => r,
        Err(error) => {
            eprintln!("quantik-portability-report: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };

    // Write output
    let text = match serde_json::to_string_pretty(&report) {
        Ok(t) => t,
        Err(error) => {
            eprintln!("quantik-portability-report: serialize report: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };

    match &cli.output {
        Some(path) => {
            // Write to file
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        eprintln!("quantik-portability-report: create output directory: {e}");
                        return std::process::ExitCode::FAILURE;
                    }
                }
            }
            if let Err(e) = std::fs::write(path, format!("{text}\n")) {
                eprintln!("quantik-portability-report: write report: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
        None => {
            // Write to stdout
            if let Err(e) = writeln!(std::io::stdout(), "{text}") {
                eprintln!("quantik-portability-report: write to stdout: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    std::process::ExitCode::SUCCESS
}

/// Find the repository root using git, following worktree links
fn find_repo_root() -> PathBuf {
    use std::process::Command;

    // Try to use git to find the repository root
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
    {
        if output.status.success() {
            if let Ok(path) = String::from_utf8(output.stdout) {
                let toplevel = PathBuf::from(path.trim());

                // If we're in a git worktree, the .git file points to the actual repo
                // We need to extract the actual repo path from the worktree info
                let git_file = toplevel.join(".git");
                if let Ok(content) = std::fs::read_to_string(&git_file) {
                    // Content looks like: "gitdir: /path/to/repo/.git/worktrees/wt-name"
                    if let Some(gitdir) = content.strip_prefix("gitdir: ") {
                        let gitdir = gitdir.trim();
                        if gitdir.contains(".git/worktrees/") {
                            // Extract the repo root from the gitdir path
                            if let Some(git_idx) = gitdir.find(".git/worktrees/") {
                                let repo_git = &gitdir[..git_idx + 4]; // Include ".git"
                                if let Some(parent) = PathBuf::from(repo_git).parent() {
                                    return parent.to_path_buf();
                                }
                            }
                        }
                    }
                }

                return toplevel;
            }
        }
    }

    // Fallback: search upwards from current directory
    let mut current = std::env::current_dir().unwrap_or_default();
    loop {
        // Check if we're at the root (contains Cargo.toml and a crates/ directory)
        if current.join("Cargo.toml").exists() && current.join("crates").exists() {
            return current;
        }

        // Move up one directory
        if !current.pop() {
            // We've reached the filesystem root without finding a repo root
            // Return the current directory as fallback
            return std::env::current_dir().unwrap_or_default();
        }
    }
}
