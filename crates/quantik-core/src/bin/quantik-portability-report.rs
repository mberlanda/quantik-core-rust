use clap::Parser;
use quantik_core::bench::portability::write_report;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "quantik-portability-report",
    about = "Emit a normalized Quantik API portability report"
)]
struct Cli {
    #[arg(long)]
    contracts_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match write_report(&cli.contracts_root, &cli.output) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("quantik-portability-report: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
