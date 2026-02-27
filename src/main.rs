mod app;
mod audio;
mod config;
mod messaging;
mod metrics;
mod model;
mod retry;
mod shutdown;
mod whisper;
mod worker;

#[tokio::main]
async fn main() {
    // ── Tracing / structured logging ──────────────────────────────────────────
    // Default level = INFO for this crate, WARN for everything else.
    // Override at runtime via the RUST_LOG environment variable:
    //   RUST_LOG=whisperust=debug,lapin=warn cargo run
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("whisperust=info,warn"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)        // omit module path from each log line
        .with_thread_ids(false)
        .init();

    // ── Run ───────────────────────────────────────────────────────────────────
    if let Err(e) = app::run().await {
        tracing::error!("❌ fatal: {e}");
        std::process::exit(1);
    }
}
