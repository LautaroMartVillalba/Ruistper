use tokio::sync::watch;

/// Sender side held by the application orchestrator.
/// Drop it or call `trigger()` to broadcast shutdown to all listeners.
pub struct ShutdownHandle {
    tx: watch::Sender<bool>,
}

/// Receiver side distributed to subsystems that must honor shutdown.
/// Clone freely — each clone independently observes the signal.
#[derive(Clone)]
pub struct ShutdownSignal {
    rx: watch::Receiver<bool>,
}

/// Construct a linked handle/signal pair.
///
/// ```rust
/// let (handle, signal) = shutdown::new_pair();
/// ```
pub fn new_pair() -> (ShutdownHandle, ShutdownSignal) {
    let (tx, rx) = watch::channel(false);
    (ShutdownHandle { tx }, ShutdownSignal { rx })
}

impl ShutdownHandle {
    /// Broadcast the shutdown signal to all outstanding [`ShutdownSignal`] receivers.
    pub fn trigger(self) {
        // Errors only if all receivers have been dropped — harmless.
        let _ = self.tx.send(true);
    }
}

impl ShutdownSignal {
    /// Asynchronously wait until the shutdown signal has been triggered.
    /// Returns immediately if the signal was already triggered before this call.
    pub async fn wait(&mut self) {
        // `changed()` fires when the value changes.
        // `wait_for(|&v| v)` also resolves immediately if the channel already
        // holds `true`, which covers the "already triggered" case.
        let _ = self.rx.wait_for(|&v| v).await;
    }
}

/// Wait for `SIGINT` (Ctrl-C) or `SIGTERM` (container stop / kill).
///
/// This is intentionally a free function (not a method) so it can be called
/// once in `app::run()` without any prior state.
pub async fn wait_for_os_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigint = signal(SignalKind::interrupt()).expect("failed to register SIGINT handler");
    let mut sigterm = signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");

    tokio::select! {
        _ = sigint.recv()  => tracing::info!("🔔 SIGINT received"),
        _ = sigterm.recv() => tracing::info!("🔔 SIGTERM received"),
    }
}
