use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Application-wide runtime metrics.
///
/// All counters use `Relaxed` ordering — they are independent observations;
/// no cross-variable synchronisation is required.
///
/// Share via `Arc<Metrics>`. Cloning the `Arc` is the intended usage:
/// ```rust
/// let metrics = Arc::new(Metrics::new());
/// let m = Arc::clone(&metrics);
/// ```
pub struct Metrics {
    /// Total jobs consumed from RabbitMQ since startup.
    pub jobs_received: AtomicU64,

    /// Jobs that completed successfully (published to `whisper_results`).
    pub jobs_succeeded: AtomicU64,

    /// Jobs that exhausted all retries and published a final error result.
    pub jobs_failed: AtomicU64,

    /// Jobs that were sent to the retry queue at least once.
    /// A single job can contribute multiple counts if it retries repeatedly.
    pub jobs_retried: AtomicU64,

    /// Current number of jobs actively being processed (gauge).
    /// Incremented at task start, decremented at task end regardless of outcome.
    pub jobs_in_flight: AtomicI64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            jobs_received: AtomicU64::new(0),
            jobs_succeeded: AtomicU64::new(0),
            jobs_failed: AtomicU64::new(0),
            jobs_retried: AtomicU64::new(0),
            jobs_in_flight: AtomicI64::new(0),
        }
    }

    // ── Convenience increment methods ──────────────────────────────────────────

    pub fn inc_received(&self) {
        self.jobs_received.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_succeeded(&self) {
        self.jobs_succeeded.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_failed(&self) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_retried(&self) {
        self.jobs_retried.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_in_flight(&self) {
        self.jobs_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dec_in_flight(&self) {
        self.jobs_in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    // ── Snapshot ───────────────────────────────────────────────────────────────

    /// Return a consistent point-in-time snapshot of all counters.
    /// Because reads are `Relaxed`, the snapshot is approximate but
    /// sufficient for observability purposes.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            received:  self.jobs_received.load(Ordering::Relaxed),
            succeeded: self.jobs_succeeded.load(Ordering::Relaxed),
            failed:    self.jobs_failed.load(Ordering::Relaxed),
            retried:   self.jobs_retried.load(Ordering::Relaxed),
            in_flight: self.jobs_in_flight.load(Ordering::Relaxed),
        }
    }

    /// Log a summary of all metrics via `tracing`.
    pub fn log_summary(&self) {
        let s = self.snapshot();
        tracing::info!(
            received  = s.received,
            succeeded = s.succeeded,
            failed    = s.failed,
            retried   = s.retried,
            in_flight = s.in_flight,
            "📊 metrics summary"
        );
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time snapshot of [`Metrics`] counters.
#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    pub received:  u64,
    pub succeeded: u64,
    pub failed:    u64,
    pub retried:   u64,
    pub in_flight: i64,
}
