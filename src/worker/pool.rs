use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;

use crate::audio::pipeline::AudioLimits;
use crate::messaging::{Job, RabbitProducer};
use crate::metrics::Metrics;
use crate::shutdown::ShutdownSignal;
use crate::whisper::WhisperEngine;

use super::task;

/// Concurrent worker pool.
///
/// Owns a fixed set of async worker tasks. Each worker runs a
/// `tokio::task::spawn_blocking` bridge for the CPU-bound audio + inference
/// pipeline, keeping the async executor free for I/O.
///
/// # Architecture
///
/// ```text
/// jobs_rx (mpsc from RabbitConsumer)
///     │
///     │  dispatch loop — backpressure point
///     ▼
/// internal_channel  (bounded, capacity = workers × 2)
///     │
///     │  Arc<Mutex<Receiver>>  — shared among N worker tasks
///     ▼
/// Worker-0 ──► task::process ──► spawn_blocking (audio + whisper)
/// Worker-1 ──► task::process ──► spawn_blocking (audio + whisper)
/// ...
/// Worker-N ──► task::process ──► spawn_blocking (audio + whisper)
/// ```
///
/// # Backpressure
/// The internal channel capacity (`workers × 2`) limits how far ahead the
/// dispatch loop can run. When the channel is full (all workers busy plus
/// the buffer), `send().await` in the dispatch loop blocks, which in turn
/// prevents new `recv()` calls on `jobs_rx`. Because the RabbitMQ consumer
/// has `prefetch_count = workers_count`, the broker holds back new deliveries
/// until a worker ACKs one, completing the backpressure chain.
///
/// # Load distribution
/// All workers share a single `Arc<Mutex<mpsc::Receiver<Job>>>`. A free
/// worker locks the receiver, takes the next job, and immediately releases the
/// lock to process it. This is equivalent to Go's single buffered channel read
/// by N goroutines.
///
/// # Shutdown
/// When the caller drops (or closes) `jobs_rx`, the dispatch loop exits,
/// the internal sender is dropped, workers see `None` from `recv()` and stop.
/// `run()` awaits all handles before returning, ensuring in-flight jobs
/// complete gracefully — mirrors Go's `wg.Wait()`.
pub struct WorkerPool {
    engine: WhisperEngine,
    producer: RabbitProducer,
    limits: AudioLimits,
    workers_count: usize,
    metrics: Arc<Metrics>,
}

impl WorkerPool {
    /// Create a pool with the given shared resources and concurrency level.
    ///
    /// - `engine` — shared model; cloned cheaply (Arc) into each worker.
    /// - `producer` — shared channel; cloned cheaply (Arc) into each worker.
    /// - `limits` — `Copy` value; copied into each worker.
    /// - `workers_count` — mirrors Go's `numWorkers` / `WORKERS_COUNT` env var.
    pub fn new(
        engine: WhisperEngine,
        producer: RabbitProducer,
        limits: AudioLimits,
        workers_count: usize,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            engine,
            producer,
            limits,
            workers_count,
            metrics,
        }
    }

    /// Start processing and block until `jobs_rx` closes (shutdown).
    ///
    /// Call this from `app.rs` after the consumer is running.
    pub async fn run(self, mut jobs_rx: mpsc::Receiver<Job>, mut shutdown_signal: ShutdownSignal) {
        // ── Internal bounded channel ──────────────────────────────────────────
        // Capacity = workers × 2 mirrors Go's `make(chan rabbitmq.Job, numWorkers*2)`.
        let (internal_tx, internal_rx) =
            mpsc::channel::<Job>(self.workers_count * 2);

        // Wrap receiver in Arc<Mutex> so all workers can share it.
        // tokio::sync::Mutex is required because we await (recv) while holding it.
        let shared_rx: Arc<Mutex<mpsc::Receiver<Job>>> =
            Arc::new(Mutex::new(internal_rx));

        // ── Spawn N worker tasks ──────────────────────────────────────────────
        let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(self.workers_count);

        for worker_id in 0..self.workers_count {
            let rx = Arc::clone(&shared_rx);
            let engine = self.engine.clone();    // Arc increment — no model reload
            let producer = self.producer.clone(); // Arc increment — no reconnect
            let limits = self.limits;              // Copy — two primitives
            let metrics = Arc::clone(&self.metrics);

            let handle = tokio::spawn(async move {
                tracing::debug!(worker = worker_id, "worker started");

                loop {
                    // Acquire lock → receive one job → release lock → process.
                    //
                    // The lock is held only during the recv() await, not during
                    // the (potentially long) task::process(). This means at most
                    // one worker is blocked waiting for a new job at any time;
                    // all others are either processing or queued on the mutex.
                    // For small N (≤5) the contention is negligible.
                    let job = {
                        let mut guard = rx.lock().await;
                        guard.recv().await
                    };

                    match job {
                        None => {
                            // Internal sender was dropped → shutdown signal.
                            tracing::debug!(worker = worker_id, "worker stopping");
                            break;
                        }
                        Some(job) => {
                            task::process(
                                worker_id,
                                job,
                                engine.clone(),
                                producer.clone(),
                                limits,
                                Arc::clone(&metrics),
                            )
                            .await;
                        }
                    }
                }
            });

            handles.push(handle);
        }

        tracing::info!(workers = self.workers_count, "👷 {} workers ready", self.workers_count);

        // ── Dispatch loop ─────────────────────────────────────────────────────
        // Reads from the RabbitMQ consumer receiver and forwards to the internal
        // channel. This is the **backpressure point**: when the internal channel
        // is full, send().await blocks here, which stops us from calling
        // jobs_rx.recv() — leaving those unacked messages on the broker.
        //
        // `biased` ensures the shutdown branch is always checked before trying
        // to receive a new job, so a high-throughput stream cannot starve the
        // shutdown signal.
        //
        // Mirrors Go’s:
        //   go func() { for job := range jobs { workerPool.Submit(job) } }()
        loop {
            tokio::select! {
                biased;

                _ = shutdown_signal.wait() => {
                    tracing::info!("🛑 shutdown signal received, draining in-flight jobs...");
                    break;
                }

                job = jobs_rx.recv() => {
                    match job {
                        None => break,
                        Some(job) => {
                            self.metrics.inc_received();
                            if internal_tx.send(job).await.is_err() {
                                tracing::error!("internal job channel closed unexpectedly");
                                break;
                            }
                        }
                    }
                }
            }
        }

        // ── Graceful shutdown ─────────────────────────────────────────────────
        // Drop the sender: workers will drain their current job and then see
        // None on the next recv(), causing them to break and exit.
        drop(internal_tx);

        tracing::info!("🛑 draining {} in-flight workers...", handles.len());

        // Await all workers — mirrors Go's wg.Wait().
        for handle in handles {
            if let Err(e) = handle.await {
                tracing::error!(error = %e, "worker task panicked during shutdown");
            }
        }

        tracing::info!("all workers stopped");
    }
}
