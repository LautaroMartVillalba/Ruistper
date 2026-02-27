/// Maximum number of retry attempts before a job is considered permanently failed.
///
/// `2` retries = `3` total execution attempts.
/// This is the single source of truth for the limit.
///
/// The broker-side delay between attempts is configured separately in the
/// queue topology (`x-message-ttl = 5 000 ms` in `messaging::rabbit`).
pub const MAX_RETRIES: i32 = 2;

// ── Decision ───────────────────────────────────────────────────────────────────

/// Outcome of a retry policy evaluation for a failed job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryDecision {
    /// The job should be re-queued via the retry exchange.
    /// The broker will hold it for `RETRY_TTL_MS` before routing it back
    /// to the main queue.
    Retry {
        /// The `retry_count` value that will be embedded in the next attempt's message.
        next_attempt: i32,
    },

    /// All attempts exhausted. A final error result must be published.
    GiveUp,
}

// ── Policy ─────────────────────────────────────────────────────────────────────

/// Retry policy for transcription jobs.
///
/// `Copy` so it can be passed freely to worker tasks and closures without cloning.
///
/// # Usage
/// ```rust
/// use crate::retry::{RetryPolicy, RetryDecision};
///
/// let policy = RetryPolicy::default(); // max_retries = MAX_RETRIES (2)
///
/// match policy.decide(job.retry_count) {
///     RetryDecision::Retry { next_attempt } => { /* publish to retry exchange */ }
///     RetryDecision::GiveUp               => { /* publish final error */ }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (not total attempts).
    /// A value of `2` allows up to 3 total executions: original + 2 retries.
    pub max_retries: i32,
}

impl Default for RetryPolicy {
    /// Returns a policy using the project-wide [`MAX_RETRIES`] constant.
    fn default() -> Self {
        Self {
            max_retries: MAX_RETRIES,
        }
    }
}

impl RetryPolicy {
    /// Create a policy with a custom retry limit.
    pub fn new(max_retries: i32) -> Self {
        Self { max_retries }
    }

    /// Decide what to do with a job that has just failed.
    ///
    /// `retry_count` is the number of times the job has **already been attempted**
    /// (0 = first attempt, never retried).
    ///
    /// # Mapping to Go
    /// ```go
    /// // producer.go
    /// func ShouldRetry(retryCount int) bool { return retryCount < MaxRetries }
    ///
    /// // pool.go — handleFailure
    /// if rabbitmq.ShouldRetry(request.RetryCount) { PublishRetry; Ack }
    /// else                                         { PublishError; Ack }
    /// ```
    pub fn decide(&self, retry_count: i32) -> RetryDecision {
        if retry_count < self.max_retries {
            RetryDecision::Retry {
                next_attempt: retry_count + 1,
            }
        } else {
            RetryDecision::GiveUp
        }
    }

    /// Convenience boolean wrapper over [`Self::decide`].
    ///
    /// Returns `true` while `retry_count < max_retries`.
    /// Equivalent to Go's `ShouldRetry`.
    #[inline]
    pub fn should_retry(&self, retry_count: i32) -> bool {
        retry_count < self.max_retries
    }

    /// Number of attempts remaining for a job at the given `retry_count`.
    /// Returns `0` when `retry_count >= max_retries`.
    #[inline]
    pub fn attempts_remaining(&self, retry_count: i32) -> i32 {
        (self.max_retries - retry_count).max(0)
    }
}
