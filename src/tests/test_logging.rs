#[test]
fn init_is_idempotent() {
    super::init_logging();
    super::init_logging();
}

#[test]
fn flush_is_safe_before_and_after_init() {
    // Pre-init (or twice-failed init in a sandbox): must be a no-op.
    super::flush_logging();
    super::init_logging();
    super::flush_logging();
}
