#[test]
fn init_is_idempotent() {
    super::init_logging();
    super::init_logging();
}
