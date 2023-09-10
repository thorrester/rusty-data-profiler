use colored::Colorize;

fn main() {
    tracing_subscriber::fmt()
        .json()
        .with_target(false)
        .with_max_level(tracing::Level::TRACE)
        .with_current_span(false)
        .init();

    tracing::info!("hello");
}
