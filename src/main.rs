fn main() {
    use tracing_subscriber::{fmt::format, prelude::*};

    // Format fields using the provided closure.
    let format = format::debug_fn(|writer, field, value| {
        // We'll format the field name and value separated with a colon.
        write!(writer, "{}: {:?}", field, value)
    })
    // Separate each field with a comma.
    // This method is provided by an extension trait in the
    // `tracing-subscriber` prelude.
    .delimited(", ");

    // Create a `fmt` subscriber that uses our custom event format, and set it
    // as the default.
    tracing_subscriber::fmt()
        .fmt_fields(format)
        .json()
        .with_current_span(false)
        .init();

    // Shave some yaks!
    let number_of_yaks = 3;
    // this creates a new event, outside of any spans.
    tracing::info!(number_of_yaks, "preparing to shave yaks");
}
