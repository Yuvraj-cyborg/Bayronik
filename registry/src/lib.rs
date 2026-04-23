//! Bayronik model registry: types, builder CLI, frozen-threshold tests.
//!
//! No `tch` / `axum` / numpy here, so:
//!   * the `registry` CLI rebuilds metadata without libtorch,
//!   * `cargo test -p registry` runs the regression suite in seconds.

mod model;
pub mod report;
pub mod sha;

pub use model::{
    FrozenMetrics, ModelRegistry, SplitMetrics, default_conditions, default_conditions_units,
    default_limitations,
};
