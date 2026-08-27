//! Python bindings via PyO3

use pyo3::prelude::*;

/// Python module
///
/// PyO3 0.21 moved module and object arguments to the `Bound<'py, T>` smart
/// pointer; the older `&PyModule` form no longer implements the conversion the
/// `#[pymodule]` macro needs, so this signature is not stylistic.
#[pymodule]
fn compressr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // TODO: Add Python bindings for Model, QuantizationConfig, etc.

    Ok(())
}

// TODO: Implement Python API classes
// - Model (lazy loading)
// - QuantizationStrategy
// - CompressedModel
