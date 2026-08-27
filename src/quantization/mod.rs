//! Quantization methods and utilities
//!
//! This module provides a user-friendly interface for model quantization.
//! It is a thin orchestration layer - ALL quantization algorithms live in boostr.
//!
//! # Architecture
//!
//! ```text
//! compressr (this crate)           boostr (ML framework)
//! ┌────────────────────┐           ┌─────────────────────────────┐
//! │ QuantizationConfig │──────────>│ quant::                     │
//! │ Quantizer          │           │   quantize()                │
//! │ QuantizedTensor    │           │   dequantize()              │
//! │ QuantType          │           │   QuantFormat, QuantConfig  │
//! └────────────────────┘           │   Backend dispatch          │
//!                                  └─────────────────────────────┘
//! ```
//!
//! # What lives here (thin layer)
//!
//! - CLI/API configuration types
//! - Quantizer orchestrator (dispatches to boostr)
//! - QuantizedTensor for format writing
//!
//! # What lives in boostr (heavy lifting)
//!
//! - All quantization algorithms (NF4, GPTQ, AWQ, K-quants)
//! - SIMD-optimized dequantization (AVX2, NEON)
//! - GPU kernels for quantization
//! - Block structures (BlockQ4K, BlockQ5K, etc.)

mod config;
mod quantizer;
mod tensor;
mod types;

// Re-export main types
pub use config::{QuantizationConfig, QuantizationMethod};
pub use quantizer::Quantizer;
pub use tensor::QuantizedTensor;
pub use types::QuantType;
