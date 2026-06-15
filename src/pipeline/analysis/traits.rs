//! `JunctionAnalyzer` — strategy interface for collecting junction data.
//!
//! The assembler calls `analyze` every time it reaches a branching node.
//! Implementations decide what to record and how to store it.
//! A `NoOpAnalyzer` is provided for the `--no-analyze` path with zero overhead.

use std::path::Path;
use anyhow::Result;
use super::features::{JunctionContext, JunctionFeatures};

/// Strategy interface for junction analysis.
pub trait JunctionAnalyzer {
    /// Called by the assembler each time it reaches a branching node.
    ///
    /// Implementations should compute features from `ctx` and store them.
    fn analyze(&mut self, ctx: &JunctionContext) -> JunctionFeatures;

    /// Write all collected features to `path` (e.g. CSV).
    fn export(&self, path: &Path) -> Result<()>;

    /// Number of junctions recorded so far.
    fn junction_count(&self) -> usize;
}

/// A zero-overhead placeholder used when `--analyze` is not passed.
///
/// Satisfies the type system without allocating anything.
pub struct NoOpAnalyzer;

impl JunctionAnalyzer for NoOpAnalyzer {
    fn analyze(&mut self, ctx: &JunctionContext) -> JunctionFeatures {
        JunctionFeatures {
            depth_in_contig: ctx.depth_in_contig,
            out_degree: ctx.out_degree,
            in_degree: ctx.in_degree,
            ..Default::default()
        }
    }

    fn export(&self, _path: &Path) -> Result<()> {
        Ok(())
    }

    fn junction_count(&self) -> usize {
        0
    }
}
