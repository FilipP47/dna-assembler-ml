//! Junction analysis: feature extraction and ML data collection.
pub mod collector;
pub mod features;
pub mod options;
pub mod traits;

pub use collector::MlDataCollector;
pub use features::{JunctionContext, JunctionFeatures};
pub use options::AnalysisOptions;
pub use traits::{JunctionAnalyzer, NoOpAnalyzer};