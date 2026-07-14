//! Training-data generation: CSV export of labeled junction features.
pub mod generator;
pub mod ground_truth;

pub use generator::{GeneratorStats, TrainingDataConfig, TrainingDataGenerator};
pub use ground_truth::{find_ground_truth, GroundTruth};
