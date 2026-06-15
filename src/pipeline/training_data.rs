//! Training-data generation: CSV export of labeled junction features.
pub mod generator;
pub mod ground_truth;

pub use generator::{TrainingDataConfig, TrainingDataGenerator, GeneratorStats};
pub use ground_truth::{GroundTruth, find_ground_truth};