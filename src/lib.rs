#[derive(Debug)]
pub struct NeuralNetConfig {
    pub input_nodes: usize,
    pub hidden_nodes: Vec<usize>,
    pub output_nodes: usize,
    pub random_weights: bool,
}

impl NeuralNetConfig {
    pub fn new(input_nodes: usize, hidden_nodes: Vec<usize>, output_nodes: usize) -> Self {
        Self {
            input_nodes,
            hidden_nodes,
            output_nodes,
            random_weights: true,
        }
    }
}

use nalgebra::DMatrix;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

pub struct NeuralNet {
    pub net_config: NeuralNetConfig,
}

impl NeuralNet {
    pub fn new(net_config: NeuralNetConfig) -> Self {
        Self { net_config }
    }

    fn gen_random_weight_matrix(
        &self,
        rows: usize,
        cols: usize,
        rng: &mut impl Rng,
    ) -> DMatrix<f32> {
        let elements = rows * cols;
        let range = Uniform::new_inclusive(-1.0, 1.0);
        DMatrix::from_iterator(rows, cols, (0..elements).map(|_| range.sample(rng)))
    }
}
