use nalgebra::DMatrix;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

pub struct NeuralNet {
    layers: Vec<DMatrix<f64>>,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
    activation: fn(f64) -> f64,
}

impl NeuralNet {
    // TODO: Add check to see if more than 1 layer was supplied
    pub fn new(node_counts: Vec<usize>, activation: fn(f64) -> f64, rng: &mut impl Rng) -> Self {
        Self {
            layers: node_counts.iter().map(|c| DMatrix::zeros(*c, 1)).collect(),
            weights: (0..node_counts.len() - 1)
                .map(|i| gen_random_matrix(node_counts[i + 1], node_counts[i], rng))
                .collect(),
            biases: (1..node_counts.len())
                .map(|i| gen_random_matrix(node_counts[i], 1, rng))
                .collect(),
            activation,
        }
    }

    // pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) {}

    // TODO: Check if the inputs length does not match up with the amount of rows
    // in the input layer DMatrix
    pub fn guess(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let num_layers = self.layers.len();
        self.layers[0] = DMatrix::from_row_slice(inputs.len(), 1, &inputs);

        for i in 1..num_layers {
            let mut value = &self.weights[i - 1] * &self.layers[i - 1];
            value += &self.biases[i - 1];

            for x in value.iter_mut() {
                *x = (self.activation)(*x);
            }

            self.layers[i] = value;
        }

        self.layers[num_layers - 1].iter().cloned().collect()
    }

    fn backpropagate(&mut self, guess: Vec<f64>, target: Vec<f64>) {}
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn gen_random_matrix(rows: usize, cols: usize, rng: &mut impl Rng) -> DMatrix<f64> {
    let elements = rows * cols;
    let range = Uniform::new_inclusive(-1.0, 1.0);
    DMatrix::from_iterator(rows, cols, (0..elements).map(|_| range.sample(rng)))
}
