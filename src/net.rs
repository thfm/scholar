use crate::dataset::Dataset;
use nalgebra::DMatrix;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

pub struct NeuralNet {
    layers: Vec<DMatrix<f64>>,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
    errors: Vec<DMatrix<f64>>,
    activation: Activation,
}

impl NeuralNet {
    pub fn new(node_counts: &[usize], activation: Activation) -> Self {
        let num_layers = node_counts.len();
        if num_layers < 2 {
            panic!(
                "not enough layers supplied (expected at least 2, found {})",
                num_layers
            );
        }

        let mut rng = rand::thread_rng();
        Self {
            layers: node_counts.iter().map(|c| DMatrix::zeros(*c, 1)).collect(),
            weights: (1..num_layers)
                .map(|i| gen_random_matrix(node_counts[i], node_counts[i - 1], &mut rng))
                .collect(),
            biases: node_counts
                .iter()
                .skip(1)
                .map(|c| gen_random_matrix(*c, 1, &mut rng))
                .collect(),
            errors: node_counts
                .iter()
                .skip(1)
                .map(|c| DMatrix::zeros(*c, 1))
                .collect(),
            activation,
        }
    }

    pub fn learn(&mut self, dataset: &Dataset, iterations: usize, learning_rate: f64) {
        for _ in 1..iterations {
            for (inputs, targets) in dataset {
                let guesses = self.guess(inputs);
                self.backpropagate(&guesses, targets, learning_rate);
            }
        }
    }

    pub fn guess(&mut self, inputs: &[f64]) -> Vec<f64> {
        let num_inputs = inputs.len();
        let num_input_layer_rows = self.layers[0].row_iter().len();
        if num_inputs != num_input_layer_rows {
            panic!(
                "incorrect number of inputs supplied (expected {}, found {})",
                num_input_layer_rows, num_inputs
            );
        }

        let num_layers = self.layers.len();
        self.layers[0] = convert_slice_to_matrix(inputs);

        for i in 1..num_layers {
            let mut value = &self.weights[i - 1] * &self.layers[i - 1];
            value += &self.biases[i - 1];

            for x in value.iter_mut() {
                *x = (self.activation.function)(*x);
            }

            self.layers[i] = value;
        }

        self.layers[num_layers - 1].iter().cloned().collect()
    }

    fn backpropagate(&mut self, guesses: &[f64], targets: &[f64], learning_rate: f64) {
        let guesses = convert_slice_to_matrix(guesses);
        let targets = convert_slice_to_matrix(targets);

        let num_layers = self.layers.len();
        self.errors[num_layers - 2] = targets - guesses;

        for (i, layer) in self.layers.iter().enumerate().skip(1).rev() {
            let mut gradients = layer.map(self.activation.derivative);
            gradients.component_mul_assign(&self.errors[i - 1]);
            gradients *= learning_rate;

            let deltas = &gradients * self.layers[i - 1].transpose();
            self.weights[i - 1] += deltas;

            self.biases[i - 1] += gradients;

            if i != 1 {
                self.errors[i - 2] = self.weights[i - 1].transpose() * &self.errors[i - 1];
            }
        }
    }
}

pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

// TODO: Derivative function assumes that the original function
// has already been applied
pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + (-x).exp()),
    derivative: |x| x * (1.0 - x),
};

fn gen_random_matrix(rows: usize, cols: usize, rng: &mut impl Rng) -> DMatrix<f64> {
    let elements = rows * cols;
    let range = Uniform::new_inclusive(-1.0, 1.0);
    DMatrix::from_iterator(rows, cols, (0..elements).map(|_| range.sample(rng)))
}

fn convert_slice_to_matrix(slice: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(slice.len(), 1, slice)
}
