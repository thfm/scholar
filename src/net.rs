use nalgebra::DMatrix;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

pub struct NeuralNet {
    layers: Vec<DMatrix<f64>>,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
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
            biases: (1..num_layers)
                .map(|i| gen_random_matrix(node_counts[i], 1, &mut rng))
                .collect(),
            activation,
        }
    }

    // pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) {}

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

    fn backpropagate(&mut self, guess: Vec<f64>, target: Vec<f64>) {}
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
