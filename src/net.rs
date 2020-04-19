use crate::dataset::Dataset;
use crate::utils::{convert_slice_to_matrix, gen_random_matrix};
use nalgebra::DMatrix;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{fs, marker::PhantomData, path::Path};

/// A fully-connected neural network.
#[derive(Serialize, Deserialize)]
pub struct NeuralNet<A: Activation> {
    layers: Vec<DMatrix<f64>>,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
    errors: Vec<DMatrix<f64>>,
    activation: PhantomData<A>,
}

impl<A: Activation + Serialize + DeserializeOwned> NeuralNet<A> {
    /// Creates a new network with the given node configuration and activation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scholar::net::{NeuralNet, Sigmoid};
    ///
    /// // Creates a neural network with two input nodes,
    /// // a single hidden layer with two nodes, and one output node
    /// let brain = NeuralNet::new(&[2, 2, 1], Sigmoid);
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the number of layers
    /// (i.e. the length of the given `node_counts` slice)
    /// is less than 2.
    pub fn new(node_counts: &[usize]) -> Self {
        let num_layers = node_counts.len();
        if num_layers < 2 {
            panic!(
                "not enough layers supplied (expected at least 2, found {})",
                num_layers
            );
        }

        Self {
            layers: node_counts.iter().map(|c| DMatrix::zeros(*c, 1)).collect(),
            weights: (1..num_layers)
                .map(|i| gen_random_matrix(node_counts[i], node_counts[i - 1]))
                .collect(),
            biases: node_counts
                .iter()
                .skip(1)
                .map(|c| gen_random_matrix(*c, 1))
                .collect(),
            errors: node_counts
                .iter()
                .skip(1)
                .map(|c| DMatrix::zeros(*c, 1))
                .collect(),
            activation: PhantomData,
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, LoadErr> {
        let file = fs::File::open(path)?;
        let decoded: NeuralNet<A> = bincode::deserialize_from(file)?;

        Ok(decoded)
    }

    /// Trains the network on the given dataset for the given number of iterations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scholar::dataset::Dataset;
    /// use scholar::net::{NeuralNet, Sigmoid};
    ///
    /// let dataset = Dataset::from_csv("iris.csv", false, 4);
    ///
    /// let mut brain = NeuralNet::new(&[4, 10, 10, 1], Sigmoid);
    /// brain.train(dataset, 10_000, 0.01);
    /// ```
    pub fn train(&mut self, mut training_dataset: Dataset, iterations: u64, learning_rate: f64) {
        let progress_bar = indicatif::ProgressBar::new(iterations);
        progress_bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("Training [{bar:30}] {percent:>3}% ETA: {eta}")
                .progress_chars("=> "),
        );
        let percentile = iterations / 100;

        for i in 1..iterations {
            training_dataset.shuffle();
            for (inputs, targets) in &training_dataset {
                let guesses = self.guess(inputs);
                self.backpropagate(&guesses, targets, learning_rate);
            }

            if i % percentile == 0 {
                progress_bar.inc(percentile);
            }
        }

        progress_bar.finish_and_clear();
    }

    /// Calculates the average cost of the network.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scholar::dataset::Dataset;
    /// use scholar::net::{NeuralNet, Sigmoid};
    ///
    /// let dataset = Dataset::from_csv("iris.csv", false, 4);
    /// let (training_data, testing_data) = dataset.split(0.75);
    ///
    /// let mut brain = NeuralNet::new(&[4, 10, 10, 1], Sigmoid);
    /// brain.train(training_data, 10_000, 0.01);
    ///
    /// let avg_cost = brain.test(testing_data);
    /// println!("Accuracy: {:.2}%", (1.0 - avg_cost) * 100);
    /// ```
    pub fn test(&mut self, testing_dataset: Dataset) -> f64 {
        let mut avg_cost = 0.0;
        for (inputs, targets) in &testing_dataset {
            let guesses = self.guess(inputs);
            let cost_sum: f64 = guesses
                .iter()
                .zip(targets)
                .map(|(i, t)| (t - i).abs())
                .sum();
            avg_cost += cost_sum / guesses.len() as f64;
        }
        avg_cost /= testing_dataset.rows() as f64;

        avg_cost
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SaveErr> {
        let encoded = bincode::serialize(&self)?;
        std::fs::write(path, encoded)?;

        Ok(())
    }

    /// Performs the feedforward algorithm on the given input slice,
    /// and returns the value of the output layer as a vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scholar::net::{NeuralNet, Sigmoid};
    ///
    /// let mut brain = NeuralNet::new(&[3, 10, 2], Sigmoid);
    /// let result = brain.guess(&[1.0, 0.0, -0.5]);
    ///
    /// assert_eq!(result.len(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// This method panics if the number of given input values is not equal
    /// to the number of nodes in the network's input layer.
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
                *x = A::activate(*x);
            }

            self.layers[i] = value;
        }

        self.layers[num_layers - 1].iter().cloned().collect()
    }

    /// Performs the backpropagation algorithm using the network's guessed values
    /// for a particular input, and the real target values.
    fn backpropagate(&mut self, guesses: &[f64], targets: &[f64], learning_rate: f64) {
        let guesses = convert_slice_to_matrix(guesses);
        let targets = convert_slice_to_matrix(targets);

        let num_layers = self.layers.len();
        self.errors[num_layers - 2] = targets - guesses;

        for (i, layer) in self.layers.iter().enumerate().skip(1).rev() {
            let mut gradients = layer.map(A::derivative);
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

/// An activation for a network, including a function and a 'derivative' function.
///
/// Note that the 'derivative' function assumes that the original function
/// has already been applied to it's input (as this is always the case when
/// used in the context of neural networks).
pub trait Activation {
    fn activate(x: f64) -> f64;
    fn derivative(x: f64) -> f64;
}

/// The sigmoid activation function.
#[derive(Serialize, Deserialize)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SaveErr {
    #[error("failed to serialize network")]
    Serialize(#[from] bincode::Error),
    #[error("failed to write to file")]
    FileWrite(#[from] std::io::Error),
}

#[derive(thiserror::Error, Debug)]
pub enum LoadErr {
    #[error("failed to deserialize network")]
    Deserialize(#[from] bincode::Error),
    #[error("failed to read from file")]
    FileRead(#[from] std::io::Error),
}
