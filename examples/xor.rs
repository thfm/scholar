use scholar::{Dataset, NeuralNet, Sigmoid};

fn main() {
    let data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let dataset = Dataset::from(data);

    let mut brain: NeuralNet<Sigmoid> = NeuralNet::new(&[2, 10, 10, 1]);
    brain.train(dataset, 250_000, 0.01);

    println!("Prediction: {:.2}", brain.guess(&[1.0, 1.0])[0]);
}
