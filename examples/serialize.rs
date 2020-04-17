use scholar::dataset::Dataset;
use scholar::net::{NeuralNet, Sigmoid};

fn main() -> anyhow::Result<()> {
    let data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];

    let dataset = Dataset::from(data);

    let mut brain = NeuralNet::new(&[2, 2, 1], Sigmoid);
    brain.train(dataset, 250_000, 0.01);

    let encoded = bincode::serialize(&brain)?;
    let _decoded: NeuralNet<Sigmoid> = bincode::deserialize(&encoded[..])?;

    println!("Prediction: {:.2}", brain.guess(&[1.0, 1.0])[0]);

    Ok(())
}
