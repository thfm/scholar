#[derive(Debug)]
pub struct Dataset {
    data: Vec<(Vec<f64>, Vec<f64>)>,
}

impl Dataset {
    pub fn new(data: Vec<(Vec<f64>, Vec<f64>)>) -> Self {
        Self { data }
    }
}
