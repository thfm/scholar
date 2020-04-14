type Data = Vec<(Vec<f64>, Vec<f64>)>;

#[derive(Debug)]
pub struct Dataset {
    data: Data,
}

impl From<Data> for Dataset {
    fn from(data: Data) -> Self {
        Self { data }
    }
}
