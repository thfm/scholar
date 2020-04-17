use rand::seq::SliceRandom;

type Row = (Vec<f64>, Vec<f64>);
type Data = Vec<Row>;

/// A collection of input vectors matched with their expected output.
#[derive(Debug)]
pub struct Dataset {
    data: Data,
}

impl Dataset {
    /// Parses a dataset from a CSV file.
    ///
    /// # Examples
    /// ```rust
    /// // Parses the first four columns of 'iris.csv' as inputs,
    /// // and the remaining columns as target outputs
    /// let dataset = scholar::dataset::Dataset::from_csv("iris.csv", false, 4);
    /// ```
    pub fn from_csv(
        file_path: impl AsRef<std::path::Path>,
        includes_headers: bool,
        num_inputs: usize,
    ) -> Result<Self, ParseCsvError> {
        use std::str::FromStr;

        let file = std::fs::File::open(file_path)?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(includes_headers)
            .from_reader(file);

        let data: Result<Data, ParseCsvError> = reader
            .records()
            .map(|row| {
                let row = row?;
                let row = row
                    .iter()
                    .map(|val| {
                        let val = val.trim();
                        f64::from_str(val)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let mut input = row;
                let output = input.split_off(num_inputs);
                Ok((input, output))
            })
            .collect();
        Ok(Dataset::from(data?))
    }

    /// Returns the number of rows in the dataset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // Data for the XOR problem
    /// let data = vec![
    ///     (vec![0.0, 0.0], vec![0.0]),
    ///     (vec![0.0, 1.0], vec![1.0]),
    ///     (vec![1.0, 0.0], vec![1.0]),
    ///     (vec![1.0, 1.0], vec![0.0]),
    /// ];
    ///
    /// let dataset = scholar::dataset::Dataset::from(data);
    /// assert_eq!(dataset.rows(), 4);
    /// ```
    pub fn rows(&self) -> usize {
        self.data.len()
    }

    /// Shuffles the rows in the dataset.
    pub(crate) fn shuffle(&mut self) {
        self.data.shuffle(&mut rand::thread_rng());
    }

    /// Splits the dataset into two, with the size of each determined by
    /// the given `train_portion`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let dataset = scholar::dataset::Dataset::from_csv("iris.csv", false, 4);
    ///
    /// // Randomly allocates 75% of the original dataset to 'training_data',
    /// // and the rest to 'testing_data'
    /// let (training_data, testing_data) = dataset.split(0.75);
    /// ```
    ///
    /// # Panics
    ///
    /// This method panics if the given `train_portion` isn't between 0 and 1.
    pub fn split(mut self, train_portion: f64) -> (Self, Self) {
        if train_portion > 1.0 || train_portion < 0.0 {
            panic!(
                "training portion must be between 0 and 1 (found {})",
                train_portion
            );
        }

        self.shuffle();

        let index = self.data.len() as f64 * train_portion;
        let test_split = self.data.split_off(index.round() as usize);

        (self, Self::from(test_split))
    }

    /// Returns a reference to the row at the specified index.
    fn get(&self, index: usize) -> Option<&Row> {
        self.data.get(index)
    }
}

/// A collection of the possible errors when parsing a dataset from a CSV.
#[derive(thiserror::Error, Debug)]
pub enum ParseCsvError {
    #[error("failed to read file")]
    Read(#[from] std::io::Error),
    #[error("failed to parse CSV")]
    Parse(#[from] csv::Error),
    #[error("failed to convert value into float")]
    Convert(#[from] std::num::ParseFloatError),
}

impl From<Data> for Dataset {
    fn from(data: Data) -> Self {
        Self { data }
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = &'a Row;
    type IntoIter = DatasetIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DatasetIterator {
            dataset: self,
            index: 0,
        }
    }
}

/// An iterator over a `Dataset`.
pub struct DatasetIterator<'a> {
    dataset: &'a Dataset,
    index: usize,
}

impl<'a> Iterator for DatasetIterator<'a> {
    type Item = &'a Row;
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.dataset.get(self.index);
        self.index += 1;
        result
    }
}
