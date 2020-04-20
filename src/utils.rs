use nalgebra::DMatrix;
use rand::distributions::{Distribution, Uniform};

/// Generates a matrix with the specified dimensions and random
/// values between -1 and 1.
pub(crate) fn gen_random_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
    let elements = rows * cols;
    let range = Uniform::new_inclusive(-1.0, 1.0);
    DMatrix::from_iterator(
        rows,
        cols,
        (0..elements).map(|_| range.sample(&mut rand::thread_rng())),
    )
}

/// Converts a slice to a one-column matrix.
pub(crate) fn convert_slice_to_matrix(slice: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(slice.len(), 1, slice)
}
