use ndarray::{arr2, array, aview1, Axis};
use ndarray_stats::MaybeNan;
use ndarray_stats::{
    errors::{EmptyInput, MinMaxError, QuantileError},
    interpolate::{Higher, Interpolate, Linear, Lower, Midpoint, Nearest},
    Quantile1dExt, QuantileExt,
};
use noisy_float::types::n64;
use noisy_float::types::N64;

fn main() {
    let data = arr2(&[[1., 2., 4.0, 7.], [2.0, 3., 5., 8.], [3., 4., 6., 9.]]);
    let bool_ = data.map(|x| x.is_nan() as u64);
    let axis = Axis(0);
    let qs = &[n64(0.5)];
    let quantiles = data
        .view()
        .map(|x| n64(*x))
        .quantiles_axis_mut(axis, &aview1(qs), &Nearest)
        .unwrap();

    println!("{:?}", quantiles);
}
