use ndarray::{Array3, Array2, s};

/// Core fn for calculating similarity
pub fn sim(x: Array3<f32>) -> Array2<f32> {

    let arr2: Array2<f32> = x.slice(s![0, .., ..]).to_owned();

    arr2
}

