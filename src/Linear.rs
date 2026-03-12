use ndarray::{Array1, Array2, Array3, Array4, ArrayBase, Data, ArrayView3, Axis, Ix3};
use rand::RngExt;

pub struct Linear {
    pub weights: Array2<f32>, // (out_dims, in_dims)
    pub bias: Array1<f32>, // (out_dims)
    pub activation: fn(f32) -> f32,
}

// torch has nn.Linear(input_d, output_d)
impl Linear {
    pub fn new(
        in_dims: usize,
        out_dims: usize,
    ) -> Self {
        let mut rng = rand::rng();
        let weights = Array2::<f32>::from_shape_fn((out_dims, in_dims), |_| {rng.random_range(-0.1..0.1)});
        let bias = Array1::<f32>::from_shape_fn((out_dims), |_| {rng.random_range(-0.1..0.1)});
        Self {
            weights,
            bias,
            activation: relu,
        }
    }

    pub fn forward(&self, input: &Array1<f32>,) -> Array1<f32>{
        let in_dims = input.len();
        let (weight_rows, weight_cols) = self.weights.dim();

        assert_eq!(in_dims, weight_cols, "Input size must match Weight columns");
        
        // weight*input + bias
        let mut output = self.weights.dot(input) + &self.bias;
        output.map(|x| {(self.activation)(*x)}); //Apply activation

        return output;
    }
}


pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}