use ndarray::{Array1, Array2, Array3, Array4, ArrayBase, Data, ArrayView3, Axis, Ix3};
use rand::{Rng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

#[derive(Debug)]
pub struct Conv2d {
    // (out_channels, in_channels, kernel_h, kernel_w)
    pub kernels: Array4<f32>, // each filter is (in_channels, k_h, k_w)
    pub bias: Array1<f32>,         // (out_channels)
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub activation: fn(f32) -> f32,
}

impl Conv2d {
    pub fn new_random(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let kernels = he_init_kernels(out_channels, in_channels, kernel_size, kernel_size);
        let bias = zero_bias(out_channels);

        Self {
            kernels,
            bias,
            kernel_size,
            stride,
            padding,
            activation: relu,
        }
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32>{
        //Num of kernels = output channels. 1 kernel produce 2nd order tensor
        let (out_channels, k_in_channels, k_h, k_w) =  self.kernels.dim();
        let (in_channels, in_h, in_w) = input.dim();
        
        assert_eq!(in_channels, k_in_channels, "Kernel channel count must match input channel cout!");
        assert!((in_h >= k_h) && (in_w >= k_w), "Kernels larger than input!");

        let out_h = 1 + (in_h - k_h + (2*self.padding))/self.stride;
        let out_w = 1 + (in_w - k_w + (2*self.padding))/self.stride;
        
        let mut output = Array3::<f32>::zeros((out_channels, out_h, out_w));
        for oc in 0..out_channels {
            let k: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, _, f32> = self.kernels.index_axis(Axis(0), oc);
            let mut feature_map = convolve(&k, input, self.stride, self.padding);
            
            let b = self.bias[oc];
            feature_map.map_inplace(|v| {
                // Add bias and then apply activation
                *v = (self.activation)(*v + b);
            });
            
            //Fill output channel with the featuremap
            output.index_axis_mut(Axis(0), oc).assign(&feature_map);
        }

        return output;
    }
}


//num of outputs = num of kernels
//kernel spans all channels. one input per channel
//Works for any type T as long as its f32 and is 3D
pub fn convolve<T: Data<Elem = f32>>(kernel: &ArrayBase<T, Ix3>, input: &Array3<f32>, stride: usize, padding: usize) -> Array2<f32> {
    let (in_channels, in_h, in_w) = input.dim();
    let (k_channels, k_h, k_w) = kernel.dim();
    
    assert_eq!(in_channels, k_channels, "Kernel channels must match input channels!");
    assert!(((in_h >= k_h) && (in_w >= k_w)), "Kernel bigger than input!");
    println!("Input has size {in_w}, {in_h}");
    
    let out_h = 1 + (in_h + 2 * padding - k_h) / stride;
    let out_w = 1 + (in_w + 2 * padding - k_w) / stride;
    
    let mut output = Array2::<f32>::zeros((out_h, out_w));

    for outy in 0..out_h {
        for outx in 0..out_w { //Do every pixel in output
            let mut sum = 0.0; 

            let base_y = outy*stride;
            let base_x = outx*stride;

            for channel in 0..in_channels {
                for ky in 0..k_h {
                    for kx in 0..k_w { // For every element in kernel
                        let iy = base_y + ky;
                        let ix = base_x + kx;

                        // stride changes where each output cell starts sampling from
                        // padding shifts the sampling window so it can land partly outside the image
                        // outside image samples are treated as zero
                        let iy = iy as isize - padding as isize;
                        let ix = ix as isize - padding as isize;

                        // possibly mask here in the future so we dont gotta spam branch instructions
                        if iy >= 0
                            && iy < in_h as isize
                            && ix >= 0
                            && ix < in_w as isize
                        {
                            sum += input[(channel, iy as usize, ix as usize)] * kernel[(channel, ky, kx)];
                        }
                    }
                }
            }
            //println!("Inserting sum: {sum} into position({outx}, {outy})");
            output[(outy, outx)] = sum;
        }
    }

    return output;    
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn he_init_kernels(
    out_channels: usize,
    in_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Array4<f32> {
    let fan_in = (in_channels * kernel_h * kernel_w) as f32;
    let std = (2.0 / fan_in).sqrt();

    let normal = Normal::<f32>::new(0.0, std)
        .expect("failed to create normal distribution");
    let mut rng = rand::rng();

    Array4::from_shape_fn(
        (out_channels, in_channels, kernel_h, kernel_w),
        |_| normal.sample(&mut rng),
    )
}

pub fn zero_bias(out_channels: usize) -> Array1<f32> {
    Array1::zeros(out_channels)
}