use ndarray::{Array1, Array2, Array3, Array4};

#[derive(Debug)]
pub struct Conv2d {
    // (out_channels, in_channels, kernel_h, kernel_w)
    pub kernel_weights: Vec<Array3<f32>>, // each filter is (in_channels, k_h, k_w)
    pub bias: Array1<f32>,         // (out_channels)
    pub kernel_size: usize,
    pub stride: usize,
}

//num of outputs = num of kernels
//kernel spans all channels. one input per channel
fn convolve(kernel: &Array3<f32>, input: &Array3<f32>, stride: usize, padding: usize) -> Array2<f32>{
    let (in_channels, in_h, in_w) = input.dim();
    let (k_channels, k_h, k_w) = kernel.dim();
    assert_eq!(in_channels, k_channels, "Kernel channels must match input channels!");
    assert!(((in_h >= k_h) && (in_w >= k_w)), "Kernel bigger than input!");

    let out_h = in_h - k_h +1;
    let out_w = in_w - k_w + 1;
    let mut output = Array2::<f32>::zeros((out_h, out_w));

    for outy in (0..out_h).step_by(stride) {
        for outx in 0..out_w { //Do every pixel in output

            let mut sum = 0.0; 
            for channel in 0..in_channels{
                for ky in 0..k_h {
                    for kx in 0..k_w{ // For every element in kernel
                        //sum += input[channel][outy+ky][outx+kx] + kernel[channel][ky][kx];
                        //ndarray is indexed with tuple since its stored contiguous
                        sum += input[(channel, outx+kx, outy+ky)] + kernel[(channel, outy+ky, outx+kx)];
                    }
                }
            }
            output[(outx, outy)] = sum;
        }
    }

    output    
}


impl Conv2d {
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32>{
        let (in_channels, in_h, in_w) = input.dim();
        
    }
}






