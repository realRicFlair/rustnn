use ndarray::{Array2, Array3};

pub fn load_image(path: &str) -> image::ImageResult<Array3<f32>> {
    let img = image::ImageReader::open(path)?.decode()?.to_rgb8();
    let (w, h) = img.dimensions();
    //img already decoded to RGBRGBRGB row-major
    let raw = img.into_raw();
    //turn raw to iter, and then use map to turn to f32 between 0 to 1. Collect the iter into a vector
    let data: Vec<f32> = raw.into_iter().map(|v| v as f32 / 255.0).collect();
    // formatted as H W 3
    let pixel_grid = Array3::from_shape_vec((h as usize, w as usize, 3), data)
        .expect("failed to create Array3 from image!");
    // fix to 3 H W
    let chw = pixel_grid.permuted_axes([2, 0, 1]);
    Ok(chw)
}


pub fn arr2to3(gray: &Array2<f32>) -> Array3<f32> {
    //let (h, w) = gray.dim();
    let (h, w) = gray.dim();

    let mut rgb = Array3::<f32>::zeros((3, h, w));

    for y in 0..h {
        for x in 0..w {
            let v = gray[[y, x]];

            rgb[[0, y, x]] = v; // R
            rgb[[1, y, x]] = v; // G
            rgb[[2, y, x]] = v; // B
        }
    }

    return rgb;
}