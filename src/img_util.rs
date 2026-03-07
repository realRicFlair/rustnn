use ndarray::Array3;

pub fn load_image(path: &str) -> image::ImageResult<Array3<f32>> {
    let img = image::ImageReader::open(path)?.decode()?.to_rgb8();
    let (w, h) = img.dimensions();

    //img already decoded to RGBRGBRGB row-major
    let raw = img.into_raw();
    //turn raw to iter, and then use map to turn to f32 between 0 to 1. Collect the iter into a vector
    let data: Vec<f32> = raw.into_iter().map(|v| v as f32 / 255.0).collect();

    // format into (H, W, 3)
    let pixel_grid = Array3::from_shape_vec((h as usize, w as usize, 3), data)
        .expect("failed to create Array3 from image!");
    Ok(pixel_grid)
}