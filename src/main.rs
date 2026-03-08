mod img_util;
mod gui;
mod Conv2d;

use ndarray::Array3;
use std::{fs, path::Path};



const DATASET_PATH: &str = "/home/adi/Documents/rustnn/caltech101/101_ObjectCategories";

pub fn load_image(path: &str) -> image::ImageResult<Array3<f32>> {
    let img = image::ImageReader::open(path)?.decode()?.to_rgb8();
    let (w, h) = img.dimensions();

    //img already decoded to RGBRGBRGB row-major
    let raw = img.into_raw();

    //turn raw to iter, and then use map to turn to f32 between 0 to 1. Collect the iter into a vector
    let mut tensor = Array3::<f32>::zeros((3, h as usize, w as usize));

    for (i, pixel) in raw.chunks_exact(3).enumerate() {
        let y = i / w as usize;
        let x = i % w as usize;

        tensor[[0, y, x]] = pixel[0] as f32 / 255.0; // R
        tensor[[1, y, x]] = pixel[1] as f32 / 255.0; // G
        tensor[[2, y, x]] = pixel[2] as f32 / 255.0; // B
    }

    // outputs (3, H, W) order
    Ok(tensor)
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut names: Vec<String> = fs::read_dir(DATASET_PATH)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();

    let img_classes: Vec<(u16, String)> = names
        .into_iter()
        .enumerate()
        .map(|(i, name)| (i as u16, name))
        .collect();
    for (i, entry) in img_classes {
        println!("Found class {entry} with id:{i}");
    }

    let fp = Path::new(DATASET_PATH)
        .join("airplanes")
        .join("image_0001.jpg");

    let arr = load_image(&fp.to_str().unwrap())?;
    println!("shape = {:?}", arr.dim()); // (H, W, 3)
    println!("pixel (0,0) rgb = {:?}", arr.slice(ndarray::s![0, 0, ..]));

    /* 
    // show image
    let sample_img_fp = Path::new(DATASET_PATH).join("airplanes").join("image_0001.jpg");
    gui::show_image(&sample_img_fp)?;
    */



    Ok(())
}
