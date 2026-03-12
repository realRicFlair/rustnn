mod img_util;
mod gui;
mod Conv2d;
mod Linear;

use ndarray::{Array3, Array2, array};
use std::{fs, path::Path};
use Conv2d::convolve;

use crate::gui::show_image_ndarray;



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
        .join("image_0077.jpg");

    let arr = load_image(&fp.to_str().unwrap())?;
    //println!("shape = {:?}", arr.dim()); // (H, W, 3)
    //println!("pixel (0,0) rgb = {:?}", arr.slice(ndarray::s![0, 0, ..]));

    /* 
    // show image
    let sample_img_fp = Path::new(DATASET_PATH).join("airplanes").join("image_0001.jpg");
    gui::show_image(&sample_img_fp)?;
    */
    //show_image_ndarray(&arr)?;
    let gaussian_k: Array3<f32> = array![
        [   // channel 0
            [1.0/16.0, 1.0/8.0, 1.0/17.0],
            [1.0/8.0, 1.0/4.0, 1.0/8.0],
            [1.0/16.0, 1.0/8.0, 1.0/16.0],
        ],
        [   // channel 1
            [1.0/16.0, 1.0/8.0, 1.0/17.0],
            [1.0/8.0, 1.0/4.0, 1.0/8.0],
            [1.0/16.0, 1.0/8.0, 1.0/16.0],
        ],
        [   // channel 2
            [1.0/16.0, 1.0/8.0, 1.0/17.0],
            [1.0/8.0, 1.0/4.0, 1.0/8.0],
            [1.0/16.0, 1.0/8.0, 1.0/16.0],
        ],
    ];

    let sobel_k: Array3<f32> = array![
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
    ];

    show_image_ndarray(&arr)?;
    let output2d = convolve(&sobel_k, &arr, 1, 1);
    let output3d = arr2to3(&output2d);
    show_image_ndarray(&output3d)?;


    Ok(())
}

pub fn arr2to3(gray: &Array2<f32>) -> Array3<f32> {
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

    rgb
}
