mod img_util;
mod gui;

use ndarray::Array3;
use std::{fs, path::Path};

use img_util::load_image;

const DATASET_PATH: &str = "/home/adi/Documents/rustnn/caltech101/101_ObjectCategories";


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


    // show image
    let sample_img_fp = Path::new(DATASET_PATH)
        .join("airplanes")
        .join("image_0001.jpg");
    gui::show_image(&sample_img_fp)?;




    Ok(())
}
