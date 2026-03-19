mod utils;
mod gui;
mod Conv2d;
mod Linear;

use ndarray::{Array3, Array2, array};
use std::io::empty;
use std::{env, io, fs, path::Path};
use std::error::Error;
use Conv2d::convolve;
use crate::gui::show_image_ndarray;

const DATASET_PATH: &str = "/home/adi/Documents/rustnn/caltech101/101_ObjectCategories";


fn predict() -> Result<(), Box<dyn Error>> {
    
    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let INFERENCE: bool = (args.get(1).unwrap() == "-predict");
    let TRAINING: bool = (args.get(1).unwrap() == "-train");
    if !(INFERENCE ^ TRAINING) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "pick either -predict or -train").into());
    }




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

    let arr = utils::load_image(&fp.to_str().unwrap())?;




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
    let output3d = utils::arr2to3(&output2d);
    show_image_ndarray(&output3d)?;







    Ok(())
}

fn add_length(s: &str) -> Vec<String> {
    let mut wordvec: Vec<String> = Vec::new();
    let iter = s.split_whitespace();

    for word in iter {
        let word = word.to_owned() + " " + &word.len().to_string();
        wordvec.push(word);
    }
    return wordvec;
}



/*
//random kernels to test stuff with

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


*/