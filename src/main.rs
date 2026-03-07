mod img_util;

use ndarray::Array3;
use std::{fs, path::Path};
use eframe::egui;
use egui::{ColorImage, TextureHandle, TextureOptions};

use img_util::load_image;

const DATASET_PATH: &str = "/home/adi/Documents/rustnn/caltech101/101_ObjectCategories";

struct MyApp {
    texture: Option<TextureHandle>,
}

impl MyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let fp = Path::new("/home/adi/Documents/rustnn/caltech101/101_ObjectCategories")
            .join("airplanes")
            .join("image_0001.jpg");

        let image = image::ImageReader::open(fp)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();

        let (w, h) = image.dimensions();
        let raw = image.into_raw();

        let color_image = ColorImage::from_rgba_unmultiplied(
            [w as usize, h as usize],
            &raw,
        );

        let texture = cc.egui_ctx.load_texture(
            "my-image",
            color_image,
            TextureOptions::default(),
        );

        Self {
            texture: Some(texture),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(texture) = &self.texture {
                ui.image(texture);
            }
        });
    }
}


fn main() -> eframe::Result<()> {


    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|cc| Ok(Box::new(MyApp::new(cc)))),
    )

    //println!("Hello world!");
}
