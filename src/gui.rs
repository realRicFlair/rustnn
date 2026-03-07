use eframe::egui;
use egui::{ColorImage, TextureHandle, TextureOptions};
use std::path::{PathBuf};

pub fn show_image(path: impl Into<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let path = path.into();

    // Load image first, outside the app-creator closure
    let img = image::ImageReader::open(&path)?
        .decode()?
        .to_rgba8();

    let (w, h) = img.dimensions();
    let raw = img.into_raw();

    let color_image = ColorImage::from_rgba_unmultiplied(
        [w as usize, h as usize],
        &raw,
    );

    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(move |cc| {
            Ok(Box::new(ImageApp::new(
                cc,
                path.clone(),
                color_image.clone(),
                egui::vec2(w as f32, h as f32),
            )))
        }),
    )?;

    Ok(())
}

struct ImageApp {
    texture: TextureHandle,
    image_size: egui::Vec2,
}

impl ImageApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        path: PathBuf,
        color_image: ColorImage,
        image_size: egui::Vec2,
    ) -> Self {
        let texture = cc.egui_ctx.load_texture(
            path.to_string_lossy(),
            color_image,
            TextureOptions::default(),
        );

        Self { texture, image_size }
    }
}

impl eframe::App for ImageApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add(
                    egui::Image::new(&self.texture)
                        .fit_to_exact_size(self.image_size)
                );
            });
        });
    }
}