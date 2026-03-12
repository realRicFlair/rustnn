use std::path::PathBuf;

use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use ndarray::Array3;

pub fn show_image_ndarray(
    image_chw: &Array3<f32>,
) -> Result<(), Box<dyn std::error::Error>> {

    let (c, h, w) = image_chw.dim();

    if c != 3 {
        return Err(format!("expected image shape (3, H, W), got ({c}, {h}, {w})").into());
    }

    // ---- compute normalization range ----
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for v in image_chw.iter() {
        min_val = min_val.min(*v);
        max_val = max_val.max(*v);
    }

    let range = (max_val - min_val).max(1e-8);

    // ---- build RGBA buffer ----
    let mut raw_rgba = Vec::with_capacity(w * h * 4);

    for y in 0..h {
        for x in 0..w {

            let r = (image_chw[[0, y, x]] - min_val) / range;
            let g = (image_chw[[1, y, x]] - min_val) / range;
            let b = (image_chw[[2, y, x]] - min_val) / range;

            raw_rgba.push((r * 255.0).clamp(0.0, 255.0) as u8);
            raw_rgba.push((g * 255.0).clamp(0.0, 255.0) as u8);
            raw_rgba.push((b * 255.0).clamp(0.0, 255.0) as u8);
            raw_rgba.push(255);
        }
    }

    let color_image = ColorImage::from_rgba_unmultiplied([w, h], &raw_rgba);
    let image_size = egui::vec2(w as f32, h as f32);

    eframe::run_native(
        "Image Viewer",
        eframe::NativeOptions::default(),
        Box::new(move |cc| {
            Ok(Box::new(ImageApp::new(
                cc,
                "ndarray_image".to_string(),
                color_image.clone(),
                image_size,
            )))
        }),
    )?;

    Ok(())
}

fn float_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

struct ImageApp {
    texture: TextureHandle,
    image_size: egui::Vec2,
}

impl ImageApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        texture_name: String,
        color_image: ColorImage,
        image_size: egui::Vec2,
    ) -> Self {
        let texture = cc.egui_ctx.load_texture(
            texture_name,
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
                        .fit_to_exact_size(self.image_size),
                );
            });
        });
    }
}