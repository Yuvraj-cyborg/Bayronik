#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_transparent(false),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };

    eframe::run_native(
        "Bayronik - Baryonic Field Emulator",
        native_options,
        Box::new(|cc| Ok(Box::new(bayronik_web::BayronikApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // WASM entry point is handled by lib.rs
}
