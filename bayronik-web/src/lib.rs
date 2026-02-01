mod app;
mod inference;
mod visualization;

pub use app::BayronikApp;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let web_options = eframe::WebOptions::default();
    
    wasm_bindgen_futures::spawn_local(async {
        let start_result = eframe::WebRunner::new()
            .start(
                "bayronik_canvas",
                web_options,
                Box::new(|cc| Ok(Box::new(BayronikApp::new(cc)))),
            )
            .await;

        if let Err(e) = start_result {
            log::error!("Failed to start eframe: {e:?}");
        }
    });

    Ok(())
}
