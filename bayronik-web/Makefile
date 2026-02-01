.PHONY: all build run wasm serve clean install-deps

all: build

build:
	cargo build --release

run:
	cargo run --release

wasm:
	wasm-pack build --target web --out-dir pkg --release

serve: wasm
	python3 -m http.server 8080

clean:
	cargo clean
	rm -rf pkg

install-deps:
	cargo install wasm-pack
	rustup target add wasm32-unknown-unknown
