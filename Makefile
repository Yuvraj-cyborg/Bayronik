# Bayronik Makefile
# Build, train, and run the baryonic field emulator

.PHONY: all setup train infer build clean help

# Default target
all: setup

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

setup: setup-model setup-infer
	@echo "Setup complete"

setup-model:
	@echo "Setting up bayronik-model..."
	cd bayronik-model && uv venv && uv add torch numpy scipy einops tqdm
	@echo "Model environment ready"

setup-infer:
	@echo "Building bayronik-infer..."
	cd bayronik-infer && cargo build --release
	@echo "Inference binary ready"

setup-core:
	@echo "Building bayronik-core..."
	cd bayronik-core && cargo build --release
	@echo "Core simulation ready"

#------------------------------------------------------------------------------
# Training
#------------------------------------------------------------------------------

train: train-cv

train-cv:
	@echo "Training on CV dataset..."
	cd bayronik-model && source .venv/bin/activate && uv run python train.py --model ufno --dataset CV --epochs 50

train-lh:
	@echo "Training on LH dataset..."
	cd bayronik-model && source .venv/bin/activate && uv run python train.py --model ufno --dataset LH --epochs 100

train-fno:
	@echo "Training FNO model..."
	cd bayronik-model && source .venv/bin/activate && uv run python train.py --model fno --dataset LH --epochs 100

train-conditional:
	@echo "Training conditional U-FNO..."
	cd bayronik-model && source .venv/bin/activate && uv run python train.py --model ufno_cond --conditional --dataset LH --epochs 100

#------------------------------------------------------------------------------
# Data
#------------------------------------------------------------------------------

download-cv:
	@echo "Downloading CV dataset..."
	cd bayronik-model && source .venv/bin/activate && uv run python download_data.py --dataset CV

download-lh:
	@echo "Downloading LH dataset..."
	cd bayronik-model && source .venv/bin/activate && uv run python download_data.py --dataset LH

download-all:
	@echo "Downloading all datasets..."
	cd bayronik-model && source .venv/bin/activate && uv run python download_data.py --dataset all

#------------------------------------------------------------------------------
# Inference & Demo
#------------------------------------------------------------------------------

infer: build-infer
	@echo "Running inference TUI..."
	cd bayronik-infer && \
		source ../bayronik-model/.venv/bin/activate && \
		export DYLD_LIBRARY_PATH=$$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))") && \
		export LIBTORCH_USE_PYTORCH=1 && \
		export LIBTORCH_BYPASS_VERSION_CHECK=1 && \
		cargo run --release

run: infer

server:
	@echo "Starting inference server on http://localhost:8000"
	cd bayronik-model && uv run --extra server uvicorn server:app --host 0.0.0.0 --port 8000 --reload

server-prod:
	@echo "Starting production server..."
	cd bayronik-model && uv run --extra server uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4

webapp:
	@echo "Starting Streamlit web app on http://localhost:8501"
	cd bayronik-model && uv run --extra webapp streamlit run webapp.py

#------------------------------------------------------------------------------
# Build
#------------------------------------------------------------------------------

build: build-core build-infer
	@echo "All builds complete"

build-core:
	cd bayronik-core && cargo build --release

build-infer:
	cd bayronik-infer && cargo build --release

build-web:
	cd bayronik-web && cargo build --release

build-wasm:
	cd bayronik-web && wasm-pack build --target web --out-dir pkg --release

run-web:
	cd bayronik-web && cargo run --release

serve-web: build-wasm
	cd bayronik-web && python3 -m http.server 8080

#------------------------------------------------------------------------------
# Export
#------------------------------------------------------------------------------

export-onnx:
	@echo "Exporting model to ONNX..."
	cd bayronik-model && source .venv/bin/activate && uv run python export.py \
		--weights weights/best_ufno_CV_IllustrisTNG.pth \
		--output weights/bayronik.onnx \
		--model ufno

export-torchscript:
	@echo "Exporting model to TorchScript..."
	cd bayronik-model && source .venv/bin/activate && uv run python export.py \
		--weights weights/best_ufno_CV_IllustrisTNG.pth \
		--output weights/bayronik.pt \
		--model ufno

#------------------------------------------------------------------------------
# Generate
#------------------------------------------------------------------------------

generate-map:
	@echo "Generating N-body map..."
	cd bayronik-core && cargo run --release --example generate_map

#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------

test: test-core test-model
	@echo "All tests passed"

test-core:
	cd bayronik-core && cargo test

test-model:
	cd bayronik-model && source .venv/bin/activate && uv run python -c "from bayronik_model import UFNO2d; print('Model imports work')"

#------------------------------------------------------------------------------
# Clean
#------------------------------------------------------------------------------

clean:
	@echo "Cleaning build artifacts..."
	rm -rf target/
	rm -rf bayronik-model/.venv
	rm -rf bayronik-model/__pycache__
	rm -rf bayronik-model/src/bayronik_model/__pycache__
	find . -name "._*" -delete
	@echo "Clean complete"

clean-weights:
	rm -rf bayronik-model/weights/

clean-data:
	rm -rf bayronik-model/data/

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help:
	@echo "Bayronik - Baryonic Field Emulator"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Setup all environments"
	@echo "  setup-model    - Setup Python model environment"
	@echo "  setup-infer    - Build Rust inference binary"
	@echo ""
	@echo "Training:"
	@echo "  train-cv       - Train on CV dataset (27 sims)"
	@echo "  train-lh       - Train on LH dataset (1000 sims)"
	@echo "  train-fno      - Train FNO model"
	@echo "  train-conditional - Train conditional U-FNO"
	@echo ""
	@echo "Data:"
	@echo "  download-cv    - Download CV dataset (~300MB)"
	@echo "  download-lh    - Download LH dataset (~15GB)"
	@echo ""
	@echo "Run:"
	@echo "  run / infer    - Run inference TUI"
	@echo "  server         - Start inference API server (localhost:8000)"
	@echo "  webapp         - Start Streamlit web app (localhost:8501)"
	@echo "  generate-map   - Generate N-body map"
	@echo ""
	@echo "Export:"
	@echo "  export-onnx    - Export to ONNX (for WASM)"
	@echo "  export-torchscript - Export to TorchScript"
	@echo ""
	@echo "Build:"
	@echo "  build          - Build all Rust binaries"
	@echo "  build-web      - Build native web app"
	@echo "  build-wasm     - Build WASM frontend"
	@echo "  run-web        - Run native web app"
	@echo "  serve-web      - Build WASM and serve locally"
	@echo ""
	@echo "Clean:"
	@echo "  clean          - Remove build artifacts"
