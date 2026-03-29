# Bayronik Makefile
# Build, train, and run the baryonic field emulator

.PHONY: all setup demo server infer train build build-nbody clean help

# Paths
VENV         := bayronik-model/.venv
ACTIVATE     := source $(VENV)/bin/activate
PYTORCH_LIB   = $$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
LIBTORCH_ENV := export LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1

DATA_DIR     := bayronik-model/data
WEIGHTS_DIR  := bayronik-model/weights
CAMELS_CMD   := https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/IllustrisTNG

# Binaries (skip rebuild if up to date)
INFER_BIN    := target/release/bayronik-infer
WEB_BIN      := target/release/bayronik-web

# Default
all: help

#------------------------------------------------------------------------------
# One-command demo: installs deps, starts server + webapp
#------------------------------------------------------------------------------

demo: setup-server $(WEB_BIN)
	@echo "Starting Bayronik demo..."
	@echo "  Server:   http://localhost:8000"
	@echo "  Desktop:  launching bayronik-web"
	@echo ""
	@echo "Press Ctrl+C to stop."
	cd bayronik-model && uv run --extra server uvicorn server:app --host 0.0.0.0 --port 8000 &
	@sleep 2
	./$(WEB_BIN)

setup-server:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating venv..."; \
		cd bayronik-model && uv venv && uv sync --extra server; \
	else \
		echo "Syncing deps..."; \
		cd bayronik-model && uv sync --extra server; \
	fi

#------------------------------------------------------------------------------
# Run individual services
#------------------------------------------------------------------------------

server: setup-server
	@echo "Starting inference server on http://localhost:8000"
	cd bayronik-model && uv run --extra server uvicorn server:app --host 0.0.0.0 --port 8000 --reload

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

setup: setup-model
	@echo "Setup complete"

setup-model:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating venv..."; \
		cd bayronik-model && uv venv && uv sync; \
	else \
		echo "Venv exists, syncing..."; \
		cd bayronik-model && uv sync; \
	fi

setup-infer: setup-model $(INFER_BIN)
	@echo "Inference binary ready"

$(INFER_BIN): bayronik-infer/src/*.rs bayronik-infer/Cargo.toml
	@echo "Building bayronik-infer..."
	cd bayronik-infer && \
		$(ACTIVATE) && \
		$(LIBTORCH_ENV) && \
		export DYLD_LIBRARY_PATH=$(PYTORCH_LIB) && \
		cargo build --release

$(WEB_BIN): bayronik-web/src/*.rs bayronik-web/Cargo.toml
	@echo "Building bayronik-web..."
	cd bayronik-web && cargo build --release

#------------------------------------------------------------------------------
# Data download (LH = Latin Hypercube with varying params)
#------------------------------------------------------------------------------

download-lh:
	@mkdir -p $(DATA_DIR)
	@echo "Downloading LH dark matter maps..."
	@if [ ! -f "$(DATA_DIR)/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy" ]; then \
		wget -c -P $(DATA_DIR) "$(CAMELS_CMD)/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"; \
	else echo "  Already exists, skipping."; fi
	@echo "Downloading LH total matter maps..."
	@if [ ! -f "$(DATA_DIR)/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy" ]; then \
		wget -c -P $(DATA_DIR) "$(CAMELS_CMD)/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy"; \
	else echo "  Already exists, skipping."; fi
	@echo "Downloading LH parameter file..."
	@if [ ! -f "$(DATA_DIR)/params_LH_IllustrisTNG.txt" ]; then \
		wget -O $(DATA_DIR)/_raw_params.txt "https://raw.githubusercontent.com/franciscovillaescusa/CAMELS/master/docs/params/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt" && \
		python3 -c "import numpy as np; d=np.genfromtxt('$(DATA_DIR)/_raw_params.txt',dtype=str,comments='#'); np.savetxt('$(DATA_DIR)/params_LH_IllustrisTNG.txt',d[:,1:7].astype(float),fmt='%.5f')" && \
		rm -f $(DATA_DIR)/_raw_params.txt && \
		echo "  Params saved"; \
	else echo "  Already exists, skipping."; fi
	@echo "LH data download complete"

download-cv:
	@mkdir -p $(DATA_DIR)
	@echo "Downloading CV datasets..."
	cd bayronik-model && uv run python download_data.py --dataset CV

download-all: download-cv download-lh

#------------------------------------------------------------------------------
# Training
#------------------------------------------------------------------------------

train: train-conditional

train-cv:
	cd bayronik-model && uv run python train.py --model ufno --dataset CV --epochs 50

train-lh:
	cd bayronik-model && uv run python train.py --model ufno --dataset LH --epochs 100 --no-amp

train-conditional:
	cd bayronik-model && uv run python train.py --model ufno_cond --conditional --dataset LH \
		--epochs 100 --batch-size 16 --patience 20 --no-amp \
		--spectral-weight 0.5 --mass-weight 0.01 --verbose

#------------------------------------------------------------------------------
# Inference TUI (Rust + libtorch)
#------------------------------------------------------------------------------

infer: setup-infer
	@echo "Running inference TUI..."
	cd bayronik-infer && \
		$(ACTIVATE) && \
		$(LIBTORCH_ENV) && \
		export DYLD_LIBRARY_PATH=$(PYTORCH_LIB) && \
		cargo run --release

run: infer

#------------------------------------------------------------------------------
# Build
#------------------------------------------------------------------------------

build: build-core build-nbody $(INFER_BIN) $(WEB_BIN)
	@echo "All builds complete"

build-core:
	unset CARGO_TARGET_DIR && cargo build --release -p bayronik-core

build-nbody:
	unset CARGO_TARGET_DIR && cargo build --release -p bayronik-core --examples

build-infer: $(INFER_BIN)

build-web: $(WEB_BIN)

build-wasm:
	cd bayronik-web && wasm-pack build --target web --out-dir pkg --release

run-web: $(WEB_BIN)
	cd bayronik-web && cargo run --release

serve-web: build-wasm
	cd bayronik-web && python3 -m http.server 8080

#------------------------------------------------------------------------------
# Export
#------------------------------------------------------------------------------

export-torchscript:
	cd bayronik-model && uv run python -m bayronik_model.export \
		--weights weights/best_ufno_cond_LH_IllustrisTNG.pth \
		--output weights/bayronik.pt \
		--model ufno_cond

#------------------------------------------------------------------------------
# Generate
#------------------------------------------------------------------------------

generate-map:
	cd bayronik-core && cargo run --release --example generate_map

#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------

test: test-core test-model

test-core:
	cd bayronik-core && cargo test

test-model:
	cd bayronik-model && uv run python -c "from bayronik_model import UFNO2d; print('OK')"

#------------------------------------------------------------------------------
# GCP Training
#------------------------------------------------------------------------------

GCP_PROJECT  := bayronik-core
GCP_ZONE     := us-central1-a
GCP_VM       := bayronik-train

gcp-create:
	gcloud compute instances create $(GCP_VM) \
		--project=$(GCP_PROJECT) \
		--zone=$(GCP_ZONE) \
		--machine-type=g2-standard-8 \
		--image-family=pytorch-latest-gpu \
		--image-project=deeplearning-platform-release \
		--boot-disk-size=100GB \
		--maintenance-policy=TERMINATE \
		--restart-on-failure

gcp-ssh:
	gcloud compute ssh $(GCP_VM) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE)

gcp-upload:
	gcloud compute scp --recurse \
		bayronik-model/src bayronik-model/train.py bayronik-model/pyproject.toml \
		$(GCP_VM):~/bayronik-model/ \
		--project=$(GCP_PROJECT) --zone=$(GCP_ZONE)

gcp-download-weights:
	mkdir -p $(WEIGHTS_DIR)
	gcloud compute scp \
		$(GCP_VM):~/bayronik-model/weights/best_ufno_cond_LH_IllustrisTNG.pth \
		$(WEIGHTS_DIR)/ \
		--project=$(GCP_PROJECT) --zone=$(GCP_ZONE)

gcp-stop:
	gcloud compute instances stop $(GCP_VM) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE)

gcp-delete:
	gcloud compute instances delete $(GCP_VM) --project=$(GCP_PROJECT) --zone=$(GCP_ZONE)

#------------------------------------------------------------------------------
# Deploy
#------------------------------------------------------------------------------

deploy-info:
	@echo "Deployment options:"
	@echo ""
	@echo "  1. Vercel (static WASM frontend):"
	@echo "     - Build WASM: make build-wasm"
	@echo "     - Deploy bayronik-web/ as static site"
	@echo "     - Point API_URL to your inference server"
	@echo ""
	@echo "  2. Server (FastAPI inference backend):"
	@echo "     - Deploy server.py on Railway, Fly.io, or Modal.com"
	@echo "     - Model weights must be in bayronik-model/weights/"
	@echo ""
	@echo "  3. Desktop (local):"
	@echo "     - make demo (starts server + desktop app)"

#------------------------------------------------------------------------------
# Clean
#------------------------------------------------------------------------------

clean:
	rm -rf target/
	rm -rf bayronik-model/__pycache__
	rm -rf bayronik-model/src/bayronik_model/__pycache__
	find . -name "._*" -delete 2>/dev/null || true

clean-venv:
	rm -rf $(VENV)

clean-weights:
	rm -rf $(WEIGHTS_DIR)

clean-data:
	rm -rf $(DATA_DIR)

clean-all: clean clean-venv clean-weights clean-data

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help:
	@echo "Bayronik - Baryonic Field Emulator"
	@echo ""
	@echo "Quick start:"
	@echo "  make demo        - Start server + desktop app (one command)"
	@echo "  make server      - Start FastAPI inference server (localhost:8000)"
	@echo "  make run-web     - Run native desktop frontend"
	@echo "  make serve-web   - Build WASM and serve (localhost:8080)"
	@echo ""
	@echo "Data:"
	@echo "  make download-lh - Download LH dataset with params (~15GB)"
	@echo "  make download-cv - Download CV dataset (~300MB)"
	@echo ""
	@echo "Training:"
	@echo "  make train       - Train conditional U-FNO on LH"
	@echo "  make train-cv    - Train on CV dataset"
	@echo ""
	@echo "Inference:"
	@echo "  make infer       - Run Rust TUI (needs libtorch)"
	@echo ""
	@echo "Build:"
	@echo "  make build       - Build all Rust binaries"
	@echo "  make build-wasm  - Build WASM frontend"
	@echo ""
	@echo "Deploy:"
	@echo "  make deploy-info - Show deployment options"
	@echo ""
	@echo "Clean:"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make clean-all   - Remove everything (venv, data, weights)"
