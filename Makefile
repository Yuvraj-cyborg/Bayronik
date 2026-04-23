# Bayronik
# Rust runtime (engine, infer, registry, server, client) + Python (training,
# scientific validation). One Makefile, prod targets only.

.PHONY: help setup build test train validate phase2 server client wasm clean

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VENV         := model/.venv
ACTIVATE     := source $(VENV)/bin/activate
PYTORCH_LIB   = $$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
LIBTORCH_ENV := export LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
TCH_ENV       = $(ACTIVATE) && $(LIBTORCH_ENV) && export DYLD_LIBRARY_PATH=$(PYTORCH_LIB)

DATA_DIR     := model/data
WEIGHTS_DIR  := model/weights
CAMELS_BASE  := https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/IllustrisTNG

SERVER_BIN   := target/release/server
INFER_BIN    := target/release/infer
CLIENT_BIN   := target/release/client
REGISTRY_BIN := target/release/registry

# ---------------------------------------------------------------------------
# Help (default)
# ---------------------------------------------------------------------------

help:
	@echo "Bayronik"
	@echo ""
	@echo "  setup        Sync Python venv (model/.venv) via uv"
	@echo "  build        Build all Rust crates in release"
	@echo "  test         All tests (engine + registry + fast model)"
	@echo ""
	@echo "  server       Run inference HTTP server on :8000 (server crate)"
	@echo "  client       Run native client (desktop egui)"
	@echo "  wasm         Build WASM client (-> client/pkg/) and serve on :8080"
	@echo "  infer        Terminal inference UI (infer crate)"
	@echo ""
	@echo "  download-lh  Download CAMELS IllustrisTNG LH maps + params"
	@echo "  download-cv  Download CAMELS CV split"
	@echo "  train        Train conditional U-FNO on LH"
	@echo "  validate     Scientific validation (LH + CV) -> reports/"
	@echo "  phase2       validate -> rebuild registry -> regression tests"
	@echo ""
	@echo "  clean        Remove build artifacts"
	@echo "  clean-all    Also remove venv, data, weights"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup:
	@if [ ! -d "$(VENV)" ]; then \
		cd model && uv venv && uv sync; \
	else \
		cd model && uv sync; \
	fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build: $(SERVER_BIN) $(REGISTRY_BIN) $(CLIENT_BIN) $(INFER_BIN)

$(SERVER_BIN): server/src/*.rs server/Cargo.toml \
               registry/src/*.rs registry/src/bin/*.rs registry/Cargo.toml \
               setup
	@$(TCH_ENV) && cargo build --release -p server

$(REGISTRY_BIN): registry/src/*.rs registry/src/bin/*.rs registry/Cargo.toml
	@cargo build --release -p registry --bin registry

$(CLIENT_BIN): client/src/*.rs client/Cargo.toml engine/src/*.rs engine/Cargo.toml
	@cargo build --release -p client

$(INFER_BIN): infer/src/*.rs infer/Cargo.toml engine/src/*.rs engine/Cargo.toml setup
	@$(TCH_ENV) && cargo build --release -p infer

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

server: $(SERVER_BIN)
	@$(TCH_ENV) && ./$(SERVER_BIN) --bind 127.0.0.1:8000

client: $(CLIENT_BIN)
	@./$(CLIENT_BIN)

wasm:
	@cd client && wasm-pack build --target web --out-dir pkg --release
	@cd client && python3 -m http.server 8080

infer: $(INFER_BIN)
	@$(TCH_ENV) && ./$(INFER_BIN)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test: test-engine test-registry test-model

test-engine:
	@cargo test -p engine --release

test-registry:
	@cargo test -p registry --release

test-model: setup
	@uv run --project model --extra dev pytest model/tests -m "not slow"

test-model-slow: setup
	@uv run --project model --extra dev pytest model/tests -m slow

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

download-lh:
	@mkdir -p $(DATA_DIR)
	@for f in Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy Maps_Mtot_IllustrisTNG_LH_z=0.00.npy; do \
		if [ ! -f "$(DATA_DIR)/$$f" ]; then \
			wget -c -P $(DATA_DIR) "$(CAMELS_BASE)/$$f"; \
		fi; \
	done
	@if [ ! -f "$(DATA_DIR)/params_LH_IllustrisTNG.txt" ]; then \
		wget -O $(DATA_DIR)/_raw_params.txt \
			"https://raw.githubusercontent.com/franciscovillaescusa/CAMELS/master/docs/params/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt" && \
		python3 -c "import numpy as np; d=np.genfromtxt('$(DATA_DIR)/_raw_params.txt',dtype=str,comments='#'); np.savetxt('$(DATA_DIR)/params_LH_IllustrisTNG.txt',d[:,1:7].astype(float),fmt='%.5f')" && \
		rm -f $(DATA_DIR)/_raw_params.txt; \
	fi

download-cv:
	@mkdir -p $(DATA_DIR)
	@cd model && uv run python download_data.py --dataset CV

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

train: setup
	@cd model && uv run python train.py --model ufno_cond --conditional --dataset LH \
		--epochs 100 --batch-size 16 --patience 20 --no-amp \
		--spectral-weight 0.5 --mass-weight 0.01 --verbose

# ---------------------------------------------------------------------------
# Phase 2: scientific validation
# ---------------------------------------------------------------------------

validate: setup
	@uv run --project model python model/benchmarks/validation.py \
		--lh-samples 64 --cv-samples 32 --bins 24

build-registry: $(REGISTRY_BIN)
	@./$(REGISTRY_BIN) $${VERSION:+--version $$VERSION}

phase2: validate build-registry test-registry

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean:
	@rm -rf target/ client/pkg/
	@find . -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name '._*' -delete 2>/dev/null || true

clean-all: clean
	@rm -rf $(VENV) $(DATA_DIR) $(WEIGHTS_DIR)
