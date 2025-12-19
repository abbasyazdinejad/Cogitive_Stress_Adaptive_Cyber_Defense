# Makefile for Stress-Adaptive Defense Framework
# Simplifies common development and experiment tasks

.PHONY: help install install-dev install-all clean test format lint train loso soc reproduce

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Stress-Adaptive Defense Framework$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make install        - Install package"
	@echo "  make test           - Run tests"
	@echo "  make reproduce      - Reproduce paper results"

# ============================================================================
# Installation
# ============================================================================

install: ## Install package with core dependencies
	@echo "$(GREEN)Installing package...$(NC)"
	pip install -e .
	@echo "$(GREEN)✓ Installation complete!$(NC)"

install-dev: ## Install with development dependencies
	@echo "$(GREEN)Installing with dev dependencies...$(NC)"
	pip install -e ".[dev]"
	@echo "$(GREEN)✓ Dev installation complete!$(NC)"

install-all: ## Install with all optional dependencies
	@echo "$(GREEN)Installing with all dependencies...$(NC)"
	pip install -e ".[all]"
	@echo "$(GREEN)✓ Full installation complete!$(NC)"

install-requirements: ## Install from requirements files
	@echo "$(GREEN)Installing from requirements.txt...$(NC)"
	pip install -r requirements.txt
	pip install -r requirements-optional.txt
	@echo "$(GREEN)✓ Requirements installed!$(NC)"

# ============================================================================
# Development
# ============================================================================

clean: ## Clean build artifacts and cache
	@echo "$(YELLOW)Cleaning...$(NC)"
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	@echo "$(GREEN)✓ Cleaned!$(NC)"

test: ## Run tests with pytest
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "$(GREEN)✓ Tests complete!$(NC)"

test-fast: ## Run tests in parallel (fast)
	@echo "$(GREEN)Running tests (parallel)...$(NC)"
	pytest tests/ -n auto -v
	@echo "$(GREEN)✓ Tests complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black src/ experiments/ tests/
	isort src/ experiments/ tests/
	@echo "$(GREEN)✓ Code formatted!$(NC)"

lint: ## Lint code with flake8 and mypy
	@echo "$(GREEN)Linting code...$(NC)"
	flake8 src/ experiments/
	mypy src/ experiments/
	@echo "$(GREEN)✓ Linting complete!$(NC)"

check: format lint test ## Run all checks (format, lint, test)
	@echo "$(GREEN)✓ All checks passed!$(NC)"

# ============================================================================
# Experiments
# ============================================================================

train: ## Train stress classification model
	@echo "$(GREEN)Training stress classifier...$(NC)"
	python experiments/train_stress_model.py --model dnn_xgboost --calibrate
	@echo "$(GREEN)✓ Training complete!$(NC)"

train-all: ## Train all baseline models
	@echo "$(GREEN)Training all models...$(NC)"
	python experiments/train_stress_model.py --model all --calibrate
	@echo "$(GREEN)✓ All models trained!$(NC)"

loso: ## Run LOSO cross-validation
	@echo "$(GREEN)Running LOSO evaluation...$(NC)"
	python experiments/evaluate_loso.py --model dnn_xgboost --save-results
	@echo "$(GREEN)✓ LOSO complete!$(NC)"

loso-all: ## Run LOSO for all models
	@echo "$(GREEN)Running LOSO for all models...$(NC)"
	python experiments/evaluate_loso.py --model all
	@echo "$(GREEN)✓ LOSO complete!$(NC)"

soc: ## Run SOC simulations
	@echo "$(GREEN)Running SOC simulations...$(NC)"
	python experiments/run_soc_simulation.py --scenario both
	@echo "$(GREEN)✓ SOC simulations complete!$(NC)"

soc-synthetic: ## Run synthetic SOC scenario
	@echo "$(GREEN)Running synthetic SOC...$(NC)"
	python experiments/run_soc_simulation.py --scenario synthetic --stress cyclic
	@echo "$(GREEN)✓ Synthetic SOC complete!$(NC)"

soc-cicids: ## Run CICIDS-based SOC scenario
	@echo "$(GREEN)Running CICIDS SOC...$(NC)"
	python experiments/run_soc_simulation.py --scenario cicids
	@echo "$(GREEN)✓ CICIDS SOC complete!$(NC)"

reproduce: ## Reproduce all paper results (FULL)
	@echo "$(GREEN)Reproducing ALL paper results...$(NC)"
	@echo "$(YELLOW)This will take 2-4 hours!$(NC)"
	python experiments/reproduce_paper_results.py --all
	@echo "$(GREEN)✓ Reproduction complete! Check results/paper/$(NC)"

reproduce-quick: ## Quick reproduction (skip LOSO)
	@echo "$(GREEN)Quick reproduction (skip LOSO)...$(NC)"
	python experiments/reproduce_paper_results.py --skip-loso
	@echo "$(GREEN)✓ Quick reproduction complete!$(NC)"

reproduce-tables: ## Reproduce tables only
	@echo "$(GREEN)Reproducing tables...$(NC)"
	python experiments/reproduce_paper_results.py --tables-only
	@echo "$(GREEN)✓ Tables complete!$(NC)"

reproduce-figures: ## Reproduce figures only
	@echo "$(GREEN)Reproducing figures...$(NC)"
	python experiments/reproduce_paper_results.py --figures-only
	@echo "$(GREEN)✓ Figures complete!$(NC)"

# ============================================================================
# Data
# ============================================================================

download-wesad: ## Instructions for downloading WESAD
	@echo "$(BLUE)WESAD Dataset Download Instructions:$(NC)"
	@echo ""
	@echo "1. Visit: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/"
	@echo "2. Fill out the request form"
	@echo "3. Download the dataset (ZIP file, ~4GB)"
	@echo "4. Extract to your desired location"
	@echo "5. Update WESAD_DIR in src/config.py"
	@echo ""
	@echo "Expected structure:"
	@echo "  WESAD/"
	@echo "  ├── S2/"
	@echo "  │   ├── S2.pkl"
	@echo "  │   ├── S2_quest.csv"
	@echo "  │   └── ..."
	@echo "  ├── S3/"
	@echo "  └── ..."

download-cicids: ## Instructions for downloading CICIDS
	@echo "$(BLUE)CICIDS2017 Dataset Download Instructions:$(NC)"
	@echo ""
	@echo "1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html"
	@echo "2. Download CSV files"
	@echo "3. Extract to your desired location"
	@echo "4. Update CICIDS_DIR in src/config.py"
	@echo ""
	@echo "Note: CICIDS is optional. Synthetic data can be used instead."

verify: ## Verify installation
	@echo "$(GREEN)Verifying installation...$(NC)"
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
	@python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
	@python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
	@python -c "import xgboost as xgb; print(f'XGBoost: {xgb.__version__}')"
	@echo "$(GREEN)✓ Core packages verified!$(NC)"
	@python -c "from data.wesad_loader import WESADLoader; print('✓ Project modules OK')"

verify-gpu: ## Check GPU availability
	@echo "$(GREEN)Checking GPU...$(NC)"
	@python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'{len(gpus)} GPU(s) available') if gpus else print('No GPU detected (CPU only)')"

# ============================================================================
# Documentation
# ============================================================================

docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✓ Docs generated! Open docs/_build/html/index.html$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving docs at http://localhost:8000$(NC)"
	cd docs/_build/html && python -m http.server

# ============================================================================
# Utilities
# ============================================================================

notebook: ## Start Jupyter notebook
	@echo "$(GREEN)Starting Jupyter notebook...$(NC)"
	jupyter notebook

shell: ## Start IPython shell
	@echo "$(GREEN)Starting IPython shell...$(NC)"
	ipython

info: ## Show project information
	@echo "$(BLUE)Project Information:$(NC)"
	@echo "  Name: Stress-Adaptive Defense Framework"
	@echo "  Version: 1.0.0"
	@echo "  Python: $(shell python --version)"
	@echo "  Install location: $(shell pip show stress-adaptive-defense 2>/dev/null | grep Location | cut -d' ' -f2)"
	@echo ""
	@echo "$(BLUE)Directory Structure:$(NC)"
	@tree -L 2 -I '__pycache__|*.pyc|*.egg-info' . || ls -R

size: ## Show project size
	@echo "$(BLUE)Project Size:$(NC)"
	@du -sh . 2>/dev/null || echo "Could not determine size"
	@echo ""
	@echo "$(BLUE)Code Statistics:$(NC)"
	@find src -name "*.py" | xargs wc -l | tail -1 || echo "Could not count lines"

# ============================================================================
# Release
# ============================================================================

dist: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution...$(NC)"
	python -m build
	@echo "$(GREEN)✓ Distribution built in dist/$(NC)"

upload-test: dist ## Upload to TestPyPI
	@echo "$(YELLOW)Uploading to TestPyPI...$(NC)"
	python -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)✓ Uploaded to TestPyPI$(NC)"

upload: dist ## Upload to PyPI (PRODUCTION)
	@echo "$(RED)WARNING: This will upload to PyPI!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python -m twine upload dist/*; \
		echo "$(GREEN)✓ Uploaded to PyPI$(NC)"; \
	fi
