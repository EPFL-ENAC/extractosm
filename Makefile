.PHONY: help

help: ## Show available targets
	@echo "============================================================================="
	@echo "Root Makefile"
	@echo "============================================================================="
	@echo ""
	@echo "📋 Prerequisites:"
	@echo "  • make"
	@echo "  • uv (Python project management)"
	@echo "  • wget and osmium (Downloading and processing OpenStreetMap data)"
	@echo ""
	@echo "📦 Setup:"
	@echo "  make install          Install all dependencies"
	@echo "  make clean            Clean all artifacts"
	@echo ""
	@echo "✨ Linting and Code Formatting:"
	@echo "  make format           Format all code"
	@echo "  make lint             Run all linters"
	@echo ""
	@echo "============================================================================="

.PHONY: install install-example
install: ## Install dependencies
	uv sync --dev

install-example: ## Install dependencies for example usage
	uv sync --dev --all-extras

# Variables
UV := uv
RUFF ?= $(UV) run ruff
PYTEST ?= $(UV) run pytest

# FILES can be overridden to target specific files (for git hooks)
FILES ?= .

PY_FILES   := $(filter %.py,$(FILES))

.PHONY: format
format: ## Format code with ruff and prettier
	@set -e; \
	if [ -n "$(PY_FILES)" ]; then \
		$(RUFF) format $(PY_FILES) && \
		$(RUFF) check --fix $(PY_FILES); \
	else \
		$(RUFF) format . && \
		$(RUFF) check --fix .; \
	fi

.PHONY: lint
lint: ## Check code formatting
	@set -e; \
	if [ -n "$(PY_FILES)" ]; then \
		$(RUFF) format --check $(PY_FILES) && \
		$(RUFF) check $(PY_FILES); \
	else \
		$(RUFF) format --check . && \
		$(RUFF) check .; \
	fi


.PHONY: clean
clean: ## Clean up generated files
	rm -rf .ruff_cache
	rm -rf .data
	rm -rf data

.PHONY: pbf-download pbf-lausanne
pbf-download: ## Download latest OSM PBF file for Switzerland
	@rm -rf .data/geofabrik
	@mkdir -p .data/geofabrik
	wget https://download.geofabrik.de/europe/switzerland-latest.osm.pbf -O .data/geofabrik/switzerland-latest.osm.pbf

pbf-lausanne: ## Extract OSM data for Lausanne area
	@rm -rf .data/lausanne-*.osm.pbf
	osmium extract -b 6.5,46.5,6.8,46.6 .data/geofabrik/switzerland-latest.osm.pbf -o .data/lausanne-all.osm.pbf
	@mkdir -p data
	osmium tags-filter .data/lausanne-all.osm.pbf --overwrite -o data/lausanne-filtered.osm.pbf \
		n/public_transport,n/highway=bus_stop,n/railway \
		a/public_transport \
		nwr/route=bus,tram,train,subway,trolleybus,light_rail,ferry,monorail,trolleybus \
		r/type=route_master \
		r/public_transport=stop_area
