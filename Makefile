.PHONY: demo test test-smoke list demo-all catalog demo-orchestrator \
        install install-spatial-domain-identification install-full install-dev \
        bot-telegram bot-feishu bot-multi bot-list memory-server

## Virtual-environment + installation targets

venv:
	python3 -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install:
	pip install -e .

install-spatial-domain-identification:
	pip install -e ".[spatial-domain-identification]"

install-full:
	pip install -e ".[full]"

install-dev:
	pip install -e ".[dev]"

setup: venv
	.venv/bin/pip install -e .

setup-full: venv
	.venv/bin/pip install -e ".[full]"

## Demo & test targets

demo:
	python spatialclaw.py run spatial-preprocessing --demo --output /tmp/spatialclaw_demo

test:
	python -m pytest -v

test-smoke:
	python spatialclaw.py list
	python -m pytest tests/test_registry.py tests/test_spatialclaw_run_contracts.py tests/memory/test_memory_server_security.py -v

list:
	python spatialclaw.py list

catalog:
	python scripts/generate_catalog.py

demo-orchestrator:
	python spatialclaw.py run spatial-orchestrator --demo --output /tmp/spatialclaw_orchestrator_demo

demo-all:
	python spatialclaw.py run spatial-preprocessing --demo --output /tmp/spatial_preprocessing
	python spatialclaw.py run spatial-domain-identification --demo --output /tmp/spatial_domain_identification
	python spatialclaw.py run spatial-de --demo --output /tmp/spatial_de
	python spatialclaw.py run spatial-svg-detection --demo --output /tmp/spatial_svg_detection
	python spatialclaw.py run spatial-statistics --demo --output /tmp/spatial_statistics
	python spatialclaw.py run spatial-cell-annotation --demo --output /tmp/spatial_cell_annotation
	python spatialclaw.py run spatial-deconvolution --demo --output /tmp/spatial_deconvolution
	python spatialclaw.py run spatial-cell-communication --demo --output /tmp/spatial_cell_communication
	python spatialclaw.py run spatial-condition-comparison --demo --output /tmp/spatial_condition_comparison
	python spatialclaw.py run spatial-velocity --demo --output /tmp/spatial_velocity
	python spatialclaw.py run spatial-trajectory --demo --output /tmp/spatial_trajectory
	python spatialclaw.py run spatial-cnv --demo --output /tmp/spatial_cnv
	python spatialclaw.py run spatial-integration --demo --output /tmp/spatial_integration
	python spatialclaw.py run spatial-registration --demo --output /tmp/spatial_registration
	python spatialclaw.py run spatial-visualization --demo --output /tmp/spatial_visualization
	python spatialclaw.py run spatial-orchestrator --demo --output /tmp/spatial_orchestrator

## Bot targets

bot-telegram:
	python bot/telegram_bot.py

bot-feishu:
	python bot/feishu_bot.py

bot-multi:
	python -m bot.run --channels $(CHANNELS)

bot-list:
	python -m bot.run --list

## Memory server

memory-server:
	python spatialclaw.py memory-server
