# Thin wrappers over uv/cargo — everything here also works as a plain command.
.DEFAULT_GOAL := help
UV ?= uv
# Parts whose rungs are all implemented — the ones with figures to regenerate.
COMPLETE_PARTS ?= $(shell $(UV) run ladder status --complete)

.PHONY: help setup test lint fmt check scaffold docs status demo rust writeups writeups-check clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Create the environment and install the project with dev tools
	$(UV) sync --group dev

test:  ## Run the Python test suite
	$(UV) run pytest

lint:  ## Lint and check formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:  ## Autoformat and autofix
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

check: lint test scaffold-check docs-check rust  ## Everything CI runs

scaffold:  ## Create any parts/ files the ladder spec implies
	$(UV) run python tools/scaffold.py

scaffold-check:  ## Fail if parts/ has drifted from the ladder spec
	$(UV) run python tools/scaffold.py --check

docs:  ## Regenerate docs/ladder.md from the spec
	$(UV) run ladder docs

docs-check: docs  ## Fail if docs/ladder.md was stale
	@git diff --quiet docs/ladder.md || { echo "docs/ladder.md was stale; commit the regenerated file"; exit 1; }

writeups: $(addprefix writeup-,$(COMPLETE_PARTS))  ## Regenerate figures and metrics for every complete part

writeup-%:  ## Regenerate one part's writeup figures and metrics, e.g. make writeup-A
	$(UV) run python tools/writeup_figures.py $*

writeups-check:  ## Fail if a complete part's committed metrics have drifted
	@for part in $(COMPLETE_PARTS); do $(UV) run python tools/writeup_figures.py $$part --check || exit 1; done

status:  ## Progress across the ladder
	$(UV) run ladder status

demo:  ## Run the harness demo end to end
	$(UV) run python examples/harness_demo.py

rust:  ## Build and test the Rust workspace
	cd rust && cargo test

clean:  ## Remove run outputs and caches
	rm -rf artifacts .pytest_cache .ruff_cache rust/target
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
