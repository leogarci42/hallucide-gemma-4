# `make` on its own starts everything: the engine bridge AND the interface.
# `make help` lists the rest.
.DEFAULT_GOAL := both
.PHONY: front bridge both build lint ask measure stop clean help

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# Port 8100, not 8000: Gemma itself (vLLM/Ollama) is on 8000, and the bridge
# runs on the same box when Gemma is served locally -- 8000 would collide.
BRIDGE_PORT ?= 8100

# The engine, when it is running. Left unset, the interface says so instead of
# inventing an answer.
ENGINE_ORIGIN ?= http://localhost:$(BRIDGE_PORT)

front: stop  ## Start the interface on http://localhost:3000
	@echo "→ http://localhost:3000  (Ctrl+C to stop)"
	@[ -n "$(ENGINE_ORIGIN)" ] \
		&& echo "  engine: $(ENGINE_ORIGIN)" \
		|| echo "  no engine: set ENGINE_ORIGIN=http://host:port to connect one"
	@cd front && [ -d node_modules ] || npm install
	@cd front && ENGINE_ORIGIN=$(ENGINE_ORIGIN) npm run dev

bridge:  ## Serve the engine over HTTP on BRIDGE_PORT (default 8100)
	PORT=$(BRIDGE_PORT) $(PY) bridge.py

both:  ## Engine and interface together
	@$(MAKE) bridge & sleep 2; $(MAKE) front

measure:  ## Measure the verifier, then the routing if a model is reachable
	$(PY) -m scripts.measure_verifier
	@echo
	@$(PY) -m scripts.measure_routing --json measurements.json || \
		echo "  routing not measured: no model backend. Nothing reported."

ask:  ## Run the medical pipeline from the shell: make ask Q="your question"
	@[ -n "$(Q)" ] || (echo 'usage: make ask Q="your question"' && exit 1)
	$(PY) -m scripts.ask_medical "$(Q)"

build:  ## Production build of the interface
	@cd front && [ -d node_modules ] || npm install
	@cd front && npm run build

lint:  ## Types and lint on the interface
	@cd front && npx tsc --noEmit && npx eslint . --max-warnings=0

stop:  ## Free the port if an interface is already running
	@pkill -f "[n]ext dev" 2>/dev/null && echo "stopped the running interface" || true

clean:  ## Remove build output and dependencies
	@rm -rf front/.next front/node_modules

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t 12
