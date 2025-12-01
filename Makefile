# Global settings
RM = rm -rf
GPU_ID = 0
CUDA_TAG = cu124
VERSION = `uv version --short`


# ---- For Developer

.PHONY: reset
reset:
	rm -r ./.venv || true
	rm uv.lock || true

.PHONY: install
install:
	uv sync --extra ${CUDA_TAG} --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: lint
lint:
	uv run ruff check --output-format=full
	uv run ruff format --diff

.PHONY: mypy
mypy:
	uv run mypy src


# .PHONY: document_local
# document_local:
# 	$(RM) public
# 	uv run sphinx-build docs/ public/

.PHONY: test
test:
	uv run pytest tests -m "not gpu_test" --cov=src --cov-report term-missing --durations 5

.PHONY: gpu_test
gpu_test:
	uv run pytest tests -m "gpu_test"

# ----

# ---- Docker Images

.PHONY: push_docker_images
push_docker_images: 
	make -C docker push VERSION=${VERSION}

# ----
