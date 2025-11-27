# Global settings
RM = rm -rf
GPU_ID = 0
CUDA_TAG = cu124

VERSION = 1.4.0
PREVIOUS_VERSIONS = 1.2.0 1.3.0 1.4.0


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


.PHONY: dev_test
dev_test:
	uv run pytest tests -m "not (long_test or regression)"

.PHONY: dev_test_lf
dev_test_lf:
	uv run pytest tests -m "not (long_test or regression)" --lf

.PHONY: regression_test
regression_test:
	uv run pytest tests -m "regression" --previous-versions $(PREVIOUS_VERSIONS)

.PHONY: document_local
document_local:
	$(RM) public
	uv run sphinx-build docs/ public/

.PHONY: document
document:
	$(RM) public
	uv run sphinx-multiversion docs/ public/

# ----

# ---- Docker Images

.PHONY: build_push_image
build_push_image:
	make -C docker/ci_image build_push_ci VERSION=$(VERSION) && \
	make -C docker/ci_image build_push_release VERSION=$(VERSION)

.PHONY: build_push_release_image
build_push_release_image:
	make -C docker/ci_image build_push_release VERSION=$(VERSION)


.PHONY: build_release_sif
build_release_sif:
	make -C docker/ci_image build_release_sif VERSION=$(VERSION)

.PHONY: push_release_sif
push_release_sif:
	make -C docker/ci_image push_release_sif VERSION=$(VERSION)

.PHONY: build_push_release_sif
build_push_release_sif: build_release_sif push_release_sif

# ----
