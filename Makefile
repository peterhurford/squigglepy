# Variables
POETRY = poetry
PYTHON = $(POETRY) run python

# Install dependencies
install:
	$(POETRY) install

install-dev:
	$(POETRY) install --with dev

# Format code
format:
	$(POETRY) run black .
	$(POETRY) run ruff check . --fix

# Run linting
lint:
	$(POETRY) run ruff check .

# Run tests
test:
	$(POETRY) run pytest

# Help
help:
	@echo "Available commands:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install all dependencies including dev tools"
	@echo "  make format        Format code using Black and Ruff"
	@echo "  make lint          Run Ruff for linting"
	@echo "  make test          Run tests with pytest"