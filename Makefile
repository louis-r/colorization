.DEFAULT_GOAL := help

.PHONY: clean-pyc lint clean-logs clean help

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean-pyc: ## Remove python artifacts.
	@find . | grep -E "(__pycache__|\.pyc|\.pyo|ipynb_checkpoints)" | xargs rm -rfv
	@find . -name '*~' -exec rm -fv {} \;

lint: ## Check style with pylint.
	@find . -iname "*.py" | xargs pylint

clean-logs: ## Remove logs.
	@find . -name '*.log' -exec rm -fv {} \;

clean: ## Remove python artifacts and logs.
	@make clean-pyc
	@make clean-logs
