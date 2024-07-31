help:  ## Show this help
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST) | column -t -s :

.PHONY : setup setup_headless test coverage check clean docs cleandocs preview

setup:  ## Setup a virtual environment and install all development dependencies
	python -m venv .venv --upgrade-deps
	.venv/bin/pip install -e .[dev,test,docs]
	.venv/bin/pre-commit install
	@echo
	@echo ATTENTION: You need to activate the virtual environment in every shell with:
	@echo source .venv/bin/activate

setup_headless:  ## Install vtk-osmesa and gmsh without X11 dependencies
	.venv/bin/pip uninstall gmsh vtk -y
	.venv/bin/pip install --extra-index-url https://wheels.vtk.org vtk-osmesa
	.venv/bin/pip install -i https://gmsh.info/python-packages-dev-nox gmsh

setup_devcontainer:  ## Internal usage
	rm -rf .venv-devcontainer
	python -m venv .venv-devcontainer --upgrade-deps
	.venv-devcontainer/bin/pip install -e .[dev,test,docs,feflow]
	.venv-devcontainer/bin/pip uninstall gmsh vtk -y
	.venv-devcontainer/bin/pip install --extra-index-url https://wheels.vtk.org vtk-osmesa
	.venv-devcontainer/bin/pip install -i https://gmsh.info/python-packages-dev-nox gmsh

test:  ## Runs the unit tests
	pytest

coverage:  ## Runs the unit tests generating code coverage reports
	coverage run -m pytest
	coverage report --no-skip-covered
	coverage html
	coverage xml

check:  ## Runs various checks with pre-commit
	pre-commit run --all-files

clean:  ## Cleans up temporary files
	rm -r .coverage htmlcov

docs:  ## Builds the documentation
	make html -C docs

cleandocs:  ## Cleans up temporary documentation files
	rm -r docs/_build
	rm -r docs/auto_examples
	rm -r docs/reference/*.rst

preview:  ## Runs an auto-updating web server for the documentation
	make docs
	python docs/server.py


.PHONY: requirement
requirement:
	## conda init zsh
	## conda create --prefix /tmp/ogstools-test-env-py312  python=3.12
	## conda activate /tmp/ogstools-test-env-py312
	@version_output=$$(python --version 2>&1); \
	version=$$(echo "$$version_output" | awk '{print $$2}' | awk -F'.' '{print $$1 "_" $$2}'); \
	venv_dir=".venv_py$$version"; \
	if [ -d "$$venv_dir" ]; then \
		read -p "Virtual environment '$$venv_dir' already exists. Do you want to continue and recreate it? (y/N): " confirm; \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "Aborting."; \
			exit 1; \
		fi; \
		echo "Continuing to recreate virtual environment."; \
		rm -r "$$venv_dir"; \
	fi; \
	echo "Creating virtual environment in $$venv_dir"; \
	python -m venv $$venv_dir; \
	echo "Activating virtual environment and installing packages"; \
	. $$venv_dir/bin/activate && pip install . && pip freeze -l > requirements/requirements_py$$version.txt && \
	echo "Activating virtual environment and installing packages"; \
	. $$venv_dir/bin/activate && pip install .[dev,tests,doc] && pip freeze -l > requirements/requirements_allextras_py$$version.txt && \
	echo "Deleting virtual environment"; \
	rm -r $$venv_dir
