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
