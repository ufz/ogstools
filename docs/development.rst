Development
====================================


Development setup
-----------------

Create a virtual environment, activate it and install required packages:

    python -m venv .venv
    source .venv/bin/activate # run this in every new shell session
    pip install -e ".[test]"

    # enable basic style checks once:
    pre-commit install


CLI scripts can now be simply run:

    msh2vtu --help


Testing with `tox` and `pytest`
-------------------------------

Test environment are created and run with [`tox`](https://tox.wiki).
Tests are executed via [`pytest`](https://docs.pytest.org/en/7.2.x/)

To run the tests:

    tox # parallelize with `tox -p`

You can view a coverage report by opening `htmlcov/index.html` in a browser.

You can also run a single test environment with e.g.:

    tox -e py39

The following environments are available:

.. command-output:: tox list

Create a package
----------------


    pyproject-build


Packages can then be found in `dist/`.

Build documentation
-------------------

    tox -e docs
    make html


This will create the documentation files in `docs/_build/html`.

For development use the auto-generating and -reloading web server:

    tox -e devdocs
    # open http://127.0.0.1:5500 in a web browser
