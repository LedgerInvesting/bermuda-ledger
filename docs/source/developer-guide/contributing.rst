Contributing
------------

Dependencies
~~~~~~~~~~~~

It is recommended that you use Python > 3.11 for Bermuda development, and
that you manage Python dependencies with a virtual environment. To
install Bermuda’s Python dependencies in a virtual environment `./env`
run the following

::

   python3.11 -m venv env
   source env/bin/activate
   pip install "bermuda[dev]"


Installing Bermuda in Editable Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the dependencies have been installed, install the Bermuda package in
editable mode using the following

::

   pip install -e .


Testing
~~~~~~~

To check that Bermuda has been installed correctly, run the unit tests

::

   pytest


Developer Tools
~~~~~~~~~~~~~~~

Before contributing to Bermuda, please install the pre-commit hooks. This
will run the python linter `Black <https://github.com/psf/black>`__
before each commit.

::

   pre-commit install


Contributing
~~~~~~~~~~~~

Please follow these guidelines when contributing to Bermuda

-  All work should be done on feature branches and merged into master
   through a pull request.
-  Open a draft PR when you start working on an issue so others know
   it’s being addressed.
-  Utilize the Ruff formatter and linting tool as a `pre-commit hook <#developer-tools>`__
-  Write docstrings for user-facing code using the `Google
   format <https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings>`__.
