|build-status| |ruff| |docs|

Bermuda 
-----------------------

Bermuda is a Python package for the representation, manipulation,
and exploration of insurance loss triangles, created
by `Ledger Investing <https://ledgerinvesting.com>`_.
It offers a user-friendly interface for:

* Loading and saving insurance loss triangles using a number of formats (e.g. JSON, CSV, Pandas :code:`DataFrame` objects, binary files).
* A single :code:`Triangle` class for manipulating triangles of varying complexities (e.g. ragged, multi-program, cumulative or incremental triangles).
* An intuitive :code:`Cell` type that can hold multiple data types and metadata.
* A collection of useful :code:`Cell`- and :code:`Triangle`-level functionality, including summarizing, aggregating, extending, filtering, and bootstrapping.

If you're new to Bermuda, take a look at the `Quick Start <quick-start.rst>`_ guide
for a brief overview of its functionality, or the
`User Guide <user-guide/index.rst>`_ for a more complete explanation
of Bermuda's design decisions, insurance triangles, and Bermuda's overall architecture.
The `Tutorial <tutorials/index.rst>`_ section includes common usage
patterns.

If you're interested in contributing to Bermuda,
take a look at our `Developer Guide <developer-guide/index.rst>`_.

..  toctree::
    :titlesonly:

    install.rst
    quick-start.rst
    user-guide/index.rst
    tutorials/index.rst
    developer-guide/index.rst
    api/index.rst

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |build-status| image:: https://github.com/LedgerInvesting/bermuda/actions/workflows/test.yml/badge.svg
    :target: https://github.com/LedgerInvesting/bermuda/blob/main/.github/workflows/test.yml

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff

.. |docs| image:: https://readthedocs.com/projects/ledger-investing-bermuda/badge/?version=latest
    :target: https://ledger-investing-bermuda.readthedocs-hosted.com/en/latest/?badge=latest

