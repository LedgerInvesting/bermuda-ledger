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

Check out the `full documentation <https://ledger-investing-bermuda-ledger.readthedocs-hosted.com/en/latest/?badge=latest>`_.

If you're new to Bermuda, take a look at the 
`Quick Start <https://ledger-investing-bermuda-ledger.readthedocs-hosted.com/en/latest/quick-start.html>`_
guide
for a brief overview of its functionality, or the
`User Guide <https://ledger-investing-bermuda-ledger.readthedocs-hosted.com/en/latest/user-guide/index.html>`_ 
for a more complete explanation
of Bermuda's design decisions, insurance triangles, and Bermuda's overall architecture.
The 
`Tutorials <https://ledger-investing-bermuda-ledger.readthedocs-hosted.com/en/latest/tutorials/index.html>`_ 
section includes common usage
patterns.

If you're interested in contributing to Bermuda,
take a look at our
`Developer Guide <https://ledger-investing-bermuda-ledger.readthedocs-hosted.com/en/latest/developer-guide/index.html>`_.

Installation
-------------

Core:

..  code-block:: bash

    python3.11 -m pip install bermuda

Developing:

..  code-block:: bash

    python3.11 -m pip install 'bermuda[dev]'

Docs:

..  code-block:: bash

    python3.11 -m pip install 'bermuda[docs]'

.. |build-status| image:: https://github.com/LedgerInvesting/bermuda-ledger/actions/workflows/test.yml/badge.svg
    :target: https://github.com/LedgerInvesting/bermuda-ledger/blob/main/.github/workflows/test.yml

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff

.. |docs| image:: https://readthedocs.com/projects/ledger-investing-bermuda-ledger/badge/?version=latest
    :target: https://ledger-investing-bermuda-ledger.readthedocs-hosted.com/en/latest/?badge=latest
