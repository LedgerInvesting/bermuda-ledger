Installation
===============

User Installation
---------------------

Users can install the latest version of Bermuda via pip from PyPi by running:

..  code:: bash

    python3.11 -m pip install bermuda-ledger

We highly recommend using a virtual environment. Bermuda can
be installed using package managers, like ``uv`` and ``poetry``,
for example:

..  code:: bash

    python3.11 -m uv pip install bermuda-ledger
    python3.11 -m poetry install

You can also download Bermuda from Github directly:

..  code:: bash

    python3.11 -m pip install git+ssh://git@github.com/LedgerInvesting/bermuda-ledger.git


``dev`` and ``docs`` dependencies
------------------------------------

For those wanting to develop and/or contribute
to the documentation, you can install the additional
dependencies using the ``bermuda[docs]`` or ``bermuda[dev]``
``pip install`` syntax, e.g.

..  code:: bash

    python3.11 -m pip install "bermuda-ledger[dev]"
