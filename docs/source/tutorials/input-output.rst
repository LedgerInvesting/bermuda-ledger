Triangle Input/Output
=====================

To get started with ``Triangle`` objects, consider importing the ``meyers_tri`` object 
from Bermuda.

.. code-block:: python

   from bermuda import meyers_tri

which returns the following triangle:

..  code-block:: python

    >>> meyers_tri

           Cumulative Triangle 

     Number of slices:  1 
     Number of cells:  100 
     Triangle category:  Regular 
     Experience range:  1988-01-01/1997-12-31 
     Experience resolution:  12 
     Evaluation range:  1988-12-31/2006-12-31 
     Evaluation resolution:  12 
     Dev Lag range:  0.0 - 108.0 months 
     Fields: 
       earned_premium
       paid_loss
       reported_loss
     Common Metadata: 
       currency  USD 
       country  US 
       risk_basis  Accident 
       reinsurance_basis  Net 
       loss_definition  Loss+DCC


Bermuda ``Triangle`` objects can be exported in a variety of formats for use in other
applications. The recommended format for saving triangles to disk is to use the
``Triangle.to_binary`` method. This method saves the triangle in a binary format that
is optimized for reading into and writing from Python. The ``to_binary`` method
requires a file path as an argument, and forces the use of a ``.trib`` extension. We also
offer a compressed binary file format, which can be saved using the ``.tribc`` extension.
While this format is more memory efficient, it is much slower to read and write, so it 
is generally not recommended.

..  code-block:: python

    meyers_tri.to_binary('meyers_tri.trib')


Once saved, binary files can be read back into Python using the ``Triangle.from_binary``
method

..  code-block:: python

    from bermuda import Triangle

    meyers_tri = Triangle.from_binary('meyers_tri.trib')



Tabular Formats
-------------------------------------------------

Triangles can also be saved in a variety of commonly used tabular formats. For convenience, we've
labeled these formats as ``wide``, ``long``, and ``array`` formats. The ``wide`` format is a table where each row represents a single cell in the triangle. It has fixed columns in the order ``['period_start', 'period_end', 'evaluation_date', *fields, *metadata]``. It's considered ``wide`` because all of the fields are split out as separate columns. Take a
look at the Meyers triangle as a wide pandas ``DataFrame``:


..  code-block:: pycon

    >>> meyers_wide_df = meyers_tri.to_wide_data_frame()
    >>> meyers_wide_df.axes

    [RangeIndex(start=0, stop=100, step=1),
     Index(['period_start', 'period_end', 'evaluation_date', 'earned_premium',
            'paid_loss', 'reported_loss', 'reinsurance_basis', 'risk_basis',
            'country', 'currency', 'loss_definition'],
           dtype='object')]


This is in contrast to the ``long`` format, which is a table with a single row for each value in
each cell of the triangle. Note the following triangle is longer (300 rows instead of 100), and the columns fit the 
pattern ``['period_start', 'period_end', 'evaluation_date', *metadata, 'field', 'value']``:

..  code-block:: pycon

    >>> meyers_long_df = meyers_tri.to_long_data_frame()
    >>> meyers_long_df.axes

    [RangeIndex(start=0, stop=300, step=1),
     Index(['period_start', 'period_end', 'evaluation_date', 'reinsurance_basis',
            'risk_basis', 'country', 'currency', 'loss_definition', 'field',
            'value'],
           dtype='object')]

Both of these formats can be saved to disk using the ``to_wide_csv`` and ``to_long_csv`` methods, and 
read back into memory using ``from_wide_csv`` and ``from_long_csv``.

..  code-block:: python

    meyers_tri.to_wide_csv('meyers_tri_wide.csv')
    meyers_tri.to_long_csv('meyers_tri_long.csv')

    meyers_tri = Triangle.from_wide_csv('meyers_tri_wide.csv', detail_cols=[])
    meyers_tri = Triangle.from_long_csv('meyers_tri_long.csv')

Note that the wide format requires the user to specify either ``detail_cols`` or ``field_cols``. 
This tells Bermuda which columns in the wide format are cell fields (i.e. ``paid_loss``, ``earned_premium`` etc.) and which are metadata (i.e. ``coverage`` 

Finally, we allow export into what we call an ``array`` format -- essentially a triangle-shaped data frame (we avoid the term 'triangle' in order to avoid confusion with the ``Triangle`` class). This is the format that actuaries would typically be most familiar with, where each row represents a single 
period, and each column represents a certain development lag from the end of that period. Note that 
our convention throughout the Bermuda library is to index development lags from the *end* of the 
period rather than the beginning. Therefore, the first column of the array format is typically a
0 lag observation that takes place at the end of the period. The development lags are denoted in months, and periods are saved as date objects denoting the period start.

The array data frame can only operate on a single-sliced triangle and will only return values
for a single field. Any missing evaluation dates for a period will show up as NaN.

..  code-block:: pycon

    >>> from datetime import date
    
    >>> clipped_meyers = meyers_tri.clip(max_eval = date(1990, 12, 31))
    >>> paid_array = clipped_meyers.to_array_data_frame('paid_loss')
    >>> paid_array

           period       0         12         24
    0  1988-01-01  952000  1529000.0  2813000.0
    1  1989-01-01  849000  1564000.0        NaN
    2  1990-01-01  983000        NaN        NaN

    >>> reported_array = clipped_meyers.to_array_data_frame('reported_loss')
    >>> reported_array

           period        0         12         24
    0  1988-01-01  1722000  3830000.0  3603000.0
    1  1989-01-01  1581000  2192000.0        NaN
    2  1990-01-01  1834000        NaN        NaN

Some fields are often static with respect to evaluation date, like ``earned_premium`` or ``earned_exposure``. Rather than display these fields in a triangle it often makes sense to output them for each period at the latest evaluation date. This can be done using the ``to_right_edge_data_frame`` method, which will provide the values of all fields at the right edge of the triangle.

.. code-block:: pycon

    >>> right_edge_array = clipped_meyers.to_right_edge_data_frame()
    >>> right_edge_array

           period evaluation_date  paid_loss  reported_loss  earned_premium
    0  1988-01-01      1990-12-31    2813000        3603000         5812000
    1  1989-01-01      1990-12-31    1564000        2192000         4908000
    2  1990-01-01      1990-12-31     983000        1834000         5454000

We're frequently presented with data provided in these array formats that we'd like to load into
Bermuda ``Triangle`` objects. This can be accomplished using the ``Triange.from_array_data_frame`` method. This method requires a single field argument, but also allows for other metadata to be provided
along with the tabular data.

.. code-block:: python

   from bermuda import Metadata

   reported_tri = Triangle.from_array_data_frame(
       reported_array, 
       'reported_loss', 
       metadata = Metadata(loss_definition="Loss+DCC")
    )

Often we'll have multiple tabular triangles representing different fields, in which case we
can use the ``bermuda.io.array_triangle_builder`` helper function to build up a single
multi-field triangle.

.. code-block:: python

   from bermuda.io import array_triangle_builder

   loss_triangle = array_triangle_builder(
       dfs = [reported_array, paid_array], 
       fields = ['reported_loss', 'paid_loss'],
       metadata = Metadata(loss_definition="Loss+DCC")
    )

This triangle now has both paid and reported losses, but it's missing earned premium. Let's
read that in from a tabular format and add it to the triangle. The ``Triangle.from_statics_data_frame`` 
function assumes the first column represents the period of the associated data. 
Note that the ``static_data_tri`` must have metadata matching the existing 
loss triangle or the static values will not be attached correctly.

.. code-block:: python

   >>> static_df = right_edge_array[['period', 'earned_premium']]

   >>> static_data_tri = Triangle.from_statics_data_frame(
   ...     static_df, 
   ...     metadata = Metadata(loss_definition="Loss+DCC")
   ... )
   >>> full_triangle = loss_triangle.add_statics(static_data_tri)
   >>> full_triangle

           Cumulative Triangle


     Number of slices:  1
     Number of cells:  6
     Triangle category:  Regular
     Experience range:  1988-01-01/1990-12-31
     Experience resolution:  12
     Evaluation range:  1988-12-31/1990-12-31
     Evaluation resolution:  12
     Dev Lag range:  0.0 - 24.0 months
     Fields:
       earned_premium
       paid_loss
       reported_loss
     Common Metadata:
       risk_basis  Accident
       loss_definition  Loss+DCC


Other Formats
-------------------------------------------------
Bermuda also supports reading and writing triangles in a JSON format. This format is 
particularly useful for interacting with our upcoming modeling API - more to come on that soon!

.. code-block:: python

    meyers_tri.to_json('meyers_tri.json')
    meyers_tri = Triangle.from_json('meyers_tri.json')

Finally, we support reading triangles from and exporting to the ``chainladder`` package.


.. code-block:: python

    import chainladder as cl
    from bermuda import chain_ladder_to_triangle

    chain_ladder_tri = cl.load_sample("clrd")
    bermuda_tri = chain_ladder_to_triangle(chain_ladder_tri)
    chain_ladder_tri = bermuda_tri.to_chain_ladder()
