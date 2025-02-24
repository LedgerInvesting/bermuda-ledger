Architecture
=====================

While almost all end-user interaction with triangle data
can be accomplished with public methods available on ``Triangle``
objects, it helps to understand major elements in the internal
structure of ``Triangle``.

``Cell``
------------

The basic building block class in Bermuda triangles is the ``Cell``.
There are three types of cells: :code:`Cell` and its subclasses
:code:`CumulativeCell` and :code:`IncrementalCell`.
All cells consists of an experience period
start date and end date, a development lag, one or more observed
values, and a :code:`Metadata` object. 
A loose representation of a single cell may look
something like this:

-  **Experience Start Date**: ``2017-07-01``
-  **Experience End Date**: ``2017-07-31``
-  **Evaluation Date**: ``2018-10-31`` (development lag of 15 months)
-  **Metadata**:

   -  **Country**: US
   -  **Currency**: USD
   -  **Risk Basis**: Accident
   -  **Reinsurance Basis**: Gross
   -  **Per Occurrence Limit**: $1M
   -  **Loss Definition**: Loss+DCC
   -  **Details**:

      -  **State**: Texas
      -  **Coverage**: Bodily Injury

-  **Values**:

   -  **Paid Loss**: $1,234,567
   -  **Reported Loss**: $2,345,678
   -  **Earned Premium**: $3,456,789

If this data structure looks very similar to a single row of a tabular
triangle, that’s not by coincidence. The structure of cells
intentionally mirrors tabular triangle rows. We refer to individual
observed values within a cell as “fields”. In the example
above, the fields are paid loss, reported loss, and earned
premium. :code:`Cell.values` are implemented as a Python dictionary,
so there are essentially no restrictions on the fields that can be
stored within an observation. In the example above, all of the fields
are amounts of money, but we could just as easily have included reported
claim counts, closed claim counts, or any number of other fields.
Furthermore, since each cell’s values are independent, there is
no requirement that all observations have same set of fields. If paid
loss is present in every cell, but earned premium is only present
in some cell, that’s not a problem.

Every observation contains a set of metadata
associated with it, including items such as the country the risk is in,
the currency that loss and premium amounts are denominated in, whether
the exposure period is on accident-basis or policy-basis, and so forth.
The set of attributes is extensible via the :code:`details` dictionary attribute.
The example above shows state and coverage, but this is just for
illustrative purposes; state and coverage are not required members of
the details field, and any other arbitrary attributes can be added to
the details field if relevant.

We track all of this data at the cell level because it can be critical
for appropriate modeling when mixing data from several different
sources. For example, it would obviously be inappropriate to fit a
combined loss development model with volume-weighting to data from two
different portfolios, one of which is measured in dollars and the other
in yen, without converting currencies first! Similarly, mixing
accident-basis and policy-basis experience periods is usually a recipe
for disaster, unless special mitigation measures are taken. With that
being said, the metadata on each observation is not burdensome for
end-users. If an end-user doesn’t care about one or more metadata fields
in a given analytical context, they can simply omit them and Bermuda
will gracefully supply sensible defaults.

Triangles
---------

Collections of ``Cells`` are aggregated into a ``Triangle``. From the
end-user’s perspective, a ``Triangle`` is an undifferentiated
agglomeration of ``Cell``\ s. Under the hood, the ``Triangle`` class
indexes and groups cells by common metadata attributes; we refer to
these internal groups of ``Cell``\ s as “slices”. A slice consists of a
list of ``Cell``\ s, all of which pertain to the same logical group of
exposures. For example, a slice may contain cells for the accident-month
triangle for Company X, or for the policy-quarter triangle for private
passenger bodily injury claims in the state of Missouri for Product Y
written by Company Z. Slice grouping is automatically determined based
on the metadata associated with each cell.

Operations on Triangles
-----------------------

We stated earlier that one of the design goals of ``bermuda.Triangle``
is ergonomics. To that end, triangles include a rich set of operations
out of the box. We summarize some of the most common operations below.
In general, when we have a choice between implementing a behavior as a
function or as a method, we prefer the method in almost all cases. There
are a few reasons for this. First, all methods on triangles are
non-mutating/non-destructive, so there’s no semantic distinction between
functions and methods. Second, it tends to be easier and more natural to
express a sequence of operations on a triangle as a sequence of chained
method calls than as a nested sequence of function calls. Finally, from
a rhetorical perspective, we think of ``Triangle`` objects as having a
convenient and tidy namespace for holding operations on triangular data,
so we don’t have to import functions from another namespace or qualify
the function names.

Operators
---------

-  **Equality**: The ``==`` operator on triangles returns ``True`` if
   the contents of both operands are identical (not if the two operands
   are references to the same object, as the default behavior for Python
   objects).
-  **Concatenation**: The ``+`` operator on two triangles returns a
   single triangle with the concatenated contents of the two operands.

Properties
----------

Any given triangle ``triangle`` has the following basic properties:

-  ``triangle.slices`` returns a dictionary of slices contained in the
   triangle.
-  ``triangle.cells`` returns a list of all cells in the triangle.
-  ``triangle.periods`` is the sorted list of all distinct experience periods
   in the triangle.
-  ``triangle.dev_lags()`` is the sorted list of all distinct development
   lags in the triangle. ``dev_lag`` accepts ``unit`` as a keyword
   argument that can be ``month``, ``day`` or ``timedelta``.
-  ``triangle.evaluation_dates`` is the sorted list of all distinct
   evaluation dates in the triangle.
-  ``triangle.evaluation_date`` is the latest evaluation date in the
   triangle.
-  ``triangle.fields`` is the sorted list of all distinct fields in cells in
   the triangle.
-  ``triangle.metadata`` is the sorted list of all distinct metadata in the
   triangle.
-  ``triangle.common_metadata`` returns a single metadata element common to
   all cells in the triangle.
-  ``triangle.metadata_differences`` returns a list of unique metadata in the
   triangle, that are not in ``triangle.common_metadata``.

Triangles also implement several higher-order properties. For
explanation of Bermuda-specific triangle terminology, see the discussion
on triangle philosophy and terminology.

-  ``triangle.is_empty`` returns ``True`` if there are no cells in the
   triangle, and ``False`` otherwise.
-  ``triangle.is_disjoint`` returns ``True`` if all experience periods in the
   triangle are disjoint, and ``False`` if the triangle is erratic.
-  ``triangle.is_semi_regular`` tests whether the triangle is semi-regular.
-  ``triangle.is_regular`` tests whether the triangle is regular.
-  ``triangle.has_consistent_currency`` and ``triangle.has_consistent_risk_basis``
   test whether every cell in the triangle has the same currency or risk
   basis, respectively. These two pieces of metadata are the most common
   showstoppers for invalidating a modeling approach.
- ``triangle.is_incremental`` returns ``True`` if the triangle is incremental,
  otherwise ``False``.

Basic Mutators
--------------

Triangles have the following methods that return modified triangles:

-  ``triangle.select()`` accepts a list of field names. For each cell in the
   triangle, any fields that are not in the supplied list of names are
   removed from the cell’s set of values. If any cells don’t have any
   values in the list, then those cells are removed entirely.
-  ``triangle.clip()`` filters a triangle based on cutoff dates. For example,
   ``triangle.clip(max_eval=datetime.date(2018, 12, 31))`` removes all cells
   with an evaluation date after December 31st, 2018. ``clip`` accepts
   the keyword arguments ``min_eval``, ``max_eval``, ``min_period``,
   ``max_period``, ``min_dev``, ``max_dev``, and ``dev_lag_unit``.
   Multiple arguments can be supplied – if so, only those cells that
   satisfy all supplied conditions are returned.
-  ``triangle.right_edge`` returns the rightmost edge of the triangle – i.e.,
   for each distinct experience period within each slice, the cell with
   the latest evaluation date is retained and all other cells are
   dropped.

Representations
---------------

-  ``triangle.to_data_frame()`` returns a ``pandas.DataFrame`` representation
   of a triangle, for ease of graphing, exporting, and ad hoc
   manipulation. There are the I/O functions ``triangle.to_long_data_frame``
   and ``triangle.to_wide_data_frame`` used for transforming triangles
   to, and from, wide and long CSVs, respectively.
   Similarly, there are ``triangle.to_json()`` and ``triangle.to_binary()``
   output functions.
-  ``triangle._repr_html_()`` provides a friendlier rich-HTML representation
   of a triangle for use in Jupyter notebooks.

Intermediate Mutators
---------------------

Some triangle mutators require direct manipulation of individual cells.
Cells are fairly straightforward to work with, so this does not pose too
much of an obstacle. An individual cell ``cell``\ ’s experience period
start, experience period end, and development lag can be accessed via
``cell.period_start``, ``cell.period_end``, and ``cell.dev_lag``. The
internal cell representation of these values may be unintuitive, so be
warned. We can access individual fields within cells as (for example)
``cell.values["paid_loss"]``, or just ``cell["paid_loss"]`` for short.

-  ``triangle.filter()`` allows for filtering of triangles based on arbitrary
   cell-level predicates. For example,
   ``triangle.filter(lambda cel: cel["paid_loss"] > 0)`` removes all cells
   with zero (or negative) paid loss. The predicate function passed to
   ``filter`` must take a single argument (a single cell), and the
   predicate is then applied to every cell in the triangle, one by one.
   This means ``filter`` cannot be used to express conditions that
   depend on multiple cells.
-  ``triangle.derive_fields()`` allows for adding new fields to cells that
   are transformations of existing cells. For example,
   ``triangle.derive_fields(paid_LR=lambda ob: ob["paid_loss"] / ob["earned_premium"])``
   would add a new field ``paid_LR`` to every observation that contains
   the paid loss ratio according to the definition provided.
   ``derive_fields`` can also be used to overwrite existing fields.
- ``triangle.aggregate()`` allows for aggregation of a triangle's experience period
  or evaluation date resolution, such as turning quarterly triangles into annual
  triangles.
- ``triangle.summarize()`` allows for turning multi-slice triangles into a smaller
  number of triangles that share common metadata. Bermuda will automatically
  work out the greatest-common-denominator of ``Metadata`` objects in the triangle,
  and will try to combine fields using default aggregation functions for commonly-used
  fields (e.g. ``paid_loss``, ``reported_loss``, ``earned_premium`` etc.). Alternatively,
  users can pass in their own set of summarization functions.
- ``triangle.blend()`` allows for the blending of multiple triangles with the same cell fields
  using either a linear weighted average, or a 'mixture blend' that samples randomly from
  the different triangle fields according to weights passed in by the user. This is particularly
  useful if your triangle holds samples from upstream stochastic modelling.
- ``triangle.split()`` splits triangles by metadata attributes. For instance, if your triangle
  holds multiple lines of business triangles, you can split by the line of business metadata
  identifying attribute to obtain a dictionary of separate triangles.
- ``triangle.merge()`` offers triangle cell value joining functionality, where the ``join_type`` argument
  can be used to specify full, inner, left, right, left-anti or right-anti joining operations.
- ``triangle.coalesce()`` is similar to merge, but can take more than two triangles as input,
  where earlier triangles' cell fields take precedence over later triangles' cell fields.
  This is similar to an iterated left-join on multiple triangles.
- ``triangle.to_incremental()`` turns a cumulative triangle into an incremental triangle, or
  returns a no-op if the triangle is already incremental. ``triangle.to_cumulative()``
  provides the opposite functionality.
- ``triangle.add_statics()`` adds static field values from one triangle to the current triangle.
  Similar functionality might be achieved with a left-join ``merge`` operation or even ``derive_fields``,
  but ``add_statics`` offers greater control over merging single cell fields into the base triangle.
- ``triangle.make_right_triangle()`` creates a lower-diagonal of the existing triangle with empty cell
  field values.
- ``triangle.make_right_diagonal()`` creates a new triangle diagonal for user-specified evaluation dates.

Plots
-------

The ``Triangle`` class currently has a couple of useful visualizations using Plotly, but better
visualization functionality will be added in the future.

``triangle.plot_data_completeness()`` shows the triangular data structure as a scatter
plot in ``(experience_period, development_lag)`` coordinate space.
Each point represents a cell, colored proportional to the proportion of cell field values that are
present in the cell. If all cells have the same number of cell fields, they will all be the same
color.

``triangle.plot_right_edge()`` plots the most recent ('right edge') for, by default, paid and/or reported
loss ratios (using earned premium), but users can pass their own functions of cell values.
