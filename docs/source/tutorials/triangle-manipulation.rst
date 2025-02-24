Common triangle manipulations
================================

This tutorial will guide you through some common triangle
manipulation functions available in Bermuda. First,
let's load some sample data to work with.

..  code-block:: python

    from bermuda import meyers_tri

``meyers_tri`` is one of the triangles from Glenn Meyers' monograph
`Stochastic Loss Reserving using Bayesian MCMC Models
<https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf>`_.

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

The Meyers triangle has 100 cells with 10 annual experience periods starting
in 1988 and 10 evaluation years ending in 2006.
The development lags range from 0 to 108 months, because Bermuda
assumes development lags are time since the end of each
experience period. The triangle also provides metadata, telling us
the denomination is US dollars, the domain of business is the US,
the triangle is accident basis, the values are net of reinsurance
ceded expenses, and the loss definition is loss plus defence and cost
containment expenses (DCC).

Clipping, filtering and aggregating triangles
-------------------------------------------------

It's common to need to remove portions of a triangle based
on experience period and/or evaluation date.
Bermuda offers a ``clip`` method that makes these operations
easy. For instance, imagine we wanted to turn Meyers' 10x10 triangle
into an upper diagonal triangle.
We can do so by clipping evaluation dates at a maximum of the
latest experience period end.

..  code-block:: python

    # Triangle.periods returns a list of (period_start, period_end) tuples
    _, latest_period_end = max(meyers_tri.periods)
    clipped = meyers_tri.clip(max_eval=latest_period_end)

The new triangle, ``clipped``, now has 55 cells
with an evaluation date range from 1998 through
1997.

..  code-block:: python

    >>> clipped

           Cumulative Triangle 

     Number of slices:  1 
     Number of cells:  55 
     Triangle category:  Regular 
     Experience range:  1988-01-01/1997-12-31 
     Experience resolution:  12 
     Evaluation range:  1988-12-31/1997-12-31 
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

``Triangle.clip`` can similarly clip for
minimum and maximum experience periods as well
as evaluation dates.

A more powerful, lower-level operation is ``Triangle.filter``,
which takes any function of cells that returns a boolean, and 
filters cells accordingly. For instance, the same clipping
operation as above could be performed with ``filter``:

..  code-block:: python

    clipped = meyers_tri.filter(
        lambda cell: cell.evaluation_date <= meyers_tri.periods[-1][1]
    )

Bermuda triangles can also be aggregated across their experience period
and evaluation period axes. For example, we could turn the Meyers
triangle into a single 10-year period using ``Triangle.aggregate``.

..  code-block:: python

    import datetime

    aggregated = meyers_tri.aggregate(
        period_resolution=(10, "year"),
        period_origin=datetime.date(1987, 12, 31),
    )

For this to work, we use the ``period_origin``
argument to tell Bermuda that we want the aggregation
to happen from 1987-12-31 onwards, which will sum values
for all cells until the last period through 1997-12-31.
By default, all cells are summed.
The result is a triangle with negative development lags
but a single period of 19 cells:

..  code-block:: python

    >>> (aggregated.periods, aggregated)

    ([(datetime.date(1988, 1, 1), datetime.date(1997, 12, 31))],
            Cumulative Triangle 
     
     
      Number of slices:  1 
      Number of cells:  19 
      Triangle category:  Regular 
      Experience range:  1988-01-01/1997-12-31 
      Experience resolution:  120 
      Evaluation range:  1988-12-31/2006-12-31 
      Evaluation resolution:  12 
      Dev Lag range:  -108.0 - 108.0 months 
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
     )

The negative development lags indicate that the first cell
for experience period 1988-1-1 to 1997-12-31 is evaluated
at 1988-12-31, 10 years prior to the end of the period.

Merging and summarizing multi-slice triangles
-------------------------------------------------------

Imagine we now have separate triangles
for paid losses and premiums, which might arise if someone has loaded
triangle data from different sources. We can create two separate triangles
by using the ``Triangle.select`` method.

..  code-block:: python

    meyers_paid = meyers_tri.select("paid_loss")
    meyers_premium = meyers_tri.select("earned_premium")

We can combine these two triangles in a number of ways. Simply
concatenating the two triangles will result in a multi-slice triangle:

..  code-block:: python

    >>> meyers_paid + meyers_premium

           Cumulative Triangle 

     Number of slices:  1 
     Number of cells:  200 
     Triangle category:  Regular 
     Experience range:  1988-01-01/1997-12-31 
     Experience resolution:  12 
     Evaluation range:  1988-12-31/2006-12-31 
     Evaluation resolution:  12 
     Dev Lag range:  0.0 - 108.0 months 
     Optional Fields: 
       earned_premium (50.0% coverage)
       paid_loss (50.0% coverage)
     Common Metadata: 
       currency  USD 
       country  US 
       risk_basis  Accident 
       reinsurance_basis  Net 
       loss_definition  Loss+DCC

The ``__repr__`` method now tells us that there are 200 cells,
and two ``Optional Fields`` that each have 50% coverage across
triangle cells.

To create the original triangle again, the canonical method is ``merge``,
which is available as a method on ``Triangle``.

..  code-block:: python

    >>> meyers_merged = meyers_paid.merge(meyers_premium)
    >>> meyers_merged

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
     Common Metadata: 
       currency  USD 
       country  US 
       risk_basis  Accident 
       reinsurance_basis  Net 
       loss_definition  Loss+DCC

There is an exception to this operation, which occurs if the two
triangles have different metadata. For instance, imagine we had
the paid and earned premium triangles above but distinct metadata.
We can create these triangles with help from the ``derive_metadata``
triangle method.

..  code-block:: python

    meyers_paid = meyers_tri.select("paid_loss").derive_metadata(
        details=dict(slice=1)
    )

    meyers_premium = meyers_tri.select("earned_premium").derive_metadata(
        details=dict(slice=2)
    )

A ``merge`` operation would now return the same as ``meyers_paid + meyers_premium``
because merging shouldn't take place across distinct metadata or triangle
slices. In this case, the canonical pattern is to ``summarize`` the
combined, multi-slice triangle.

..  code-block:: python

    >>> combined = meyers_paid + meyers_premium
    >>> combined.summarize()

which returns the single triangle that we started with.
``summarize`` works by figuring out the greatest common
denominator of metadata elements, and using that to summarize
triangles using a set of pre-defined field aggregation functions,
which for loss and premium fields are all summed.
If there is an unrecognized field, Bermuda will error. For instance,
let's create a new field called ``paid_losses`` rather than ``paid_loss``,
using the ``derive_fields`` method, and try to summarize the triangles:

..  code-block:: python

    >>> meyers_paid_2 = meyers_paid.derive_fields(paid_losses = lambda cell: cell["paid_loss"])
    >>> combined = meyers_paid_2 + meyers_premium
    >>> combined.summarize()

     ...
     TriangleError: Don't know how to aggregate `paid_losses` values

The result is a ``TriangleError`` that indicates ``summarize`` does not know
how to summarize, a priori, ``paid_losses``.
However, we can pass in a custom function to help tell Bermuda what to do.
Currently, this functionality is reserved for people comfortable with looking
in the internals of ``bermuda.utils.summarize.SUMMARIZE_DEFAULTS``, since the
custom function requires on Bermuda summarization logic:

..  code-block:: python

    >>> from bermuda.utils.summarize import _conforming_sum

    >>> combined.summarize(
    ...     summary_fns={"paid_losses": lambda v: _conforming_sum(v["paid_losses"])}
    ... )

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
       paid_losses                                                
     Common Metadata:                                             
       currency  USD                                              
       country  US                                                
       risk_basis  Accident                                       
       reinsurance_basis  Net                                     
       loss_definition  Loss+DCC

This procedure now returns a summarized triangle with both ``paid_loss`` and
``paid_losses``.

In some cases, we might have premium present in
two triangles that should not be summarized.
For instance, imagine we had two triangles for paid and reported losses,
each with earned premium as a field.
We can use ``summarize_premium=False`` in our call to ``summarize``
to ensure that premium fields are not summed.

..  code-block:: python

    paid = meyers_tri.select(["paid_loss", "earned_premium"]).derive_metadata(
        details=dict(slice=1)
    )
    reported = meyers_tri.select(["reported_loss", "earned_premium"]).derive_metadata(
        details=dict(slice=2)
    )

    summarized = (paid + reported).summarize(summarize_premium=False)

How can we check that the triangle ``summarized`` has the same earned premium as both
``paid`` and ``reported`` triangles? Triangle cells are easy to iterate over,
so one option is to zip both triangles, and iterate and compare their values, such as:

..  code-block:: python

    assert all(
        cell1["earned_premium"] == cell2["earned_premium"]
        for cell1, cell2
        in zip(summarized, paid)
    )

But Bermuda also provides an ``extract()`` method on triangles,
which returns a Numpy array and can make this easier.

..  code-block:: python

    assert (summarized.extract("earned_premium") == paid.extract("earned_premium")).all()
