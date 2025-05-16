Plotting triangle data
===========================

Bermuda has a range of plots built upon the
`Altair package <https://altair-viz.github.io/index.html>`_.
Our plots allow exploring the structure of triangles,
loss development dynamics, and predictive distributions
and uncertainty. Plots, by default, open in browser windows
because they are rendered to HTML, and all plots
are interactive, offering deeper insights into the triangle
data.

Plots are available as methods on triangles directly, 
or can be imported from the ``bermuda.plot`` module
and used directly on a triangle.
Most plots can be created with custom metrics, although
there are a standard set of sensible defaults for
all plots. This tutorial will explain how to interact
with the plots and customize them for your own
use cases.

In all cases, we'll be basing the plots below
on simulated data using the structure of the
Meyers triangle, which is created in the code box below.
You can largely ignore this data simulation and move
on to the plot demonstrations.

.. altair-plot::

   import numpy as np
   from bermuda import meyers_tri

   rng = np.random.default_rng(1234)

   # Subset to a more typical 'training' data set
   raw = meyers_tri.clip(max_eval=max(meyers_tri.periods)[1])

   # Create a triangle of fake data
   def logistic(x, alpha: float = 1.0, beta: float = 0.0):
       """Assume a logistic curve for loss development patterns"""
       return 1 / (1 + np.exp(-alpha * (x - beta)))

   paid_percent = logistic(np.array(raw.dev_lags()), alpha=0.1, beta=48)
   reported_percent = logistic(np.array(raw.dev_lags()), alpha=0.05, beta=36)
   paid_share = {lag_months: share for lag_months, share in zip(raw.dev_lags(), paid_percent)}
   reported_share = {lag_months: share for lag_months, share in zip(raw.dev_lags(), reported_percent)}
   open_share = {lag_months: share for lag_months, share in zip(raw.dev_lags(), reported_percent[::-1])}

   ultimates = {cell.period: rng.normal(1e6, 1e5) for cell in raw}
   loss_volatility = {lag_months: 0.1 * share for lag_months, share in zip(raw.dev_lags(), paid_percent[::-1])} 
   total_claims = 10_000

   triangle = raw.derive_fields(
       paid_loss = (
           lambda cell: rng.lognormal(np.log(ultimates[cell.period] * paid_share[cell.dev_lag()]), loss_volatility[cell.dev_lag()])
       ),
       earned_premium = 1.2e6,
       open_claims = lambda cell: rng.poisson(total_claims * open_share[cell.dev_lag()]),
       reported_claims = total_claims,
    ).derive_fields(
       reported_loss = (
           lambda cell: max(
               cell["paid_loss"], 
               rng.lognormal(np.log(ultimates[cell.period] * reported_share[cell.dev_lag()]), loss_volatility[cell.dev_lag()])
            )
        ),
    )

Triangle structure and simple EDA
------------------------------------------
Users can get explore the overall structure of a triangle
by using the 'data completeness' plot, which shows the number
of data fields per cell in the triangle.

.. altair-plot::
    triangle.plot_data_completeness()

We can clearly see which periods and which lags have data
from this plot. If there are different numbers of data fields
per cell, then these are illustrated as different colors:

.. altair-plot::
    (triangle[:10].select(["paid_loss"]) + triangle[10:]).plot_data_completeness()

Users can also hover over all the available plot marks (i.e. points, bars, lines etc.) 
to further explore the triangle data.

Actuaries and insurance data scientists might frequently want to see the latest
loss ratios in a triangle, which can be visualized by the 'right edge' plot:

.. altair-plot::
    triangle.plot_right_edge()

It is also common to visualize triangle data as a heatmap, such as the loss
ratios. By default, ``plot_heatmap`` plots ``paid_loss`` ratios, but users can
pass in functions of their own too (see below).

.. altair-plot::
    triangle.plot_heatmap()

Similarly, users can plot the age-to-age (ATA) factors as a Tukey boxplot
using:

.. altair-plot::
    triangle.plot_atas()

By default, this plot shows the ``paid_loss`` ATAs.

More advanced loss development dynamics
------------------------------------------

Bermuda includes a number of more advanced plots showing
loss development dynamics. The growth curve plot is a typical
plot of loss ratios by development lag and accident year:

.. altair-plot::
    triangle.plot_growth_curve()

Users can select individual accident years to highlight
them clearly, as well as zoom in on harder-to-see or
overlapping areas of the plot.

We can also display similar information as a 'mountain' plot,
where each line represents a development lag in the triangle:

.. altair-plot::
    triangle.plot_mountain()

This plot can be more useful for detecting trends across
accident years.

The ballistic and broom plots allow comparing paid and reported
loss ratios directly, and the assumption that paid and reported
losses should converge within accident periods at higher development
lags. The ballistic plot 
compares paid to reported loss ratios
on the x- and y-axes, respectively, (named after the accident periods' 'missile-like' trajectories),
while the broom plot
plots the ratio of paid-to-reported losses to the paid
loss ratio (named after the tendency for spread out at higher
development lags as paid loss ratios develop towards ultimate).

.. altair-plot::
   triangle.plot_ballistic()

.. altair-plot::
   triangle.plot_broom()

The hose plot displays incremental paid loss ratios
against reported loss ratios. A common pattern is for
experience period's incremental
paid loss ratios to grow in size in the medium-term
and slowly decay towards reported loss ratios, making
the image of a hose spraying water.

.. altair-plot::
   triangle.plot_hose()

Rather than loss ratios, we can also look at claim count development
patterns. The 'drip' plot shows the proportion of open claims ('open claim
share') for each experience period across development lags against
the reported loss ratio.
The lines create a dripping effect as claims are closed
and loss ratios approach their ultimate values.

.. altair-plot::
    triangle.plot_drip()

Finally, another visualization of the age-to-age (ATA) factors
is the sunset plot, where ATAs are plotted for each development
lag separately against evaluation period. By default we use
a boxcox transform to make the ATAs more visible at higher development
lags (when ATAs are converging to 1.0), and the plot overlays LOESS
curves on the ATAs per development lag to illustrate the patterns
across experience periods.

.. altair-plot::
    triangle.plot_sunset()

Reference table
--------------------

The table below summarizes the plots available, a short
description, and their default behaviour.

.. list-table::

   * - Plot name
     - Description
     - Defaults
   * - ``plot_data_completness``
     - Plot the number of data fields in each cell of the triangle
       as a scatter plot.
     - By default, plots all fields in each cell. Uncertainty indicated
       using ribbons/bands or vertical line segments.
   * - ``plot_right_edge``
     - Plot the latest diagonal of loss ratios and premium in the triangle.
     - Plots all loss fields as loss ratios and earned premium as a bar chart.
   * - ``plot_heatmap``
     - Heatmap plot of triangle quantities. Predictions/uncertainty shown with black borders.
     - By default, plots paid loss ratios.
   * - ``plot_atas``
     - Plots age-to-age factors as Tukey boxplots. Individual ATAs are shown as points.
     - By default, plots paid loss ATAs.
   * - ``plot_growth_curve``
     - Plots triangle metrics per experience period by development lag. Loss metrics typically
       demonstrate a growth curve pattern as they approach ultimate.
     - Plots paid loss ratios, by default.
   * - ``plot_mountain``
     - Plots triangle metrics per development lag across experience periods. Loss metrics typically
       show a mountain ridge-type pattern.
     - Plots paid loss ratios by default.
   * - ``plot_ballistic``
     - Plots paid/reported triangle metrics per experience period. Loss metrics typically create a
       missile-like trajectory as paid and reported losses converge.
     - Defaults to paid loss ratios (x-axis) and reported loss ratios (y-axis) 
   * - ``plot_broom``
     - Plots paid losses against paid-to-reported ratios. Typically creates a broom-type pattern
       as the paid-to-reported ratio converges on 1.0. 
     - Defaults to paid loss ratio against paid-to-reported ratios, with a vertical x-axis line at 1.0. 
   * - ``plot_hose``
     - Plots incremental paid loss ratios against paid loss ratios, typically forming a hose-spray
       pattern.
     - Incremental paid loss ratios on the y-axis aginst paid loss ratios on the x-axis.
   * - ``plot_drip``
     - Plots the proportion of open claims against reported loss ratios, typically forming a dripping
       pattern.
     - The proportion of open claims ('open claim share') against reported loss ratios.
   * - ``plot_sunset``
     - Plots boxcox-transformed ATAs per development lag across calendar or evaluation period, typically
       forming a sky-line pattern.
     - Boxcox-transformed paid ATAs.

Plotting custom metric functions
------------------------------------------------

While the plots above all have their default metrics, users can change which metrics are shown,
depending on the plot type. There are currently two ways of displaying custom
metrics. The ``plot_heatmap``, ``plot_growth_curve``, ``plot_mountain``, ``plot_atas``,
and ``plot_sunset`` plots each have a ``metric_dict`` keyword argument which is a dictionary
of metric name-metric function key-value pairs. Multiple metrics are displayed as faceted/panel
plots. For instance, imagine we want to plot paid and reported losses as growth curves. We
can do so by filling the ``metric_dict`` argument with two (lambda) functions of each cell
in the triangle.

.. altair-plot::

   triangle.plot_growth_curve(
       metric_dict = {
           "Paid LR": lambda cell: cell["paid_loss"] / cell["earned_premium"],
           "Reported LR": lambda cell: cell["reported_loss"] / cell["earned_premium"],
        },
        width=250,
        height=200,
    )

Notice that the keys of the dictionary are mapped to the plot names when there are multiple
metrics.

The metric functions can either be a function of each cell, or a function of each cell and
the previous cell. This option is only applied to cells within the same experience period.
For instance, if we wanted to plot paid and reported ATAs, we could utilize this pattern:

.. altair-plot::

   triangle.plot_atas(
       metric_dict = {
           "Paid ATAs": lambda cell, prev_cell: cell["paid_loss"] / prev_cell["paid_loss"],
           "Reported ATAs": lambda cell, prev_cell: cell["reported_loss"] / prev_cell["reported_loss"],
        },
        width=300,
        height=200,
    )

The ``plot_ballistic``, ``plot_broom``, ``plot_drip``, and ``plot_hose`` plots can't plot multiple
metrics, but users can plot custom x- and y-axis metrics on the single plot by
specifying the ``axis_metrics`` dictionary. This assumes the first argument
is the x-axis metric and the second is the y-axis metric. For example, here's
the drip plot with open claim share plotted against reported losses rather than
reported loss ratios:

.. altair-plot::

   triangle.plot_drip(
       axis_metrics = {
           "Reported Loss": lambda cell: cell["reported_loss"],
           "Open Claim Share": lambda cell: cell["open_claims"] / cell["reported_claims"],
       }
    )


Multi-slice triangles and faceting
------------------------------------

We've already seen one version of faceting in the previous section where multiple
metrics are plotted. The Bermuda plots will also automatically facet based on
triangle slices. Users can therefore control faceting by creating or summarizing
multiple slices and have fine-grained control of the plots they want.
When there are multiple metrics as well, both metrics and slices are faceted.
As an example, here's the heatmap plot for multiple slices and multiple metrics.

.. altair-plot::

   (triangle.derive_metadata(id=1) + triangle.derive_metadata(id=2)).plot_heatmap(
       metric_dict = {
           "Paid LR": lambda cell: cell["paid_loss"] / cell["earned_premium"],
           "Reported LR": lambda cell: cell["reported_loss"] / cell["earned_premium"],
        },
        width=250,
        height=200,
    )

Users can use the ``width`` and ``height`` arguments available in each plot to control
the plot size. In the presence of multiple slices, these values refer to each plot.
Users can also specify the ``ncols`` argument to change how many columns are faceted.

.. altair-plot::

   (triangle.derive_metadata(id=1) + triangle.derive_metadata(id=2)).plot_data_completeness(
       ncols=1,
       width=400,
       height=300,
   )

Prediction uncertainty
-------------------------

Most Bermuda plots are able to handle cells with predictions as well as point estimates,
without any other input from the user. Taking the ``plot_right_edge`` plot as an example,
here's some simulated predictions for the last few diagonals in the data:

.. altair-plot::

   prediction_evals = triangle.evaluation_dates[-3:]
   triangle_predictions = triangle.derive_fields(
       paid_loss = lambda cell: (
          rng.normal(cell["paid_loss"], 1e3 * cell.dev_lag(), size=1000) 
          if cell.evaluation_date in prediction_evals
          else cell["paid_loss"]
       ),
       reported_loss = lambda cell: (
          rng.normal(cell["reported_loss"], 1e3 * cell.dev_lag(), size=1000) 
          if cell.evaluation_date in prediction_evals
          else cell["reported_loss"]
       ),
   )

   triangle_predictions.plot_right_edge(width=500)

Users can also switch how the uncertainty is plotted in certain plots using the ``uncertainty_type``
argument.

.. altair-plot::

   triangle_predictions.plot_right_edge(width=500, uncertainty_type="segments")

In general, uncertainty is difficult to show in plots, and users will need to make their
own judgements as to whether a plot illustrates uncertainty clearly for their own
use-case. A good example is the ``plot_growth_curve`` plot where uncertainty 
bands will quickly overlap:

.. altair-plot::

   triangle_predictions.plot_growth_curve()

Users can make use of the interactive features of the plot to help explore predictions
for individual experience periods by clicking on individual periods' lines to highlight
them. We can also use the multi-slice faceting to our advantage in these plots
by separating the triangle into different slices based on experience period:

.. altair-plot::

   triangle_predictions.derive_metadata(
       id = lambda cell: cell.period_start
   ).plot_growth_curve(width=250, height=200, ncols=2)

Customizing plot aesthetics
------------------------------

Each of Bermuda's plot return either an ``altair.LayerChart`` object
or an ``altair.ConcatChart`` object. Users familiar with Altair can 
make changes to the charts, although some changes are harder to make than
others.

Titles, colors and top-level configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way of changing the chart title is to use the ``.properties``
method on chart objects:

.. altair-plot::

   triangle.plot_right_edge().properties(title="My chart")

If you want to customize the title, or any other parts of plots, then Altair's 
`top-level chart configuration <https://altair-viz.github.io/user_guide/configuration.html#top-level-chart-configuration>`_
tools are the main methods.

.. altair-plot::

   triangle.plot_right_edge().configure(
       background="#eeeeee",
    ).configure_title(
       font="monospace"
    ).configure_axisX(
       titleColor="blue",
       labelColor="green",
       tickColor="red",
       labelFontSize=10,
   ).configure_legend(
       direction="horizontal",
       orient="bottom",
   ).configure_bar(
       stroke="black",
   ) 

If a change doesn't take effect, it might be that the particular configuration
has been set at a higher-precedence level of encodings, following Altair's order
of 
`global configuration v local configuration v encoding precedence <https://altair-viz.github.io/user_guide/customization.html#which-to-use>`_.
More features will be added for users to easily control the finer-grained
chart details.

Resolving scales in faceted plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the faceted plot above showing growth curves by accident years,
it is hard to compare across plots because they have different scales.
Luckily, we can use Altair's ``resolve_scale`` method to share x- and y-axis
scales across subplots. We also make this plot easier to read by removing the facet
titles and changing the legend positioning.

.. altair-plot::

   triangle_predictions.derive_metadata(
       id = lambda cell: cell.period_start
   ).plot_growth_curve(
       width=150, height=100, ncols=5, facet_titles=[""]*len(triangle.periods)
   ).resolve_scale(
       x="shared", y="shared", 
   ).resolve_axis(
       y="shared"
   ).properties(
       background="#eeeeee",
   ).configure_legend(
       orient="top",
       direction="horizontal",
       titleFontSize=10,
       labelFontSize=10,
       offset=1,
   ).configure_concat(
       spacing=2,
   ).interactive()


Users can also use the ``resolve_*`` functions to change how legend and color
schemes are handled. For instance, the multi-slice and multi-metric
heatmap plot shown above, there is a legend per plot and the color schemes
are treated differently. That is, if you hover over equivalent loss ratio values,
you'll see that the color schemes diverge in each plot.
Instead, we might want to share the gradient across plots and
have a single legend. This could be handled by the following: 

.. altair-plot::

   (triangle.derive_metadata(id=1) + triangle.derive_metadata(id=2)).plot_heatmap(
       metric_dict = {
           "Paid LR": lambda cell: cell["paid_loss"] / cell["earned_premium"],
           "Reported LR": lambda cell: cell["reported_loss"] / cell["earned_premium"],
        },
        width=300,
        height=200,
    ).resolve_scale(
        color="shared",
    ).resolve_legend(
        color="shared",
    ).configure_legend(
        direction="horizontal",
        orient="top",
    )

