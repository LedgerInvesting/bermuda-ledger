Triangles
===============

Traditional Triangles
-----------------------

Most of the models used in actuarial science are designed to work with aggregated
data, meaning that Bermuda does not assume any knowledge of losses at the level
of individual claims. The actuarial community traditionally aggregates
loss data into a form known as a “loss triangle”. An example of a
traditional loss triangle is shown below:

============= ======= ======= ======= =======
Accident Year AY+0    AY+1    AY+2    AY+3
============= ======= ======= ======= =======
1988          952000  1529000 2813000 3647000
1989          849000  1564000 2202000 
1990          983000  2211000         
1991          1657000                 
============= ======= ======= ======= =======

The etymology of the term “triangle” should be fairly obvious from this
view – the set of observed cells is a triangular region in the upper
left of the table. In some cases, we have historical losses from many
accident years, but we don’t have any data past a fixed development lag
cutoff. In such cases, the data in the table forms a trapezoid instead
of a triangle, but the generic term triangle is commonly used to loss
“trapezoids” as well.

Sometimes, loss triangles are presented in a format where the column
headers are evaluation dates instead of development lags. In the example
above, we could relabel the columns headers as 1988, 1989, 1990, 1991,
and move the data so that it forms a triangle in the upper right instead
of the upper left of the table. It should be readily apparent that we
can convert from development lags to evaluation dates and back without
any issues, and that both formats contain exactly the same information.

The example loss triangle above presents paid losses, but it should be
clear that we could just as easily present another metric – perhaps
incurred losses, case reserves or claim counts – in the same format. The
tabular format above can only handle one metric at a time, so actuaries
typically speak of a “paid loss triangle”, an “incurred loss triangle”,
and so forth.

Tabular Triangles
-----------------------

The traditional triangular data format is clean and compact, but it
doesn’t generalize easily. The table below presents a triangle in a
**tabular** format. (This is not the data format that Bermuda uses
internally, but it is a useful mental model for reasoning about
triangles.)

================ ============== =============== ========= =============
Experience Start Experience End Evaluation Date Paid Loss Reported Loss
================ ============== =============== ========= =============
1988-01-01       1988-12-31     1988-12-31      952000    1722000
1988-01-01       1988-12-31     1989-12-31      1529000   3830000
1988-01-01       1988-12-31     1990-12-31      2813000   3603000
1988-01-01       1988-12-31     1991-12-31      3647000   3835000
1989-01-01       1989-12-31     1989-12-31      849000    1581000
1989-01-01       1989-12-31     1990-12-31      1564000   2192000
1989-01-01       1989-12-31     1991-12-31      2202000   2528000
1990-01-01       1990-12-31     1990-12-31      983000    1834000
1990-01-01       1990-12-31     1991-12-31      2211000   3009000
1991-01-01       1991-12-31     1991-12-31      1657000   2305000
================ ============== =============== ========= =============

This format offers a few advantages over the traditional format. First,
the traditional format encourages sloppy thinking about data types. Each
cell of a traditional loss triangle is defined by two coordinates: an
experience period (which is an interval in time) and an evaluation date
(which is a single point in time). In the traditional format, both
experience period and evaluation date are represented with ambiguous
labels like “1993”, which could refer to either an interval in time or
some undefined point within that interval. Some authors refer to the
interval [``1998-01-01``, ``1998-12-31``] as “AY1998” and the evaluation
date ``1998-12-31`` as “CY1998”, but this sort of terminology encourages
sloppy thinking. In the tabular format, we represent the experience
period with the pair of values “Experience Start” and “Experience End”,
and the evaluation date as a date. This may seem like useless pedantry
at this point, but we will see shortly how some real-world triangle-like
data makes this distinction important.

(As an aside, we define our coordinates with an experience period and
evaluation date, whereas the traditional format uses experience period
and development lag. As noted above, there is no substantive difference
between evaluation date and development lag. However, there two
competing standards for the definition of development lag. We use the
convention that development lag is equal to the signed difference
between the evaluation date and the *end* of the experience period.
Other authors use the convention that development lag is the signed
difference between the evaluation date and the *start* of the experience
period. In this presentation, we use evaluation date mostly because
standard conventions for representing intervals between dates
unambiguously are not as robust as conventions for representing dates
themselves.)

Another advantage is that the tabular format can easily be extended to
accomodate multiple fields. In the example table above, we show both
paid loss and reported loss. The traditional format would require two
separate triangles to store the same information, whereas this format
stores arbitrarily many attributes for each cell within the same table
row.

We can extend this idea by adding columns to the tabular format with
arbitrary metadata. Say we are modeling loss development for a private
passenger auto portfolio, and our loss development model could vary by
state and coverage. Rather than try to manage :math:`50 \times 12 = 600`
separate triangles, we can simply add State and Coverage as columns in
our tabular format, and concatenate all 600 triangles into a single
table.

We can take this idea further still by including metadata columns that
represent whether the experience periods are accident-basis or policy
basis; whether the losses are gross or net of reinsurance; whether
individual claims are capped, and if so, at what value; whether the
losses include DCC or LAE; and so forth. (Of course, for many of these
metadata columns, constructing a model that appropriately handles mixed
data may be non-trivial.)

Taxonomy of Triangle Data
-------------------------

The final advantage of the tabular format is that it is able to handle
unusual loss development data much more naturally. The traditional
triangular format carries with it a strong implication that every
experience period will be the same length, the interval between every
pair of successive development lags will be the same, and every cell in
the upper-left portion of the table will be observed. Some actuaries
believe that all triangles satisfy all of the above criteria. People who
have done consulting work or have analyzed reinsurance submissions know
that all of the above assumptions can be violated on a regular basis.
Therefore, a format that can handle violations of the above assumptions
is not a mere theoretical nicety, but an essential prerequisite to
modeling real-world data.

Essentially all of the actuarial literature on loss development assumes
triangles that satisfy all of the above criteria. It’s understandable
why this is the case: most of the literature is focused on new
techniques for loss development, and it’s easier to present a new
technique as applied to clean data, rather than spending a great deal of
effort handling edge cases. Since unusual loss development data is
rarely discussed, there is no standardized terminology for describing
unusual triangles.

We at Ledger Investing have developed a taxonomy for these unusual
triangles, which we have found greatly facilitates communication. We
describe our terms bloew.

A **square** triangle has the same amount of time between successive
experience periods as between successive evaluation dates. In the
examples shown above, both the experience periods and evaluation date
intervals are annual. If this condition does not hold (for example,
experience periods are annual but evaluation date intervals are
quarterly), the triangle is **non-square**. (Yes, we are aware of the
incongruity of discussing square triangles.)

A **complete** triangle has observations at every cell, except for cases
where a cell’s evaluation date is greater than the maximum evaluation
date in the triangle. An **incomplete** triangle has some cells with
missing observations.

A **regular** triangle has the same length for every experience period,
and the same interval of time between successive evaluation dates or
development lags. A **semi-regular** triangle has every experience
period the same length, but the interval of time between successive
evaluation dates or development lags is inconsistent. For example, a
semi-regular triangle may contain observations from the following set of
evaluation dates: [``2018-12-31``, ``2019-06-30``, ``2019-09-30``,
``2020-06-30``, ``2020-12-31``, ``2021-03-31``].

An **irregular** triangle has evaluation periods with inconsistent
lengths. For example, an irregular triangle may contain observations
from the following set of experience periods: [(``2018-01-01``,
``2018-09-30``), (``2018-10-01``, ``2019-12-31``), (``2020-01-01``,
``2020-06-30``)]. It’s important to note that a triangle is considered
irregular if it has inconsistent experience period lengths, even if its
evaluation dates are consistent. This is because many loss development
models are able to handle semi-regular triangles much more easily and
naturally than irregular triangles.

An **erratic** triangle has evaluation periods that are not strictly
disjoint. For example, an erratic triangle may contain observations from
the following set of experience periods: [(``2018-01-01``,
``2018-12-31``), (``2018-07-01``, ``2019-09-30``), (``2019-07-01``,
``2019-12-31``)]. As with irregular triangles, the only criteria for a
triangle qualifying as erratic is whether there is any overlap in
experience periods. A triangle is erratic (and therefore not
semi-regular or irregular) even if all of its experience periods are the
same length and its evaluation dates follow a consistent pattern. The
principal challange that erratic triangles pose compared to irregular
triangles is that common statistical assumptions about independent and
identically distributed (i.i.d.) noise terms go out the window.

Finally, there are a couple of terms that we at Ledger use in a way that
is not entirely consistent with the broader actuarial community. First,
as we noted earlier, traditional triangles can only hold one type of
data – .e.g., paid losses or open claims. Given that tabular
representations of triangles don’t have this restriction, we can speak
of a single triangle with multiple attributes: we talk about a triangle
with paid losses and open claims, rather than a paid loss triangle and
an open claim triangle.

Second, the actuarial community often refers to the sum of paid losses
and outstanding case reserves as “incurred” losses. The problem with
this terminology is that it excludes IBNR, which is uncomfortable –
after all, IBNR stands for incurred but not reported! IBNR therefore
definitionally belongs in incurred losses. We use the term “reported”
losses to refer to the sum of paid losses and cases reserves, and
“incurred” losses to refer to the sum of reported losses and IBNR. These
definitions imply the identity ``incurred - reported = IBNR``, which
happens to align very nicely with names of the terms.

As an aside, this also means that we rarely work with incurred losses
directly as modeling inputs. Usually, one of our primary objectives in
modeling is to develop an independent estimate of IBNR. We reserve the
term “incurred losses” to refer specifically to the estimate of ultimate
losses that was booked by an insurer at a specific point in time. We
refer to our estimates of ultimate losses as “estimated ultimate
losses”, to avoid confusion with insurer estimates of IBNR, which are
ontologically privileged by virtue of the fact that they are reflected
in the insurer’s financial statements.
