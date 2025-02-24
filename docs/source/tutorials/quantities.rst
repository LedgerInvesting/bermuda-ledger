Deriving key triangle quantities
==========================================

Bermuda currently has no utility functions specifically
for extracting quantities such as age-to-age factors or loss
ratios. Modeling functionality is part
of the upcoming Ledger Investing infrastructure,
which will include numerous advanced methods for the
modeling of loss development patterns.

If you want to extract deterministic quantities from
triangles, it's recommended to use base Python
routines and the low-level, core Bermuda
functionality, which offers users considerable power
and flexibility in triangle manipulation and exploration.
For instance, age-to-age factors could be extracted using: 

..  code:: python

    from bermuda import meyers_tri

    atas = {
        (cell.period_start.year, prev.dev_lag()): cell["paid_loss"] / prev["paid_loss"]
        for cell, prev
        in zip(meyers_tri[1:], meyers_tri[:-1])
        if cell.period == prev.period
    }

This extracts the quantities into a dictionary:

..  code:: python

    >>> atas
    {(1988, 0.0): 1.60609243697479,                                                                                                                                                                                                                         
     (1988, 12.0): 1.8397645519947678,                                                                                                                                                                                                                      
     (1988, 24.0): 1.2964806256665482,                                                                                                                                                                                                                      
     (1988, 36.0): 1.021113243761996,                                                                                                                                                                                                                       
     (1988, 48.0): 1.029001074113856,                                                                                                                                                                                                                       
     (1988, 60.0): 1.0174843423799582,                                                                                                                                                                                                                      
     (1988, 72.0): 1.0020518081559375,                                                                                                                                                                                                                      
     (1988, 84.0): 1.0010238034297414,                                                                                                                                                                                                                      
     (1988, 96.0): 1.0002556890820762,                                                                                                                                                                                                                      
     (1989, 0.0): 1.8421672555948174,                                                                                                                                                                                                                       
     ...
    }


These values can easily be used to create a Pandas
``DataFrame`` for further manipulation. For instance:

..  code:: python

    >>> from pandas import DataFrame
    >>> df = DataFrame.from_dict(atas, orient="index", columns=["ata"])
    >>> df

                       ata
    (1988, 0.0)   1.606092
    (1988, 12.0)  1.839765
    (1988, 24.0)  1.296481
    (1988, 36.0)  1.021113
    (1988, 48.0)  1.029001
    ...                ...
    (1997, 48.0)  1.052562
    (1997, 60.0)  1.003745
    (1997, 72.0)  1.018657
    (1997, 84.0)  1.009035
    (1997, 96.0)  1.001694

    [90 rows x 1 columns]

For volume-weighted age-to-age factors, users can use more
complex dict- and list-comprehensions along with Bermuda
utilities. For example, here we use the ``extract``, ``filter`` and 
``dev_lags()`` methods:

..  code:: python

    lags = meyers_tri.dev_lags()

    volume_weighted = {
        lag: (
            meyers_tri.filter(lambda cell: cell.dev_lag() == lag).extract("paid_loss").sum() / 
            meyers_tri.filter(lambda cell: cell.dev_lag() == prev_lag).extract("paid_loss").sum() 
        )
        for lag, prev_lag
        in zip(lags[1:], lags[:-1])
    }


Users can also use Bermuda's ``extract`` utility to extract
other quantities of interest. For instance, loss ratios could
be calculated using:

..  code:: python

    >>> lrs = meyers_tri.extract("paid_loss") / meyers_tri.extract("earned_premium")
    >>> lrs

    array([0.16379904, 0.26307639, 0.48399862, 0.62749484, 0.64074329,
           0.65932553, 0.67085341, 0.67222987, 0.6729181 , 0.67309016,
           0.17298289, 0.31866341, 0.44865526, 0.49551752, 0.50285249,
           0.50672372, 0.51202119, 0.51466993, 0.51568867, 0.51487368,
           0.18023469, 0.40539054, 0.51888522, 0.70260359, 0.74055739,
           0.74532453, 0.75210854, 0.76182618, 0.78254492, 0.78364503,
           0.32081317, 0.51984511, 0.61355276, 0.69699903, 0.75508228,
           0.83639884, 0.83872217, 0.83988383, 0.84046467, 0.84046467,
           0.17874952, 0.37207518, 0.50364404, 0.63904871, 0.6459532 ,
           0.66954354, 0.67721519, 0.67894131, 0.67894131, 0.68718834,
           0.22217973, 0.45927342, 0.53518164, 0.57284895, 0.58011472,
           0.58164436, 0.61759082, 0.61912046, 0.61969407, 0.6248566 ,
           0.29607372, 0.59695513, 0.79026442, 0.9443109 , 1.09415064,
           1.13782051, 1.13822115, 1.13842147, 1.13862179, 1.13862179,
           0.22685693, 0.38053421, 0.47694841, 0.56348335, 0.67288694,
           0.73216246, 0.75320161, 0.7546652 , 0.75521405, 0.75521405,
           0.25373134, 0.46153846, 0.64427861, 0.73536165, 0.75870647,
           0.78970532, 0.79085343, 0.79238423, 0.79257558, 0.79295829,
           0.28476421, 0.54070939, 0.6394599 , 0.74042725, 0.76682789,
           0.80713422, 0.81015719, 0.82527207, 0.83272874, 0.83413946])

``Triangle`` objects also have a ``derive_fields`` method that makes
adding new ``Cell`` fields easy:

..  code:: python

    >>> meyers_tri.derive_fields(paid_lr = lambda cell: cell["paid_loss"] / cell["earned_premium"])

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
       paid_lr
       reported_loss
     Common Metadata: 
       currency  USD 
       country  US 
       risk_basis  Accident 
       reinsurance_basis  Net 
       loss_definition  Loss+DCC

Note the new triangle field ``paid_lr`` in the ``Fields`` summary.
