Data Structure Design  
======================

The data structures in Bermuda were created
with the following design objectives in mind:

-  **Unity/Consistency**: The primary user-facing class is
   ``bermuda.Triangle``. Any amount of triangle data, from a
   small paid loss triangle for a single book of business to a massive,
   heterogeneous collection of industry loss data is representable as a
   single ``Triangle`` object. There are no artificial distinctions
   between a single triangle and a collection of triangles, or
   restrictions on what sorts of triangle data can be concatenated. Even
   predictive distributions of unobserved quantities can be handled by
   this class.
-  **Immutability**: Objects in ``bermuda.triangle`` are immutable,
   meaning that there are no methods or functions that change the value
   of an object in-place. Methods on triangles such as ``filter``,
   ``clip``, and ``select`` return modified copies of the original
   triangle. This makes reasoning about the behavior of triangle objects
   significantly easier.
-  **Ergonomics**: Common manipulations on triangles are provided, and
   the internals of triangles are designed to make more common
   manipulations and transformations relatively straightforward.
-  **Performance**: The ``bermuda.Triangle`` class is designed to
   scale reasonably well, and to not be a performance or memory
   bottleneck in the modeling pipeline.
