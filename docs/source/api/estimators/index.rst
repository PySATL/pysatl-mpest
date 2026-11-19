.. _api.estimators:

###############
mpest.estimators
###############

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   direct/index
   iterative/index

.. automodule:: pysatl_mpest.estimators
   :no-members:

------------------

Facades (High-Level API)
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: em.rst

    ~EM

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

    ~ECM
    ~ECME

.. note::

   ``EM`` is the standalone Expectation-Maximization estimator. ``ECM`` performs conditional maximization over ordered parameter groups passed to ``fit`` as ``parameter_groups`` and matches ``EM`` when they are omitted; ``ECME`` additionally allows per-component Q-function or observed-data-likelihood maximization.

Abstract Classes
================

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~BaseEstimator
