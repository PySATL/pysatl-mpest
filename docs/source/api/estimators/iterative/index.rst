.. _api.estimators.direct:

###############
mpest.estimators.iterative
###############

.. automodule:: rework_pysatl_mpest.estimators.iterative
   :no-members:

Abstract Classes
================

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~Pipeline
   ~PipelineStep
   ~PipelineState
   ~Breakpointer
   ~Pruner

Available Implementations
-------------------------

The concrete implementations of the base classes, available for use when
building a pipeline, will be listed below.

Steps (`PipelineStep`):
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~ExpectationStep
   ~MaximizationStep

Breakpointers (`Breakpointer`):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~StepBreakpointer

Pruners (`Pruner`):
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~PriorThresholdPruner

Logging Implementations
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~IterationsHistory
   ~IterationRecord

Usage Example
-------------
