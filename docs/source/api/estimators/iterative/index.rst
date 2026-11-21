.. _api.estimators.iterative:

##########################
mpest.estimators.iterative
##########################

.. automodule:: pysatl_mpest.estimators.iterative
   :no-members:

Pipeline
========

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~Pipeline
   ~PipelineState

Base Classes
============

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~PipelineStep
   ~Breakpointer
   ~Pruner

Implementations
===============

Concrete steps, stopping criteria and pruners for building a pipeline.

Steps (`PipelineStep`):
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~ExpectationStep
   ~WeightsStep
   ~MaximizationStep

M-step configuration (`MaximizationStep`):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~OptimizationBlock
   ~MaximizationStrategy

Breakpointers (`Breakpointer`):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~StepBreakpointer
   ~LikelihoodBreakpointer

Pruners (`Pruner`):
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~PriorThresholdPruner
