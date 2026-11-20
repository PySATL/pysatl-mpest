{# templates/autosummary/em.rst #}
{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
   :show-inheritance:

.. autosummary::
   :toctree: .
   :template: method.rst

   ~{{ name }}.fit
