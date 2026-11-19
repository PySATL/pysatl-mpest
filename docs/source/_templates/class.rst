{# templates/autosummary/class.rst #}
{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
   :show-inheritance:

{% set public_methods = [] %}
{% for item in methods %}
    {% if not item.startswith('__') %}
        {% set _ = public_methods.append(item) %}
    {% endif %}
{% endfor %}

{% if public_methods %}

.. autosummary::
   :toctree: .
   :template: method.rst
   :hidden:

{% for item in public_methods %}
   ~{{ name }}.{{ item }}
{% endfor %}


{% endif %}
