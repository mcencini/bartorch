{{ (fullname | replace("bartorch.", "", 1) if fullname.count(".") > 1 else fullname) | escape | underline }}

.. currentmodule:: {{ module }}

{% set page = class_pages.get(fullname, {"members": [], "inherits": []}) %}
.. autoclass:: {{ objname }}
   :show-inheritance:
{% for directive, member in page["members"] %}
   .. {{ directive }}:: {{ member }}
{% endfor %}
{% if page["inherits"] %}
Also has the methods and properties of {% for base in page["inherits"] %}:class:`~{{ base }}`{% if not loop.last %}, {% endif %}{% endfor %}.
{% endif %}
{% if fullname in gallery_backreferences %}
.. minigallery:: {{ fullname }}
   :add-heading: Examples using ``{{ objname }}``
{% endif %}
