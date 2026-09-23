{{ (fullname | replace("bartorch.", "", 1) if fullname.count(".") > 1 else fullname) | escape | underline }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}
{% if fullname in gallery_backreferences %}
.. minigallery:: {{ fullname }}
   :add-heading: Examples using ``{{ objname }}``
{% endif %}
