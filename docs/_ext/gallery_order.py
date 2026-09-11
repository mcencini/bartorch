"""Sort key for the gallery: by lesson number, ignoring the ``plot_`` or ``optional_`` prefix."""

from pathlib import Path


class ExampleOrder:
    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __call__(self, filename):
        return Path(filename).name.partition("_")[2]
