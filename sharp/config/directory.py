"""
The canonical way to share variables across modules, that can be edited at
runtime: a separate module.
(https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules)

sharp.cmdline.worker sets config_dir, after which other modules can use it.
(Notably sharp.config.load, which adds the config_dir to the Python path,
so that we can import from the user specified config.py file).
"""

from pathlib import Path

config_dir: Path = ...
# Globally usable object.
