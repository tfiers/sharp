from preload import preload

preload(["scipy.signal", "fklab.segments", "matplotlib.pyplot"])


from matplotlib.figure import Figure

import fileflow
from sharp.config_spec import SharpConfig
from sharp.datatypes.base import FigureFile


config = SharpConfig.load()
sharp_workflow = fileflow.Workflow(config)
sharp_workflow.register_filetype(Figure, FigureFile)
