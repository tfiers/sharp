from typing import List

from matplotlib.figure import Figure
import numpy as np

import fileflow
from sharp.config.spec import SharpConfig
from sharp.datatypes.base import ArrayFile, FigureFile, MultichannelArrayFile
from sharp.datatypes.segments import SegmentArray, MultichannelSegmentArrayFile


config = SharpConfig.load()

sharp_workflow = fileflow.Workflow(config)

sharp_workflow.register_filetype([float, int], ArrayFile)
sharp_workflow.register_filetype(List[np.ndarray], MultichannelArrayFile)
sharp_workflow.register_filetype(
    List[SegmentArray], MultichannelSegmentArrayFile
)
sharp_workflow.register_filetype(Figure, FigureFile)
